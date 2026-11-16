"""Bulk download of facts from Wikidata (the "scraper").

We query QLever's Wikidata endpoint rather than query.wikidata.org: WDQS kills
queries after 60 s, which rules out bulk extraction (e.g. ~1.5M humans with a
place of birth and an English Wikipedia article), while QLever returns them in
one query in seconds. Results are streamed as TSV.

Queries per relation:
    fetch_facts            (subject, value) rows for subjects of the right class with an enwiki article
    fetch_subject_aliases  English aliases of those subjects (for name lookup)
    fetch_answer_info      labels and aliases of the answer entities
    fetch_place_ancestors  (places only) enclosing units that also count as correct
plus fetch_coarse_places once per build (countries / first-level divisions are never correct
place answers).

QLever serves one recent Wikidata snapshot; build.py records the build date next to the
store, because the store is a dated snapshot.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from typing import Iterator, Optional

import requests

from wikifacts.relations import Relation

# Generic client name: Wikimedia asks automated clients to identify themselves.
USER_AGENT = "wikifacts/0.1 (academic research; python-requests)"

ENDPOINT = "https://qlever.dev/api/wikidata"
# QLever is a shared public service and rate-limits bursts of heavy queries (HTTP 429),
# sometimes for hours. Only the one-time store build queries it, unattended, so we are
# polite and very patient: a 30 s pause between queries, and up to MAX_RETRIES retries
# with exponential backoff capped at MAX_WAIT (~4-5 h of throttling per query in total).
PAUSE_BETWEEN_QUERIES = 30  # seconds
MAX_RETRIES = 12
MAX_WAIT = 1800  # seconds

PREFIXES = """PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX schema: <http://schema.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
"""

# Label languages we accept, in order of preference. Wikidata increasingly stores names only
# as the language-neutral "mul" label (e.g. Marie Curie has no "en" label), so requiring
# "en" silently drops many entities -- famous ones most of all.
LABEL_LANGS = ("en", "mul")


def _label(var: str, out: str, required: bool = True) -> str:
    """SPARQL snippet binding ?out to ?var's English label, falling back to its 'mul' label."""
    snippet = (f'OPTIONAL {{ ?{var} rdfs:label ?{out}_en . FILTER(LANG(?{out}_en) = "en") }}\n'
               f'  OPTIONAL {{ ?{var} rdfs:label ?{out}_mul . FILTER(LANG(?{out}_mul) = "mul") }}\n'
               f'  BIND(COALESCE(?{out}_en, ?{out}_mul) AS ?{out})')
    return snippet + (f"\n  FILTER(BOUND(?{out}))" if required else "")


def _alias(var: str, out: str) -> str:
    """SPARQL snippet binding ?out to each English or 'mul' alias of ?var (required)."""
    return f'?{var} skos:altLabel ?{out} . FILTER(LANG(?{out}) IN ("en", "mul"))'


# Subjects of the relation: instance of the class *or a subclass of it* (Spirited Away is an
# "anime film", not directly a "film"), has the property, has an English Wikipedia article.
_SUBJECTS = """
  ?s wdt:P31/wdt:P279* wd:{cls} ; wdt:{pid} ?v .
  ?article schema:about ?s ; schema:isPartOf <https://en.wikipedia.org/> .
"""


# --------------------------------------------------------------------------- transport

def _cell(raw: str) -> str:
    """One TSV cell -> plain value: '<http://.../Q42>' -> 'Q42', '"Paris"@en' -> 'Paris'."""
    if raw.startswith("<") and raw.endswith(">"):
        return raw[1:-1].rsplit("/", 1)[-1]
    if raw.startswith('"'):
        end = raw.rfind('"')
        return raw[1:end].replace('\\"', '"').replace("\\t", "\t").replace("\\n", " ")
    return raw


def _lines(response) -> Iterator[str]:
    """Split a streamed UTF-8 response on '\\n' only.

    (requests' iter_lines uses str.splitlines, which also splits on Unicode line
    separators such as U+2028 that occur inside Wikidata labels.)
    """
    buf = ""
    for chunk in response.iter_content(chunk_size=1 << 20, decode_unicode=True):
        buf += chunk
        *complete, buf = buf.split("\n")
        yield from complete
    if buf:
        yield buf


def run_query(query: str, retries: int = MAX_RETRIES) -> list[dict]:
    """Run one SELECT query; return all rows as dicts {variable: plain value} (missing = '').

    Rows are collected before returning, so a retry after a dropped connection never
    leaves partial results behind. NOTE: a dropped stream can also end *silently*, so
    this function alone cannot guarantee completeness -- large downloads must go
    through run_select, which checks row counts.
    """
    time.sleep(PAUSE_BETWEEN_QUERIES)
    for attempt in range(retries):
        try:
            with requests.post(ENDPOINT, data={"query": PREFIXES + query},
                               headers={"Accept": "text/tab-separated-values", "User-Agent": USER_AGENT},
                               stream=True, timeout=1800) as r:
                if r.status_code == 429:  # rate limited: wait as asked (or back off), then retry
                    retry_after = r.headers.get("Retry-After", "")
                    wait = int(retry_after) if retry_after.isdigit() else min(60 * 2**attempt, MAX_WAIT)
                    print(f"  rate limited by QLever, waiting {wait}s")
                    time.sleep(wait)
                    continue
                if 400 <= r.status_code < 500:  # a bad query never succeeds: fail now, with QLever's reason
                    raise ValueError(f"QLever rejected the query ({r.status_code}): {r.text[:500]}")
                r.raise_for_status()
                r.encoding = "utf-8"           # TSV has no charset header; requests would assume Latin-1
                lines = _lines(r)
                header = [h.lstrip("?") for h in next(lines).split("\t")]
                rows = []
                for line in lines:
                    if not line:
                        continue
                    cells = [_cell(c) for c in line.split("\t")]
                    cells += [""] * (len(header) - len(cells))   # unbound OPTIONAL vars may be dropped
                    rows.append(dict(zip(header, cells)))
                return rows
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            wait = min(30 * 2**attempt, MAX_WAIT)
            print(f"  SPARQL error ({e.__class__.__name__}), retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("QLever query failed: still rate limited after all retries")


PAGE_SIZE = 250_000


def _projected_vars(select: str) -> list[str]:
    """Output variables of a SELECT clause: plain ?vars plus the alias of each
    "(expression AS ?alias)" -- never the variables inside an aggregate like COUNT(?s)."""
    aliases = re.findall(r"AS\s+(\?\w+)\s*\)", select)
    plain = re.sub(r"\((?:[^()]|\([^()]*\))*\)", " ", select)   # drop (...) incl. one nested level
    return list(dict.fromkeys(re.findall(r"\?\w+", plain) + aliases))


def run_select(select: str, where: str, order_by: str, limit: Optional[int] = None,
               page_size: Optional[int] = None, group_by: str = "") -> Iterator[dict]:
    """Download a large result completely, in verified pages.

    1. COUNT the rows the query should return (capped at `limit`).
    2. Fetch pages with ORDER BY `order_by` (must make the order total) + LIMIT/OFFSET.
    3. Check each page has exactly the expected number of rows; retry short pages.
    4. Raise if a page stays incomplete -- never silently return a partial store.
    """
    inner = f"SELECT {select} WHERE {{\n{where}\n}} {group_by}"   # group_by: "GROUP BY ... HAVING ..."
    page_size = page_size or PAGE_SIZE
    # Paging is only safe with a *total* order: rows that tie on `order_by` (e.g. a subject
    # with two descriptions) could be skipped or repeated at page boundaries. So every
    # selected variable is appended as a tie-breaker.
    order_by = " ".join([order_by, *(v for v in _projected_vars(select) if v not in order_by.split())])
    total = int(run_query(f"SELECT (COUNT(*) AS ?_rows) WHERE {{ {inner} }}")[0]["_rows"])
    if limit is not None:
        total = min(total, limit)
    for offset in range(0, total, page_size):
        expected = min(page_size, total - offset)
        for attempt in range(4):
            rows = run_query(f"{inner}\nORDER BY {order_by} LIMIT {expected} OFFSET {offset}")
            if len(rows) == expected:
                break
            print(f"  incomplete page at offset {offset:,}: {len(rows):,}/{expected:,} rows, retrying")
        else:
            raise RuntimeError(f"page at offset {offset:,} still incomplete after retries; aborting")
        if total > page_size:
            print(f"  {offset + expected:,}/{total:,} rows")
        yield from rows


# --------------------------------------------------------------------------- queries

def fetch_facts(rel: Relation, limit: Optional[int] = None) -> Iterator[dict]:
    """Yield one dict per (subject, value) row.

    Keys: qid, label, description, sitelinks, enwiki_title, answer_qid, answer_value.
    For year relations answer_qid is None and answer_value is the year.
    With `limit` (smoke tests) the most popular subjects come first.
    """
    where = f"""
  {_SUBJECTS.format(cls=rel.subject_class, pid=rel.pid)}
  ?s wikibase:sitelinks ?sl .
  {_label("s", "sLabel")}
  OPTIONAL {{ ?s schema:description ?sDesc . FILTER(LANG(?sDesc) = "en") }}
  {_label("v", "vLabel", required=False)}"""
    order = "DESC(?sl) ?s ?v" if limit else "?s ?v"
    for b in run_select("?s ?sLabel ?sDesc ?sl ?article ?v ?vLabel", where, order, limit=limit):
        if rel.answer_type == "year":
            year = b["v"][:4]                    # '1974-05-01T00:00:00Z' -> '1974'
            if not year.isdigit():               # BCE dates ('-0469-...') and odd values
                continue
            answer_qid, answer_value = None, year
        else:
            if not b["vLabel"]:                  # answer has no English label: unusable
                continue
            answer_qid, answer_value = b["v"], b["vLabel"]
        yield {
            "qid": b["s"], "label": b["sLabel"], "description": b["sDesc"],
            "sitelinks": int(b["sl"] or 0), "enwiki_title": b["article"],
            "answer_qid": answer_qid, "answer_value": answer_value,
        }


def fetch_subject_aliases(rel: Relation) -> Iterator[tuple[str, str]]:
    """Yield (subject_qid, english_alias) for all subjects of the relation."""
    where = f"""
  {_SUBJECTS.format(cls=rel.subject_class, pid=rel.pid)}
  {_alias("s", "alias")}"""
    for b in run_select("DISTINCT ?s ?alias", where, "?s ?alias"):
        yield b["s"], b["alias"]


def fetch_answer_info(rel: Relation) -> dict[str, dict]:
    """{answer_qid: {label, aliases}} for every answer entity of the relation."""
    where = f"""
  {{ SELECT DISTINCT ?v WHERE {{ {_SUBJECTS.format(cls=rel.subject_class, pid=rel.pid)} }} }}
  {_label("v", "label")}
  OPTIONAL {{ {_alias("v", "alias")} }}"""
    info: dict[str, dict] = defaultdict(lambda: {"label": "", "aliases": set()})
    for b in run_select("DISTINCT ?v ?label ?alias", where, "?v ?alias"):
        d = info[b["v"]]
        d["label"] = b["label"]
        if b.get("alias"):
            d["aliases"].add(b["alias"])
    return dict(info)


# ------------------------------------------------------------------- place answers
#
# For places, an enclosing unit also counts as correct, but not the country. We generalize "one level up" to: every enclosing unit
# (following P131 "located in the administrative territorial entity") up to, but not
# including, the first COARSE unit, i.e. a first-level division (state, province) or a
# country. Example: Benimàmet -> Poblados del Oeste -> Valencia -> Province of Valencia
# are accepted; Valencian Community and Spain are not. Hamlet, NC -> Richmond County is
# accepted; North Carolina is not.
#
# Cities that are *also* first-level divisions (Berlin, Vienna, Moscow, Tokyo) are not
# coarse: anything that is a human settlement is never coarse.

COARSE_PLACE_CLASSES = ["Q6256", "Q3624078", "Q5107", "Q7275", "Q10864048"]
# country, sovereign state, continent, state, first-level administrative division
HUMAN_SETTLEMENT = "Q486972"
MAX_PARENT_DEPTH = 4


def fetch_coarse_places() -> set[str]:
    """All countries, continents and first-level divisions that are not also settlements (~11k)."""
    classes = " ".join(f"wd:{c}" for c in COARSE_PLACE_CLASSES)
    where = f"""
  ?x wdt:P31/wdt:P279* ?c . VALUES ?c {{ {classes} }}
  MINUS {{ ?x wdt:P31/wdt:P279* wd:{HUMAN_SETTLEMENT} }}"""
    return {b["x"] for b in run_select("DISTINCT ?x", where, "?x")}


def fetch_place_ancestors(rel: Relation, coarse: set[str]) -> dict[str, list[str]]:
    """{answer_qid: [accepted enclosing units]} for the place answers of `rel`.

    We download the direct P131 edges of every place in the answers' ancestor graph,
    then walk up from each answer, one level at a time, stopping at coarse units,
    at dissolved entities (P576, e.g. "Kingdom of Prussia" above Königsberg), and
    after MAX_PARENT_DEPTH levels.
    """
    where = f"""
  {{ SELECT DISTINCT ?v WHERE {{ {_SUBJECTS.format(cls=rel.subject_class, pid=rel.pid)} }} }}
  ?v wdt:P131* ?x .
  ?x wdt:P131 ?p .
  OPTIONAL {{ ?p wdt:P576 ?d . }}
  BIND(BOUND(?d) AS ?dissolved)"""
    parents: dict[str, set[str]] = defaultdict(set)
    blocked = set(coarse)
    for b in run_select("DISTINCT ?x ?p ?dissolved", where, "?x ?p ?dissolved"):
        parents[b["x"]].add(b["p"])
        if b["dissolved"] in ("true", "1"):
            blocked.add(b["p"])

    out: dict[str, list[str]] = {}
    for start in list(parents):
        accepted = walk_ancestors(start, parents, blocked)
        if accepted:
            out[start] = accepted
    return out


def walk_ancestors(start: str, parents: dict[str, set[str]], blocked: set[str],
                   max_depth: int = MAX_PARENT_DEPTH) -> list[str]:
    """Enclosing units of `start`, walking up level by level; never passes through a blocked unit."""
    accepted: set[str] = set()
    frontier = {start}
    for _ in range(max_depth):
        frontier = {p for x in frontier for p in parents.get(x, ()) if p not in blocked} - accepted
        if not frontier:
            break
        accepted |= frontier
    return sorted(accepted)


def fetch_labels(qids: list[str], batch: int = 2000) -> dict[str, dict]:
    """{qid: {label, aliases}} for a list of QIDs (used for parent admin units)."""
    out: dict[str, dict] = defaultdict(lambda: {"label": "", "aliases": set()})
    for i in range(0, len(qids), batch):
        values = " ".join(f"wd:{q}" for q in qids[i:i + batch])
        query = f"""
SELECT ?x ?label ?alias WHERE {{
  VALUES ?x {{ {values} }}
  {_label("x", "label")}
  OPTIONAL {{ {_alias("x", "alias")} }}
}}
"""
        for b in run_query(query):
            out[b["x"]]["label"] = b["label"]
            if b.get("alias"):
                out[b["x"]]["aliases"].add(b["alias"])
    return dict(out)
