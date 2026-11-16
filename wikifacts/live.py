"""Live Wikidata API fallback for names that are not in the local store.

If lookup in the local store fails, we search the live API. Any entity found there is checked with the
same rules as cached ones and then *written into the store*, so it is only
fetched once and later runs stay reproducible.

Use it for evaluation only. Inside RL keep `allow_live=False` (the default in claims.py):
a store-only lookup keeps the reward fixed and fast.
"""

from __future__ import annotations

import time

import requests

from wikifacts.relations import Relation
from wikifacts.sparql import USER_AGENT
from wikifacts.store import FactStore

API = "https://www.wikidata.org/w/api.php"

MIN_INTERVAL = 0.5   # seconds between API calls (the API rate-limits bursts with HTTP 429)
_last_call = [0.0]


def _get(params: dict, retries: int = 5) -> dict:
    """GET the Wikidata API, pacing calls and retrying rate-limited / failed requests."""
    for attempt in range(retries):
        wait = _last_call[0] + MIN_INTERVAL - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
        r = requests.get(API, params={**params, "format": "json"},
                         headers={"User-Agent": USER_AGENT}, timeout=30)
        if r.status_code == 429 or r.status_code >= 500:
            retry_after = r.headers.get("Retry-After", "")
            time.sleep(int(retry_after) if retry_after.isdigit() else 5 * 2**attempt)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()
    raise RuntimeError(f"Wikidata API still failing after {retries} tries")


def search(name: str, limit: int = 5) -> list[str]:
    """wbsearchentities: candidate QIDs for an English name (matches labels and aliases)."""
    data = _get({"action": "wbsearchentities", "search": name, "language": "en",
                 "type": "item", "limit": limit})
    return [hit["id"] for hit in data.get("search", [])]


def get_entities(qids: list[str]) -> dict[str, dict]:
    """wbgetentities: full JSON for up to 50 QIDs."""
    if not qids:
        return {}
    data = _get({"action": "wbgetentities", "ids": "|".join(qids[:50]),
                 "props": "labels|aliases|descriptions|claims|sitelinks", "languages": "en|mul"})
    return data.get("entities", {})


def _claim_values(entity: dict, pid: str) -> list[str]:
    """Values of a property: QIDs for items, year strings for times."""
    out = []
    for claim in entity.get("claims", {}).get(pid, []):
        snak = claim.get("mainsnak", {})
        if snak.get("snaktype") != "value":
            continue
        dv = snak["datavalue"]
        if dv["type"] == "wikibase-entityid":
            out.append(dv["value"]["id"])
        elif dv["type"] == "time":
            out.append(dv["value"]["time"][1:5])  # '+1974-05-01T00:00:00Z' -> '1974'
    return out


def _label(entity: dict) -> str:
    """English label, falling back to the language-neutral 'mul' label (see sparql.LABEL_LANGS)."""
    labels = entity.get("labels", {})
    return (labels.get("en") or labels.get("mul") or {}).get("value", "")


def _aliases(entity: dict) -> list[str]:
    """English and 'mul' aliases."""
    aliases = entity.get("aliases", {})
    return [a["value"] for lang in ("en", "mul") for a in aliases.get(lang, [])]


def is_instance_of(entity: dict, cls: str, depth: int = 3) -> bool:
    """P31 is `cls` or a subclass of it (following P279 up to `depth` levels), like
    the bulk download's `wdt:P31/wdt:P279*`."""
    frontier = set(_claim_values(entity, "P31"))
    seen: set[str] = set()
    for _ in range(depth + 1):
        if cls in frontier:
            return True
        seen |= frontier
        frontier = {c for e in get_entities(sorted(frontier)[:50]).values()
                    for c in _claim_values(e, "P279")} - seen
        if not frontier:
            return False
    return False


def fetch_into_store(name: str, rel: Relation, store: FactStore) -> list[str]:
    """Search `name` live; add matching subjects (with their facts for `rel`) to the store.

    Returns the QIDs that were added. Only entities that are instances of the
    relation's subject class and have an English Wikipedia article are kept,
    exactly like the bulk SPARQL download.
    """
    added = []
    for qid, ent in get_entities(search(name)).items():
        if not is_instance_of(ent, rel.subject_class):
            continue
        enwiki = ent.get("sitelinks", {}).get("enwiki")
        if not enwiki:
            continue
        store.add_entity(
            qid=qid, label=_label(ent),
            description=ent.get("descriptions", {}).get("en", {}).get("value", ""),
            subject_class=rel.subject_class, sitelinks=len(ent.get("sitelinks", {})),
            enwiki_title=enwiki["title"].replace(" ", "_"), aliases=_aliases(ent),
        )
        values = _claim_values(ent, rel.pid)
        if rel.answer_type == "year":
            for year in values:
                store.add_fact(qid, rel.name, year)
        else:
            answers = get_entities(values)
            # Simplification vs the bulk build: only the direct P131 parent is accepted here,
            # and coarse answers are not filtered (live lookups are for evaluation only).
            parents = get_entities([p for a in answers.values() for p in _claim_values(a, "P131")[:1]])
            for pqid, pent in parents.items():
                store.add_answer(pqid, _label(pent), _aliases(pent))
            for aqid, aent in answers.items():
                if not _label(aent):
                    continue
                store.add_answer(aqid, _label(aent), _aliases(aent))
                if rel.is_place:
                    store.add_answer_parents(aqid, [p for p in _claim_values(aent, "P131")[:1] if p in parents])
                store.add_fact(qid, rel.name, _label(aent), answer_qid=aqid)
        added.append(qid)
    store.commit()
    return added
