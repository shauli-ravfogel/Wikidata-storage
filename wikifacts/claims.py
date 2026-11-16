"""Check claims against the fact store.

Two questions are answered here:

  1. Is the entity a claim talks about in the store, with a usable gold answer?
         "I don't know the place of birth of Juninho Pernambucano."
         -> ClaimCheck(valid, fact=<Juninho, place_of_birth, gold={Recife, Pernambuco, ...}>)
     (`ClaimChecker.check`, `ClaimChecker.check_pair` for "I know ... / I don't know ..." pairs)

  2. Is a stated answer true?
         ("Juninho Pernambucano", place_of_birth, "Recife") -> AnswerCheck(correct=True)
     (`ClaimChecker.verify_answer`)

A claim is any text whose last relevant sentence contains "<relation phrase> of <name>"
("I know the director of Alphaville (1965).", "What is the place of birth of Pelé?").
A year after a title ("Brothers (2015)") is used to tell same-named works apart.

Rejection reasons (`ClaimCheck.reason`), checked in this order:
    unparseable    no "<phrase> of <name>" sentence (or the name is an echoed placeholder)
    not_found      no entity of the relation's subject class has this name
    lookup_failed  the live Wikidata lookup failed (network): unknown, NOT "not found"
    ambiguous      several entities share the name and neither the year nor popularity singles one out
    no_answer      the entity has no value for the relation
    multi_valued   several values for a relation marked single-valued
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from wikifacts.grading import string_match
from wikifacts.normalize import norm
from wikifacts.relations import Relation
from wikifacts.store import Fact, FactStore

# --------------------------------------------------------------------------- parsing

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
# Trailing junk after the name: " with 50% confidence.", "?", quotes, periods, markdown stars.
_TAIL = re.compile(r"\s+with\s+\d+\s*%.*$|[\s\"'.?!*]+$", re.IGNORECASE)
_PLACEHOLDER = re.compile(r"<([^<>]*)>")
# Placeholder words a model may echo instead of naming something ("<full name>", "<film title>").
GENERIC_PLACEHOLDERS = {"full name", "name", "title", "film title", "book title", "title of the work",
                        "name of the region", "release year", "publication year", "year"}
_YEAR_SUFFIX = re.compile(r"\s*\((\d{4})[^()]*\)$")   # "Brothers (2015)", "Brothers (2015 film)"
MAX_NAME_WORDS = 12


@dataclass
class ParsedClaim:
    name: Optional[str]          # entity name; None if the text makes no parseable claim
    reasoning: str = ""          # text before the claim sentence (e.g. chain of thought)
    year: Optional[str] = None   # "(1968)" after a title, used to disambiguate


def parse_claim(text: str, relation: Relation) -> ParsedClaim:
    """Extract the entity name from the LAST sentence containing "<relation.phrase> of <name>"."""
    lines = [line.strip() for line in _THINK.sub("", text).splitlines() if line.strip()]
    # Greedy prefix: the LAST "<phrase> of" on a line wins, since models often put reasoning and
    # the final sentence on one line ("... by Dario Argento. I don't know the director of Suspiria.").
    pattern = re.compile(r"^(.*)" + re.escape(relation.phrase) + r"\s+of\s+(.+)$", re.IGNORECASE)
    placeholders = GENERIC_PLACEHOLDERS | {w.lower() for w in _PLACEHOLDER.findall(relation.placeholder)}

    for i in range(len(lines) - 1, -1, -1):         # with reasoning first, the claim comes last
        m = pattern.search(lines[i])
        if not m:
            continue
        # Same-line reasoning ends at the last sentence boundary before the claim.
        prefix = m.group(1)
        cut = max(prefix.rfind(c) for c in ".!?")
        reasoning = "\n".join([*lines[:i], prefix[:cut + 1] if cut >= 0 else ""]).strip()

        name = m.group(2)
        while (stripped := _TAIL.sub("", name)) != name:   # strip trailing punctuation / confidence clauses
            name = stripped
        name = name.strip(" \"'*")
        year = None
        if m_year := _YEAR_SUFFIX.search(name):
            year, name = m_year.group(1), name[:m_year.start()].strip(" \"'*")
        if name.startswith("<") and name.endswith(">"):     # "<Zindagi Na Milegi Dobara>" -> unwrap
            name = name[1:-1].strip()
        if (not name or len(name.split()) > MAX_NAME_WORDS or name.lower() in placeholders
                or _PLACEHOLDER.search(name)):              # an echoed placeholder is not a name
            return ParsedClaim(None, reasoning)
        return ParsedClaim(name, reasoning, year)
    return ParsedClaim(None, "\n".join(lines))


_KNOWN_LINE = re.compile(r"^\W*I\s+know\b", re.IGNORECASE)
_UNKNOWN_LINE = re.compile(r"^\W*I\s+(?:don['’]t|do\s+not)\s+know\b", re.IGNORECASE)


def split_pair(text: str, relation: Relation) -> tuple[str, str]:
    """The last "I know ..." line and the last "I don't know ..." line that name an entity (either order).

    Falls back to the last line of each kind (which will then fail to parse); "" if there is none.
    """
    lines = [line.strip() for line in _THINK.sub("", text).splitlines() if line.strip()]

    def last(kind: re.Pattern) -> str:
        candidates = [ln for ln in reversed(lines) if kind.match(ln)]
        for ln in candidates:
            if parse_claim(ln, relation).name:
                return ln
        return candidates[0] if candidates else ""

    return last(_KNOWN_LINE), last(_UNKNOWN_LINE)


# --------------------------------------------------------------------------- checking

# An ambiguous name resolves to its most popular candidate if that one has >= 10 sitelinks and
# at least 5x as many as the runner-up ("Martin Luther": the reformer, not the politician).
MIN_DOMINANT_SITELINKS = 10
DOMINANCE_RATIO = 5


@dataclass
class ClaimCheck:
    """Result of checking one claim. Valid iff `fact` is set."""

    text: str
    name: Optional[str] = None        # parsed entity name
    fact: Optional[Fact] = None       # the fact the claim is about (with gold answers)
    reason: Optional[str] = None      # why the claim was rejected (None if valid)
    reasoning: str = ""               # text before the claim sentence

    @property
    def valid(self) -> bool:
        return self.fact is not None


@dataclass
class AnswerCheck:
    """Result of checking an answer ("Recife") to a question about a named subject."""

    in_store: bool                    # the subject was found, with a gold answer
    correct: Optional[bool] = None    # None if not in the store (truth unknown)
    fact: Optional[Fact] = None
    reason: Optional[str] = None      # why the subject could not be used (see ClaimCheck reasons)


class ClaimChecker:
    """Checks claims and answers against a FactStore.

        checker = ClaimChecker(FactStore("store.sqlite"), load_relations())
        checker.check("I don't know the director of Suspiria (1977).", "film_director")
        checker.verify_answer("Suspiria (1977)", "film_director", "Dario Argento")

    `allow_live=True` searches the live Wikidata API for names missing from the store
    (and adds what it finds to the store). Keep it off inside RL: slow, and the reward
    would change as the store grows.
    """

    def __init__(self, store: FactStore, relations: dict[str, Relation], allow_live: bool = False):
        self.store, self.relations, self.allow_live = store, relations, allow_live
        # (relation, normalized name, year) -> (fact, reason). RL samples repeat the same names
        # thousands of times; looking each up once is much faster. Not used with live lookups,
        # which change the store as they go.
        self._memo: dict[tuple, tuple[Optional[Fact], Optional[str]]] = {}

    def _relation(self, relation: str | Relation) -> Relation:
        return relation if isinstance(relation, Relation) else self.relations[relation]

    def resolve(self, name: str, relation: Relation, year: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
        """Name -> (qid, None) or (None, reason). `year` only breaks ties, never rejects a unique match."""
        qids = self.store.lookup(name, relation.subject_class)
        if not qids and self.allow_live:
            from wikifacts.live import fetch_into_store     # network; imported only when needed
            try:
                fetch_into_store(name, relation, self.store)
            except Exception as e:                           # we don't know whether it exists
                print(f"  live lookup failed for {name!r}: {e}")
                return None, "lookup_failed"
            qids = self.store.lookup(name, relation.subject_class)
        if not qids:
            return None, "not_found"
        if len(qids) == 1:
            return qids[0], None

        entities = {q: self.store.entity(q) for q in qids}
        if year:      # keep candidates whose description / label mentions the year ("1968 film by ...")
            same_year = [q for q, e in entities.items() if year in f"{e['description']} {e['label']}"]
            if len(same_year) == 1:
                return same_year[0], None
        ranked = sorted(qids, key=lambda q: entities[q]["sitelinks"] or 0, reverse=True)
        top, second = (entities[q]["sitelinks"] or 0 for q in ranked[:2])
        if top >= MIN_DOMINANT_SITELINKS and top >= DOMINANCE_RATIO * max(second, 1):
            return ranked[0], None
        return None, "ambiguous"

    def find_fact(self, name: str, relation: str | Relation,
                  year: Optional[str] = None) -> tuple[Optional[Fact], Optional[str]]:
        """Entity name -> (Fact, None) if it is in the store with a usable answer, else (None, reason)."""
        rel = self._relation(relation)
        key = (rel.name, norm(name), year)
        if not self.allow_live and key in self._memo:
            return self._memo[key]
        fact, reason = None, None
        qid, reason = self.resolve(name, rel, year)
        if qid is not None:
            fact = self.store.get_fact(qid, rel)
            if fact is None:
                reason = "no_answer"
            elif not rel.multi_valued and len(self.store.answer_values(qid, rel.name)) > 1:
                fact, reason = None, "multi_valued"
        if not self.allow_live:
            self._memo[key] = (fact, reason)
        return fact, reason

    def check(self, text: str, relation: str | Relation) -> ClaimCheck:
        """Parse a claim ("I don't know the <phrase> of <name>.") and look its entity up."""
        rel = self._relation(relation)
        parsed = parse_claim(text, rel)
        if parsed.name is None:
            return ClaimCheck(text, reason="unparseable", reasoning=parsed.reasoning)
        fact, reason = self.find_fact(parsed.name, rel, parsed.year)
        return ClaimCheck(text, parsed.name, fact, reason, parsed.reasoning)

    def check_pair(self, text: str, relation: str | Relation) -> tuple[ClaimCheck, ClaimCheck]:
        """Check a two-line answer: ("I know <phrase> of A.", "I don't know <phrase> of B.")."""
        rel = self._relation(relation)
        known_line, unknown_line = split_pair(text, rel)
        return self.check(known_line, rel), self.check(unknown_line, rel)

    def verify_answer(self, subject: str, relation: str | Relation, answer: str) -> AnswerCheck:
        """Is `answer` the correct value of `relation` for the subject named `subject`?

        `subject` may carry a year for works ("Brothers (2015)"). String matching only;
        see grading.llm_judge for a fallback on answers this rejects.
        """
        rel = self._relation(relation)
        year = None
        if m := _YEAR_SUFFIX.search(subject):
            year, subject = m.group(1), subject[:m.start()].strip()
        fact, reason = self.find_fact(subject, rel, year)
        if fact is None:
            return AnswerCheck(in_store=False, reason=reason)
        return AnswerCheck(True, string_match(answer, fact.gold_answers, rel.answer_type), fact)
