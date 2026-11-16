"""SQLite fact store: the single source of gold answers.

Schema (one file; ~2 GB for ~40 relations):

    entities(qid, label, description, subject_class, sitelinks, enwiki_title, pageviews)
        Subjects we can ask about (people, films, ...). `sitelinks` = number of Wikipedia
        language editions, a popularity proxy.
    names(qid, name_norm, name_sorted)
        Label + aliases of each subject, normalized two ways (see normalize.py), so
        lookup tolerates accents, punctuation and word order.
    facts(subject_qid, relation, answer_qid, answer_value)
        One row per (subject, relation, value). answer_qid is NULL for years.
    answers(qid, label, aliases_json)
        Answer entities (cities, people, ...) with their aliases.
    answer_parents(qid, parent_qid)
        For places: enclosing administrative units that also count as correct
        (Benimàmet -> Valencia), up to but not including states and countries.

Writes go through the `add_*` methods (used by build.py and live.py); reads return `Fact`s.
"""

from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

from wikifacts.normalize import norm, sorted_key
from wikifacts.relations import Relation

# Very short aliases ("CA", "NY") match unrelated answers after normalization
# ("Los Angeles, CA"), so only answer aliases of 4+ characters are accepted.
MIN_ALIAS_CHARS = 4

SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    qid TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    description TEXT,
    subject_class TEXT,
    sitelinks INTEGER,
    enwiki_title TEXT,
    pageviews REAL
);
CREATE TABLE IF NOT EXISTS names (
    qid TEXT NOT NULL,
    name_norm TEXT NOT NULL,
    name_sorted TEXT NOT NULL,
    UNIQUE (qid, name_norm)
);
CREATE INDEX IF NOT EXISTS idx_names_norm ON names(name_norm);
CREATE INDEX IF NOT EXISTS idx_names_sorted ON names(name_sorted);
CREATE TABLE IF NOT EXISTS facts (
    subject_qid TEXT NOT NULL,
    relation TEXT NOT NULL,
    answer_qid TEXT,
    answer_value TEXT NOT NULL,
    UNIQUE (subject_qid, relation, answer_value)
);
CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject_qid, relation);
CREATE INDEX IF NOT EXISTS idx_facts_relation ON facts(relation);
CREATE TABLE IF NOT EXISTS answers (
    qid TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    aliases_json TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS answer_parents (
    qid TEXT NOT NULL,
    parent_qid TEXT NOT NULL,
    UNIQUE (qid, parent_qid)
);
"""


@dataclass
class Fact:
    """A verifiable (subject, relation, gold answers) triple."""

    subject_qid: str              # Wikidata id, e.g. "Q312837"
    subject_label: str            # e.g. "Juninho Pernambucano"
    subject_description: str      # e.g. "Brazilian footballer" (disambiguates the question)
    relation: str                 # relation name, e.g. "place_of_birth"
    gold_answers: list[str]       # every accepted string: labels, aliases, enclosing units, ...
    answer_qids: list[str] = field(default_factory=list)  # Wikidata ids of the answer(s), if entities
    popularity: Optional[float] = None                    # sitelinks of the subject

    @property
    def key(self) -> str:
        """Stable id, e.g. "Q312837|place_of_birth" (used to cache answers)."""
        return f"{self.subject_qid}|{self.relation}"

    def to_dict(self) -> dict:
        return asdict(self)


class FactStore:
    """Thin wrapper around the SQLite file. Use ':memory:' for tests."""

    def __init__(self, path: str):
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    # ------------------------------------------------------------------ writes

    def add_entity(self, qid: str, label: str, description: str = "", subject_class: str = "",
                   sitelinks: int = 0, enwiki_title: str = "", aliases: Iterable[str] = ()) -> None:
        """Insert or update a subject entity together with its searchable names."""
        self.conn.execute(
            "INSERT INTO entities(qid, label, description, subject_class, sitelinks, enwiki_title) "
            "VALUES (?,?,?,?,?,?) ON CONFLICT(qid) DO UPDATE SET "
            "label=excluded.label, description=excluded.description, sitelinks=excluded.sitelinks",
            (qid, label, description, subject_class, sitelinks, enwiki_title),
        )
        self.add_names(qid, [label, *aliases])

    def add_names(self, qid: str, names: Iterable[str]) -> None:
        """Index extra names (aliases) for a subject."""
        rows = [(qid, norm(n), sorted_key(n)) for n in names if n and norm(n)]
        self.conn.executemany("INSERT OR IGNORE INTO names VALUES (?,?,?)", rows)

    def add_fact(self, subject_qid: str, relation: str, answer_value: str,
                 answer_qid: Optional[str] = None) -> None:
        """One (subject, relation, value) row. For years: answer_value '1974', answer_qid None."""
        self.conn.execute("INSERT OR IGNORE INTO facts VALUES (?,?,?,?)",
                          (subject_qid, relation, answer_qid, answer_value))

    def add_answer(self, qid: str, label: str, aliases: Iterable[str] = ()) -> None:
        """Insert or update an answer entity (e.g. a city) with its aliases."""
        self.conn.execute(
            "INSERT INTO answers VALUES (?,?,?) ON CONFLICT(qid) DO UPDATE SET "
            "label=excluded.label, aliases_json=excluded.aliases_json",
            (qid, label, json.dumps(sorted(set(aliases)))),
        )

    def add_answer_parents(self, qid: str, parent_qids: Iterable[str]) -> None:
        """Enclosing units of a place answer that also count as correct (their names via add_answer)."""
        self.conn.executemany("INSERT OR IGNORE INTO answer_parents VALUES (?,?)",
                              [(qid, p) for p in parent_qids])

    def remove_facts_with_answer(self, relation: str, answer_qid: str) -> None:
        """Drop facts whose answer is unusable (e.g. a country for an "in what city" question)."""
        self.conn.execute("DELETE FROM facts WHERE relation=? AND answer_qid=?", (relation, answer_qid))

    def set_pageviews(self, qid: str, views: float) -> None:
        self.conn.execute("UPDATE entities SET pageviews=? WHERE qid=?", (views, qid))

    # ------------------------------------------------------------------- reads

    def lookup(self, name: str, subject_class: Optional[str] = None) -> list[str]:
        """QIDs whose label or alias matches `name` (optionally only of one subject class).

        Tries the exact normalized form first; only if that finds nothing, the
        word-order-insensitive key.
        """
        for column, key in (("name_norm", norm(name)), ("name_sorted", sorted_key(name))):
            if not key:
                continue
            sql = f"SELECT DISTINCT n.qid FROM names n JOIN entities e ON e.qid = n.qid WHERE n.{column} = ?"
            args: list = [key]
            if subject_class:
                sql += " AND e.subject_class = ?"
                args.append(subject_class)
            qids = [r[0] for r in self.conn.execute(sql, args)]
            if qids:
                return qids
        return []

    def entity(self, qid: str) -> Optional[dict]:
        """Row of the entities table as a dict, or None."""
        keys = ["qid", "label", "description", "subject_class", "sitelinks", "enwiki_title", "pageviews"]
        row = self.conn.execute(f"SELECT {', '.join(keys)} FROM entities WHERE qid=?", (qid,)).fetchone()
        return dict(zip(keys, row)) if row else None

    def answer_values(self, subject_qid: str, relation: str) -> list[tuple[Optional[str], str]]:
        """All (answer_qid, answer_value) rows for a subject and relation."""
        return list(self.conn.execute(
            "SELECT answer_qid, answer_value FROM facts WHERE subject_qid=? AND relation=?",
            (subject_qid, relation)))

    def answer_names(self, answer_qid: str, include_parents: bool) -> list[str]:
        """Label + (long enough) aliases of an answer entity, plus its accepted enclosing units' names."""
        row = self.conn.execute("SELECT label, aliases_json FROM answers WHERE qid=?", (answer_qid,)).fetchone()
        if row is None:
            return []
        label, aliases_json = row
        names = [label, *(a for a in json.loads(aliases_json) if len(norm(a)) >= MIN_ALIAS_CHARS)]
        if include_parents:
            parents = self.conn.execute("SELECT parent_qid FROM answer_parents WHERE qid=?", (answer_qid,))
            for (parent,) in parents.fetchall():
                names += self.answer_names(parent, include_parents=False)
        return names

    def get_fact(self, subject_qid: str, relation: Relation) -> Optional[Fact]:
        """The Fact (with the full gold answer set) for a subject, or None if it has no value."""
        ent = self.entity(subject_qid)
        values = self.answer_values(subject_qid, relation.name)
        if ent is None or not values:
            return None
        gold: list[str] = []
        for answer_qid, value in values:
            gold.append(value)
            if answer_qid:
                gold += self.answer_names(answer_qid, include_parents=relation.is_place)
        return Fact(
            subject_qid=subject_qid,
            subject_label=ent["label"],
            subject_description=ent["description"] or "",
            relation=relation.name,
            gold_answers=sorted(set(gold)),
            answer_qids=sorted({q for q, _ in values if q}),
            popularity=ent["sitelinks"],
        )

    def sample_subjects(self, relation: str, n: int, seed: int = 0) -> list[str]:
        """n random subject QIDs that have a value for `relation` (deterministic given `seed`)."""
        rows = self.conn.execute("SELECT DISTINCT subject_qid FROM facts WHERE relation=?", (relation,))
        qids = sorted(r[0] for r in rows)
        random.Random(seed).shuffle(qids)
        return qids[:n]

    def counts(self) -> dict[str, int]:
        """Number of distinct subjects per relation."""
        return dict(self.conn.execute(
            "SELECT relation, COUNT(DISTINCT subject_qid) FROM facts GROUP BY relation").fetchall())
