"""Relation definitions (which Wikidata property, which subjects, how answers are graded).

The default set ships with the package in `relations.yaml`; pass your own file to
`load_relations` to use different relations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_RELATIONS_FILE = Path(__file__).with_name("relations.yaml")
ANSWER_TYPES = ("entity", "place", "year")


@dataclass(frozen=True)
class Relation:
    """One relation, e.g. place_of_birth = (P19, humans, place). See relations.yaml for field docs."""

    name: str
    pid: str                 # Wikidata property, e.g. "P19"
    subject_class: str       # Wikidata class of the subjects, e.g. "Q5" (human)
    answer_type: str         # "entity" | "place" | "year"
    phrase: str              # "the place of birth": claims read "... <phrase> of <subject>"
    question: str            # "In what city was {subject} ({description}) born?"
    placeholder: str = "<name>"   # subject slot in claim templates, e.g. "<full name>"
    multi_valued: bool = False    # if False, subjects with several values are rejected

    @property
    def is_place(self) -> bool:
        return self.answer_type == "place"


def load_relations(path: str | Path = DEFAULT_RELATIONS_FILE) -> dict[str, Relation]:
    """Return {relation_name: Relation} from a YAML file."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    relations = {}
    for name, fields in raw.items():
        rel = Relation(name=name, **fields)
        if rel.answer_type not in ANSWER_TYPES:
            raise ValueError(f"{name}: answer_type must be one of {ANSWER_TYPES}, got {rel.answer_type!r}")
        relations[name] = rel
    return relations
