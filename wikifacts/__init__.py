"""wikifacts: a local Wikidata fact store, claim checking, and rewards for "do you know this fact?" claims.

    sparql.py     download facts from Wikidata (QLever SPARQL endpoint)
    build.py      CLI: build / resume the SQLite store            (python -m wikifacts.build)
    store.py      FactStore: name lookup and gold answer sets (Fact)
    live.py       live Wikidata API fallback for names missing from the store
    claims.py     ClaimChecker: is a claimed entity in the store? is an answer true?
    grading.py    answer vs gold answers (string match, optional LLM judge)
    answering.py  Answerer: ask a model about facts k times -> accuracy
    reward.py     claim rewards (knows what it claims to know?) as TRL/GRPO reward functions
    relations.py  relation definitions (relations.yaml ships with the package)
"""

from wikifacts.claims import AnswerCheck, ClaimCheck, ClaimChecker, parse_claim
from wikifacts.grading import string_match
from wikifacts.relations import Relation, load_relations
from wikifacts.store import Fact, FactStore

__all__ = ["AnswerCheck", "ClaimCheck", "ClaimChecker", "Fact", "FactStore", "Relation",
           "load_relations", "parse_claim", "string_match"]
