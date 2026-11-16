"""Reward for knowledge claims: does the model know what it says it knows, and not know what it
says it doesn't?

A claim names an entity: "I know the director of Alphaville (1965)." or "I don't know the
director of Le Doulos (1962)." The claim is VALID if it parses and the entity is in the fact
store with a gold answer (claims.ClaimChecker). The model's accuracy `acc` on the fact is then
measured by asking it the templated question k times (answering.Answerer).

    "I know X"        reward = acc            (high if it really knows X)
    "I don't know X"  reward = 1 - acc        (high if it really doesn't)
    invalid claim     reward = INVALID_REWARD (made-up / unknown / ambiguous entity)

For a paired answer ("I know A." + "I don't know B.") the reward is the mean of the two claim
rewards, in [0, 1]; it is INVALID_REWARD if either claim is invalid or both name the same entity.

`PairedClaimReward` and `ClaimReward` are reward functions for TRL's GRPOTrainer.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from wikifacts.answering import Answerer
from wikifacts.claims import ClaimChecker
from wikifacts.relations import Relation
from wikifacts.store import Fact

# Below every valid reward (>= 0), so naming a made-up or unknown entity never pays.
INVALID_REWARD = -0.5

# The model's accuracy on each fact (fraction of correct samples), batched.
AccuracyFn = Callable[[list[Fact]], list[float]]


def claim_reward(claims_known: bool, acc: Optional[float]) -> float:
    """acc for "I know X", 1 - acc for "I don't know X"; INVALID_REWARD if acc is None (invalid claim)."""
    if acc is None:
        return INVALID_REWARD
    return acc if claims_known else 1.0 - acc


def pair_reward(acc_known: Optional[float], acc_unknown: Optional[float]) -> float:
    """Mean of the two claim rewards: (acc_known + 1 - acc_unknown) / 2, or INVALID_REWARD."""
    if acc_known is None or acc_unknown is None:
        return INVALID_REWARD
    return (claim_reward(True, acc_known) + claim_reward(False, acc_unknown)) / 2


def answerer_accuracy(answerer: Answerer, relations: dict[str, Relation], k: int = 8,
                      temperature: float = 1.0) -> AccuracyFn:
    """AccuracyFn that asks the model itself (via an Answerer, cached if it has a Cache)."""
    def accuracy(facts: list[Fact]) -> list[float]:
        if not facts:
            return []
        return [r.acc for r in answerer.answer(facts, relations, k=k, temperature=temperature)]
    return accuracy


def _text(completion) -> str:
    """TRL passes plain strings, or (conversational datasets) a list of chat messages."""
    return completion if isinstance(completion, str) else completion[-1]["content"]


def _accuracies(accuracy: AccuracyFn, facts: list[Fact]) -> dict[str, float]:
    """fact_key -> accuracy, with each distinct fact answered once."""
    unique = list({f.key: f for f in facts}.values())
    return dict(zip([f.key for f in unique], accuracy(unique)))


class PairedClaimReward:
    """TRL reward for paired answers: "I know <phrase> of A." + "I don't know <phrase> of B."

        checker = ClaimChecker(FactStore("store.sqlite"), load_relations())
        reward = PairedClaimReward(checker, answerer_accuracy(answerer, checker.relations))
        GRPOTrainer(..., reward_funcs=[reward], train_dataset=ds)   # ds needs a "relation" column
    """

    __name__ = "paired_claim_reward"      # TRL logs rewards under the function's name

    def __init__(self, checker: ClaimChecker, accuracy: AccuracyFn):
        self.checker, self.accuracy = checker, accuracy

    def __call__(self, prompts, completions, relation, **_) -> list[float]:
        pairs = [self.checker.check_pair(_text(c), rel) for c, rel in zip(completions, relation)]
        # Usable: both claims valid, about two different entities.
        usable = [k.valid and u.valid and k.fact.subject_qid != u.fact.subject_qid for k, u in pairs]
        acc = _accuracies(self.accuracy, [c.fact for (k, u), ok in zip(pairs, usable) if ok for c in (k, u)])
        return [pair_reward(acc[k.fact.key], acc[u.fact.key]) if ok else INVALID_REWARD
                for (k, u), ok in zip(pairs, usable)]


_DONT_KNOW = re.compile(r"\bI\s+(?:don['’]t|do\s+not)\s+know\b", re.IGNORECASE)


def is_unknown_claim(text: str) -> bool:
    """True if the claim (the last line of `text`) says "I don't know" / "I do not know"."""
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    return bool(lines) and _DONT_KNOW.search(lines[-1]) is not None


class ClaimReward:
    """TRL reward for single claims: "I know <phrase> of X." or "I don't know <phrase> of X."

    The claim type is read from the last line ("I don't know" / "I do not know" -> unknown claim,
    anything else -> known claim); pass `claims_known=True/False` to fix it instead.
    """

    __name__ = "claim_reward"

    def __init__(self, checker: ClaimChecker, accuracy: AccuracyFn, claims_known: Optional[bool] = None):
        self.checker, self.accuracy, self.claims_known = checker, accuracy, claims_known

    def __call__(self, prompts, completions, relation, **_) -> list[float]:
        texts = [_text(c) for c in completions]
        checks = [self.checker.check(t, rel) for t, rel in zip(texts, relation)]
        acc = _accuracies(self.accuracy, [c.fact for c in checks if c.valid])
        rewards = []
        for text, check in zip(texts, checks):
            known = self.claims_known if self.claims_known is not None else not is_unknown_claim(text)
            rewards.append(claim_reward(known, acc[check.fact.key] if check.valid else None))
        return rewards
