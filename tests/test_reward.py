"""Claim rewards and the GRPO reward functions (with fake accuracies)."""

from wikifacts.answering import Answerer
from wikifacts.reward import (INVALID_REWARD, ClaimReward, PairedClaimReward, answerer_accuracy,
                              claim_reward, is_unknown_claim, pair_reward)

KNOWN = "I know the place of birth of Frédéric Chopin."
UNKNOWN = "I don't know the place of birth of Juninho Pernambucano."
ACC = {"Q6|place_of_birth": 1.0, "Q1|place_of_birth": 0.25}     # Chopin known, Juninho mostly not
accuracy = lambda facts: [ACC[f.key] for f in facts]             # noqa: E731


def test_scalar_rewards():
    assert claim_reward(True, 0.75) == 0.75 and claim_reward(False, 0.75) == 0.25
    assert claim_reward(True, None) == INVALID_REWARD
    assert pair_reward(1.0, 0.25) == (1.0 + 0.75) / 2
    assert pair_reward(None, 0.25) == INVALID_REWARD


def test_paired_claim_reward(checker):
    reward = PairedClaimReward(checker, accuracy)
    swap = lambda t: t.replace("Frédéric Chopin", "TMP").replace("Juninho Pernambucano", "Frédéric Chopin").replace("TMP", "Juninho Pernambucano")  # noqa: E731
    completions = [f"{KNOWN}\n{UNKNOWN}",                                   # right both times
                   swap(f"{KNOWN}\n{UNKNOWN}"),                             # wrong both times
                   f"{KNOWN}\n" + UNKNOWN.replace("Juninho Pernambucano", "Frédéric Chopin"),   # same entity
                   f"{KNOWN}\n" + UNKNOWN.replace("Juninho Pernambucano", "Carlos Inventado"),  # not in store
                   "no idea"]
    out = reward(["p"] * 5, completions, ["place_of_birth"] * 5)
    assert out == [0.875, 0.125, INVALID_REWARD, INVALID_REWARD, INVALID_REWARD]


def test_single_claim_reward(checker):
    reward = ClaimReward(checker, accuracy)
    chat = [{"role": "assistant", "content": UNKNOWN}]                    # conversational completion
    out = reward(["p"] * 4, [KNOWN, UNKNOWN, chat, "I know the place of birth of John Smith."], ["place_of_birth"] * 4)
    assert out == [1.0, 0.75, 0.75, INVALID_REWARD]                      # John Smith is ambiguous
    assert is_unknown_claim("thinking...\nI do not know the place of birth of X.")
    assert not is_unknown_claim("I don't know much.\nI know the place of birth of X.")


def test_answerer_accuracy(checker, relations):
    model = lambda prompts, n, **kw: [["Recife", "Rio"] for _ in prompts]   # noqa: E731
    reward = ClaimReward(checker, answerer_accuracy(Answerer("fake", model), relations, k=2))
    assert reward(["p"], [UNKNOWN], ["place_of_birth"]) == [0.5]
