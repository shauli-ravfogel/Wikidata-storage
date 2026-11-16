"""Is an answer correct? Compare it with a fact's gold answer set.

1. String match (always): a normalized gold answer appears in the normalized answer as a
   whole phrase, or matches it up to word order. Years: compare the first year mentioned.
2. LLM judge (optional): re-check answers the string match rejected, to recover false
   negatives ("NYC" vs "New York City").
"""

from __future__ import annotations

from typing import Callable

from wikifacts.normalize import extract_year, norm, sorted_key


def string_match(answer: str, gold_answers: list[str], answer_type: str = "entity") -> bool:
    """True if `answer` matches any of `gold_answers`.

    >>> string_match("He was born in Recife, Pernambuco.", ["Recife"])
    True
    >>> string_match("January 30, 1975", ["1975"], answer_type="year")
    True
    """
    if answer_type == "year":
        year = extract_year(answer)
        return year is not None and year in {extract_year(g) for g in gold_answers}

    # Only the first line counts: models sometimes keep talking after the answer.
    first_line = answer.strip().split("\n")[0]
    padded = f" {norm(first_line)} "
    answer_sorted = sorted_key(first_line)
    for gold in gold_answers:
        g = norm(gold)
        if g and (f" {g} " in padded or sorted_key(gold) == answer_sorted):
            return True
    return False


JUDGE_TEMPLATE = """Question: {question}
Accepted answers: {gold}
Model answer: {answer}

Does the model answer refer to the same thing as one of the accepted answers? Reply with "yes" or "no"."""


def llm_judge(generate: Callable[[list[str]], list[str]],
              items: list[tuple[str, str, list[str]]]) -> list[bool]:
    """LLM-judge verdicts for [(question, answer, gold_answers)].

    `generate` maps a list of prompts to one (greedy, short) completion each.
    """
    if not items:
        return []
    prompts = [JUDGE_TEMPLATE.format(question=q, gold="; ".join(g[:10]), answer=a) for q, a, g in items]
    return [out.strip().lower().startswith("yes") for out in generate(prompts)]
