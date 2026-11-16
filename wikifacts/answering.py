"""Measure how often a model answers a fact correctly (its accuracy, the input to the reward).

The model is asked a TEMPLATED question built from the Fact alone, in a fresh context
("In what city was Juninho Pernambucano (Brazilian footballer) born?"), k times, and each
sample is graded against the gold answers.

Bring your own model: `Answerer` takes any function
    generate(prompts: list[str], n: int, temperature: float, max_tokens: int) -> list[list[str]]
(n samples per prompt). See the README for a vLLM example.

Only the raw samples are cached (per model, question, k, temperature), never the grades:
fixing a gold answer or the grader never requires re-running the model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from wikifacts.cache import Cache, make_key
from wikifacts.grading import llm_judge, string_match
from wikifacts.normalize import norm
from wikifacts.relations import Relation
from wikifacts.store import Fact

MAX_ANSWER_TOKENS = 24
ANSWER_INSTRUCTION = "Answer the question with just the answer (a name or a year), no explanation."

GenerateFn = Callable[..., list[list[str]]]

# ------------------------------------------------------------------------- questions

_YEARS_IN_PARENS = re.compile(r"\s*\([^)]*\d{3,4}[^)]*\)")
_ANY_YEAR = re.compile(r"\b\d{3,4}\b")


def safe_description(fact: Fact, relation: Relation) -> str:
    """The subject's Wikidata description, unless it gives the answer away.

    Descriptions often contain the answer ("1968 film by Stanley Kubrick", "Spanish artist
    (1904–1989)"). Parenthesised years are dropped; then the whole description is dropped if
    a gold answer (or, for year relations, any year) is still in it.
    """
    desc = _YEARS_IN_PARENS.sub("", fact.subject_description).strip()
    if relation.answer_type == "year" and _ANY_YEAR.search(desc):
        return ""
    padded = f" {norm(desc)} "
    if any(norm(g) and f" {norm(g)} " in padded for g in fact.gold_answers):
        return ""
    return desc


def question_for(fact: Fact, relation: Relation) -> str:
    """'In what city was Juninho Pernambucano (Brazilian footballer) born?'"""
    q = relation.question.format(subject=fact.subject_label, description=safe_description(fact, relation))
    return q.replace(" ()", "")


def answer_prompt(question: str, completion_mode: bool = False) -> str:
    """Wrap a question for the answering model (base models get a 'Q: ... A:' prompt)."""
    return f"Q: {question}\nA:" if completion_mode else f"{ANSWER_INSTRUCTION}\n\n{question}"


# ------------------------------------------------------------------------- answering

@dataclass
class AnswerResult:
    """k samples for one fact, graded."""

    model: str
    fact_key: str
    question: str
    samples: list[str]
    correct: list[bool]

    @property
    def acc(self) -> float:
        return sum(self.correct) / len(self.correct) if self.correct else float("nan")


class Answerer:
    """Asks one model about facts and grades its answers.

        answerer = Answerer("qwen3-14b", my_generate, cache=Cache("cache.sqlite"))
        results = answerer.answer(facts, relations, k=8)     # -> [AnswerResult]
        accs = [r.acc for r in results]

    `name` identifies the model in the cache: give fine-tuned variants their own name.
    `completion_mode=True` for base (non-chat) models: 'Q: ... A:' prompt, stop at newline.
    `judge`, if given, maps prompts to one completion each and re-checks string-match misses.
    """

    def __init__(self, name: str, generate: GenerateFn, cache: Optional[Cache] = None,
                 completion_mode: bool = False, judge: Optional[Callable[[list[str]], list[str]]] = None,
                 judge_name: str = "judge"):
        self.name, self.generate, self.cache = name, generate, cache
        self.completion_mode, self.judge, self.judge_name = completion_mode, judge, judge_name

    def sample(self, questions: list[str], k: int, temperature: float) -> list[list[str]]:
        """k answer samples per question, from the cache where possible."""
        keys = [make_key(kind="samples", model=self.name, question=q, k=k, temperature=temperature,
                         max_tokens=MAX_ANSWER_TOKENS) for q in questions]
        samples = {i: hit for i, key in enumerate(keys) if self.cache and (hit := self.cache.get(key)) is not None}
        todo = [i for i in range(len(questions)) if i not in samples]
        if todo:
            prompts = [answer_prompt(questions[i], self.completion_mode) for i in todo]
            kwargs = {"stop": ["\n"]} if self.completion_mode else {}
            outs = self.generate(prompts, n=k, temperature=temperature, max_tokens=MAX_ANSWER_TOKENS, **kwargs)
            for i, out in zip(todo, outs):
                samples[i] = [s.strip() for s in out]
            if self.cache:
                self.cache.put_many({keys[i]: samples[i] for i in todo})
        return [samples[i] for i in range(len(questions))]

    def _judge(self, items: list[tuple[str, str, list[str]]]) -> list[bool]:
        """LLM-judge verdicts, cached per (judge, question, answer, gold set)."""
        keys = [make_key(kind="judge", judge=self.judge_name, question=q, answer=a, gold=sorted(g))
                for q, a, g in items]
        verdicts = {i: v for i, key in enumerate(keys) if self.cache and (v := self.cache.get(key)) is not None}
        todo = [i for i in range(len(items)) if i not in verdicts]
        for i, v in zip(todo, llm_judge(self.judge, [items[i] for i in todo])):
            verdicts[i] = v
        if self.cache and todo:
            self.cache.put_many({keys[i]: verdicts[i] for i in todo})
        return [verdicts[i] for i in range(len(items))]

    def answer(self, facts: list[Fact], relations: dict[str, Relation], k: int = 8,
               temperature: float = 1.0) -> list[AnswerResult]:
        """One graded AnswerResult per fact (same order)."""
        rels = [relations[f.relation] for f in facts]
        questions = [question_for(f, r) for f, r in zip(facts, rels)]
        samples = self.sample(questions, k, temperature)
        correct = [[string_match(s, f.gold_answers, r.answer_type) for s in outs]
                   for f, r, outs in zip(facts, rels, samples)]
        if self.judge is not None:
            misses = [(i, j) for i, c in enumerate(correct) for j, ok in enumerate(c) if not ok]
            verdicts = self._judge([(questions[i], samples[i][j], facts[i].gold_answers) for i, j in misses])
            for (i, j), v in zip(misses, verdicts):
                correct[i][j] = v
        return [AnswerResult(self.name, f.key, q, s, c) for f, q, s, c in zip(facts, questions, samples, correct)]
