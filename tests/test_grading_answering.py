"""Grading, question templates and the cached answering pass (with a fake model)."""

from wikifacts.answering import Answerer, question_for
from wikifacts.cache import Cache
from wikifacts.grading import string_match
from wikifacts.store import Fact


class FakeModel:
    """Canned answers; counts calls."""

    def __init__(self, answers):
        self.answers, self.calls = answers, 0

    def __call__(self, prompts, n=1, temperature=1.0, max_tokens=64, stop=None):
        self.calls += 1
        return [[self.answers[i % len(self.answers)] for i in range(n)] for _ in prompts]


def test_string_match():
    gold = ["Recife", "Pernambuco"]
    assert string_match("He was born in Recife, Pernambuco.", gold)
    assert string_match("Pernambuco", gold)
    assert not string_match("São Paulo", gold)
    assert string_match("January 30, 1975", ["1975"], "year")
    assert not string_match("1976", ["1975"], "year")


def test_question_is_templated_and_never_leaks(relations, store):
    fact = store.get_fact("Q1", relations["place_of_birth"])
    assert question_for(fact, relations["place_of_birth"]) == \
        "In what city was Juninho Pernambucano (Brazilian footballer) born?"
    film = Fact("Q9", "2001: A Space Odyssey", "1968 film by Stanley Kubrick", "film_director",
                gold_answers=["Stanley Kubrick"])
    assert question_for(film, relations["film_director"]) == "Who directed 2001: A Space Odyssey?"
    dali = Fact("Q8", "Salvador Dalí", "Spanish artist (1904–1989)", "year_of_birth", gold_answers=["1904"])
    assert question_for(dali, relations["year_of_birth"]) == "In what year was Salvador Dalí (Spanish artist) born?"


def test_answerer_grades_and_caches(relations, store, tmp_path):
    fact = store.get_fact("Q1", relations["place_of_birth"])
    model = FakeModel(["Recife", "São Paulo", "Recife", "Rio"])
    answerer = Answerer("fake", model, cache=Cache(str(tmp_path / "c.sqlite")))
    [res] = answerer.answer([fact], relations, k=4)
    assert res.correct == [True, False, True, False] and res.acc == 0.5
    [res2] = answerer.answer([fact], relations, k=4)          # served from the cache
    assert model.calls == 1 and res2.acc == 0.5


def test_llm_judge_recovers_misses(relations, store):
    fact = store.get_fact("Q1", relations["place_of_birth"])
    judge = lambda prompts: ["yes" if "Model answer: Recifé-PE" in p else "no" for p in prompts]  # noqa: E731
    answerer = Answerer("fake", FakeModel(["Recifé-PE", "Rio"]), judge=judge)
    assert answerer.answer([fact], relations, k=2)[0].correct == [True, False]
