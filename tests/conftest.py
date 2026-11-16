"""Shared fixtures: the packaged relations and a tiny in-memory fact store."""

import pytest

from wikifacts import ClaimChecker, FactStore, load_relations


@pytest.fixture
def relations():
    return load_relations()


@pytest.fixture
def store():
    s = FactStore(":memory:")
    # Juninho: one birthplace (Recife, enclosing unit Pernambuco).
    s.add_entity("Q1", "Juninho Pernambucano", "Brazilian footballer", "Q5", sitelinks=25,
                 aliases=["Antônio Augusto Ribeiro Reis Júnior"])
    s.add_answer("Q100", "Pernambuco", ["PE"])
    s.add_answer("Q101", "Recife", ["Recife, Brazil"])
    s.add_answer_parents("Q101", ["Q100"])
    s.add_fact("Q1", "place_of_birth", "Recife", answer_qid="Q101")
    s.add_fact("Q1", "year_of_birth", "1975")
    # Frédéric Chopin: a second valid subject (for pairs).
    s.add_entity("Q6", "Frédéric Chopin", "Polish composer", "Q5", sitelinks=150)
    s.add_answer("Q104", "Żelazowa Wola")
    s.add_fact("Q6", "place_of_birth", "Żelazowa Wola", answer_qid="Q104")
    # Two different people named "John Smith" -> ambiguous.
    s.add_entity("Q2", "John Smith", "English footballer", "Q5", sitelinks=3)
    s.add_entity("Q3", "John Smith", "American politician", "Q5", sitelinks=4)
    # Two birthplaces recorded -> multi_valued.
    s.add_entity("Q4", "Mystery Person", "Hungarian footballer", "Q5", sitelinks=2)
    s.add_answer("Q102", "Budapest")
    s.add_answer("Q103", "Szeged")
    s.add_fact("Q4", "place_of_birth", "Budapest", answer_qid="Q102")
    s.add_fact("Q4", "place_of_birth", "Szeged", answer_qid="Q103")
    # In the store, but no birthplace.
    s.add_entity("Q5", "Nobody Known", "", "Q5", sitelinks=1)
    s.commit()
    return s


@pytest.fixture
def checker(store, relations):
    return ClaimChecker(store, relations)
