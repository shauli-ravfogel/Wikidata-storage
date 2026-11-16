"""Parsing claims and checking them against the store."""

from wikifacts import FactStore, parse_claim
from wikifacts.claims import ClaimChecker, split_pair
from wikifacts.normalize import norm, sorted_key


def test_normalize():
    assert norm("São Paulo, Brazil!") == "sao paulo brazil"
    assert sorted_key("Pernambucano, Juninho") == sorted_key("Juninho Pernambucano")


def test_parse_basic(relations):
    rel = relations["place_of_birth"]
    assert parse_claim("I don't know the place of birth of Juninho Pernambucano.", rel).name == "Juninho Pernambucano"
    assert parse_claim('"I know the place of birth of **Pelé** with 90% confidence."', rel).name == "Pelé"
    assert parse_claim("What is the place of birth of Dener?", rel).name == "Dener"


def test_parse_reasoning_and_failures(relations):
    rel = relations["place_of_birth"]
    p = parse_claim("<think>hmm</think>He played for Lyon.\nI don't know the place of birth of Juninho Pernambucano.", rel)
    assert (p.name, p.reasoning) == ("Juninho Pernambucano", "He played for Lyon.")
    assert parse_claim("I don't know much about Juninho.", rel).name is None
    assert parse_claim("I don't know the place of birth of <full name>.", rel).name is None


def test_parse_titles_and_years(relations):
    film = relations["film_director"]
    p = parse_claim("I don't know the director of Brothers (2015).", film)
    assert (p.name, p.year) == ("Brothers", "2015")
    p = parse_claim("I don't know the director of The Church (1989 film).", film)
    assert (p.name, p.year) == ("The Church", "1989")
    p = parse_claim("I don't know the director of <Zindagi Na Milegi Dobara> (2011).", film)
    assert (p.name, p.year) == ("Zindagi Na Milegi Dobara", "2011")
    assert parse_claim("I don't know the director of <film title> (<release year>).", film).name is None


def test_claim_on_same_line_as_reasoning(relations):
    film = relations["film_director"]
    p = parse_claim("Suspiria by Dario Argento is a classic of the director of horror films. "
                    "I don't know the director of Suspiria (1977).", film)
    assert (p.name, p.year) == ("Suspiria", "1977")
    assert "Dario Argento" in p.reasoning


def test_split_pair(relations):
    rel = relations["place_of_birth"]
    known, unknown = split_pair("I do not know the place of birth of B.\nI know the place of birth of A.", rel)
    assert (parse_claim(known, rel).name, parse_claim(unknown, rel).name) == ("A", "B")
    # Reasoning lines starting with "I know" are skipped when they make no claim.
    text = ("I know Chopin well; I don't know much about minor pianists.\n"
            "I know the place of birth of Frédéric Chopin.\nI don't know the place of birth of Juninho Pernambucano.")
    known, unknown = split_pair(text, rel)
    assert parse_claim(known, rel).name == "Frédéric Chopin"
    assert split_pair("I know the place of birth of A.", rel)[1] == ""


def test_check_valid_and_gold_set(checker):
    c = checker.check("I don't know the place of birth of juninho pernambucano.", "place_of_birth")
    assert c.valid, c.reason
    assert {"Recife", "Recife, Brazil", "Pernambuco"} <= set(c.fact.gold_answers)   # enclosing unit counts
    assert "PE" not in c.fact.gold_answers                                            # too-short alias dropped


def test_check_rejection_reasons(checker):
    reason = lambda t: checker.check(t, "place_of_birth").reason  # noqa: E731
    assert reason("blah") == "unparseable"
    assert reason("I don't know the place of birth of Carlos Inventado.") == "not_found"
    assert reason("I don't know the place of birth of John Smith.") == "ambiguous"
    assert reason("I don't know the place of birth of Mystery Person.") == "multi_valued"
    assert reason("I don't know the place of birth of Nobody Known.") == "no_answer"


def test_year_and_popularity_break_ties(relations):
    s = FactStore(":memory:")
    s.add_entity("F1", "Brothers", "2009 film by Jim Sheridan", "Q11424")
    s.add_entity("F2", "Brothers", "2015 Indian film", "Q11424")
    s.add_entity("P1", "Martin Luther", "German reformer", "Q5", sitelinks=260)
    s.add_entity("P2", "Martin Luther", "American politician", "Q5", sitelinks=19)
    s.add_entity("P3", "Leif Larsen", "a", "Q5", sitelinks=10)
    s.add_entity("P4", "Leif Larsen", "b", "Q5", sitelinks=9)
    c = ClaimChecker(s, relations)
    film, person = relations["film_director"], relations["place_of_birth"]
    assert c.resolve("Brothers", film) == (None, "ambiguous")
    assert c.resolve("Brothers", film, year="2015") == ("F2", None)
    assert c.resolve("Brothers", film, year="1999") == (None, "ambiguous")
    assert c.resolve("Martin Luther", person) == ("P1", None)          # one dominant candidate
    assert c.resolve("Leif Larsen", person) == (None, "ambiguous")


def test_check_pair(checker):
    known, unknown = checker.check_pair("I know the place of birth of Frédéric Chopin.\n"
                                        "I don't know the place of birth of Juninho Pernambucano.", "place_of_birth")
    assert known.fact.subject_qid == "Q6" and unknown.fact.subject_qid == "Q1"


def test_verify_answer(checker):
    assert checker.verify_answer("Juninho Pernambucano", "place_of_birth", "Recife").correct is True
    assert checker.verify_answer("Juninho Pernambucano", "place_of_birth", "Pernambuco").correct is True
    assert checker.verify_answer("Juninho Pernambucano", "place_of_birth", "São Paulo").correct is False
    assert checker.verify_answer("Juninho Pernambucano", "year_of_birth", "born January 30, 1975").correct is True
    missing = checker.verify_answer("Carlos Inventado", "place_of_birth", "Recife")
    assert (missing.in_store, missing.correct, missing.reason) == (False, None, "not_found")
