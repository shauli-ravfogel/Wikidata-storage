"""Which enclosing units count as a correct place answer (mirrors real Wikidata chains)."""

from wikifacts.sparql import walk_ancestors


def test_stops_at_first_level_division():
    # Hamlet, NC -> Richmond County -> North Carolina (coarse) -> USA (coarse)
    parents = {"hamlet": {"richmond"}, "richmond": {"nc"}, "nc": {"usa"}}
    assert walk_ancestors("hamlet", parents, blocked={"nc", "usa"}) == ["richmond"]


def test_accepts_city_above_neighbourhood():
    # Benimàmet -> Poblados del Oeste -> Valencia -> Province of Valencia -> Valencian Community
    parents = {"benimamet": {"poblados"}, "poblados": {"valencia"}, "valencia": {"province"},
               "province": {"community"}, "community": {"spain"}}
    assert walk_ancestors("benimamet", parents, blocked={"community", "spain"}) == ["poblados", "province", "valencia"]


def test_dissolved_units_block_the_path():
    parents = {"konigsberg": {"kingdom_prussia"}, "kingdom_prussia": {"german_reich"}}
    assert walk_ancestors("konigsberg", parents, blocked={"kingdom_prussia"}) == []


def test_depth_limit():
    parents = {f"u{i}": {f"u{i + 1}"} for i in range(10)}
    assert len(walk_ancestors("u0", parents, blocked=set(), max_depth=4)) == 4
