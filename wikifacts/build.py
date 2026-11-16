"""Build the fact store (SQLite) from Wikidata. CPU + network only.

    python -m wikifacts.build --store data/store.sqlite                       # all relations
    python -m wikifacts.build --store data/smoke.sqlite --limit 500           # smoke test: 500 most popular subjects
    python -m wikifacts.build --store data/store.sqlite --relations place_of_birth film_director
    python -m wikifacts.build --store data/store.sqlite --relations-file my_relations.yaml

Per relation:
  1. (subject, value) rows          -> entities + facts
  2. subject aliases                -> names (for lookup)
  3. answer labels / aliases        -> answers
  4. places only: facts whose answer is a country / first-level division are dropped,
     and the enclosing units that also count as correct are stored (with their names)

A full build takes hours (QLever rate-limits; see sparql.py). It is resumable: finished
relations are listed in <store>.meta.json and skipped on a rerun (--force rebuilds them).
The build date is recorded there too: the store is a dated snapshot of Wikidata.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

from wikifacts.relations import DEFAULT_RELATIONS_FILE, Relation, load_relations
from wikifacts.sparql import (fetch_answer_info, fetch_coarse_places, fetch_facts, fetch_labels,
                              fetch_place_ancestors, fetch_subject_aliases)
from wikifacts.store import FactStore


def build_relation(store: FactStore, rel: Relation, limit: Optional[int], coarse: Optional[set[str]]) -> None:
    """Download one relation into the store. `coarse` (place relations only): places that are never answers."""
    name = rel.name
    t0 = time.time()

    # 1. Facts (and the subject entities they are about).
    subjects, answers = set(), set()
    for row in fetch_facts(rel, limit=limit):
        store.add_entity(row["qid"], row["label"], row["description"], rel.subject_class,
                         row["sitelinks"], row["enwiki_title"])
        store.add_fact(row["qid"], name, row["answer_value"], row["answer_qid"])
        subjects.add(row["qid"])
        if row["answer_qid"]:
            answers.add(row["answer_qid"])
    store.commit()
    print(f"[{name}] {len(subjects):,} subjects, {len(answers):,} answer entities ({time.time() - t0:.0f}s)")

    # 2. Subject aliases (the query returns all subjects; with --limit keep only the stored ones).
    n = 0
    for qid, alias in fetch_subject_aliases(rel):
        if qid in subjects:
            store.add_names(qid, [alias])
            n += 1
    store.commit()
    print(f"[{name}] {n:,} subject aliases")

    # 3. Answer entities (year relations have none).
    if not answers:
        return
    info = fetch_answer_info(rel)
    for qid in answers & info.keys():
        store.add_answer(qid, info[qid]["label"], info[qid]["aliases"])
    store.commit()

    # 4. Places: drop too-coarse answers ("born in Bavaria" does not answer "in what city"), and
    #    store the enclosing units that also count as correct.
    if rel.is_place:
        dropped = answers & coarse
        for qid in dropped:
            store.remove_facts_with_answer(name, qid)
        ancestors = fetch_place_ancestors(rel, coarse)
        units = set()
        for qid in (answers - dropped) & ancestors.keys():
            store.add_answer_parents(qid, ancestors[qid])
            units.update(ancestors[qid])
        for qid, p in fetch_labels(sorted(units)).items():
            store.add_answer(qid, p["label"], p["aliases"])
        store.commit()
        print(f"[{name}] dropped {len(dropped):,} coarse answers; {len(units):,} enclosing units accepted")


def _sidecar(store_path: str, suffix: str) -> Path:
    """<store>.meta.json / <store>.coarse.json next to the store file."""
    return Path(store_path).with_suffix(suffix)


def coarse_places(store_path: str) -> set[str]:
    """Countries / first-level divisions: fetched once and cached next to the store (for resumes)."""
    path = _sidecar(store_path, ".coarse.json")
    if path.exists():
        return set(json.loads(path.read_text()))
    coarse = fetch_coarse_places()
    path.write_text(json.dumps(sorted(coarse)))
    return coarse


def build(store_path: str, relations: dict[str, Relation], names: list[str],
          limit: Optional[int] = None, force: bool = False) -> None:
    """Build (or resume building) the store for the relations in `names`."""
    Path(store_path).parent.mkdir(parents=True, exist_ok=True)
    store = FactStore(store_path)
    meta_path = _sidecar(store_path, ".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if meta.get("limit") != limit:          # a smoke-test store is never resumed as a full one
        meta["completed"] = []
    done = set() if force else set(meta.get("completed", []))
    todo = [n for n in names if n not in done]
    if done & set(names):
        print(f"resuming: skipping completed relations {sorted(done & set(names))}")

    coarse = coarse_places(store_path) if any(relations[n].is_place for n in todo) else None
    for name in todo:
        build_relation(store, relations[name], limit, coarse)
        meta["completed"] = sorted(set(meta.get("completed", [])) | {name})
        meta.update({"built": time.strftime("%Y-%m-%d"), "endpoint": "qlever-wikidata", "limit": limit})
        meta_path.write_text(json.dumps(meta, indent=2))

    meta["counts"] = store.counts()
    meta_path.write_text(json.dumps(meta, indent=2))
    store.close()
    print("\nSubjects per relation:")
    for name, n in sorted(meta["counts"].items()):
        print(f"  {name:28s} {n:>10,d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True, help="output SQLite file")
    ap.add_argument("--relations-file", default=str(DEFAULT_RELATIONS_FILE))
    ap.add_argument("--relations", nargs="*", help="relation names (default: all in the relations file)")
    ap.add_argument("--limit", type=int, default=None, help="max rows per relation, most popular first (smoke test)")
    ap.add_argument("--force", action="store_true", help="rebuild relations already marked completed")
    args = ap.parse_args()
    relations = load_relations(args.relations_file)
    unknown = set(args.relations or []) - relations.keys()
    if unknown:
        ap.error(f"unknown relations: {sorted(unknown)}")
    build(args.store, relations, args.relations or list(relations), args.limit, args.force)


if __name__ == "__main__":
    main()
