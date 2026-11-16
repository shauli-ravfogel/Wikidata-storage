# wikifacts

A local store of Wikidata facts, a checker for claims made about them, and a reward for
"do you know this fact?" claims. You can use it on its own in other projects.

- **Build** an SQLite store of `(subject, relation, gold answers)` facts from Wikidata
  (people's birthplaces, film directors, book authors, ...: 38 relations ship by default).
- **Check claims.** Is the entity a claim names in the store, with a usable answer? Is a
  given answer to a fact correct?
- **Reward** a model for knowing the facts it says it knows and not knowing the facts it
  says it doesn't. The reward functions plug into TRL's `GRPOTrainer`.

## Install

```bash
pip install -e path/to/wikifacts          # deps: pyyaml, requests
pip install -e "path/to/wikifacts[dev]"   # + pytest
pytest path/to/wikifacts                  # no network or GPU needed
```

## 1. Build the fact store

```bash
python -m wikifacts.build --store data/store.sqlite --limit 500      # smoke test, ~minutes
python -m wikifacts.build --store data/store.sqlite                  # everything, hours
python -m wikifacts.build --store data/store.sqlite --relations place_of_birth film_director
python -m wikifacts.build --store data/store.sqlite --relations-file my_relations.yaml
```

- The build runs on a CPU with network access. It queries the public
  [QLever](https://qlever.dev) Wikidata endpoint. WDQS times out on bulk queries, so it is
  not used.
- Pages are size-checked, and the build waits out rate limits patiently.
- A build is **resumable**. Finished relations are listed in `<store>.meta.json` and skipped
  on a rerun (`--force` rebuilds them). That file also records the build date, because the
  store is a dated snapshot of Wikidata.
- Only subjects with an English Wikipedia article are kept.
- Place answers follow some extra rules:
  - Countries and states are never accepted as answers.
  - Enclosing units below the state level also count as correct. For example, "Valencia" is
    accepted for a neighbourhood of Valencia.

Existing stores from the `kgap` project (`data/store.sqlite`, `data/store_stage1.sqlite`) use
the same schema and work as they are.

### Relations

Relations are defined in [`wikifacts/relations.yaml`](wikifacts/relations.yaml). To add one,
add an entry to that file (or to your own file) and build it. No code changes are needed.

```yaml
place_of_birth:
  pid: P19                    # Wikidata property
  subject_class: Q5           # subjects: instances of this class or of any subclass
  answer_type: place          # entity | place | year: decides how answers are graded
  phrase: the place of birth  # claims read "... <phrase> of <subject>"
  question: In what city was {subject} ({description}) born?
  placeholder: <full name>    # subject slot in claim templates; an echoed placeholder is no claim
  multi_valued: false         # if false, subjects with several values are rejected
```

## 2. Check claims

```python
from wikifacts import ClaimChecker, FactStore, load_relations

checker = ClaimChecker(FactStore("data/store.sqlite"), load_relations())

# Is the entity in the store, with a usable gold answer?
c = checker.check("I don't know the director of Suspiria (1977).", "film_director")
c.valid               # True
c.fact.gold_answers   # ['Argento, Dario', 'D. Argento', 'Dario Argento', ...]

c = checker.check("I don't know the place of birth of Carlos Inventado.", "place_of_birth")
c.valid, c.reason     # (False, 'not_found')

# A two-line answer: "I know ... of A." + "I don't know ... of B."
known, unknown = checker.check_pair(text, "place_of_birth")

# Is an answer true?
checker.verify_answer("Suspiria (1977)", "film_director", "Dario Argento").correct   # True
checker.verify_answer("Frédéric Chopin", "place_of_birth", "Warsaw").correct         # False
```

A claim is any text whose last relevant sentence contains `<phrase> of <name>`. Anything
before that sentence, such as chain of thought, is ignored. A year after a title, as in
`Brothers (2015)`, picks out one of several works with the same name. Name lookup ignores
accents, punctuation and word order.

When a claim is rejected, `c.reason` says why:

| reason | meaning |
|---|---|
| `unparseable` | no `<phrase> of <name>` sentence, or the name is an echoed placeholder |
| `not_found` | no entity of the relation's subject class has this name |
| `ambiguous` | several entities share the name, and neither the year nor popularity picks one |
| `no_answer` | the entity has no value for this relation |
| `multi_valued` | there are several values, but the relation is single-valued |
| `lookup_failed` | the live lookup failed (only with `allow_live=True`) |

`ClaimChecker(..., allow_live=True)` also searches the live Wikidata API for names that are
missing from the store, and writes what it finds into the store. Use this for evaluation
only. For RL, keep it off so the reward stays fixed and fast.

Answers are graded by normalized string match against all labels and aliases. Years are
compared by the year alone. `grading.llm_judge` is an optional fallback for answers the
string match rejects, such as "NYC" for "New York City".

## 3. Measure what a model knows

The model is asked a templated question about the fact in a fresh context, k times, and each
answer is graded. For example: *"In what city was Juninho Pernambucano (Brazilian footballer)
born?"* The subject's description is left out of the question whenever it would give the
answer away.

Bring any generation function that returns `n` samples per prompt:

```python
from vllm import LLM, SamplingParams
from wikifacts.answering import Answerer
from wikifacts.cache import Cache

llm = LLM("Qwen/Qwen3-14B")

def generate(prompts, n, temperature, max_tokens, stop=None):
    params = SamplingParams(n=n, temperature=temperature, max_tokens=max_tokens, stop=stop)
    msgs = [[{"role": "user", "content": p}] for p in prompts]
    return [[o.text for o in out.outputs] for out in llm.chat(msgs, params)]

answerer = Answerer("qwen3-14b", generate, cache=Cache("data/cache.sqlite"))
results = answerer.answer(facts, relations, k=8)      # one AnswerResult per fact
[r.acc for r in results]                              # fraction of correct samples
```

The cache keeps the raw samples per model, question and sampling settings, and never the
grades. Fixing a gold answer therefore never requires rerunning the model. Give a fine-tuned
model its own `name` so its answers are cached separately.

## 4. Reward

The reward checks two things:

1. Both claims name real entities that are in the store.
2. The model's own accuracy on each fact matches what it claimed.

| claim | reward |
|---|---|
| "I know X" | `acc(X)` |
| "I don't know X" | `1 - acc(X)` |
| paired: "I know A" + "I don't know B" | `(acc(A) + 1 - acc(B)) / 2` |
| invalid: unparseable, not in the store, ambiguous, or the same entity twice in a pair | `INVALID_REWARD = -0.5` |

Here `acc` is the model's accuracy on the fact over k samples. Every valid reward lies in
[0, 1], so the invalid reward is always worse than any valid claim. Naming a made-up entity
never pays.

```python
from trl import GRPOTrainer
from wikifacts.reward import PairedClaimReward, answerer_accuracy

reward = PairedClaimReward(checker, answerer_accuracy(answerer, checker.relations, k=8))
trainer = GRPOTrainer(model=..., reward_funcs=[reward], train_dataset=ds, ...)
# ds: a "prompt" column plus a "relation" column (relation name per prompt)
```

- `ClaimReward` handles single claims. It reads the claim type from the last line ("I don't
  know" or "I do not know" means an unknown claim). Pass `claims_known=True` or `False` to
  fix the type instead.
- `accuracy` can be any function `list[Fact] -> list[float]`, for example a lookup in
  precomputed accuracies. `answerer_accuracy` asks the frozen model online. Give that model
  a cache, because the same facts come up again and again.
- The scalar functions are `claim_reward(claims_known, acc)` and `pair_reward(acc_known, acc_unknown)`.

## Package layout

| module | role |
|---|---|
| `sparql.py` | Wikidata download over QLever: facts, aliases, answer names, place hierarchy |
| `build.py` | CLI that builds or resumes the store |
| `store.py` | `FactStore` (SQLite) and the `Fact` record |
| `live.py` | live Wikidata API fallback for names missing from the store |
| `claims.py` | claim parsing and `ClaimChecker` (in the store? answer true?) |
| `grading.py` | answer vs gold answers: string match, optional LLM judge |
| `answering.py` | question templates and `Answerer` (k samples, graded, cached) |
| `reward.py` | claim rewards and TRL reward functions |
| `normalize.py`, `cache.py`, `relations.py` | helpers and the relation config |
