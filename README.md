# Redrob — Intelligent Candidate Discovery & Ranking

A ranking engine for the Redrob hackathon: given the *Senior AI Engineer —
Founding Team* job description, rank the 100,000-candidate pool and emit the
top-100 as `submission.csv`.

> **Design thesis (from the JD's own note):** the right answer is **not** "find
> the candidates with the most AI keywords." That is a trap baked into the
> dataset. The right answer is to **read the gap between what a profile claims
> and what the career actually shows**, then weigh real availability. So this is
> a *structured-reasoning* ranker, not an embedding-similarity ranker.

---

## Why not "just use an LLM + embeddings"?

Because the rules forbid it and it would lose:

- **`submission_spec.pdf` §3 — Network OFF, CPU only, ≤ 5 min, ≤ 16 GB** for
  100K candidates. No OpenAI / Anthropic / NVIDIA / any hosted model at ranking
  time. An LLM-per-candidate re-ranker cannot fit the budget.
- **`submission_spec.pdf` §7 — honeypots.** Pure embedding similarity ranks the
  keyword-stuffed honeypots highly; >10 % honeypots in the top-100 is an
  automatic **disqualification**. Embedding the skills list *is* the trap.

So the ranking step is **pure Python standard library** — zero third-party
imports, deterministic, and reproduces in seconds inside a clean container.

## Reproduce the submission

```bash
# Real pool (gzipped JSONL is read directly):
python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv

# Validate format before uploading:
python validate_submission.py submission.csv

# Quick run on the bundled 50-candidate sample:
python rank.py --candidates data/sample_candidates.json --out sample.csv
```

No setup, no pre-computation, no network. `requirements.txt` lists only
`streamlit`, which is used **solely** by the optional demo (`app.py`) and is
never imported by `rank.py`.

---

## Architecture

```
candidates.jsonl(.gz)
        │
        ▼
  features.extract()        3 separate text views per candidate:
        │                     identity  = headline + title + summary  (CLAIM)
        │                     evidence  = career-history descriptions (DID)
        │                     skills    = skill names                 (GAMEABLE)
        ▼
  honeypot.detect()         internal-consistency checks → tier-0 floor
        │
        ▼
  scoring.score()
        ├─ coherence gate    engineer vs keyword-stuffing non-engineer  [×]
        ├─ technical fit      weighted JD "absolutely need" blend        [+]
        └─ do-NOT-want        explicit JD exclusion penalties            [×]
        │
        ▼
  features.behavioral_modifier()   availability: recency, response rate… [×]
        │
        ▼
  sort by (score desc, candidate_id asc) → top 100
        │
        ▼
  reasoning.build()         grounded, varied, concern-aware 1–2 sentences
        │
        ▼
  submission.csv
```

### 1. Three text views (trap defense by construction)
The dataset deliberately (a) scrambles career descriptions and (b) stuffs AI
keywords into the **skills list** of non-engineers (Marketing Managers,
Accountants…). We never trust one field. `evidence` (what they *did*) is
weighted above `skills` (most easily gamed) everywhere in scoring.

### 2. Coherence gate — the primary discriminator
A multiplicative factor in `[0.12, 1.0]`. It asks: *does this person identify
and read as an engineer?* A "Marketing Manager" whose summary says *"I've spent
my career in marketing… experimented with ChatGPT"* is gated **down hard** no
matter how many AI skills they list. An engineer whose summary, title, and
career descriptions all cohere around ML/retrieval is gated up — but only if the
evidence text actually backs the claim.

### 3. Technical fit — the JD's "absolutely need" list
Weighted blend (weights in `src/jd_profile.py`, `FIT_WEIGHTS`):

| Signal | Weight | JD basis |
|---|---:|---|
| retrieval / ranking / search / recsys | 0.26 | *"own the ranking, retrieval, matching systems"* |
| production ML depth | 0.14 | embeddings/PyTorch/transformers, pre-LLM depth |
| product vs services company | 0.12 | *"applied ML at product companies, not pure services"* |
| **skill-trust depth** | 0.10 | relevant skills weighted by endorsements + months + assessment scores |
| vector DB / hybrid search | 0.10 | *"absolutely need"* (FAISS, Pinecone, Qdrant…) |
| **applied-ML tenure** | 0.08 | *"4–5 years in applied ML"* — years in ML roles, not just total |
| evaluation frameworks | 0.08 | *"absolutely need"* (NDCG, MRR, MAP, A/B) |
| experience band (5–9, peak 6–8) | 0.06 | *"what we mean by 5–9 years"* |
| location | 0.06 | Pune/Noida preferred; India relocatable; outside = no visa |

**Skill-trust** is the differentiator among genuine engineers: a skill claimed
"expert" with 0 endorsements / 0 months counts for almost nothing, while one
backed by endorsements, real duration, and a Redrob assessment score counts
fully — so depth-validated candidates rise above thin keyword lists.

### 4. "Things we explicitly do NOT want" — penalties
Multiplicative discounts, each mapped to a JD bullet: CV/speech/robotics primary
without NLP/IR; **entire** career at consulting firms (current consulting +
prior product is fine); recent-GenAI-keywords without pre-LLM ML depth
(framework enthusiast); frequent short tenures (title-chaser); a current
manager/architect title with no recent hands-on coding evidence (*"this role
writes code"*); and outside India (no visa sponsorship).

### 5. Behavioral availability modifier
A bounded `[0.45, 1.10]` multiplier. *"A perfect-on-paper candidate who hasn't
logged in for 6 months with a 5 % response rate is, for hiring, not available."*
Uses `last_active_date` recency (relative to the pool's latest activity),
`recruiter_response_rate`, `open_to_work_flag`, interview reliability, profile
completeness, and notice period.

### 6. Honeypot rejection (`src/honeypot.py`)
No hard-coded IDs. We detect **impossibility**: a skill used longer than the
whole career; ≥3 expert skills with 0 months used; a single role outlasting the
career; history months far exceeding stated experience; education ending before
it begins. Two independent impossibilities → flagged and floored, so honeypots
cannot reach the top-100.

### 7. Grounded reasoning (`src/reasoning.py`)
Every clause is built from fields that exist on the candidate (no hallucinated
skills/employers). Templates rotate by a stable hash of `candidate_id` for
variation, and the tone matches the rank — top ranks state strengths, lower
ranks lead with honest concerns. This targets the six Stage-4 reasoning checks
directly.

### Matching detail
Vocabulary matching uses a **leading word boundary + open trailing stem**
(`src/features.py`), so `rank` matches *ranking/ranked* but not *frank*, and
`search` does not match *research*. Ambiguous bare tokens (`map`, `ctr`, …) were
removed in favour of precise phrases (`mean average precision`, `a/b test`).

---

On the 50-candidate sample, the lone genuine retrieval/ranking engineer
(Recommendation Systems Engineer at Swiggy/Uber, FAISS/Pinecone, 91 % response)
tops the list; the keyword-stuffers (Marketing/Operations Managers with AI
skills) are gated down and labelled *reads as non-engineer / keyword-stuffed*.

## Repository layout

```
rank.py                     entry point
src/jd_profile.py           vocab sets + weights from the JD
src/features.py             text views, vocab matching, behavioral modifier
src/honeypot.py             internal-consistency honeypot detection
src/scoring.py              coherence gate + fit blend + penalties
src/reasoning.py            reasoning generator
src/io_utils.py             jsonl/gz/json loading, CSV writing
app.py                      Streamlit sandbox demo
data/sample_candidates.json 50-candidate sample (sandbox input)
submission_metadata.yaml    portal metadata
validate_submission.py      bundle format validator
```
