# RAG Evaluation Lab

Which RAG design choices actually matter? I built a small question-answering
system over biomedical abstracts and ran a 2x2x2 factorial experiment on it:
chunk size (150 vs 450 words) x top-k (3 vs 8) x query rewriting (on/off).

The short answer: on this corpus, none of the three factors mattered much.
Retrieval sat near the ceiling in all 8 conditions (recall@k 0.850 to 0.975),
and the largest effect was query rewriting *hurting* recall by about 6 points.
Answer faithfulness was at ceiling (0.993 over 76 judged answers). Details and
statistics below.

## The problem

When you build a retrieval-augmented generation system, you face a pile of
design choices: how to split documents into chunks, how many chunks to retrieve,
whether to rewrite the user's query first. Blog posts give strong opinions
about each. I wanted measured answers with uncertainty attached, on a fixed
test set, with the analysis planned before I saw any results.

## Data

* **Corpus:** 579 recent PubMed abstracts, fetched via Entrez (no API key) with
  four fixed queries: type 2 diabetes treatment (149 abstracts), hypertension
  management (139), asthma inhaler therapy (145), migraine prevention (146).
  Only records with a non-empty abstract were kept. See `data/build_corpus.py`.
* **Test questions:** 40 questions, 10 per topic. Each was drafted by Gemini
  from one 150-word passage, then kept only if a verbatim evidence quote from
  the passage supported the answer (56 drafted, 52 passed verification, 40 kept
  in a balanced fixed-seed sample). `data/questions.json`.
* **Query rewrites:** one Gemini rewrite per question, generated once and reused
  across conditions so rewriting is a clean experimental factor.
  `data/rewrites.json`.

## Methods

**System.** TF-IDF retriever (1-2 word grams, cosine similarity) over the
chunked corpus, plus a Gemini generator instructed to answer only from the
retrieved passages.

**Design.** Fully within-subjects: all 40 questions run through all 8
conditions, so each question is its own control. This is the repeated-measures
analogue of a randomized block design, and it is what lets 40 questions carry
real statistical weight.

**Metrics.**
* Retrieval (all 40 questions x 8 conditions): recall@k, MRR, nDCG@k. The
  relevance judgment is the single chunk containing the passage the question
  was drawn from (binary relevance).
* Generation (pre-registered random subset of 10 questions x 8 conditions, seed
  20261002): claim-level faithfulness. An independent judge model splits each
  answer into atomic claims and marks each supported / contradicted /
  unverifiable against the retrieved passages. Abstentions ("not stated in the
  provided context") make no factual claim and are counted separately, not
  scored.

**Statistics.** Per metric: repeated-measures ANOVA with the three factors
(questions as blocks), reporting F, p, and partial eta-squared. Then three
planned paired contrasts (marginal means, paired by question) with mean
difference, 95% CI, and Cohen's dz. The factors, metrics, and tests were fixed
before the experiment ran. The analysis plan is written out in the notebook
before any results appear.

## Key results

All numbers below are computed, not illustrative. The notebook re-runs every
statistic from the saved data files.

**Retrieval (n = 40 questions).** Nothing reached p < 0.05.

| Contrast | Mean diff | 95% CI | Cohen's dz | p |
|---|---|---|---|---|
| Chunk: 150 vs 450 words (recall@k) | +0.025 | [-0.029, +0.079] | +0.15 | 0.35 |
| Top-k: 3 vs 8 (recall@k) | -0.025 | [-0.055, +0.005] | -0.26 | 0.10 |
| Rewrite: off vs on (recall@k) | +0.062 | [-0.005, +0.130] | +0.30 | 0.067 |

Same pattern for MRR and nDCG@k: no significant effects; partial eta-squared
values between 0.015 and 0.083. The best condition was small chunks / k=3 /
no rewrite (recall@k 0.975, nDCG@k 0.910); the worst was large chunks / k=3 /
rewrite (recall@k 0.850, nDCG@k 0.813).

What this means: the TF-IDF retriever already finds the source passage 85-98%
of the time, so there is almost no headroom for any factor to matter (a ceiling
effect). The largest observed difference favors *not* rewriting: my
keyword-style rewrites stripped filler words the retriever was using as
discriminative terms. It is not significant at 0.05, but the direction is
consistent across all three metrics.

**Generation (n = 10 questions, 80 answers).** Mean faithfulness 0.993 across
76 scored answers; 75 of 76 fully supported. The single exception: one answer
added "(or monthly headache days)" when the passage only said migraine days
(faithfulness 0.5), a mild overgeneralization the judge caught. With only one
deviating answer, the ANOVA is uninformative (all p = 0.35); I report it in the
notebook but do not lean on it.

**Abstentions: 4 of 80.** One question was never retrieved under the strict
single-chunk relevance rule in any condition. Under small chunks the generator
correctly abstained all four times; under large chunks it answered correctly
from *other* retrieved chunks that also contained the answer. That exposed the
weakest part of my setup: the single-relevant-chunk judgment is too strict,
since an answer can live in more than one chunk.

## Limitations and threats to validity

* The relevance judgment (one correct chunk per question) is too strict, as q35
  showed. A graded or multi-chunk judgment would be fairer.
* Ceiling effects everywhere: with questions drawn from the passages and a
  clean corpus, this setup cannot separate the factors well. Harder questions
  or a noisier corpus would.
* Only 40 questions for retrieval and 10 for generation; confidence intervals
  are wide throughout.
* TF-IDF only. Nothing here compares retriever families.
* The LLM judge can be wrong. Two of its outputs needed re-parsing, and it
  ignored the abstention instruction on four answers (handled by excluding
  abstentions from faithfulness, as pre-registered).
* Biomedical abstracts only; results may not transfer to other domains.
* Drafting, rewriting, generation, and judging used Gemini flash-tier models
  (3.5-flash-lite, plus 3.1-flash-lite for three re-judged questions after the
  former hit its daily API quota). Same-model-family evaluation is a known
  bias risk, and within each question all 8 conditions used the same judge.

## How to run

Requires Python 3 with numpy, pandas, scikit-learn, scipy, matplotlib,
statsmodels. The analysis needs no API calls; the data files are committed.

```bash
python3 src/analysis.py        # ANOVA + contrasts -> data/analysis_results.json
python3 src/plots.py           # figures -> figures/
python3 src/make_dashboard.py  # dashboard -> dashboard.html
```

To rebuild the corpus and re-run the experiment from scratch (needs a Gemini
API key via the skill setup and takes a while):

```bash
python3 data/build_corpus.py   # fetch PubMed abstracts -> data/corpus.jsonl
python3 src/build_questions.py # draft + verify questions
python3 src/build_rewrites.py  # one rewrite per question
python3 src/run_experiment.py  # retrieval + generation + judging
```

## Repo structure

```
src/
  corpus.py            corpus loading + deterministic chunking
  retrieval.py         TF-IDF retriever, recall@k / MRR / nDCG@k
  gemini_client.py     cached Gemini client (no raw keys anywhere)
  build_questions.py   draft questions, verify against source passage
  build_rewrites.py    one cached rewrite per question
  run_experiment.py    the 2x2x2 experiment: retrieval, generation, judging
  finish_judging.py    completed judging after a quota interruption
  analysis.py          repeated-measures ANOVA + planned contrasts
  plots.py             publication-style figures
  make_dashboard.py    builds dashboard.html
notebooks/analysis.ipynb   executed analysis with narrative
data/
  corpus.jsonl         579 PubMed abstracts (title, abstract, MeSH, PMID)
  build_corpus.py      reproducible corpus build
  questions.json       40 verified questions (56 drafted, 52 passed)
  rewrites.json        40 cached query rewrites
  retrieval_results.csv  320 rows: 40 questions x 8 conditions
  generation.json      80 generated answers (10 questions x 8)
  judge.json           80 claim-level judgments
  analysis_results.json  ANOVA tables + contrasts
  cache/               raw API responses (reproducibility)
figures/               interaction plots, condition means, distributions
dashboard.html         self-contained Plotly dashboard
REPORT.md              full write-up
```
