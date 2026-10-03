# RAG Evaluation Lab: full report

## Background

Retrieval-augmented generation is now the default architecture for
question-answering over private corpora: retrieve some passages, stuff them
into a language model's context, generate an answer. Three choices come up in
every implementation I have seen:

1. **Chunk size.** Split documents into small pieces (precise matches, but an
   answer can be split across a boundary) or large pieces (more context per
   hit, but noisier matches)?
2. **Top-k.** Hand the generator 3 passages or 8? More context helps recall but
   adds noise and cost.
3. **Query rewriting.** Rewrite the user's question into a cleaner search query
   first, or retrieve with the raw question?

The usual advice is anecdotal. I ran a factorial experiment to get measured
answers with uncertainty attached.

## Methodology

**Corpus.** 579 recent PubMed abstracts fetched through the Entrez API with
four fixed queries (type 2 diabetes treatment: 149, hypertension management:
139, asthma inhaler therapy: 145, migraine prevention: 146). Entrez's default
ordering is most-recent-first, so the corpus is "recent literature on these
topics as of October 2026". Only records with a non-empty abstract of at
least 40 words were kept. The build script (`data/build_corpus.py`) documents
every choice; re-running it reproduces the corpus up to PubMed's index
changing over time.

**Chunking.** Deterministic word windows, no randomness: small = 150 words
with 30 overlap (1631 chunks), large = 450 words with 60 overlap (634 chunks).

**Test questions.** I sampled one 150-word chunk per abstract, stratified by
topic (fixed seed 20261002), and had Gemini draft one question, one answer,
and one verbatim evidence quote per passage. Then I verified each draft
automatically: the quote had to appear verbatim (whitespace/case normalized)
in the passage, and the answer had to share at least 3 content words with the
quote. 56 drafted, 52 passed, and I kept a balanced 40 (10 per topic). The
verification step is the load-bearing part of the pipeline: an unverified test
question can punish the system for a bad test rather than a bad answer.

**Query rewriting.** One Gemini rewrite per question ("keep the key medical
terms, drop filler words"), generated once and cached, identical across the
two rewrite-on conditions. In practice the rewrites came out as keyword
strings, e.g. "erenumab treatment response definition study". That is a
legitimate rewrite strategy, but it matters for interpreting the results.

**System.** TF-IDF retriever (unigrams + bigrams, English stop words,
min_df=2, cosine similarity) plus a Gemini generator (flash tier) instructed
to answer only from the provided passages, or to say "Not stated in the
provided context."

**Experimental design.** 2 (chunk size) x 2 (top-k) x 2 (rewrite), fully
within-subjects. All 40 questions go through all 8 conditions: 320 retrieval
runs. For generation, a pre-registered random subset of 10 questions
(seed 20261002) x 8 conditions = 80 answers.

**Metrics.**
* Retrieval: recall@k, MRR, nDCG@k per question per condition. Relevance is
  binary and strict: the single chunk containing the passage the question was
  drawn from.
* Generation: claim-level faithfulness. An independent judge model splits each
  answer into atomic claims and labels each supported / contradicted /
  unverifiable against the retrieved passages only. Faithfulness = supported /
  (total - abstained). Answers that abstain make no factual claim and are
  excluded from faithfulness, counted separately.

**Statistical analysis.** For each metric: repeated-measures ANOVA
(statsmodels AnovaRM) with the three factors, questions as the blocking unit,
reporting F, p, and partial eta-squared (computed as F*df1 / (F*df1 + df2)).
Then three planned paired contrasts on the marginal means (paired by
question), each with the mean difference, a t-based 95% CI, and Cohen's dz.
The factors, metrics, and tests were fixed before the experiment ran; the
analysis plan is written out in the notebook ahead of the results. No other
tests were run.

**Practical notes.** The Gemini free tier quota interrupted the work twice:
gemini-2.5-flash allows 20 requests/day (I switched before using it for
anything kept), and gemini-3.5-flash-lite was exhausted during judging. The
remaining judgments (questions q23, q32, q35, all 8 conditions each) were done
with gemini-3.1-flash-lite. Within each question, all 8 conditions used the
same judge model, so the within-question comparisons stay clean. All raw API
responses are cached in `data/cache/` (253 calls total).

## Results

### Retrieval

Condition means (n = 40 questions):

| Chunk | k | Rewrite | Recall@k | MRR | nDCG@k |
|---|---|---|---|---|---|
| 150 words | 3 | off | 0.975 | 0.888 | 0.910 |
| 150 words | 8 | off | 0.975 | 0.888 | 0.910 |
| 450 words | 3 | off | 0.925 | 0.833 | 0.857 |
| 450 words | 8 | off | 0.950 | 0.840 | 0.868 |
| 150 words | 3 | on | 0.900 | 0.812 | 0.835 |
| 150 words | 8 | on | 0.900 | 0.812 | 0.835 |
| 450 words | 3 | on | 0.850 | 0.800 | 0.813 |
| 450 words | 8 | on | 0.925 | 0.815 | 0.842 |

Repeated-measures ANOVA (recall@k): chunk_size F(1,39) = 0.89, p = 0.35,
partial eta2 = 0.022; top_k F(1,39) = 2.79, p = 0.10, eta2 = 0.067; rewrite
F(1,39) = 3.55, p = 0.067, eta2 = 0.083. No interaction reached p < 0.20.
MRR and nDCG@k give the same picture (all p > 0.07, eta2 0.015-0.080).

Planned contrasts (recall@k, paired by question):

| Contrast | Mean diff | 95% CI | Cohen's dz | p |
|---|---|---|---|---|
| 150 vs 450 words | +0.025 | [-0.029, +0.079] | +0.15 | 0.35 |
| k=3 vs k=8 | -0.025 | [-0.055, +0.005] | -0.26 | 0.10 |
| no rewrite vs rewrite | +0.062 | [-0.005, +0.130] | +0.30 | 0.067 |

For MRR: chunk +0.028 (p = 0.44, dz = +0.12), top-k -0.005 (p = 0.11,
dz = -0.26), rewrite +0.052 (p = 0.099, dz = +0.27). For nDCG@k: chunk +0.027
(p = 0.40, dz = +0.13), top-k -0.010 (p = 0.11, dz = -0.26), rewrite +0.055
(p = 0.073, dz = +0.29).

### Generation

76 scored answers (4 abstentions excluded): mean faithfulness 0.993, minimum
0.5. 75 of 76 answers fully supported. The ANOVA on the 9 questions with
complete data is uninformative (all F = 1.00, p = 0.35, driven by the single
deviating answer); I report it for completeness in the notebook and do not
interpret it.

The one partial failure: under small chunks / k=8 / no rewrite, the generator
answered a question about erenumab response definition with ">= 50% reduction
in monthly migraine days (or monthly headache days)" when the passage only
mentioned migraine days. The judge marked the headache-days claim
unverifiable: faithfulness 0.5. A small overgeneralization, and exactly what
claim-level judging is meant to catch.

Abstentions: 4 of 80, all for one question (q35) under small chunks. The
generator correctly said "not stated in the provided context" in all four.
Under large chunks it answered correctly with faithfulness 1.0 in all four,
because other retrieved chunks also contained the answer even though the
single designated "relevant" chunk was never retrieved (recall@k = 0 for q35
in all 8 conditions).

## Interpretation

1. **Nothing I tested mattered much for retrieval, and that is the finding.**
   A plain TF-IDF retriever finds the source passage 85-98% of the time on
   this kind of corpus, leaving almost no headroom. Anyone tuning chunk size
   or top-k on a clean corpus with in-distribution questions is optimizing
   noise. The effort belongs on harder retrieval problems: noisier corpora,
   questions needing synthesis across documents, ambiguous queries.
2. **Rewriting can hurt.** The largest effect in the study (about 6 points of
   recall, dz = 0.30, p = 0.067) favored *not* rewriting. Keyword-style
   rewrites drop the filler words that TF-IDF uses as discriminative terms.
   Rewriting is not a free upgrade; a bad rewrite is worse than none, and its
   value depends on the retriever.
3. **The generator's failure mode was silence, not hallucination.** When the
   context lacked the answer, it abstained correctly. The one unfaithful
   answer was a mild overgeneralization, not a fabrication. Constraining the
   generator to the retrieved context works as intended here.
4. **My relevance judgment was too strict.** q35 showed the answer living in
   several chunks while the metric demanded one specific chunk. Graded or
   multi-chunk relevance would be fairer, and answerability (can the system
   answer from what it retrieved?) is arguably the more honest retrieval
   metric for RAG.

## Limitations

Same as the README, in brief: strict single-chunk relevance; ceiling effects;
40/10 questions; TF-IDF only; LLM judge error (two outputs needed re-parsing,
abstention instruction ignored on four answers); biomedical abstracts only;
same-model-family evaluation (Gemini throughout, with the quota-forced judge
switch documented above).

## References

* Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive
  NLP Tasks. NeurIPS 2020.
* Cuconasu et al. (2024). The Power of Noise: Redefining Retrieval for RAG
  Systems. (On top-k and context noise trade-offs.)
* NCBI Entrez Programming Utilities. https://www.ncbi.nlm.nih.gov/books/NBK25501/
* Lakens (2013). Calculating and reporting effect sizes to facilitate
  cumulative science. Frontiers in Psychology. (Why this report leads with dz
  and eta-squared, not p-values.)
