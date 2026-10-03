# dashboard-data: rag-eval-lab

Key results from `notebooks/analysis.ipynb` ("RAG Evaluation Lab"), exported
for the interactive dashboard. All numbers come from the notebook's executed
outputs. A 2x2x2 factorial experiment (chunk size x top-k x query rewriting)
on a small QA system over 579 biomedical abstracts. Honest null result: no
factor reached p < 0.05.

## Files

- **condition_means.json** — mean retrieval metrics per condition.
  Fields: `n_questions`, `conditions[]`: `chunk_size` ("small" = 150 words,
  "large" = 450 words), `top_k`, `rewrite` ("on"/"off"), `recall_at_k`,
  `mrr`, `ndcg_at_k`.
- **anova.json** — repeated-measures ANOVA + planned paired contrasts.
  Fields: `n_questions`, `metrics[]`: `metric`,
  `anova[]`: `factor`, `F`, `p`, `partial_eta2`,
  `contrasts[]`: `factor`, `level_a`, `level_b`, `mean_diff_a_minus_b`,
  `ci95` ([low, high]), `cohens_dz`, `p`.
- **generation.json** — faithfulness of generated answers.
  Fields: `scored_answers`, `mean_faithfulness`, `abstentions`,
  `abstention_denominator`, `by_condition[]`: `chunk_size`, `top_k`,
  `rewrite`, `faithfulness`, `abstentions`.
- **meta.json** — experiment setup.
  Fields: `corpus_abstracts`, `corpus_topics` (counts per query),
  `abstract_length_words` (median/min/max), `questions`,
  `questions_drafted`, `questions_passed_verification`, `factors`, `seed`,
  `retriever`, `judge`.
