"""Statistical analysis of the factorial experiment.

Design: 2 (chunk_size) x 2 (top_k) x 2 (rewrite) fully within-subjects;
questions are the blocking unit (repeated measures). For each metric:
  - statsmodels AnovaRM with the three factors, subject=question_id.
  - Partial eta-squared computed from F: eta2 = F*df1 / (F*df1 + df2).
  - Planned paired contrasts on marginal means (one per factor), paired by
    question, with Cohen's dz and 95% CI of the mean difference
    (t-based, scipy.stats.ttest_rel confidence interval).

No model selection or p-hacking: the three factors and the metrics were fixed
before the experiment ran (see the analysis-plan cell in the notebook).
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.anova import AnovaRM

BASE = "/home/hatch/workspace/resume-projects/rag-eval-lab/data"
OUT = f"{BASE}/analysis_results.json"

CONTRASTS = [
    ("chunk_size", "small", "large"),
    ("top_k", "3", "8"),
    ("rewrite", "off", "on"),
]


def partial_eta2(F, df1, df2):
    return float(F * df1 / (F * df1 + df2))


def paired_contrast(df, metric, factor, a, b):
    """Marginal means per question for each level, paired t-test + dz + CI."""
    sub = df[df[factor].isin([a, b])]
    piv = sub.pivot_table(index="question_id", columns=factor, values=metric)
    x = piv[a].to_numpy(dtype=float)
    y = piv[b].to_numpy(dtype=float)
    diff = x - y
    n = len(diff)
    res = stats.ttest_rel(x, y)
    dz = float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else 0.0
    se = stats.sem(diff)
    ci = stats.t.interval(0.95, n - 1, loc=np.mean(diff), scale=se)
    return {
        "factor": factor,
        "level_a": a,
        "level_b": b,
        "n": n,
        "mean_a": float(np.mean(x)),
        "mean_b": float(np.mean(y)),
        "mean_diff_a_minus_b": float(np.mean(diff)),
        "ci95_low": float(ci[0]),
        "ci95_high": float(ci[1]),
        "t": float(res.statistic),
        "p": float(res.pvalue),
        "cohens_dz": dz,
    }


def analyze_metric(df, metric):
    d = df.copy()
    d["top_k"] = d["top_k"].astype(str)
    aov = AnovaRM(
        d, metric, subject="question_id",
        within=["chunk_size", "top_k", "rewrite"],
    ).fit()
    anova_rows = []
    for factor, row in aov.anova_table.iterrows():
        F, df1, df2, p = (
            row["F Value"], row["Num DF"], row["Den DF"], row["Pr > F"]
        )
        anova_rows.append(
            {
                "factor": factor,
                "F": float(F),
                "df1": float(df1),
                "df2": float(df2),
                "p": float(p),
                "partial_eta2": partial_eta2(F, df1, df2),
            }
        )
    contrasts = [paired_contrast(d, metric, *c) for c in CONTRASTS]
    cond_means = (
        d.groupby(["chunk_size", "top_k", "rewrite"])[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    return {
        "n_questions": int(d["question_id"].nunique()),
        "anova": anova_rows,
        "planned_contrasts": contrasts,
        "condition_means": cond_means.to_dict(orient="records"),
    }


def main():
    retr = pd.read_csv(f"{BASE}/retrieval_results.csv")
    with open(f"{BASE}/judge.json") as f:
        judged = json.load(f)
    gen_rows = []
    abstentions = []
    for key, g in judged.items():
        is_abstention = (
            g["answer"].strip().lower().rstrip(".")
            == "not stated in the provided context"
        )
        abstentions.append(
            {
                "question_id": g["question_id"],
                "chunk_size": g["chunk_size"],
                "top_k": g["top_k"],
                "rewrite": g["rewrite"],
                "abstained": is_abstention,
            }
        )
        if is_abstention or g["faithfulness"] is None:
            continue
        gen_rows.append(
            {
                "question_id": g["question_id"],
                "chunk_size": g["chunk_size"],
                "top_k": g["top_k"],
                "rewrite": g["rewrite"],
                "faithfulness": g["faithfulness"],
            }
        )
    gen = pd.DataFrame(gen_rows)
    abst = pd.DataFrame(abstentions)

    results = {"seed": 20261002}
    for metric in ["recall_at_k", "mrr", "ndcg_at_k"]:
        results[metric] = analyze_metric(retr, metric)
        print(f"{metric}: n={results[metric]['n_questions']}")
    # Faithfulness ANOVA needs balanced cells: keep only questions with all 8
    # conditions scored (q35 abstained under small chunks and is excluded here;
    # abstentions are reported separately below).
    cell_counts = gen.groupby("question_id").size()
    complete_qids = cell_counts[cell_counts == 8].index
    gen_bal = gen[gen["question_id"].isin(complete_qids)].copy()
    results["faithfulness"] = analyze_metric(gen_bal, "faithfulness")
    results["faithfulness"]["n_questions_excluded"] = int(
        gen["question_id"].nunique() - gen_bal["question_id"].nunique()
    )
    results["faithfulness"]["excluded_question_ids"] = sorted(
        set(gen["question_id"]) - set(complete_qids)
    )
    results["abstention"] = {
        "n_total": int(len(abst)),
        "n_abstained": int(abst["abstained"].sum()),
        "by_condition": abst.groupby(["chunk_size", "top_k", "rewrite"])["abstained"]
        .agg(["sum", "count"])
        .reset_index()
        .to_dict(orient="records"),
    }
    print(f"abstentions: {int(abst['abstained'].sum())}/{len(abst)}")
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
