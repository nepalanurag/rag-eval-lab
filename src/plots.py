"""Publication-style figures for the experiment."""
import itertools
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
FIG = f"{BASE}/figures"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    }
)

METRICS = {
    "recall_at_k": "Recall@k",
    "mrr": "MRR",
    "ndcg_at_k": "nDCG@k",
    "faithfulness": "Faithfulness (claim-level)",
}
PAIRS = [
    ("chunk_size", "top_k", {"small": "150 words", "large": "450 words"}),
    ("chunk_size", "rewrite", {"small": "150 words", "large": "450 words"}),
    ("top_k", "rewrite", {}),
]
LEVEL_LABELS = {
    "chunk_size": {"small": "150 words", "large": "450 words"},
    "top_k": {"3": "k=3", "8": "k=8"},
    "rewrite": {"off": "no rewrite", "on": "rewrite"},
}


def load():
    retr = pd.read_csv(f"{BASE}/data/retrieval_results.csv")
    retr["top_k"] = retr["top_k"].astype(str)
    with open(f"{BASE}/data/judge.json") as f:
        judged = json.load(f)
    rows = [
        {
            "question_id": g["question_id"],
            "chunk_size": g["chunk_size"],
            "top_k": str(g["top_k"]),
            "rewrite": g["rewrite"],
            "faithfulness": g["faithfulness"],
        }
        for g in judged.values()
        if g["faithfulness"] is not None
    ]
    gen = pd.DataFrame(rows)
    return {"retrieval": retr, "faithfulness": gen}


def interaction_plots(df, metric, tag):
    ylabel = METRICS[metric]
    for f1, f2, _ in PAIRS:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        lv1 = sorted(df[f1].unique())
        lv2 = sorted(df[f2].unique())
        x = np.arange(len(lv1))
        for j, b in enumerate(lv2):
            means, lo, hi = [], [], []
            for a in lv1:
                v = df[(df[f1] == a) & (df[f2] == b)][metric].to_numpy()
                m = v.mean()
                se = v.std(ddof=1) / np.sqrt(len(v))
                means.append(m)
                lo.append(m - 1.96 * se)
                hi.append(m + 1.96 * se)
            ax.errorbar(
                x + (j - (len(lv2) - 1) / 2) * 0.08,
                means,
                yerr=[np.array(means) - np.array(lo), np.array(hi) - np.array(means)],
                marker="o",
                capsize=4,
                label=f"{f2} = {LEVEL_LABELS[f2].get(b, b)}",
            )
        ax.set_xticks(x)
        ax.set_xticklabels([LEVEL_LABELS[f1].get(a, a) for a in lv1])
        ax.set_xlabel(f1.replace("_", " "))
        ax.set_ylabel(f"{ylabel} (mean per question, 95% CI)")
        ax.set_title(f"{ylabel}: {f1.replace('_', ' ')} x {f2.replace('_', ' ')}")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(f"{FIG}/{tag}_{metric}_{f1}_x_{f2}.png")
        plt.close(fig)


def condition_means_plot(df, metric, tag):
    ylabel = METRICS[metric]
    g = df.groupby(["chunk_size", "top_k", "rewrite"])[metric]
    means = g.mean()
    ses = g.std(ddof=1) / np.sqrt(g.count())
    labels = [
        f"{LEVEL_LABELS['chunk_size'][a]}\n{LEVEL_LABELS['top_k'][b]}\n{LEVEL_LABELS['rewrite'][c]}"
        for a, b, c in means.index
    ]
    order = np.argsort(means.values)[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = np.arange(len(means))
    ax.bar(
        x,
        means.values[order],
        yerr=1.96 * ses.values[order],
        capsize=3,
        color="#3b6ea5",
        edgecolor="white",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([labels[i] for i in order], fontsize=8)
    ax.set_ylabel(f"{ylabel} (mean +/- 95% CI)")
    ax.set_title(f"{ylabel} by experimental condition (n={df['question_id'].nunique()} questions)")
    fig.tight_layout()
    fig.savefig(f"{FIG}/{tag}_{metric}_conditions.png")
    plt.close(fig)


def distribution_plot(df, metric, tag):
    ylabel = METRICS[metric]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    conds = sorted(df.groupby(["chunk_size", "top_k", "rewrite"]).groups.keys())
    data = [
        df[
            (df["chunk_size"] == a)
            & (df["top_k"] == b)
            & (df["rewrite"] == c)
        ][metric].to_numpy()
        for a, b, c in conds
    ]
    labels = [f"{a[0].upper()}/k={b}/{'rw' if c=='on' else 'no'}" for a, b, c in conds]
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for box in bp["boxes"]:
        box.set(facecolor="#bcd3e8", edgecolor="#3b6ea5")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("condition (chunk / top-k / rewrite)")
    ax.set_title(f"Per-question {ylabel} across conditions")
    fig.tight_layout()
    fig.savefig(f"{FIG}/{tag}_{metric}_distribution.png")
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    data = load()
    for metric in ["recall_at_k", "mrr", "ndcg_at_k"]:
        interaction_plots(data["retrieval"], metric, "retrieval")
        condition_means_plot(data["retrieval"], metric, "retrieval")
    distribution_plot(data["retrieval"], "ndcg_at_k", "retrieval")
    interaction_plots(data["faithfulness"], "faithfulness", "generation")
    condition_means_plot(data["faithfulness"], "faithfulness", "generation")
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
