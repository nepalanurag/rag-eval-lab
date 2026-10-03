"""Generate dashboard.html: self-contained Plotly dashboard (CDN only)."""
import json
import html

BASE = "/home/hatch/workspace/resume-projects/rag-eval-lab"
OUT = BASE + "/dashboard.html"
PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.27.0.min.js"

METRICS = [
    ("recall_at_k", "Recall@k",
     "fraction of questions where the source chunk was in the top-k"),
    ("mrr", "MRR", "mean reciprocal rank of the source chunk"),
    ("ndcg_at_k", "nDCG@k",
     "rank-discounted gain for the single relevant chunk"),
    ("faithfulness", "Faithfulness",
     "share of answer claims supported by the retrieved context "
     "(judged subset, n=10 questions)"),
]
LEVEL_LABELS = {
    "chunk_size": {"small": "150 words", "large": "450 words"},
    "top_k": {"3": "k=3", "8": "k=8"},
    "rewrite": {"off": "no rewrite", "on": "rewrite"},
}


def cond_label(c):
    return (LEVEL_LABELS["chunk_size"][c["chunk_size"]] + " / " +
            LEVEL_LABELS["top_k"][str(c["top_k"])] + " / " +
            LEVEL_LABELS["rewrite"][c["rewrite"]])


CSS = """
body{font-family:Georgia,serif;max-width:960px;margin:0 auto;padding:24px;
color:#222;line-height:1.55}
h1{font-size:28px}
h2{font-size:22px;border-bottom:2px solid #3b6ea5;padding-bottom:6px;margin-top:40px}
h3{font-size:18px;margin-bottom:4px}
h4{font-size:15px;margin:18px 0 6px}
.card{border:1px solid #ccc;border-radius:6px;padding:16px;margin:18px 0;background:#fafafa}
.desc{color:#555;font-size:14px;margin-top:0}
.best{font-size:15px}
table{border-collapse:collapse;margin:8px 0 20px;font-size:14px}
th,td{border:1px solid #bbb;padding:6px 10px;text-align:left}
th{background:#eef3f8}
.note{background:#fff8e6;border-left:4px solid #d9a441;padding:10px 14px;margin:16px 0;font-size:14px}
"""


def main():
    with open(BASE + "/data/analysis_results.json") as f:
        res = json.load(f)
    with open(BASE + "/data/questions.json") as f:
        qmeta = json.load(f)["meta"]

    specs = []
    cards = []
    for metric, title, desc in METRICS:
        r = res[metric]
        cms = sorted(r["condition_means"], key=lambda c: -c["mean"])
        best = cms[0]
        labels = [cond_label(c) for c in cms]
        means = [round(c["mean"], 4) for c in cms]
        err = [round(1.96 * c["std"] / (c["count"] ** 0.5), 4) for c in cms]
        specs.append({"id": "chart_" + metric, "labels": labels,
                      "means": means, "err": err,
                      "title": title + " by condition (mean +/- 95% CI)"})
        cards.append(
            '<div class="card"><h3>' + title + '</h3>'
            '<p class="desc">' + desc + '</p>'
            '<div id="chart_' + metric + '"></div>'
            '<p class="best">Best condition: <b>' + html.escape(cond_label(best)) +
            '</b> at ' + format(best["mean"], ".3f") +
            ' (n=' + str(r["n_questions"]) + ' questions).</p></div>'
        )

    anova_html = []
    for metric, title, _ in METRICS:
        rows = "".join(
            "<tr><td>" + a["factor"] + "</td><td>" + format(a["F"], ".2f") +
            "</td><td>" + format(a["df1"], ".0f") + ", " + format(a["df2"], ".0f") +
            "</td><td>" + format(a["p"], ".4f") + "</td><td>" +
            format(a["partial_eta2"], ".3f") + "</td></tr>"
            for a in res[metric]["anova"]
        )
        anova_html.append(
            "<h4>" + title + " (n=" + str(res[metric]["n_questions"]) + " questions)</h4>"
            "<table><tr><th>Effect</th><th>F</th><th>df</th><th>p</th>"
            "<th>partial eta^2</th></tr>" + rows + "</table>"
        )

    contrast_html = []
    for metric, title, _ in METRICS:
        rows = "".join(
            "<tr><td>" + c["factor"] + ": " + c["level_a"] + " vs " + c["level_b"] +
            "</td><td>" + format(c["mean_a"], ".3f") + " vs " + format(c["mean_b"], ".3f") +
            "</td><td>" + format(c["mean_diff_a_minus_b"], "+.3f") +
            " [" + format(c["ci95_low"], "+.3f") + ", " + format(c["ci95_high"], "+.3f") + "]" +
            "</td><td>" + format(c["cohens_dz"], "+.2f") + "</td><td>" +
            format(c["p"], ".4f") + "</td></tr>"
            for c in res[metric]["planned_contrasts"]
        )
        contrast_html.append(
            "<h4>" + title + "</h4>"
            "<table><tr><th>Contrast</th><th>Means</th><th>Diff [95% CI]</th>"
            "<th>Cohen's dz</th><th>p</th></tr>" + rows + "</table>"
        )

    parts = []
    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="en"><head><meta charset="utf-8">')
    parts.append("<title>RAG Evaluation Lab: does chunk size, top-k, or query "
                 "rewriting matter?</title>")
    parts.append('<script src="' + PLOTLY_CDN + '"></script>')
    parts.append("<style>" + CSS + "</style></head><body>")
    parts.append("<h1>RAG Evaluation Lab</h1>")
    parts.append(
        "<p>Which RAG design choices actually move the needle? A 2x2x2 factorial "
        "experiment: chunk size (150 vs 450 words) x top-k (3 vs 8) x query "
        "rewriting (on/off), with test questions as the blocking unit. "
        "Corpus: 579 PubMed abstracts. " + str(qmeta["n_drafted"]) +
        " drafted, " + str(qmeta["n_final"]) + " kept after verification.</p>")
    parts.append("<h2>Overview</h2>")
    parts.append(
        '<div class="note">Headline: retrieval sat near the ceiling in all 8 '
        "conditions (recall@k 0.85 to 0.98), and no factor reached statistical "
        "significance. The largest effect was query rewriting <i>hurting</i> "
        "recall by about 6 points (p = 0.067, Cohen's dz = 0.30): keyword-style "
        "rewrites dropped terms the retriever needed. Chunk size (150 vs 450 "
        "words) and top-k (3 vs 8) moved retrieval by only 1-3 points. Answer "
        "faithfulness was at ceiling (0.993 over 76 judged answers); the "
        "generator failed by correctly abstaining when the context was missing, "
        "not by hallucinating.</div>")
    parts.extend(cards)
    parts.append("<h2>Methods</h2>")
    parts.append(
        "<p><b>Corpus.</b> 579 recent PubMed abstracts on four topics (type 2 "
        "diabetes treatment, hypertension management, asthma inhaler therapy, "
        "migraine prevention), fetched via Entrez with fixed queries. "
        "<b>Questions.</b> 40 test questions, 10 per topic. Each was drafted by "
        "Gemini from one 150-word passage, then kept only if a verbatim evidence "
        "quote from the passage supported the answer (56 drafted, 40 kept). "
        "<b>System.</b> TF-IDF retriever (1-2 grams, cosine) over the chunked "
        "corpus, plus a Gemini generator instructed to answer only from the "
        "retrieved passages. <b>Relevance.</b> The single relevant chunk per "
        "question is the chunk containing the passage the question was drawn "
        "from. <b>Design.</b> All 40 questions run through all 8 conditions "
        "(fully within-subjects). Retrieval metrics: recall@k, MRR, nDCG@k. "
        "Generation: a pre-registered random subset of 10 questions x 8 "
        "conditions, each answer judged claim-by-claim by an independent "
        "Gemini judge with a written rubric (supported / contradicted / "
        "unverifiable). <b>Statistics.</b> Repeated-measures ANOVA per metric "
        "(questions as blocks), plus planned paired t-tests with Cohen's dz "
        "and 95% CIs on the three marginal contrasts. Factors and metrics were "
        "fixed before the experiment ran.</p>")
    parts.append("<h2>Results</h2>")
    parts.append(
        "<p>Charts above show condition means with 95% CIs. ANOVA tables and "
        "planned contrasts below. Effect sizes (partial eta-squared, Cohen's dz) "
        "tell you how much each factor matters, not just whether p &lt; "
        "0.05.</p><h3>Repeated-measures ANOVA</h3>")
    parts.extend(anova_html)
    parts.append("<h3>Planned paired contrasts (marginal means, paired by "
                 "question)</h3>")
    parts.extend(contrast_html)
    parts.append("<h2>Limitations</h2>")
    parts.append("""<ul>
<li>Single relevant chunk per question: the source passage. Real users have
information needs spread across documents; this setup rewards finding one
specific passage and may overstate top-k effects.</li>
<li>TF-IDF only. Dense embeddings were not tested, so nothing here says which
retriever family is best in general.</li>
<li>Faithfulness judged by an LLM judge on 10 questions. The judge can be wrong;
spot checks were done during development, but judge error is a real threat.</li>
<li>Query rewriting was one fixed rewrite per question from the same model family
as the generator. A different rewriter might behave differently.</li>
<li>Biomedical abstracts only. Results may not transfer to other domains or to
long documents where chunking choices bite harder.</li>
</ul>
<p>Code, data, and the executed analysis notebook are in the
<a href="https://github.com/nepalanurag/rag-eval-lab">rag-eval-lab</a> repo.</p>""")
    parts.append("<script>")
    parts.append("var SPECS = " + json.dumps(specs) + ";")
    parts.append("""SPECS.forEach(function(s){
  Plotly.newPlot(s.id,
    [{x: s.labels, y: s.means,
      error_y: {type: "data", array: s.err, visible: true},
      type: "bar", marker: {color: "#3b6ea5"}}],
    {title: s.title, margin: {b: 150},
     xaxis: {tickangle: -25}, yaxis: {title: "score"}});
});""")
    parts.append("</script></body></html>")

    with open(OUT, "w") as f:
        f.write("\n".join(parts))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
