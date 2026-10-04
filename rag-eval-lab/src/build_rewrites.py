"""Query rewriting: one Gemini rewrite per question, cached, reused everywhere.

The rewrite is generated ONCE per question and is identical across the two
rewrite-on conditions (small/big chunks), so 'query rewriting' is a clean
experimental factor with no re-randomization noise.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

sys.path.insert(0, os.path.join(ROOT, "src"))
from gemini_client import generate, set_model  # noqa: E402
set_model("gemini-3.5-flash-lite")  # free-tier quota is per model; 2.5-flash exhausted

OUT = os.path.join(ROOT, "data", "rewrites.json")

REWRITE_PROMPT = """Rewrite the medical question below as a short standalone search
query: keep the key medical terms, drop filler words, do not answer it.
Reply with ONLY the rewritten query, no quotes, no other text.

Question: {question}"""


def main():
    with open(
        os.path.join(ROOT, "data", "questions.json")
    ) as f:
        questions = json.load(f)["questions"]
    out = {}
    for q in questions:
        rw = generate(
            REWRITE_PROMPT.format(question=q["question"]), temperature=0.0
        ).strip().strip('"')
        out[q["question_id"]] = rw
        print(q["question_id"], "->", rw[:90])
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {len(out)} rewrites -> {OUT}")


if __name__ == "__main__":
    main()
