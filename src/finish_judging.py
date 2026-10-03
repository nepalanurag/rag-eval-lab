"""Finish judging: re-judge q23, q32, q35 fully with gemini-3.1-flash-lite
(the 3.5-flash-lite daily quota was exhausted mid-run), keep the 7 already
fully-judged questions on 3.5-flash-lite. Within each question, all 8
conditions use the same judge model. Writes judge.json incrementally.

Run: python3 src/finish_judging.py
"""
import json
import re
import sys

sys.path.insert(0, "/home/hatch/workspace/resume-projects/rag-eval-lab/src")
from corpus import load_corpus, build_chunks  # noqa: E402
from gemini_client import generate, set_model, QuotaExhausted  # noqa: E402

BASE = "/home/hatch/workspace/resume-projects/rag-eval-lab/data"
OUT = f"{BASE}/judge.json"

REJUDGE_MODEL = "gemini-3.1-flash-lite"
REJUDGE_QIDS = {"q23", "q32", "q35"}

JUDGE_PROMPT = """You are grading a question-answering system for faithfulness.
Given the question, a model answer, and the context passages the model was
allowed to use, break the model answer into atomic factual claims. For each
claim, judge it against the context passages ONLY:
- "supported": the passage states it directly
- "contradicted": the passage says the opposite
- "unverifiable": the passage does not say it

If the model answer says "Not stated in the provided context", give it a
single claim with verdict "abstained".

Reply with ONLY a JSON object:
{{"claims": [{{"claim": "...", "verdict": "supported|contradicted|unverifiable|abstained"}}]}}

Question: {question}

Model answer: {answer}

Context passages:
{context}"""


def strip_fences(t):
    t = t.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t)
    return t


def parse_judge(raw):
    """Return list of claim dicts. JSON first, regex fallback on malformed JSON."""
    txt = strip_fences(raw)
    try:
        return json.loads(txt).get("claims", [])
    except Exception:
        pass
    verdicts = re.findall(
        r'"verdict"\s*:\s*"(supported|contradicted|unverifiable|abstained)"', txt
    )
    claims = re.findall(r'"claim"\s*:\s*"((?:[^"\\]|\\.)*)"', txt)
    out = []
    for i, v in enumerate(verdicts):
        out.append({"claim": claims[i] if i < len(claims) else "", "verdict": v})
    if not out:
        raise ValueError(f"unparseable judge output: {raw[:200]}")
    return out


def faithfulness(claims):
    n_abs = sum(1 for c in claims if c.get("verdict") == "abstained")
    n_sup = sum(1 for c in claims if c.get("verdict") == "supported")
    denom = len(claims) - n_abs
    return (n_sup / denom) if denom > 0 else None


def main():
    docs = load_corpus()
    chunk_text = {}
    for name, (cs, ov) in {"small": (150, 30), "large": (450, 60)}.items():
        for c in build_chunks(docs, cs, ov, prefix=name[0]):
            chunk_text[c["chunk_id"]] = c["text"]
    with open(f"{BASE}/generation.json") as f:
        gen_data = json.load(f)
    generations = gen_data["generations"]
    with open(f"{BASE}/questions.json") as f:
        q_by_id = {q["question_id"]: q for q in json.load(f)["questions"]}

    judged = {}
    if __import__("os").path.exists(OUT):
        judged = json.load(open(OUT))

    n_done = 0
    for key in sorted(generations):
        g = generations[key]
        qid = g["question_id"]
        if key in judged and qid not in REJUDGE_QIDS:
            continue  # already judged with the original model
        model = REJUDGE_MODEL if qid in REJUDGE_QIDS else "gemini-3.5-flash-lite"
        set_model(model)
        q = q_by_id[qid]
        ctx = "\n\n".join(
            f"[{i+1}] {chunk_text[cid]}" for i, cid in enumerate(g["context_ids"])
        )
        raw = generate(
            JUDGE_PROMPT.format(
                question=q["question"], answer=g["answer"], context=ctx
            ),
            temperature=0.0,
        )
        claims = parse_judge(raw)
        judged[key] = {**g, "claims": claims, "faithfulness": faithfulness(claims),
                       "judge_model": model}
        n_done += 1
        print(f"judge {key} [{model}]: faith={judged[key]['faithfulness']}", flush=True)
        with open(OUT, "w") as f:
            json.dump(judged, f, indent=1)
    print(f"judged {n_done} new; total {len(judged)} -> {OUT}")


if __name__ == "__main__":
    main()
