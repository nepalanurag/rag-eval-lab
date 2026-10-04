"""Run the 2x2x2 factorial experiment.

Factors: chunk_size {small: 150 words, large: 450 words}
         x top_k {3, 8}
         x query_rewrite {off, on}

Retrieval metrics: all 40 questions x 8 conditions (no LLM calls needed).
Generation + faithfulness judging: pre-registered random subset of 10
questions (seed 20261002) x 8 conditions = 80 answers + 80 judge calls.

Relevance judgment: the single relevant chunk is the chunk (in that
condition's chunking) containing the passage the question was drawn from.
This is stated as a limitation in the README/REPORT.
"""
import itertools
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

sys.path.insert(0, os.path.join(ROOT, "src"))
from corpus import load_corpus, build_chunks  # noqa: E402
from retrieval import Retriever, retrieval_metrics  # noqa: E402
from gemini_client import generate, set_model  # noqa: E402
set_model("gemini-3.5-flash-lite")  # free-tier quota is per model; 2.5-flash exhausted

SEED = 20261002
BASE = os.path.join(ROOT, "data")
CHUNK_LEVELS = {"small": (150, 30), "large": (450, 60)}
TOPK_LEVELS = [3, 8]
REWRITE_LEVELS = ["off", "on"]
N_GEN_SUBSET = 10

GEN_PROMPT = """Answer the question using ONLY the numbered context passages below.
If the answer is not stated in the passages, reply exactly: Not stated in the provided context.
Keep your answer to 2-3 sentences. Do not use outside knowledge.

Context passages:
{context}

Question: {question}

Answer:"""

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


def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def find_large_for_small(large_chunks, small_chunk):
    """Large chunk (same doc) whose text contains the small chunk's head."""
    head = norm(small_chunk["text"])[:60]
    for c in large_chunks:
        if c["doc_id"] == small_chunk["doc_id"] and head in norm(c["text"]):
            return c["chunk_id"]
    return None


def main():
    rng = random.Random(SEED)
    docs = load_corpus()
    chunk_sets = {
        name: build_chunks(docs, cs, ov, prefix=name[0])
        for name, (cs, ov) in CHUNK_LEVELS.items()
    }
    for name, chunks in chunk_sets.items():
        print(f"{name}: {len(chunks)} chunks")
    retrievers = {name: Retriever(chunks) for name, chunks in chunk_sets.items()}

    with open(f"{BASE}/questions.json") as f:
        questions = json.load(f)["questions"]
    with open(f"{BASE}/rewrites.json") as f:
        rewrites = json.load(f)

    small_by_id = {c["chunk_id"]: c for c in chunk_sets["small"]}
    large_map = {}
    for q in questions:
        sc = small_by_id[q["source_chunk_id_small"]]
        large_map[q["question_id"]] = find_large_for_small(chunk_sets["large"], sc)
        assert large_map[q["question_id"]] is not None, q["question_id"]

    conditions = list(
        itertools.product(CHUNK_LEVELS, TOPK_LEVELS, REWRITE_LEVELS)
    )

    # ---- retrieval metrics: all questions x all conditions ----
    rows = []
    contexts = {}  # (qid, cond) -> list of chunk ids, for generation step
    for q in questions:
        qid = q["question_id"]
        for cs, k, rw in conditions:
            query = rewrites[qid] if rw == "on" else q["question"]
            ranked = retrievers[cs].rank(query, top_n=k)
            rel = q["source_chunk_id_small"] if cs == "small" else large_map[qid]
            m = retrieval_metrics([cid for cid, _ in ranked], rel, k)
            rows.append(
                {
                    "question_id": qid,
                    "chunk_size": cs,
                    "top_k": k,
                    "rewrite": rw,
                    **m,
                }
            )
            contexts[(qid, cs, k, rw)] = [cid for cid, _ in ranked]
    import csv

    with open(f"{BASE}/retrieval_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"retrieval rows: {len(rows)}")

    # ---- generation: pre-registered subset ----
    subset = sorted(rng.sample([q["question_id"] for q in questions], N_GEN_SUBSET))
    print("generation subset:", subset)
    chunk_text = {}
    for chunks in chunk_sets.values():
        for c in chunks:
            chunk_text[c["chunk_id"]] = c["text"]
    q_by_id = {q["question_id"]: q for q in questions}

    generations = {}
    for qid in subset:
        q = q_by_id[qid]
        for cs, k, rw in conditions:
            cids = contexts[(qid, cs, k, rw)]
            ctx = "\n\n".join(
                f"[{i+1}] {chunk_text[cid]}" for i, cid in enumerate(cids)
            )
            answer = generate(
                GEN_PROMPT.format(context=ctx, question=q["question"]),
                temperature=0.0,
            )
            generations[f"{qid}|{cs}|{k}|{rw}"] = {
                "question_id": qid,
                "chunk_size": cs,
                "top_k": k,
                "rewrite": rw,
                "context_ids": cids,
                "answer": answer,
            }
            print(f"gen {qid} {cs} k={k} rw={rw}: {answer[:70]}...")
    with open(f"{BASE}/generation.json", "w") as f:
        json.dump(
            {"seed": SEED, "subset": subset, "generations": generations}, f, indent=1
        )

    # ---- judging ----
    judged = {}
    for key, g in generations.items():
        q = q_by_id[g["question_id"]]
        ctx = "\n\n".join(
            f"[{i+1}] {chunk_text[cid]}" for i, cid in enumerate(g["context_ids"])
        )
        raw = generate(
            JUDGE_PROMPT.format(
                question=q["question"], answer=g["answer"], context=ctx
            ),
            temperature=0.0,
        )
        try:
            txt = raw.strip()
            if txt.startswith("```"):
                txt = re.sub(r"^```[a-zA-Z]*\n?", "", txt)
                txt = re.sub(r"\n?```$", "", txt)
            parsed = json.loads(txt)
            claims = parsed.get("claims", [])
        except Exception as e:
            print(f"judge parse failed {key}: {e}")
            claims = []
        n = len(claims)
        n_sup = sum(1 for c in claims if c.get("verdict") == "supported")
        n_abs = sum(1 for c in claims if c.get("verdict") == "abstained")
        denom = n - n_abs
        faith = (n_sup / denom) if denom > 0 else None
        judged[key] = {
            **g,
            "claims": claims,
            "faithfulness": faith,
        }
        print(f"judge {key}: {n_sup}/{denom} supported -> faith={faith}")
    with open(f"{BASE}/judge.json", "w") as f:
        json.dump(judged, f, indent=1)
    print(f"judged {len(judged)} answers")


if __name__ == "__main__":
    main()
