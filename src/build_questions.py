"""Draft test questions with Gemini, then verify each against its passage.

Pipeline (honest by construction):
1. Sample candidate passages: one 150-word chunk per abstract, stratified by
   topic, fixed seed 20261002. Oversample (56 drafted) so we can discard.
2. Gemini drafts one question + answer + verbatim evidence quote per passage.
3. Verification (automatic, strict): the evidence quote must appear verbatim
   (whitespace/case normalized) in the passage, and the answer must share at
   least 3 content words with the quote. Anything failing is discarded.
4. Keep a fixed-seed balanced sample: 10 per topic, 40 total.

Drafted/kept counts are printed and recorded in the output file.
"""
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

sys.path.insert(0, os.path.join(ROOT, "src"))
from corpus import load_corpus, build_chunks  # noqa: E402
from gemini_client import generate, set_model, regenerate  # noqa: E402
set_model("gemini-3.5-flash-lite")  # free-tier quota is per model; 2.5-flash exhausted

SEED = 20261002
OUT = os.path.join(ROOT, "data", "questions.json")
N_DRAFT = 56
N_KEEP_PER_TOPIC = 10

DRAFT_PROMPT = """You are helping build a test set for a medical question-answering system.
Read the abstract passage below. Write ONE question that a clinician or patient
might ask, whose answer is directly stated in the passage. Then give the answer
in one or two sentences, and quote the exact sentence(s) from the passage that
support the answer.

Reply with ONLY a JSON object, no other text:
{{"question": "...", "answer": "...", "evidence": "exact quote from the passage"}}

Passage:
\"\"\"{passage}\"\"\""""

STOP = set(
    "the a an and or of to in on for with is are was were be by as at from that this it its their his her our your as".split()
)


def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def content_words(s):
    return {w.strip(".,;:!?()\"'").lower() for w in s.split()} - STOP


def verify(draft, passage):
    """Return True only if the answer is grounded in the passage."""
    try:
        ev = norm(draft["evidence"])
        q = draft["question"].strip()
        a = draft["answer"].strip()
    except (KeyError, TypeError, AttributeError):
        return False
    if len(q.split()) < 5 or len(a.split()) < 4 or len(ev.split()) < 5:
        return False
    if ev not in norm(passage):
        return False
    if len(content_words(a) & content_words(ev)) < 3:
        return False
    return True


def parse_draft(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return json.loads(text)


def main():
    rng = random.Random(SEED)
    docs = load_corpus()
    small_chunks = build_chunks(docs, chunk_size=150, overlap=30, prefix="s")
    # one chunk per abstract, stratified by topic
    by_topic = {}
    seen_docs = set()
    for c in small_chunks:
        if c["doc_id"] in seen_docs:
            continue
        seen_docs.add(c["doc_id"])
        by_topic.setdefault(c["topic"], []).append(c)
    candidates = []
    per = N_DRAFT // len(by_topic)
    for topic, chunks in sorted(by_topic.items()):
        candidates += rng.sample(chunks, min(per, len(chunks)))
    rng.shuffle(candidates)
    candidates = candidates[:N_DRAFT]

    drafted, kept = 0, []
    for c in candidates:
        drafted += 1
        draft = None
        for attempt_text in range(2):  # one retry on malformed JSON
            try:
                prompt = DRAFT_PROMPT.format(passage=c["text"])
                raw = regenerate(prompt, temperature=0.3) if attempt_text else generate(prompt, temperature=0.3)
                draft = parse_draft(raw)
                break
            except Exception as e:
                print(f"  draft parse failed ({c['chunk_id']}) attempt {attempt_text}: {e}", flush=True)
        if draft is None:
            continue
        if not verify(draft, c["text"]):
            print(f"  discarded ({c['chunk_id']}): failed verification", flush=True)
            continue
        kept.append(
            {
                "question_id": None,  # assigned after balancing
                "topic": c["topic"],
                "source_doc_id": c["doc_id"],
                "source_chunk_id_small": c["chunk_id"],
                "question": draft["question"].strip(),
                "reference_answer": draft["answer"].strip(),
                "evidence": draft["evidence"].strip(),
            }
        )
        print(f"  kept ({c['chunk_id']}) [{len(kept)}/{N_DRAFT}]", flush=True)

    # balanced final set, fixed seed
    final = []
    for topic in sorted(by_topic):
        pool = [k for k in kept if k["topic"] == topic]
        final += rng.sample(pool, min(N_KEEP_PER_TOPIC, len(pool)))
    final.sort(key=lambda k: (k["topic"], k["source_doc_id"]))
    for i, k in enumerate(final, 1):
        k["question_id"] = f"q{i:02d}"
    meta = {
        "seed": SEED,
        "n_drafted": drafted,
        "n_verified_kept": len(kept),
        "n_final": len(final),
    }
    with open(OUT, "w") as f:
        json.dump({"meta": meta, "questions": final}, f, indent=1)
    print(f"\ndrafted={drafted} verified_kept={len(kept)} final={len(final)} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
