"""Corpus loading and deterministic chunking."""
import json
import re

CORPUS_PATH = "/home/hatch/workspace/resume-projects/rag-eval-lab/data/corpus.jsonl"

WORD_RE = re.compile(r"\S+")


def load_corpus(path=CORPUS_PATH):
    docs = []
    with open(path) as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def chunk_words(text, chunk_size, overlap):
    """Split text into word windows. Deterministic, no randomness."""
    words = WORD_RE.findall(text)
    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(words):
        piece = words[start : start + chunk_size]
        if len(piece) < 20 and chunks:
            break  # drop trailing stub
        chunks.append(" ".join(piece))
        start += step
    return chunks


def build_chunks(docs, chunk_size, overlap, prefix):
    chunks = []
    for d in docs:
        full = d["title"] + ". " + d["abstract"]
        for i, text in enumerate(chunk_words(full, chunk_size, overlap)):
            chunks.append(
                {
                    "chunk_id": f"{prefix}_{d['doc_id']}_c{i}",
                    "doc_id": d["doc_id"],
                    "topic": d["topic_query"],
                    "text": text,
                }
            )
    return chunks
