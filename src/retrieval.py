"""TF-IDF retriever over a fixed chunk set (cosine similarity)."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Retriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.ids = [c["chunk_id"] for c in chunks]
        self.vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=2
        )
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in chunks])

    def rank(self, query, top_n):
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix).ravel()
        order = np.argsort(-sims, kind="stable")[:top_n]
        return [(self.ids[i], float(sims[i])) for i in order]


def retrieval_metrics(ranked_ids, relevant_id, k):
    """Single relevant doc: recall@k in {0,1}; MRR; nDCG@k (binary relevance)."""
    rank = None
    for i, cid in enumerate(ranked_ids[:k], 1):
        if cid == relevant_id:
            rank = i
            break
    recall_at_k = 1.0 if rank is not None else 0.0
    mrr = 1.0 / rank if rank is not None else 0.0
    ndcg = 1.0 / np.log2(rank + 1) if rank is not None else 0.0
    return {"recall_at_k": recall_at_k, "mrr": mrr, "ndcg_at_k": ndcg}
