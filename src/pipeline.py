"""RAG pipelines A (baseline) and B (candidate), driven entirely by a JSON config."""
import json
import time
import uuid
from typing import Dict, List, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import CrossEncoder, SentenceTransformer

from . import settings
from .chunking import chunk_words
from .llm import chat

SYSTEM_PROMPTS = {
    # Baseline: tight, context-only, short answer.
    "strict": (
        "Answer the question using ONLY the provided context. "
        "Reply with a short answer (a few words or one sentence). "
        "If the context is insufficient, say so."
    ),
    # Candidate: deliberately "helpful" -- invites explanation/background. This is the
    # lever that lets tangential chunks leak into the answer (groundedness regression).
    "explain": (
        "Answer the question using the provided context, then add a brief explanation "
        "with helpful background so the reader fully understands the answer."
    ),
}


class RAGPipeline:
    def __init__(self, config_path: str):
        with open(config_path, encoding="utf-8") as f:
            self.cfg: Dict = json.load(f)
        self.name = self.cfg["name"]
        self.embedder = SentenceTransformer(self.cfg["embedder"])
        self.reranker = CrossEncoder(self.cfg["reranker"]) if self.cfg.get("reranker") else None
        self.client = QdrantClient(url=settings.qdrant_url())

    # ---------- indexing ----------
    def ingest(self, corpus: List[Dict]) -> int:
        """(Re)build this pipeline's collection from corpus [{title, text}]."""
        col = self.cfg["collection"]
        chunks = []
        for doc in corpus:
            pieces = chunk_words(doc["text"], self.cfg["chunk_words"], self.cfg["chunk_overlap_words"])
            for i, piece in enumerate(pieces):
                chunks.append((doc["title"], i, piece))
        vectors = self.embedder.encode(
            [f"{t}. {p}" for t, _, p in chunks], normalize_embeddings=True, batch_size=64
        )
        if self.client.collection_exists(col):
            self.client.delete_collection(col)
        self.client.create_collection(col, vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE))
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{t}::{i}")),
                vector=v.tolist(),
                payload={"title": t, "chunk_index": i, "text": p},
            )
            for (t, i, p), v in zip(chunks, vectors)
        ]
        for s in range(0, len(points), 128):
            self.client.upsert(col, points[s:s + 128])
        return len(points)

    # ---------- retrieval ----------
    def retrieve(self, question: str) -> List[Tuple[str, str]]:
        vec = self.embedder.encode(question, normalize_embeddings=True).tolist()
        res = self.client.query_points(
            self.cfg["collection"], query=vec, limit=self.cfg["candidate_k"], with_payload=True
        ).points
        hits = [(p.payload["title"], p.payload["text"]) for p in res]
        if self.reranker is not None and hits:
            scores = self.reranker.predict([(question, f"{t}. {x}") for t, x in hits])
            hits = [h for _, h in sorted(zip(scores, hits), key=lambda z: -z[0])]
        return hits[: self.cfg["top_k"]]

    # ---------- end to end ----------
    def retrieve_and_generate(self, question: str) -> dict:
        t0 = time.perf_counter()
        hits = self.retrieve(question)
        contexts = [f"[{t}] {x}" for t, x in hits]
        user = "CONTEXT:\n" + "\n\n".join(contexts) + f"\n\nQUESTION: {question}"
        answer = chat(
            self.cfg["generator_model"],
            [{"role": "system", "content": SYSTEM_PROMPTS[self.cfg["answer_style"]]},
             {"role": "user", "content": user}],
            max_tokens=250,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        titles: List[str] = []
        for t, _ in hits:
            if t not in titles:
                titles.append(t)
        return {
            "answer": answer,
            "retrieved_contexts": contexts,
            "context_titles": titles,
            "latency_ms": latency_ms,
        }
