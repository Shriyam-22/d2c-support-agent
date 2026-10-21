"""Retrieval over the product catalogue and store policies.

Two retrievers share one interface:
  - BM25Retriever: keyword search, no API calls, no dependencies. Default.
  - EmbeddingRetriever: dense vectors from any OpenAI-compatible embeddings
    endpoint, cosine similarity with numpy. Swap in pgvector/Qdrant for scale.

Chunking strategy:
  - Products: one chunk per product (name, category, description, tags, price).
    Products are short and self-contained, so splitting them would only hurt.
  - Policies: one chunk per "## " section, so each chunk answers one topic
    (shipping, COD, returns...) and can be quoted on its own.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Sequence

from .language import HINGLISH_MARKERS, normalise_for_search

TOKEN = re.compile(r"[a-z0-9ऀ-ॿ]+")
STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "is", "are",
    "my", "me", "i", "it", "with", "do", "you", "your", "what", "which", "can",
    "have", "has", "be", "this", "that", "any", "some", "please", "pls",
} | HINGLISH_MARKERS  # Hindi function words carry no meaning for retrieval


def stem(token: str) -> str:
    """Very light English stemming so 'returns', 'returned' and 'return' match.
    Deliberately conservative: it only strips a few suffixes from longer words."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix) and not token.endswith("ss"):
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class Chunk:
    id: str
    kind: str  # "product" | "policy"
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Hit:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    def search(self, query: str, k: int = 3, kind: str | None = None) -> list[Hit]: ...


# ---------------------------------------------------------------- chunking

def product_chunks(catalogue: Sequence[dict]) -> list[Chunk]:
    chunks = []
    for p in catalogue:
        text = (
            f"{p['name']}. Category: {p['category']}. {p['description']} "
            f"Good for: {', '.join(p.get('tags', []))}. Size: {p.get('size', '')}. "
            f"Price: {p['price']}."
        )
        chunks.append(Chunk(id=p["sku"], kind="product", text=text, metadata=p))
    return chunks


def policy_chunks(markdown: str) -> list[Chunk]:
    chunks = []
    for section in re.split(r"^## ", markdown, flags=re.MULTILINE)[1:]:
        title, _, body = section.partition("\n")
        title = title.strip()
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        chunks.append(Chunk(id=f"policy:{slug}", kind="policy",
                            text=f"{title}. {body.strip()}", metadata={"title": title}))
    return chunks


def load_chunks(data_dir: Path) -> list[Chunk]:
    catalogue = json.loads((data_dir / "catalogue.json").read_text(encoding="utf-8"))
    policies = (data_dir / "policies.md").read_text(encoding="utf-8")
    return product_chunks(catalogue) + policy_chunks(policies)


# ---------------------------------------------------------------- BM25

class BM25Retriever:
    """Okapi BM25, implemented directly so the ranking is easy to inspect."""

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.docs = [tokenize(c.text) for c in self.chunks]
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_len) / max(len(self.docs), 1)
        self.tf = [Counter(d) for d in self.docs]
        df: Counter = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def _score(self, q_tokens: list[str], i: int) -> float:
        score, tf, dl = 0.0, self.tf[i], self.doc_len[i]
        for t in q_tokens:
            if t not in tf:
                continue
            f = tf[t]
            score += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return score

    def search(self, query: str, k: int = 3, kind: str | None = None) -> list[Hit]:
        q_tokens = tokenize(normalise_for_search(query))
        hits = [Hit(c, self._score(q_tokens, i)) for i, c in enumerate(self.chunks)
                if kind is None or c.kind == kind]
        hits = [h for h in hits if h.score > 0]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]


# ---------------------------------------------------------------- embeddings

EmbedFn = Callable[[list[str]], list[list[float]]]


class EmbeddingRetriever:
    """Dense retrieval. `embed` takes a list of texts and returns vectors.

    Vectors are computed once at start-up and kept in memory, which is fine
    for a catalogue of a few thousand products. For more, store them in
    pgvector or Qdrant and keep this class as the query side.
    """

    def __init__(self, chunks: Sequence[Chunk], embed: EmbedFn):
        import numpy as np

        self.np = np
        self.chunks = list(chunks)
        self.embed = embed
        vecs = np.array(embed([c.text for c in self.chunks]), dtype=float)
        self.matrix = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    def search(self, query: str, k: int = 3, kind: str | None = None) -> list[Hit]:
        np = self.np
        q = np.array(self.embed([normalise_for_search(query)])[0], dtype=float)
        q = q / np.linalg.norm(q)
        scores = self.matrix @ q
        order = np.argsort(-scores)
        hits = [Hit(self.chunks[i], float(scores[i])) for i in order
                if kind is None or self.chunks[i].kind == kind]
        return hits[:k]


def openai_embed_fn(client, model: str) -> EmbedFn:
    def embed(texts: list[str]) -> list[list[float]]:
        resp = client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]
    return embed
