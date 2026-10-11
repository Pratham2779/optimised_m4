"""Layer 1 — OPTIMIZED: Union-Find merge, batched embeddings, Gram matrix."""
import asyncio
import numpy as np
from typing import List
from src.config import AppConfig, setup_logger
from src.llm_client import OllamaClient
from src.models import RawMemoryItem, DisambiguatedMemoryClue

logger = setup_logger("m4.layer1")

class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

def compute_cosine_similarity(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def _gram_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized = embeddings / norms
    return normalized @ normalized.T

class CategoryBasedMemoryLearning:
    def __init__(self, config: AppConfig, client: OllamaClient):
        self.config = config
        self.client = client

    def _format_extraction_prompt(self, conversation_text: str) -> str:
        return f"""Here is a conversation:
{conversation_text}

Please analyze the conversation and identify distinct pieces of information to memorize.
Focus exclusively on details specific to the speakers or the situation (e.g., personal preferences, plans, specific events, relationships) and avoid summarizing common knowledge you already possess.
For each identified piece of memory, classify it into a distinct and natural memory category and provide a concise summary.
Return a JSON array of objects, where each object represents a memory item.

Example JSON Response:
[
  {{"Category": "Personal Preferences", "Summary": "The user prefers black coffee without sugar."}},
  {{"Category": "Life Events", "Summary": "The user adopted a golden retriever named Max last Friday."}}
]

Now, for the given conversation, provide the JSON response without any reasoning:"""

    async def process_session(self, conversation_text: str) -> List[DisambiguatedMemoryClue]:
        logger.info("Extracting candidate memory categories and clues...")
        prompt = self._format_extraction_prompt(conversation_text)
        raw_items: List[RawMemoryItem] = await self.client.extract_structured_json(prompt)
        if not raw_items:
            logger.warning("No memory clues identified.")
            return []

        all_texts = [item.category for item in raw_items] + [item.summary for item in raw_items]
        all_embeddings = await self.client.get_embeddings_batch(all_texts)
        n = len(raw_items)
        category_embeddings = all_embeddings[:n]
        clue_embeddings = all_embeddings[n:]

        cat_matrix = np.array(category_embeddings, dtype=np.float32)
        sim_matrix = _gram_matrix(cat_matrix)

        gamma = self.config.similarity_threshold
        uf = _UnionFind(n)
        pairs_i, pairs_j = np.where(np.triu(sim_matrix >= gamma, k=1))
        for i, j in zip(pairs_i, pairs_j):
            uf.union(int(i), int(j))


        groups: dict = {}
        for i in range(n):
            root = uf.find(i)
            groups.setdefault(root, []).append(i)

        canonical_items: List[DisambiguatedMemoryClue] = []
        for root, members in groups.items():
            canonical_cat = raw_items[root].category
            canonical_emb = category_embeddings[root]
            for idx in members:
                canonical_items.append(DisambiguatedMemoryClue(
                    category=canonical_cat,
                    summary=raw_items[idx].summary,
                    category_embedding=canonical_emb,
                    clue_embedding=clue_embeddings[idx],
                ))

        logger.info("Disambiguation: %d raw → %d consolidated clues.", n, len(canonical_items))
        return canonical_items
