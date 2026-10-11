"""Layer 2 — OPTIMIZED: batch chain lookup, batch node append, O(n) edges."""
import uuid
from datetime import datetime, timezone
from typing import List
from src.config import AppConfig, setup_logger
from src.models import DisambiguatedMemoryClue, MemoryNodeModel
from src.graph_store import Neo4jGraphStore

logger = setup_logger("m4.layer2")

class ChainBasedMemoryConstruction:
    def __init__(self, config: AppConfig, graph_store: Neo4jGraphStore):
        self.config = config
        self.graph_store = graph_store

    def build_memory_chains(self, clues: List[DisambiguatedMemoryClue], session_id: str) -> List[str]:
        if not clues:
            return []
        gamma = self.config.similarity_threshold

        cat_embeddings = [c.category_embedding for c in clues]
        match_results = self.graph_store.find_most_similar_chains_batch(cat_embeddings)

        target_categories = []
        chains_to_create = []
        for clue, match in zip(clues, match_results):
            if match and match[2] >= gamma:
                target_categories.append(match[1])
            else:
                target_categories.append(clue.category)
                chains_to_create.append({
                    "chain_id": f"chain_{uuid.uuid4().hex[:12]}",
                    "category_name": clue.category,
                    "embedding": clue.category_embedding,
                })

        if chains_to_create:
            self.graph_store.create_memory_chains_batch(chains_to_create)

        now = datetime.now(timezone.utc)
        nodes = []
        for clue, cat in zip(clues, target_categories):
            nodes.append(MemoryNodeModel(
                id=f"mem_{uuid.uuid4().hex[:12]}",
                summary=clue.summary,
                category=cat,
                session_id=session_id,
                created_at=now,
                last_used_at=now,
                embedding=clue.clue_embedding,
            ))

        self.graph_store.append_nodes_batch(nodes, target_categories)

        pairs = []
        if len(nodes) > 1:
            anchor_id = nodes[0].id
            for i in range(1, len(nodes)):
                pairs.append((anchor_id, nodes[i].id))
            self.graph_store.link_shared_nodes_batch(pairs)

        return [n.id for n in nodes]
