"""Layer 3 — OPTIMIZED: embedding pre-filter, LLMLingua, parallel LLM calls."""
import asyncio
import re
from typing import List, Optional
from datetime import datetime, timezone, timedelta
import numpy as np
from src.config import AppConfig, setup_logger
from src.models import MemoryNodeModel
from src.llm_client import OllamaClient
from src.graph_store import Neo4jGraphStore

_COMPRESSOR = None

def get_compressor():
    global _COMPRESSOR
    if _COMPRESSOR is None:
        try:
            from llmlingua import PromptCompressor
            # Paper Reference [43]: LLMLingua-2
            _COMPRESSOR = PromptCompressor(
                model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                use_llmlingua2=True,
                device_map="cpu",
            )
        except Exception:
            try:
                from llmlingua import PromptCompressor
                _COMPRESSOR = PromptCompressor()
            except Exception:
                _COMPRESSOR = None
    return _COMPRESSOR


logger = setup_logger("m4.layer3")

def _cosine(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))

class SelfMonitoringMemoryUpdating:
    def __init__(
        self,
        config: AppConfig,
        llm_client: Optional[OllamaClient] = None,
        graph_store: Optional[Neo4jGraphStore] = None,
    ):
        self.config = config
        self.llm = llm_client or OllamaClient(config)
        self.graph = graph_store or Neo4jGraphStore(config)

    async def calibrate_chain(self, chain_id: str, new_node: MemoryNodeModel) -> bool:
        existing = self.graph.get_chain_nodes(chain_id)
        if not existing:
            return False

        candidates = []
        for node in existing:
            if node["id"] == new_node.id:
                continue
            emb = node.get("embedding")
            if emb:
                sim = _cosine(emb, new_node.embedding)
                if sim >= 0.3:
                    candidates.append((sim, node))
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_k = candidates[:5]

        if not top_k:
            return False

        async def _check(node):
            prompt = f"""Previous_summary: {node['summary']}
New_summary: {new_node.summary}

Judge whether there is a conflict or update between them.
If conflict, output 1, otherwise 0 without reasoning:"""
            return await self.llm.complete_chat([{"role": "user", "content": prompt}])

        results = await asyncio.gather(*[_check(n) for _, n in top_k])
        conflicts = [
            (sim, node)
            for (sim, node), r in zip(top_k, results)
            if re.search(r"\b1\b", r.strip())
        ]

        if not conflicts:
            return False

        _, conflict_node = conflicts[0]
        merge_prompt = f"""Previous_summary: {conflict_node['summary']}
New_summary: {new_node.summary}

Merge them into a new summary without reasoning."""
        merged = await self.llm.complete_chat([{"role": "user", "content": merge_prompt}])
        merged_clean = merged.strip()

        # Step 2: Remove the inconsistent node from the chain
        self.graph.remove_node_from_chain(conflict_node["id"])
        # Step 3: Calibrate the newly introduced node with the merged summary
        self.graph.update_node_summary(new_node.id, merged_clean, calibrated=True)
        return True

    async def compress_dormant_nodes(self, days_idle: int = 14) -> int:
        threshold = (datetime.now(timezone.utc) - timedelta(days=days_idle)).isoformat()
        candidates = self.graph.get_dormant_nodes(threshold=threshold, limit=50)

        compressed_count = 0
        compressor = get_compressor()
        for node in candidates:
            if compressor:
                try:
                    compressed = compressor.compress_prompt(node["summary"], rate=0.5)["compressed_prompt"]
                except Exception:
                    compressed = node["summary"]
            else:
                compressed = node["summary"]


            orig_tokens = set(node["summary"].lower().split())
            comp_tokens = set(compressed.lower().split())
            if not orig_tokens:
                continue
            overlap = len(orig_tokens & comp_tokens) / len(orig_tokens)
            if overlap >= 0.6:
                self.graph.mark_node_compressed(node["id"], compressed)
                compressed_count += 1

        return compressed_count

