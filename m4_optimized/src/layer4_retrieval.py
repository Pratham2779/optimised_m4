"""Layer 4 — Self-Reflection-Based Memory Retrieval (Algorithm 1)
Optimized bidirectional chain traversal, confidence-gated reflection, and vector anchoring.
Based on IEEE TASLPRO 2026: Meta-Memory for Large Language Models.
"""
import json
import re
import asyncio
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from async_lru import alru_cache
from src.config import AppConfig, setup_logger
from src.llm_client import OllamaClient
from src.graph_store import Neo4jGraphStore

logger = setup_logger("m4.layer4")

class SelfReflectionMemoryRetrieval:
    def __init__(
        self,
        config: AppConfig,
        llm_client: Optional[OllamaClient] = None,
        graph_store: Optional[Neo4jGraphStore] = None,
    ):
        self.config = config
        self.llm = llm_client or OllamaClient(config)
        self.graph = graph_store or Neo4jGraphStore(config)
        self.gamma = getattr(config, "similarity_threshold", 0.50)

    async def retrieve_memory(self, query: str) -> Tuple[List[str], str]:
        logger.info("[Layer 4] Retrieval for: '%s'", query)

        available = []
        try:
            available = self.graph.get_all_categories()
        except Exception as e:
            logger.error("Category fetch failed: %s", e)

        memory_context: List[str] = []
        q_emb = list(await self.llm.get_embedding(query))
        q_vec = np.array(q_emb, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec) + 1e-9

        # Step 1: Category Selection (LLM Reflection + Semantic Cosine Ranking)
        cat_names = []
        if available:
            cat_prompt = f"""Query: {query}
Available memory categories: {available}

Select the category names likely to contain facts for the query.
Return JSON array of names, e.g. ["Daily Life", "Work"]:"""
            try:
                raw = await self.llm.complete_chat([{"role": "user", "content": cat_prompt}])
                m = re.search(r"\[.*\]", raw, re.DOTALL)
                if m:
                    cats = json.loads(m.group(0))
                    if isinstance(cats, list):
                        for c in cats:
                            c_name = c if isinstance(c, str) else c.get("Category", "")
                            if c_name in available and c_name not in cat_names:
                                cat_names.append(c_name)
            except Exception as e:
                logger.error("Category extraction failed: %s", e)

        # Semantic Ranking to supplement / fallback
        if available:
            try:
                cat_embs = await self.llm.get_embeddings_batch(available)
                c_matrix = np.array(cat_embs, dtype=np.float32)
                c_norms = np.linalg.norm(c_matrix, axis=1) + 1e-9
                sims = (c_matrix @ q_vec) / (c_norms * q_norm)
                top_indices = np.argsort(sims)[::-1][:min(3, len(available))]
                for idx in top_indices:
                    c_name = available[idx]
                    if c_name not in cat_names:
                        cat_names.append(c_name)
            except Exception as e:
                logger.error("Category semantic ranking notice: %s", e)

        if not cat_names and available:
            cat_names = list(available)[:3]

        # Step 2: Algorithm 1 Bidirectional Traversal across selected memory chains
        for cat in cat_names:
            chain_nodes = self.graph.get_chain_nodes(cat)
            if not chain_nodes:
                continue

            emb_matrix = np.array([n["embedding"] for n in chain_nodes], dtype=np.float32)
            norms = np.linalg.norm(emb_matrix, axis=1) + 1e-9
            sims = (emb_matrix @ q_vec) / (norms * q_norm)

            best_idx = int(np.argmax(sims))
            best_node = chain_nodes[best_idx]
            best_sim = float(sims[best_idx])

            if best_sim >= 0.25 and best_node["summary"] not in memory_context:
                memory_context.append(best_node["summary"])

            # Bidirectional Chain Traversal (Algorithm 1)
            pf, pb = best_idx + 1, best_idx - 1
            max_depth = 2
            step = 0

            while (pf < len(chain_nodes) or pb >= 0) and step < max_depth:
                step += 1
                forward = chain_nodes[pf] if pf < len(chain_nodes) else None
                backward = chain_nodes[pb] if pb >= 0 else None

                if backward and backward["summary"] not in memory_context:
                    b_vec = np.array(backward["embedding"], dtype=np.float32)
                    b_sim = float(np.dot(b_vec, q_vec) / ((np.linalg.norm(b_vec) + 1e-9) * q_norm))
                    if b_sim >= 0.25 or step == 1:
                        memory_context.insert(0, backward["summary"])

                if forward and forward["summary"] not in memory_context:
                    f_vec = np.array(forward["embedding"], dtype=np.float32)
                    f_sim = float(np.dot(f_vec, q_vec) / ((np.linalg.norm(f_vec) + 1e-9) * q_norm))
                    if f_sim >= 0.25 or step == 1:
                        memory_context.append(forward["summary"])

                if pf < len(chain_nodes):
                    pf += 1
                if pb >= 0:
                    pb -= 1

            if len(memory_context) >= 4 and await self._is_confident(query, tuple(memory_context)):
                break

        # Step 3: Response Generation (Fig. 8 in paper)
        if memory_context:
            context_formatted = "\n".join(f"- {m}" for m in memory_context)
            final_prompt = f"""You are an accurate, helpful AI memory assistant.
Context Memories:
{context_formatted}

Question: {query}

Instructions:
1. Answer the question directly using facts from the memories above.
2. Be concise and precise. Extract exact names, dates, or terms if asked.
3. If not stated in memories, say "Not enough information".
Answer:"""
        else:
            final_prompt = f"""Question: {query}
Answer concisely:"""

        response = await self.llm.complete_chat([{"role": "user", "content": final_prompt}])
        return memory_context, response.strip()

    @alru_cache(maxsize=512)
    async def _is_confident(self, query: str, memories: tuple) -> bool:
        if not memories:
            return False
        mem_str = "\n".join(f"- {m}" for m in memories[:5])
        prompt = f"""Query: {query}
Memories:
{mem_str}

Do the memories provide sufficient factual clues to answer the query?
Answer 1 for YES, or 0 for NO:"""
        try:
            r = await self.llm.complete_chat([{"role": "user", "content": prompt}])
            r_clean = r.strip()
            return r_clean.startswith("1") or bool(re.search(r"\b1\b", r_clean))
        except Exception:
            return False
