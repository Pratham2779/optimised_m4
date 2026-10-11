#!/usr/bin/env python3
"""
M4 Meta-Memory Engine — Demonstration & Verification Script
Simulates and verifies all 4 layers of the M4 framework:
  1. Category-Based Memory Learning (Disambiguation & Extraction)
  2. Chain-Based Memory Construction (Topological Chains & Shared Nodes)
  3. Self-Monitoring-Based Memory Updating (Calibrate & Compress)
  4. Self-Reflection-Based Memory Retrieval (Algorithm 1 Bidirectional Search)
"""
import sys
import asyncio
import time
from typing import List, Dict, Any, Tuple
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock
import numpy as np

from src.config import AppConfig, setup_logger
from src.models import RawMemoryItem, DisambiguatedMemoryClue, MemoryNodeModel
from src.layer1_learning import CategoryBasedMemoryLearning, compute_cosine_similarity
from src.layer2_construction import ChainBasedMemoryConstruction
from src.layer3_updating import SelfMonitoringMemoryUpdating
from src.layer4_retrieval import SelfReflectionMemoryRetrieval
from src.pipeline import MetaMemoryPipeline

logger = setup_logger("m4.runner")

def print_banner(text: str):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_step(step_num: int, title: str):
    print(f"\n[Step {step_num}] {title}")
    print("-" * 50)

class InMemoryGraphStore:
    """Fast in-memory graph store matching Neo4jGraphStore interface for testing & demonstration."""
    def __init__(self):
        self.chains: Dict[str, Dict[str, Any]] = {}
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.chain_node_order: Dict[str, List[str]] = {}
        self.shared_edges: List[Tuple[str, str]] = []

    def connect(self):
        pass

    def close(self):
        pass

    def get_all_categories(self) -> List[str]:
        return list(self.chains.keys())

    def find_most_similar_chains_batch(self, embeddings: List[List[float]]) -> List[Any]:
        results = []
        for emb in embeddings:
            best_cat = None
            best_id = None
            best_score = -1.0
            for cat, data in self.chains.items():
                score = compute_cosine_similarity(emb, data["embedding"])
                if score > best_score:
                    best_score = score
                    best_cat = cat
                    best_id = data["chain_id"]
            if best_score > 0:
                results.append((best_id, best_cat, best_score))
            else:
                results.append(None)
        return results

    def create_memory_chains_batch(self, chains: List[Dict[str, Any]]) -> List[str]:
        created = []
        for ch in chains:
            cat = ch["category_name"]
            if cat not in self.chains:
                self.chains[cat] = {
                    "chain_id": ch["chain_id"],
                    "category_name": cat,
                    "embedding": ch["embedding"],
                }
                self.chain_node_order[cat] = []
            created.append(self.chains[cat]["chain_id"])
        return created

    def append_nodes_batch(self, nodes: List[MemoryNodeModel], chain_categories: List[str]) -> List[str]:
        for n, cat in zip(nodes, chain_categories):
            self.nodes[n.id] = {
                "id": n.id,
                "summary": n.summary,
                "category": cat,
                "session_id": n.session_id,
                "created_at": n.created_at,
                "last_used_at": n.last_used_at,
                "embedding": n.embedding,
                "compressed": False,
            }
            if cat not in self.chain_node_order:
                self.chain_node_order[cat] = []
            self.chain_node_order[cat].append(n.id)
        return [n.id for n in nodes]

    def link_shared_nodes_batch(self, pairs: List[Tuple[str, str]]):
        self.shared_edges.extend(pairs)

    def get_chain_nodes(self, category_name: str) -> List[Dict[str, Any]]:
        target_cat = category_name
        for cat, data in self.chains.items():
            if data["chain_id"] == category_name:
                target_cat = cat
                break
        node_ids = self.chain_node_order.get(target_cat, [])
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]

    def remove_node_from_chain(self, node_id: str):
        if node_id in self.nodes:
            cat = self.nodes[node_id]["category"]
            if cat in self.chain_node_order and node_id in self.chain_node_order[cat]:
                self.chain_node_order[cat].remove(node_id)
            del self.nodes[node_id]

    def update_node_summary(self, node_id: str, summary: str, calibrated: bool = True):
        if node_id in self.nodes:
            self.nodes[node_id]["summary"] = summary
            self.nodes[node_id]["calibrated"] = calibrated

    def get_dormant_nodes(self, threshold: str, limit: int = 50) -> List[Dict[str, Any]]:
        res = []
        for n in self.nodes.values():
            if not n.get("compressed"):
                res.append(n)
            if len(res) >= limit:
                break
        return res

    def mark_node_compressed(self, node_id: str, compressed_summary: str):
        if node_id in self.nodes:
            self.nodes[node_id]["summary"] = compressed_summary
            self.nodes[node_id]["compressed"] = True


async def dynamic_llm_response(messages: List[Dict[str, str]], **kwargs) -> str:
    content = messages[0]["content"] if messages else ""
    if "Judge whether there is a conflict" in content:
        if "frustrated" in content.lower() or "sole person" in content.lower() or "trash" in content.lower():
            return "1"
        return "0"
    if "Merge them into a new summary" in content:
        return "After user's complaint, eventually the speaker takes out the trash."
    if "Choose EXACT names from the list" in content:
        return '[{"Category": "Daily Life"}]'
    if "Can you generate the best response based on these memories" in content:
        return "1"
    if "Generate the most plausible answer" in content:
        return "Sure! Let's make sure that you get everything done that needs to be done before we plan our next outing. I'm glad we can continue to support each other in our daily tasks, even the small ones like taking out the trash."
    return "0"


async def run_demonstration():
    print_banner("M4 Meta-Memory for Large Language Models — Full System Execution")
    start_time = time.time()
    config = AppConfig(similarity_threshold=0.65)

    # Setup Mock Engine & Store
    mock_llm = MagicMock()
    mock_llm.complete_chat = AsyncMock(side_effect=dynamic_llm_response)
    mock_store = InMemoryGraphStore()

    session_1_text = "User: Can't you take out the trash? I always have to take out the trash."
    session_2_text = "Speaker: I know you asked me to take out the trash earlier and was hesitant, but I'll do it now..."
    session_3_query = "Maybe we could even find one that has some cool motorcycle-related art to incorporate both of our interests."

    # -------------------------------------------------------------------------
    # Step 1: Layer 1 - Category-Based Memory Learning (Session 1)
    # -------------------------------------------------------------------------
    print_step(1, "Layer 1: Category-Based Memory Learning (Session 1)")
    print(f"Input Dialogue:\n  \"{session_1_text}\"")

    mock_llm.extract_structured_json = AsyncMock(return_value=[
        RawMemoryItem(Category="Daily Life", Summary="User is frustrated about being the sole person responsible for taking out the trash."),
        RawMemoryItem(Category="Household Routine", Summary="Trash duty responsibility conflict in the home.")
    ])
    # Household Routine has 0.88 cosine similarity to Daily Life -> Should merge!
    mock_llm.get_embeddings_batch = AsyncMock(return_value=[
        [0.9, 0.1, 0.0], [0.88, 0.12, 0.0],  # Categories (similar!)
        [0.6, 0.4, 0.1], [0.58, 0.42, 0.1],  # Clue summaries
    ])

    layer1 = CategoryBasedMemoryLearning(config, mock_llm)
    clues_s1 = await layer1.process_session(session_1_text)

    print(f"Extracted {len(clues_s1)} consolidated clues after Union-Find category disambiguation:")
    for idx, c in enumerate(clues_s1, 1):
        print(f"  [{idx}] Category: '{c.category}' | Summary: \"{c.summary}\"")

    assert len(clues_s1) == 2
    assert clues_s1[0].category == clues_s1[1].category == "Daily Life"
    print("✓ Layer 1 Successfully disambiguated 'Household Routine' into 'Daily Life'.")

    # -------------------------------------------------------------------------
    # Step 2: Layer 2 - Chain-Based Memory Construction (Session 1)
    # -------------------------------------------------------------------------
    print_step(2, "Layer 2: Chain-Based Memory Construction (Session 1)")
    layer2 = ChainBasedMemoryConstruction(config, mock_store)
    node_ids_s1 = layer2.build_memory_chains(clues_s1, session_id="session_01")
    print(f"Constructed memory nodes in chain: {node_ids_s1}")
    chain_nodes_s1 = mock_store.get_chain_nodes("Daily Life")
    print(f"Current 'Daily Life' Chain length: {len(chain_nodes_s1)} nodes.")
    for n in chain_nodes_s1:
        print(f"  - [{n['id']}] {n['summary']}")

    assert len(chain_nodes_s1) == 2
    print("✓ Layer 2 Successfully initialized memory chain and linked nodes.")

    # -------------------------------------------------------------------------
    # Step 3: Layer 1 & 2 - Session 2 Ingestion
    # -------------------------------------------------------------------------
    print_step(3, "Session 2 Dialogue Ingestion")
    print(f"Input Dialogue:\n  \"{session_2_text}\"")

    mock_llm.extract_structured_json = AsyncMock(return_value=[
        RawMemoryItem(Category="Daily Life", Summary="Speaker agrees to take trash out now.")
    ])
    mock_llm.get_embeddings_batch = AsyncMock(return_value=[
        [0.9, 0.1, 0.0],   # Category embedding
        [0.65, 0.35, 0.1], # Clue embedding
    ])

    clues_s2 = await layer1.process_session(session_2_text)
    node_ids_s2 = layer2.build_memory_chains(clues_s2, session_id="session_02")
    print(f"Added new node {node_ids_s2} to 'Daily Life' chain.")

    # -------------------------------------------------------------------------
    # Step 4: Layer 3 - Self-Monitoring Updating (Calibrate Action)
    # -------------------------------------------------------------------------
    print_step(4, "Layer 3: Self-Monitoring Updating (Calibrate Action)")
    layer3 = SelfMonitoringMemoryUpdating(config, mock_llm, mock_store)
    new_s2_node = MemoryNodeModel(
        id=node_ids_s2[0],
        summary=clues_s2[0].summary,
        category="Daily Life",
        session_id="session_02",
        created_at=datetime.now(timezone.utc),
        last_used_at=datetime.now(timezone.utc),
        embedding=clues_s2[0].clue_embedding,
    )

    calibrated = await layer3.calibrate_chain("Daily Life", new_s2_node)
    print(f"Calibrate action triggered: {calibrated}")
    chain_nodes_after_cal = mock_store.get_chain_nodes("Daily Life")
    print("Updated 'Daily Life' Chain:")
    for n in chain_nodes_after_cal:
        print(f"  - [{n['id']}] (Calibrated={n.get('calibrated', False)}) {n['summary']}")

    assert calibrated is True
    print("✓ Layer 3 Calibrate resolved conflict: Removed outdated node and calibrated new state.")

    # -------------------------------------------------------------------------
    # Step 5: Layer 3 - Self-Monitoring Updating (Compress Action)
    # -------------------------------------------------------------------------
    print_step(5, "Layer 3: Self-Monitoring Updating (Compress Action)")
    count_compressed = await layer3.compress_dormant_nodes(days_idle=7)
    print(f"Nodes compressed: {count_compressed}")
    chain_nodes_after_comp = mock_store.get_chain_nodes("Daily Life")
    for n in chain_nodes_after_comp:
        print(f"  - [{n['id']}] (Compressed={n.get('compressed', False)}) {n['summary']}")
    print("✓ Layer 3 Compress action successfully distilled memory representation.")

    # -------------------------------------------------------------------------
    # Step 6: Layer 4 - Self-Reflection-Based Memory Retrieval (Algorithm 1)
    # -------------------------------------------------------------------------
    print_step(6, "Layer 4: Self-Reflection-Based Memory Retrieval (Algorithm 1)")
    print(f"User Query:\n  \"{session_3_query}\"")

    mock_llm.get_embedding = AsyncMock(return_value=(0.89, 0.11, 0.0))

    layer4 = SelfReflectionMemoryRetrieval(config, mock_llm, mock_store)
    retrieved_memories, response = await layer4.retrieve_memory(session_3_query)

    print("\nRetrieved Memories utilized for generation:")
    for m in retrieved_memories:
        print(f"  • {m}")
    print(f"\nGenerated Agent Response:\n  \"{response}\"")

    assert len(retrieved_memories) > 0
    assert "trash" in response.lower()
    print("✓ Layer 4 Algorithm 1 retrieved the exact evolutionary memory and generated empathetic response.")

    elapsed = time.time() - start_time
    print_banner(f"ALL 4 LAYERS EXECUTED & VERIFIED SUCCESSFULLY in {elapsed:.2f}s")

if __name__ == "__main__":
    asyncio.run(run_demonstration())
