"""Optimized Neo4j store with batch UNWIND, managed transactions, vector index, and high-speed in-memory fallback for Colab/Kaggle."""
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Dict, Any, Set
import numpy as np
from neo4j import GraphDatabase, Driver, Session
from src.config import AppConfig, setup_logger
from src.models import MemoryNodeModel

logger = setup_logger("m4.graph_store")

class InMemoryGraphStore:
    """High-speed in-memory graph store engine for Google Colab / Kaggle environments."""
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.chains: Dict[str, Dict[str, Any]] = {}
        self.next_links: Dict[str, str] = {}
        self.prev_links: Dict[str, str] = {}
        self.shared_links: Set[Tuple[str, str]] = set()

    def connect(self):
        logger.info("InMemoryGraphStore initialized and active.")

    def close(self):
        pass

    def find_most_similar_chains_batch(self, embeddings: List[List[float]]) -> List[Optional[Tuple[str, str, float]]]:
        if not embeddings or not self.chains:
            return [None] * len(embeddings)
        chain_list = list(self.chains.values())
        chain_embs = np.array([c["embedding"] for c in chain_list], dtype=np.float32)
        chain_norms = np.linalg.norm(chain_embs, axis=1)
        results = []
        for emb in embeddings:
            q_vec = np.array(emb, dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)
            if q_norm == 0:
                results.append(None)
                continue
            sims = (chain_embs @ q_vec) / (chain_norms * q_norm + 1e-9)
            best_idx = int(np.argmax(sims))
            results.append((chain_list[best_idx]["id"], chain_list[best_idx]["category_name"], float(sims[best_idx])))
        return results

    def upsert_chains_and_head_nodes_batch(self, clues: List[Any], session_id: str) -> List[str]:
        now = datetime.now(timezone.utc)
        node_ids = []
        for clue in clues:
            nid = f"mem_{uuid.uuid4().hex[:12]}"
            node_ids.append(nid)
            node_data = {
                "id": nid,
                "summary": clue.summary,
                "category": clue.category,
                "session_id": session_id,
                "created_at": now.isoformat(),
                "last_used_at": now.isoformat(),
                "embedding": clue.clue_embedding,
                "compressed": False,
            }
            self.nodes[nid] = node_data

            cat = clue.category
            if cat not in self.chains:
                cid = f"chain_{uuid.uuid4().hex[:8]}"
                self.chains[cat] = {
                    "id": cid,
                    "category_name": cat,
                    "head": nid,
                    "tail": nid,
                    "embedding": clue.category_embedding,
                }
            else:
                old_tail = self.chains[cat]["tail"]
                self.next_links[old_tail] = nid
                self.prev_links[nid] = old_tail
                self.chains[cat]["tail"] = nid
    def create_memory_chains_batch(self, chains: List[Dict[str, Any]]) -> List[str]:
        created_ids = []
        for c in chains:
            cat = c["category_name"]
            cid = c.get("chain_id") or f"chain_{uuid.uuid4().hex[:12]}"
            created_ids.append(cid)
            if cat not in self.chains:
                self.chains[cat] = {
                    "id": cid,
                    "category_name": cat,
                    "head": None,
                    "tail": None,
                    "embedding": c.get("embedding", []),
                }
        return created_ids

    def append_nodes_batch(self, nodes: List[Any], categories: List[str]) -> List[str]:
        node_ids = []
        for n, cat in zip(nodes, categories):
            nid = getattr(n, "id", None) or f"mem_{uuid.uuid4().hex[:12]}"
            node_ids.append(nid)
            summary = getattr(n, "summary", "")
            session_id = getattr(n, "session_id", "")
            created_at = getattr(n, "created_at", None)
            created_at_str = created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at)
            embedding = getattr(n, "embedding", [])
            node_data = {
                "id": nid,
                "summary": summary,
                "category": cat,
                "session_id": session_id,
                "created_at": created_at_str,
                "last_used_at": created_at_str,
                "embedding": embedding,
                "compressed": False,
            }
            self.nodes[nid] = node_data

            if cat not in self.chains:
                cid = f"chain_{uuid.uuid4().hex[:12]}"
                self.chains[cat] = {
                    "id": cid,
                    "category_name": cat,
                    "head": nid,
                    "tail": nid,
                    "embedding": embedding,
                }
            else:
                chain = self.chains[cat]
                if chain.get("head") is None:
                    chain["head"] = nid
                    chain["tail"] = nid
                else:
                    old_tail = chain["tail"]
                    self.next_links[old_tail] = nid
                    self.prev_links[nid] = old_tail
                    chain["tail"] = nid
        return node_ids

    def link_shared_nodes_batch(self, pairs: List[Tuple[str, str]]) -> None:
        for a, b in pairs:
            self.shared_links.add((min(a, b), max(a, b)))

    def get_all_categories(self) -> List[str]:
        return list(self.chains.keys())

    def get_chain_nodes(self, category_name: str) -> List[Dict[str, Any]]:
        chain = self.chains.get(category_name)
        if not chain:
            return []
        nodes = []
        curr = chain.get("head")
        while curr:
            if curr in self.nodes:
                nodes.append(self.nodes[curr])
            curr = self.next_links.get(curr)
        return nodes

    def remove_node_from_chain(self, node_id: str) -> None:
        if node_id not in self.nodes:
            return
        cat = self.nodes[node_id]["category"]
        chain = self.chains.get(cat)
        prev_node = self.prev_links.get(node_id)
        next_node = self.next_links.get(node_id)

        if prev_node and next_node:
            self.next_links[prev_node] = next_node
            self.prev_links[next_node] = prev_node
        elif prev_node:
            self.next_links.pop(prev_node, None)
        elif next_node:
            self.prev_links.pop(next_node, None)

        if chain:
            if chain.get("head") == node_id:
                chain["head"] = next_node
            if chain.get("tail") == node_id:
                chain["tail"] = prev_node

        self.next_links.pop(node_id, None)
        self.prev_links.pop(node_id, None)
        self.nodes.pop(node_id, None)

    def update_node_summary(self, node_id: str, summary: str, calibrated: bool = True) -> None:
        if node_id in self.nodes:
            self.nodes[node_id]["summary"] = summary
            self.nodes[node_id]["calibrated"] = calibrated

    def get_dormant_nodes(self, threshold: str, limit: int = 50) -> List[Dict[str, Any]]:
        results = []
        for n in self.nodes.values():
            if n.get("last_used_at", "") < threshold and not n.get("compressed", False):
                results.append({"id": n["id"], "summary": n["summary"]})
                if len(results) >= limit:
                    break
        return results

    def mark_node_compressed(self, node_id: str, compressed_summary: str) -> None:
        if node_id in self.nodes:
            self.nodes[node_id]["summary"] = compressed_summary
            self.nodes[node_id]["compressed"] = True


class Neo4jGraphStore:
    def __init__(self, config: AppConfig):
        self.config = config
        self._driver: Optional[Driver] = None
        self._fallback: Optional[InMemoryGraphStore] = None

    def connect(self):
        if str(self.config.neo4j_uri).lower().startswith("memory"):
            logger.info("Configured for InMemoryGraphStore.")
            self._fallback = InMemoryGraphStore(self.config)
            return

        if not self._driver and not self._fallback:
            try:
                drv = GraphDatabase.driver(
                    self.config.neo4j_uri,
                    auth=(self.config.neo4j_user, self.config.neo4j_password),
                    max_connection_pool_size=50,
                    connection_acquisition_timeout=3.0,
                )
                drv.verify_connectivity()
                self._driver = drv
            except Exception as e:
                logger.warning("Neo4j server connection unverified (%s). Activating seamless InMemoryGraphStore fallback.", e)
                self._driver = None
                self._fallback = InMemoryGraphStore(self.config)

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None
        if self._fallback:
            self._fallback.close()
            self._fallback = None

    @contextmanager
    def session(self) -> Session:
        if not self._driver and not self._fallback:
            self.connect()
        if self._fallback:
            raise RuntimeError("Running in in-memory mode, session() not available.")
        if self._driver:
            s = self._driver.session()
            try:
                yield s
            finally:
                s.close()
        else:
            raise RuntimeError("Neo4j driver is not available.")

    def find_most_similar_chains_batch(self, embeddings: List[List[float]]) -> List[Optional[Tuple[str, str, float]]]:
        if not self._driver and not self._fallback:
            self.connect()
        if self._fallback:
            return self._fallback.find_most_similar_chains_batch(embeddings)
        if not embeddings:
            return []
        query = """
        UNWIND range(0, size($embeddings) - 1) AS idx
        WITH idx, $embeddings[idx] AS emb
        CALL db.index.vector.queryNodes('category_chain_embeddings', 1, emb)
        YIELD node, score
        RETURN idx, node.id AS chain_id, node.category_name AS category_name, score
        ORDER BY idx
        """
        results = []
        with self.session() as s:
            records = s.run(query, embeddings=embeddings).data()
            by_idx = {r["idx"]: r for r in records}
            for i in range(len(embeddings)):
                r = by_idx.get(i)
                if r:
                    results.append((r["chain_id"], r["category_name"], float(r["score"])))
                else:
                    results.append(None)
        return results

    def upsert_chains_and_head_nodes_batch(self, clues: List[Any], session_id: str) -> List[str]:
        if self._fallback:
            return self._fallback.upsert_chains_and_head_nodes_batch(clues, session_id)
        if not clues:
            return []
        now = datetime.now(timezone.utc)
        nodes = [
            MemoryNodeModel(
                summary=c.summary,
                category=c.category,
                session_id=session_id,
                created_at=now,
                last_used_at=now,
                embedding=c.clue_embedding,
            )
            for c in clues
        ]
        chain_categories = [c.category for c in clues]
        chain_embeddings = [c.category_embedding for c in clues]

        query = """
        UNWIND range(0, size($node_ids) - 1) AS idx
        WITH idx,
             $node_ids[idx] AS nid,
             $summaries[idx] AS summary,
             $categories[idx] AS cat,
             $session_ids[idx] AS sess_id,
             $created_ats[idx] AS created_at,
             $last_used_ats[idx] AS last_used_at,
             $embeddings[idx] AS emb,
             $chain_cats[idx] AS chain_cat,
             $chain_embeddings[idx] AS chain_emb

        MERGE (c:MemoryChain {category_name: chain_cat})
        ON CREATE SET c.id = 'chain_' + substring(randomUUID(), 0, 8),
                      c.embedding = chain_emb,
                      c.created_at = created_at

        CREATE (n:MemoryNode {
            id: nid,
            summary: summary,
            category: cat,
            session_id: sess_id,
            created_at: created_at,
            last_used_at: last_used_at,
            embedding: emb,
            calibrated: false
        })

        WITH c, n
        OPTIONAL MATCH (c)-[old_tail:TAIL_NODE]->(tail:MemoryNode)
        FOREACH (_ IN CASE WHEN tail IS NULL THEN [1] ELSE [] END |
            MERGE (c)-[:HEAD_NODE]->(n)
            MERGE (c)-[:TAIL_NODE]->(n)
        )
        FOREACH (_ IN CASE WHEN tail IS NOT NULL THEN [1] ELSE [] END |
            CREATE (tail)-[:NEXT_IN_CHAIN]->(n)
            DELETE old_tail
            MERGE (c)-[:TAIL_NODE]->(n)
        )
        """
        with self.session() as s:
            s.run(
                query,
                node_ids=[n.id for n in nodes],
                summaries=[n.summary for n in nodes],
                categories=[n.category for n in nodes],
                chain_embeddings=chain_embeddings,
                session_ids=[n.session_id for n in nodes],
                created_ats=[n.created_at.isoformat() for n in nodes],
                last_used_ats=[n.last_used_at.isoformat() for n in nodes],
                embeddings=[n.embedding for n in nodes],
                chain_cats=chain_categories,
            )
        return [n.id for n in nodes]

    def create_memory_chains_batch(self, chains: List[Dict[str, Any]]) -> List[str]:
        if self._fallback:
            return self._fallback.create_memory_chains_batch(chains)
        if not chains:
            return []
        query = """
        UNWIND $chains AS c
        MERGE (ch:MemoryChain {category_name: c.category_name})
        ON CREATE SET ch.id = c.chain_id, ch.embedding = c.embedding
        RETURN ch.id AS id
        """
        with self.session() as s:
            res = s.run(query, chains=chains).data()
            return [r["id"] for r in res]

    def append_nodes_batch(self, nodes: List[Any], categories: List[str]) -> List[str]:
        if self._fallback:
            return self._fallback.append_nodes_batch(nodes, categories)
        if not nodes:
            return []
        query = """
        UNWIND range(0, size($node_ids) - 1) AS idx
        WITH idx,
             $node_ids[idx] AS nid,
             $summaries[idx] AS summary,
             $categories[idx] AS cat,
             $session_ids[idx] AS sess_id,
             $created_ats[idx] AS created_at,
             $last_used_ats[idx] AS last_used_at,
             $embeddings[idx] AS emb

        MATCH (c:MemoryChain {category_name: cat})
        CREATE (n:MemoryNode {
            id: nid,
            summary: summary,
            category: cat,
            session_id: sess_id,
            created_at: created_at,
            last_used_at: last_used_at,
            embedding: emb,
            calibrated: false
        })

        WITH c, n
        OPTIONAL MATCH (c)-[old_tail:TAIL_NODE]->(tail:MemoryNode)
        FOREACH (_ IN CASE WHEN tail IS NULL THEN [1] ELSE [] END |
            MERGE (c)-[:HEAD_NODE]->(n)
            MERGE (c)-[:TAIL_NODE]->(n)
        )
        FOREACH (_ IN CASE WHEN tail IS NOT NULL THEN [1] ELSE [] END |
            CREATE (tail)-[:NEXT_IN_CHAIN]->(n)
            DELETE old_tail
            MERGE (c)-[:TAIL_NODE]->(n)
        )
        """
        with self.session() as s:
            s.run(
                query,
                node_ids=[n.id for n in nodes],
                summaries=[n.summary for n in nodes],
                categories=categories,
                session_ids=[n.session_id for n in nodes],
                created_ats=[n.created_at.isoformat() for n in nodes],
                last_used_ats=[n.last_used_at.isoformat() for n in nodes],
                embeddings=[n.embedding for n in nodes],
            )
        return [n.id for n in nodes]

    def link_shared_nodes_batch(self, pairs: List[Tuple[str, str]]) -> None:
        if self._fallback:
            return self._fallback.link_shared_nodes_batch(pairs)
        if not pairs:
            return
        query = """
        UNWIND $pairs AS pair
        MATCH (a:MemoryNode {id: pair[0]}), (b:MemoryNode {id: pair[1]})
        MERGE (a)-[:SHARED_WITH]->(b)
        """
        with self.session() as s:
            s.run(query, pairs=[[a, b] for a, b in pairs])

    def get_all_categories(self) -> List[str]:
        if not self._driver and not self._fallback:
            self.connect()
        if self._fallback:
            return self._fallback.get_all_categories()
        query = "MATCH (c:MemoryChain) RETURN c.category_name AS cat"
        with self.session() as s:
            return [r["cat"] for r in s.run(query).data()]

    def get_chain_nodes(self, category_name: str) -> List[Dict[str, Any]]:
        if not self._driver and not self._fallback:
            self.connect()
        if self._fallback:
            return self._fallback.get_chain_nodes(category_name)
        query = """
        MATCH (c:MemoryChain)
        WHERE c.category_name = $category_name OR c.id = $category_name
        MATCH (c)-[:HEAD_NODE]->(head:MemoryNode)
        MATCH path = (head)-[:NEXT_IN_CHAIN*0..]->(tail:MemoryNode)
        WHERE NOT (tail)-[:NEXT_IN_CHAIN]->()
        RETURN [n in nodes(path) | {id: n.id, summary: n.summary, category: n.category,
                session_id: n.session_id, embedding: n.embedding, compressed: coalesce(n.compressed, false)}] AS chain_nodes
        """
        with self.session() as s:
            records = s.run(query, category_name=category_name).data()
            if not records:
                return []
            return records[0].get("chain_nodes", []) or []

    def remove_node_from_chain(self, node_id: str) -> None:
        if self._fallback:
            return self._fallback.remove_node_from_chain(node_id)
        query = """
        MATCH (n:MemoryNode {id: $nid})
        OPTIONAL MATCH (c:MemoryChain)-[h:HEAD_NODE]->(n)
        OPTIONAL MATCH (c)-[t:TAIL_NODE]->(n)
        OPTIONAL MATCH (prev:MemoryNode)-[r_prev:NEXT_IN_CHAIN]->(n)
        OPTIONAL MATCH (n)-[r_next:NEXT_IN_CHAIN]->(next:MemoryNode)
        FOREACH (_ IN CASE WHEN prev IS NOT NULL AND next IS NOT NULL THEN [1] ELSE [] END |
            CREATE (prev)-[:NEXT_IN_CHAIN]->(next)
        )
        FOREACH (_ IN CASE WHEN h IS NOT NULL AND next IS NOT NULL THEN [1] ELSE [] END |
            CREATE (c)-[:HEAD_NODE]->(next)
        )
        FOREACH (_ IN CASE WHEN t IS NOT NULL AND prev IS NOT NULL THEN [1] ELSE [] END |
            CREATE (c)-[:TAIL_NODE]->(prev)
        )
        DETACH DELETE n
        """
        with self.session() as s:
            s.execute_write(lambda tx: tx.run(query, nid=node_id))

    def update_node_summary(self, node_id: str, summary: str, calibrated: bool = True) -> None:
        if self._fallback:
            return self._fallback.update_node_summary(node_id, summary, calibrated)
        query = """
        MATCH (n:MemoryNode {id: $nid})
        SET n.summary = $summary, n.calibrated = $calibrated
        """
        with self.session() as s:
            s.execute_write(lambda tx: tx.run(query, nid=node_id, summary=summary, calibrated=calibrated))

    def get_dormant_nodes(self, threshold: str, limit: int = 50) -> List[Dict[str, Any]]:
        if self._fallback:
            return self._fallback.get_dormant_nodes(threshold, limit)
        query = """
        MATCH (n:MemoryNode)
        WHERE n.last_used_at < $threshold AND (n.compressed IS NULL OR n.compressed = false)
        RETURN n.id AS id, n.summary AS summary LIMIT $limit
        """
        with self.session() as s:
            return s.run(query, threshold=threshold, limit=limit).data()

    def mark_node_compressed(self, node_id: str, compressed_summary: str) -> None:
        if self._fallback:
            return self._fallback.mark_node_compressed(node_id, compressed_summary)
        query = """
        MATCH (n:MemoryNode {id: $nid})
        SET n.summary = $summary, n.compressed = true
        """
        with self.session() as s:
            s.execute_write(lambda tx: tx.run(query, nid=node_id, summary=compressed_summary))
