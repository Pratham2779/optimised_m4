from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple
from src.config import AppConfig, setup_logger
from src.models import MemoryNodeModel
from src.llm_client import OllamaClient
from src.graph_store import Neo4jGraphStore
from src.layer1_learning import CategoryBasedMemoryLearning
from src.layer2_construction import ChainBasedMemoryConstruction
from src.layer3_updating import SelfMonitoringMemoryUpdating
from src.layer4_retrieval import SelfReflectionMemoryRetrieval

logger = setup_logger("m4.pipeline")

class MetaMemoryPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self.llm_client = OllamaClient(config)
        self.graph_store = Neo4jGraphStore(config)
        self.layer1 = CategoryBasedMemoryLearning(config, self.llm_client)
        self.layer2 = ChainBasedMemoryConstruction(config, self.graph_store)
        self.layer3 = SelfMonitoringMemoryUpdating(config, self.llm_client, self.graph_store)
        self.layer4 = SelfReflectionMemoryRetrieval(config, self.llm_client, self.graph_store)

    def startup(self):
        self.graph_store.connect()

    async def shutdown(self):
        await self.llm_client.close()
        self.graph_store.close()

    async def ingest_session(
        self, session_id: str, conversation_text: str, calibrate: bool = True
    ) -> Dict[str, Any]:
        """Ingest conversation: Layer 1 (Learning) -> Layer 2 (Construction) -> Layer 3 (Calibrate)."""
        logger.info("=== Ingesting Session [%s] ===", session_id)
        clues = await self.layer1.process_session(conversation_text)
        if not clues:
            return {
                "session_id": session_id,
                "clues_extracted": 0,
                "nodes_created": [],
                "calibrated_count": 0,
            }

        node_ids = self.layer2.build_memory_chains(clues, session_id)

        calibrated_count = 0
        if calibrate:
            now = datetime.now(timezone.utc)
            for clue, node_id in zip(clues, node_ids):
                node_model = MemoryNodeModel(
                    id=node_id,
                    summary=clue.summary,
                    category=clue.category,
                    session_id=session_id,
                    created_at=now,
                    last_used_at=now,
                    embedding=clue.clue_embedding,
                )
                if await self.layer3.calibrate_chain(clue.category, node_model):
                    calibrated_count += 1

        return {
            "session_id": session_id,
            "clues_extracted": len(clues),
            "nodes_created": node_ids,
            "calibrated_count": calibrated_count,
            "status": "Committed to Neo4j",
        }

    async def query(self, query_text: str) -> Tuple[List[str], str]:
        """Layer 4: Self-reflection-based memory retrieval and answer generation."""
        logger.info("=== Querying Pipeline: '%s' ===", query_text)
        return await self.layer4.retrieve_memory(query_text)

    async def compress(self, days_idle: int = 14) -> int:
        """Layer 3: Self-monitoring memory compression for dormant nodes."""
        logger.info("=== Running Compression on Dormant Nodes (>= %d days) ===", days_idle)
        return await self.layer3.compress_dormant_nodes(days_idle=days_idle)

