import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock
from src.config import AppConfig
from src.models import RawMemoryItem, DisambiguatedMemoryClue, MemoryNodeModel
from src.llm_client import OllamaClient
from src.layer1_learning import CategoryBasedMemoryLearning, compute_cosine_similarity, _UnionFind
from src.layer2_construction import ChainBasedMemoryConstruction
from src.layer3_updating import SelfMonitoringMemoryUpdating
from src.layer4_retrieval import SelfReflectionMemoryRetrieval
from src.pipeline import MetaMemoryPipeline

# =============================================================================
# Mathematical & Utility Tests
# =============================================================================

def test_union_find_basic():
    uf = _UnionFind(5)
    uf.union(0, 1)
    uf.union(2, 3)
    uf.union(1, 3)
    assert uf.find(0) == uf.find(2)
    assert uf.find(4) != uf.find(0)

def test_cosine_identity():
    assert pytest.approx(compute_cosine_similarity([1, 0, 0], [1, 0, 0]), 1e-6) == 1.0

def test_cosine_orthogonal():
    assert pytest.approx(compute_cosine_similarity([1, 0], [0, 1]), 1e-6) == 0.0

def test_cosine_zero_vectors():
    assert compute_cosine_similarity([0, 0], [1, 1]) == 0.0
    assert compute_cosine_similarity([0, 0], [0, 0]) == 0.0

# =============================================================================
# LLM Client Tests
# =============================================================================

def test_ollama_client_parse_json_markdown():
    config = AppConfig()
    client = OllamaClient(config)
    raw_md = """Here is the extracted information:
```json
[
  {"Category": "Daily Life", "Summary": "User drinks coffee every morning."},
  {"Category": "Work", "Summary": "User is a software engineer."}
]
```"""
    parsed = client._parse_json_array(raw_md)
    assert parsed is not None
    assert len(parsed) == 2
    assert parsed[0].category == "Daily Life"
    assert parsed[1].summary == "User is a software engineer."

def test_ollama_client_parse_raw_json():
    config = AppConfig()
    client = OllamaClient(config)
    raw_json = '[{"Category": "Hobbies", "Summary": "User loves hiking."}]'
    parsed = client._parse_json_array(raw_json)
    assert parsed is not None
    assert len(parsed) == 1
    assert parsed[0].category == "Hobbies"

def test_ollama_client_parse_invalid():
    config = AppConfig()
    client = OllamaClient(config)
    assert client._parse_json_array("Not a json") is None
    assert client._parse_json_array('{"key": "value"}') is None

# =============================================================================
# Layer 1: Category-Based Memory Learning Tests
# =============================================================================

async def test_layer1_union_find_merge():
    config = AppConfig(similarity_threshold=0.7)
    mock_client = MagicMock()
    mock_client.extract_structured_json = AsyncMock(return_value=[
        RawMemoryItem(Category="Dietary Preference", Summary="Drinks green tea"),
        RawMemoryItem(Category="Dietary Choices", Summary="Avoids red meat"),
    ])
    mock_client.get_embeddings_batch = AsyncMock(return_value=[
        [1.0, 0.0, 0.0], [0.99, 0.01, 0.0],
        [0.1, 0.2, 0.3], [0.4, 0.5, 0.6],
    ])
    layer1 = CategoryBasedMemoryLearning(config, mock_client)
    result = await layer1.process_session("test dialogue")
    assert len(result) == 2
    assert result[0].category == "Dietary Preference"
    assert result[1].category == "Dietary Preference"

async def test_layer1_empty_extraction():
    config = AppConfig()
    mock_client = MagicMock()
    mock_client.extract_structured_json = AsyncMock(return_value=[])
    layer1 = CategoryBasedMemoryLearning(config, mock_client)
    result = await layer1.process_session("empty conversation")
    assert result == []

# =============================================================================
# Layer 2: Chain-Based Memory Construction Tests
# =============================================================================

def test_layer2_batch_chain_linking():
    config = AppConfig(similarity_threshold=0.5)
    mock_store = MagicMock()
    mock_store.find_most_similar_chains_batch.return_value = [
        ("chain_123", "Fitness Routine", 0.85)
    ]
    clues = [DisambiguatedMemoryClue(
        category="Workout Habits", summary="Swimming every Tuesday.",
        category_embedding=[0.8, 0.2], clue_embedding=[0.1, 0.9])]
    layer2 = ChainBasedMemoryConstruction(config, mock_store)
    ids = layer2.build_memory_chains(clues, session_id="test_01")
    assert len(ids) == 1
    mock_store.append_nodes_batch.assert_called_once()
    nodes, cats = mock_store.append_nodes_batch.call_args[0]
    assert cats[0] == "Fitness Routine"

def test_layer2_creates_new_chain_and_shared_links():
    config = AppConfig(similarity_threshold=0.8)
    mock_store = MagicMock()
    mock_store.find_most_similar_chains_batch.return_value = [None, None]
    mock_store.create_memory_chains_batch.return_value = ["c1", "c2"]

    clues = [
        DisambiguatedMemoryClue(
            category="Daily Life", summary="Takes trash out",
            category_embedding=[1.0, 0.0], clue_embedding=[0.5, 0.5]
        ),
        DisambiguatedMemoryClue(
            category="Art", summary="Likes painting",
            category_embedding=[0.0, 1.0], clue_embedding=[0.2, 0.8]
        ),
    ]
    layer2 = ChainBasedMemoryConstruction(config, mock_store)
    ids = layer2.build_memory_chains(clues, session_id="test_02")
    assert len(ids) == 2
    mock_store.create_memory_chains_batch.assert_called_once()
    mock_store.append_nodes_batch.assert_called_once()
    mock_store.link_shared_nodes_batch.assert_called_once()

# =============================================================================
# Layer 3: Self-Monitoring-Based Memory Updating Tests
# =============================================================================

async def test_layer3_calibrate_with_conflict():
    config = AppConfig()
    mock_llm = MagicMock()
    mock_store = MagicMock()

    # Existing node on Daily Life chain
    mock_store.get_chain_nodes.return_value = [
        {"id": "node_old", "summary": "User complains about taking trash out.", "embedding": [1.0, 0.0]}
    ]

    # LLM detects conflict ('1') and provides merged summary
    mock_llm.complete_chat = AsyncMock(side_effect=[
        "1",  # Conflict detected
        "User previously complained, but now speaker assists with trash."  # Merged summary
    ])

    layer3 = SelfMonitoringMemoryUpdating(config, mock_llm, mock_store)
    new_node = MemoryNodeModel(
        id="node_new",
        summary="Speaker agreed to take trash out now.",
        category="Daily Life",
        session_id="s2",
        created_at=datetime.now(timezone.utc),
        last_used_at=datetime.now(timezone.utc),
        embedding=[0.95, 0.05],
    )

    calibrated = await layer3.calibrate_chain("Daily Life", new_node)
    assert calibrated is True
    mock_store.remove_node_from_chain.assert_called_once_with("node_old")
    mock_store.update_node_summary.assert_called_once_with(
        "node_new",
        "User previously complained, but now speaker assists with trash.",
        calibrated=True
    )

async def test_layer3_calibrate_no_conflict():
    config = AppConfig()
    mock_llm = MagicMock()
    mock_store = MagicMock()

    mock_store.get_chain_nodes.return_value = [
        {"id": "node_1", "summary": "User loves Italian pasta.", "embedding": [0.8, 0.2]}
    ]
    mock_llm.complete_chat = AsyncMock(return_value="0")  # No conflict

    layer3 = SelfMonitoringMemoryUpdating(config, mock_llm, mock_store)
    new_node = MemoryNodeModel(
        id="node_2",
        summary="User also enjoys pizza on weekends.",
        category="Food",
        session_id="s2",
        created_at=datetime.now(timezone.utc),
        last_used_at=datetime.now(timezone.utc),
        embedding=[0.85, 0.15],
    )

    calibrated = await layer3.calibrate_chain("Food", new_node)
    assert calibrated is False
    mock_store.remove_node_from_chain.assert_not_called()
    mock_store.update_node_summary.assert_not_called()

async def test_layer3_compress_dormant_nodes():
    from unittest.mock import patch
    config = AppConfig()
    mock_store = MagicMock()
    mock_llm = MagicMock()

    mock_store.get_dormant_nodes.return_value = [
        {"id": "node_dormant", "summary": "The user was planning to buy a motorcycle in summer."}
    ]

    mock_compressor = MagicMock()
    mock_compressor.compress_prompt.return_value = {
        "compressed_prompt": "user planning to buy a motorcycle in summer"
    }

    layer3 = SelfMonitoringMemoryUpdating(config, mock_llm, mock_store)
    with patch("src.layer3_updating.get_compressor", return_value=mock_compressor):
        count = await layer3.compress_dormant_nodes(days_idle=14)
        assert count == 1
        mock_store.mark_node_compressed.assert_called_once()


# =============================================================================
# Layer 4: Self-Reflection-Based Memory Retrieval Tests
# =============================================================================

async def test_layer4_retrieval_bidirectional_and_order():
    config = AppConfig()
    mock_llm = MagicMock()
    mock_store = MagicMock()

    mock_store.get_all_categories.return_value = ["Daily Life", "Work"]

    # Category matching prompt -> ["Daily Life"]
    mock_llm.get_embedding = AsyncMock(return_value=(0.9, 0.1))

    mock_llm.get_embeddings_batch = AsyncMock(return_value=[[0.9, 0.1], [0.1, 0.9]])
    # Chain has 3 nodes: index 0 (earliest), index 1 (middle), index 2 (latest)
    nodes = [
        {"id": "n0", "summary": "Earliest memory: Started new morning routine.", "embedding": [0.5, 0.5]},
        {"id": "n1", "summary": "Middle memory: Drinks black coffee every morning.", "embedding": [0.9, 0.1]},
        {"id": "n2", "summary": "Latest memory: Switched to pour-over coffee.", "embedding": [0.7, 0.3]},
    ]
    mock_store.get_chain_nodes.return_value = nodes

    # Responses from LLM:
    # 1. Category extraction -> [{"Category": "Daily Life"}]
    # 2. Final response generation -> "The user drinks coffee as part of their routine."
    mock_llm.complete_chat = AsyncMock(side_effect=[
        '[{"Category": "Daily Life"}]',
        "The user drinks coffee as part of their routine."
    ])

    layer4 = SelfReflectionMemoryRetrieval(config, mock_llm, mock_store)
    memories, answer = await layer4.retrieve_memory("What coffee does the user drink?")

    # Verify order preservation: backward (n0) inserted at index 0, forward (n2) appended
    assert len(memories) == 3
    assert memories[0] == "Earliest memory: Started new morning routine."
    assert memories[1] == "Middle memory: Drinks black coffee every morning."
    assert memories[2] == "Latest memory: Switched to pour-over coffee."
    assert "coffee" in answer.lower()

async def test_layer4_retrieval_immediate_confidence():
    config = AppConfig()
    mock_llm = MagicMock()
    mock_store = MagicMock()

    mock_store.get_all_categories.return_value = ["Work"]
    mock_llm.get_embedding = AsyncMock(return_value=(1.0, 0.0))
    mock_llm.get_embeddings_batch = AsyncMock(return_value=[[1.0, 0.0]])
    mock_store.get_chain_nodes.return_value = [
        {"id": "n0", "summary": "User works at Tech Corp.", "embedding": [1.0, 0.0]}
    ]

    mock_llm.complete_chat = AsyncMock(side_effect=[
        '[{"Category": "Work"}]',
        "User works at Tech Corp."
    ])

    layer4 = SelfReflectionMemoryRetrieval(config, mock_llm, mock_store)
    memories, answer = await layer4.retrieve_memory("Where does the user work?")
    assert len(memories) == 1
    assert "Tech Corp" in answer

# =============================================================================
# MetaMemoryPipeline End-to-End Orchestration Tests
# =============================================================================

async def test_pipeline_ingest_and_query_flow():
    config = AppConfig(similarity_threshold=0.6)
    pipeline = MetaMemoryPipeline(config)

    # Mock components
    pipeline.layer1.process_session = AsyncMock(return_value=[
        DisambiguatedMemoryClue(
            category="Daily Life",
            summary="User walks dog every morning at 7 AM.",
            category_embedding=[1.0, 0.0],
            clue_embedding=[0.8, 0.2],
        )
    ])
    pipeline.layer2.build_memory_chains = MagicMock(return_value=["mem_node_01"])
    pipeline.layer3.calibrate_chain = AsyncMock(return_value=False)
    pipeline.layer4.retrieve_memory = AsyncMock(return_value=(
        ["User walks dog every morning at 7 AM."],
        "The user walks their dog at 7 AM each morning."
    ))
    pipeline.layer3.compress_dormant_nodes = AsyncMock(return_value=2)

    # Ingest session
    ingest_res = await pipeline.ingest_session("session_1", "User: I walk my dog at 7 AM.")
    assert ingest_res["clues_extracted"] == 1
    assert ingest_res["nodes_created"] == ["mem_node_01"]
    assert ingest_res["calibrated_count"] == 0

    # Query
    memories, answer = await pipeline.query("When does the user walk their dog?")
    assert len(memories) == 1
    assert "7 AM" in answer

    # Compress
    compressed = await pipeline.compress(days_idle=14)
    assert compressed == 2
