#!/usr/bin/env python3
"""
Live End-to-End Verification of M4 with Real Local Ollama and Real Local Neo4j Docker.
"""
import asyncio
import time
from src.config import AppConfig, setup_logger
from src.pipeline import MetaMemoryPipeline

logger = setup_logger("m4.live_test")

async def test_live_m4():
    print("=" * 70)
    print("  M4 Live End-to-End Test: Local Ollama + Local Neo4j Docker")
    print("=" * 70)

    config = AppConfig(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="StrongPassword123!",
        ollama_base_url="http://localhost:11434",
        llm_model="qwen2.5:7b-instruct",
        embedding_model="all-minilm",
        similarity_threshold=0.65,
    )

    pipeline = MetaMemoryPipeline(config)
    pipeline.startup()
    print("✓ Successfully connected to Live Neo4j Docker on port 7687.")

    try:
        # Session 1
        print("\n--- Ingesting Session 1 via Live Ollama & Neo4j ---")
        s1_text = "User: I really love drinking iced matcha latte every afternoon around 3 PM."
        t0 = time.time()
        res1 = await pipeline.ingest_session("live_sess_01", s1_text, calibrate=True)
        t1 = time.time()
        print(f"Session 1 Result (took {t1 - t0:.2f}s): {res1}")

        # Session 2 (conflict/update)
        print("\n--- Ingesting Session 2 via Live Ollama & Neo4j ---")
        s2_text = "User: I had to stop drinking iced matcha latte because caffeine keeps me awake at night. Now I only drink chamomile tea in the afternoon."
        t2 = time.time()
        res2 = await pipeline.ingest_session("live_sess_02", s2_text, calibrate=True)
        t3 = time.time()
        print(f"Session 2 Result (took {t3 - t2:.2f}s): {res2}")

        # Query
        print("\n--- Querying Memory via Layer 4 (Algorithm 1) ---")
        query_text = "What does the user prefer to drink in the afternoon?"
        t4 = time.time()
        memories, answer = await pipeline.query(query_text)
        t5 = time.time()

        print(f"\nRetrieved Memories ({len(memories)}):")
        for m in memories:
            print(f"  • {m}")
        print(f"\nGenerated Answer (took {t5 - t4:.2f}s):")
        print(f"  \"{answer}\"")

        print("\n" + "=" * 70)
        print("  LIVE TEST COMPLETED SUCCESSFULLY!")
        print("=" * 70)

    finally:
        await pipeline.shutdown()

if __name__ == "__main__":
    asyncio.run(test_live_m4())
