#!/usr/bin/env python3
"""
M4 Custom Example Tester — CLI & Automated Custom Demonstration
Allows running custom multi-session scenarios directly through the live M4 Pipeline.
"""
import sys
import argparse
import asyncio
import time
from src.config import AppConfig, setup_logger
from src.pipeline import MetaMemoryPipeline

logger = setup_logger("m4.custom_tester")

CUSTOM_SCENARIOS = {
    "1": {
        "title": "Scenario 1: Preference Conflict & Calibration (Tea vs. Coffee)",
        "sessions": [
            {
                "id": "scenario1_s1",
                "text": "User: I always drink iced black coffee first thing at 8 AM. I can never function without coffee."
            },
            {
                "id": "scenario1_s2",
                "text": "User: My doctor advised me to stop drinking coffee due to acid reflux. Now I strictly drink warm oat milk matcha tea in the morning instead of coffee."
            }
        ],
        "query": "What does the user drink in the morning now?"
    },
    "2": {
        "title": "Scenario 2: Professional Role & Technical Architecture Update",
        "sessions": [
            {
                "id": "scenario2_s1",
                "text": "User: I am developing our company backend using monolithic Django with PostgreSQL on AWS EC2."
            },
            {
                "id": "scenario2_s2",
                "text": "User: We completely decommissioned the Django monolith. We migrated our entire architecture to microservices using Go, FastAPI, and Kubernetes."
            }
        ],
        "query": "What tech stack and architecture is the user's company backend using?"
    },
    "3": {
        "title": "Scenario 3: Pet Adoption & Shared Cross-Category Graph",
        "sessions": [
            {
                "id": "scenario3_s1",
                "text": "User: I love landscape nature photography, especially taking macro shots of flowers during weekend morning walks."
            },
            {
                "id": "scenario3_s2",
                "text": "User: Last Saturday I adopted an active golden retriever puppy named Barnaby. Now I take Barnaby on my morning walks through the botanical gardens."
            }
        ],
        "query": "Who accompanies the user during their morning walks, and what do they do?"
    }
}

async def run_scenario(scenario_key: str):
    sc = CUSTOM_SCENARIOS.get(scenario_key)
    if not sc:
        print(f"Unknown scenario key: {scenario_key}")
        return

    print("\n" + "=" * 75)
    print(f"  {sc['title']}")
    print("=" * 75)

    config = AppConfig()
    pipeline = MetaMemoryPipeline(config)
    pipeline.startup()

    try:
        for idx, sess in enumerate(sc["sessions"], 1):
            print(f"\n[Step {idx}] Ingesting Session: {sess['id']}")
            print(f"Input: \"{sess['text']}\"")
            t0 = time.time()
            res = await pipeline.ingest_session(sess["id"], sess["text"], calibrate=True)
            elapsed = time.time() - t0

            print(f"  → Layer 1 Clues Extracted: {res['clues_extracted']}")
            print(f"  → Layer 2 Node IDs Linked: {res['nodes_created']}")
            print(f"  → Layer 3 Calibrated Updates: {res['calibrated_count']}")
            print(f"  → Status: {res['status']} ({elapsed:.2f}s)")

        print(f"\n[Step {len(sc['sessions']) + 1}] Layer 4 Heuristic Retrieval (Algorithm 1)")
        print(f"Query: \"{sc['query']}\"")
        t0 = time.time()
        memories, answer = await pipeline.query(sc["query"])
        elapsed = time.time() - t0

        print(f"\nRetrieved Memory Trace ({len(memories)} clues):")
        for m in memories:
            print(f"  • {m}")

        print(f"\nGenerated Agent Response ({elapsed:.2f}s):")
        print(f"  \"{answer}\"")
        print("\n" + "-" * 75)
        print("✓ Scenario completed successfully.")

    finally:
        await pipeline.shutdown()

async def run_custom_interactive():
    config = AppConfig()
    pipeline = MetaMemoryPipeline(config)
    pipeline.startup()
    print("\n=== M4 Custom Interactive Shell ===")
    print("Enter 'exit' at any prompt to quit.\n")

    try:
        while True:
            action = input("\nChoose action: [1] Ingest Session, [2] Query Memory, [3] Exit: ").strip()
            if action in ["3", "exit", "quit"]:
                break
            elif action == "1":
                sess_id = input("Enter Session ID (e.g. sess_01): ").strip()
                if not sess_id or sess_id == "exit":
                    break
                text = input("Enter Conversation Dialogue: ").strip()
                if not text or text == "exit":
                    break
                print("\nIngesting...")
                t0 = time.time()
                res = await pipeline.ingest_session(sess_id, text, calibrate=True)
                print(f"✓ Ingested in {time.time() - t0:.2f}s:")
                print(f"  - Clues: {res['clues_extracted']}, Nodes: {res['nodes_created']}, Calibrated: {res['calibrated_count']}")
            elif action == "2":
                q = input("Enter Query: ").strip()
                if not q or q == "exit":
                    break
                print("\nRetrieving...")
                t0 = time.time()
                memories, ans = await pipeline.query(q)
                print(f"✓ Answered in {time.time() - t0:.2f}s:")
                print("Retrieved memories:")
                for m in memories:
                    print(f"  • {m}")
                print(f"\nResponse:\n{ans}")
    finally:
        await pipeline.shutdown()

def main():
    parser = argparse.ArgumentParser(description="Test M4 with Custom Examples")
    parser.add_argument("--scenario", choices=["1", "2", "3", "all"], help="Run preset custom scenario (1, 2, 3, or all)")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive CLI session")

    args = parser.parse_args()

    if args.interactive:
        asyncio.run(run_custom_interactive())
    elif args.scenario == "all":
        for k in ["1", "2", "3"]:
            asyncio.run(run_scenario(k))
    elif args.scenario:
        asyncio.run(run_scenario(args.scenario))
    else:
        print("Please choose an option:")
        print("  1. Run Scenario 1: Preference Conflict & Calibration (Tea vs. Coffee)")
        print("  2. Run Scenario 2: Technical Architecture Evolution (Monolith to Microservices)")
        print("  3. Run Scenario 3: Shared Cross-Category Graph (Nature Photography & Barnaby)")
        print("  4. Run All 3 Scenarios")
        print("  5. Interactive CLI Prompt")
        choice = input("\nEnter choice [1-5]: ").strip()
        if choice in ["1", "2", "3"]:
            asyncio.run(run_scenario(choice))
        elif choice == "4":
            for k in ["1", "2", "3"]:
                asyncio.run(run_scenario(k))
        elif choice == "5":
            asyncio.run(run_custom_interactive())
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
