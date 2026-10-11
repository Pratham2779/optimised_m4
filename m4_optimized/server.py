#!/usr/bin/env python3
"""
M4 Interactive Demonstration & Mentor Evaluation Server
Serves the web UI and provides REST APIs for live testing, custom input ingestion,
memory chain visualization, retrieval, and academic benchmark evaluations.
"""
import sys
import os
import json
import asyncio
import subprocess
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any, List

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(SCRIPT_DIR, "src")):
    PROJECT_ROOT = SCRIPT_DIR
else:
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if os.path.exists(os.path.join(SCRIPT_DIR, "index.html")):
    HTML_DIR = SCRIPT_DIR
elif os.path.exists(os.path.join(PROJECT_ROOT, "ui/index.html")):
    HTML_DIR = os.path.join(PROJECT_ROOT, "ui")
elif os.path.exists(os.path.join(PROJECT_ROOT, "index.html")):
    HTML_DIR = PROJECT_ROOT
else:
    HTML_DIR = SCRIPT_DIR

from src.config import AppConfig, setup_logger
from src.pipeline import MetaMemoryPipeline

logger = setup_logger("m4.ui_server")

config = AppConfig()
pipeline = MetaMemoryPipeline(config)

import threading

_LOOP = asyncio.new_event_loop()
def _start_background_loop():
    asyncio.set_event_loop(_LOOP)
    _LOOP.run_forever()

_LOOP_THREAD = threading.Thread(target=_start_background_loop, daemon=True)
_LOOP_THREAD.start()

def run_async(coro):
    """Run an async coroutine on the persistent server event loop."""
    future = asyncio.run_coroutine_threadsafe(coro, _LOOP)
    return future.result(timeout=180.0)

class M4RequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=HTML_DIR, **kwargs)

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/status":
            self.handle_status()
        elif path == "/api/chains":
            self.handle_get_chains()
        elif path == "/api/evaluation-results":
            self.handle_get_evaluation_results()
        else:
            if path == "/":
                self.path = "/index.html"
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            payload = json.loads(body) if body else {}
        except Exception:
            payload = {}

        if path == "/api/run-tests":
            self.handle_run_tests()
        elif path == "/api/run-demo":
            self.handle_run_demo()
        elif path == "/api/run-evaluations":
            self.handle_run_evaluations(payload)
        elif path == "/api/ingest":
            self.handle_ingest(payload)
        elif path == "/api/query":
            self.handle_query(payload)
        elif path == "/api/clear":
            self.handle_clear()
        else:
            self._send_json({"error": "Endpoint not found"}, status=404)

    def handle_status(self):
        db_connected = False
        db_chains = 0
        db_nodes = 0
        try:
            pipeline.startup()
            db_connected = True
            fb = getattr(pipeline.graph_store, "_fallback", None)
            if fb:
                db_chains = len(fb.chains)
                db_nodes = len(fb.nodes)
            elif getattr(pipeline.graph_store, "_driver", None):
                with pipeline.graph_store.session() as s:
                    c_res = s.run("MATCH (c:MemoryChain) RETURN count(c) AS count").single()
                    db_chains = c_res["count"] if c_res else 0
                    n_res = s.run("MATCH (n:MemoryNode) RETURN count(n) AS count").single()
                    db_nodes = n_res["count"] if n_res else 0
        except Exception as e:
            logger.error("DB check failed: %s", e)

        res = {
            "neo4j": {
                "uri": config.neo4j_uri,
                "connected": db_connected,
                "chains_count": db_chains,
                "nodes_count": db_nodes,
            },
            "ollama": {
                "base_url": config.ollama_base_url,
                "llm_model": config.llm_model,
                "embedding_model": config.embedding_model,
                "similarity_threshold": config.similarity_threshold,
            },
        }
        self._send_json(res)

    def handle_get_chains(self):
        try:
            categories = pipeline.graph_store.get_all_categories()
            chains_data = []
            for cat in categories:
                nodes = pipeline.graph_store.get_chain_nodes(cat)
                chains_data.append({
                    "category": cat,
                    "node_count": len(nodes),
                    "nodes": [
                        {
                            "id": n["id"],
                            "summary": n["summary"],
                            "session_id": n.get("session_id", ""),
                            "calibrated": n.get("calibrated", False),
                            "compressed": n.get("compressed", False),
                        }
                        for n in nodes
                    ]
                })
            self._send_json({"chains": chains_data})
        except Exception as e:
            self._send_json({"error": str(e), "chains": []}, status=500)

    def handle_get_evaluation_results(self):
        results_path = os.path.join(PROJECT_ROOT, "results/all_evaluation_results.json")
        if os.path.exists(results_path):
            try:
                with open(results_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._send_json({"success": True, "data": data})
                return
            except Exception as e:
                self._send_json({"success": False, "error": str(e)}, status=500)
                return
        self._send_json({"success": False, "message": "No previous evaluation results found."})

    def handle_run_evaluations(self, payload: Dict[str, Any]):
        t0 = time.time()
        suite = payload.get("suite", "all")
        locomo_samples = int(payload.get("locomo_samples", 2))
        msc_episodes = int(payload.get("msc_episodes", 2))
        try:
            py_bin = sys.executable
            cmd = [
                py_bin, "run_all_paper_evaluations.py",
                "--suite", suite,
                "--locomo_samples", str(locomo_samples),
                "--msc_episodes", str(msc_episodes),
                "--output_file", "results/all_evaluation_results.json"
            ]
            proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=900)
            elapsed = time.time() - t0

            results_path = os.path.join(PROJECT_ROOT, "results/all_evaluation_results.json")
            eval_data = {}
            if os.path.exists(results_path):
                with open(results_path, "r", encoding="utf-8") as f:
                    eval_data = json.load(f)

            self._send_json({
                "success": proc.returncode == 0,
                "elapsed": round(elapsed, 2),
                "output": proc.stdout + proc.stderr,
                "data": eval_data
            })
        except Exception as e:
            self._send_json({"success": False, "error": str(e)}, status=500)

    def handle_run_tests(self):
        t0 = time.time()
        try:
            venv_pytest = os.path.join(PROJECT_ROOT, ".venv/bin/pytest")
            cmd = [venv_pytest, "tests/test_unit.py", "-v", "--no-header"] if os.path.exists(venv_pytest) else ["pytest", "tests/test_unit.py", "-v", "--no-header"]
            proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=60)
            elapsed = time.time() - t0
            output = proc.stdout + proc.stderr

            tests = []
            for line in output.splitlines():
                if "::" in line and ("PASSED" in line or "FAILED" in line):
                    parts = line.split("::")
                    if len(parts) >= 2:
                        test_info = parts[1].split()
                        test_name = test_info[0]
                        status = "PASSED" if "PASSED" in line else "FAILED"
                        tests.append({"name": test_name, "status": status})

            self._send_json({
                "success": proc.returncode == 0,
                "elapsed": round(elapsed, 2),
                "total": len(tests),
                "passed": sum(1 for t in tests if t["status"] == "PASSED"),
                "tests": tests,
                "output": output,
            })
        except Exception as e:
            self._send_json({"success": False, "error": str(e)}, status=500)

    def handle_run_demo(self):
        t0 = time.time()
        try:
            py_bin = sys.executable
            cmd = [py_bin, "run_m4.py"]
            proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=60)
            elapsed = time.time() - t0
            self._send_json({
                "success": proc.returncode == 0,
                "elapsed": round(elapsed, 2),
                "output": proc.stdout + proc.stderr,
            })
        except Exception as e:
            self._send_json({"success": False, "error": str(e)}, status=500)

    def handle_ingest(self, payload: Dict[str, Any]):
        session_id = payload.get("session_id", f"sess_{int(time.time())}")
        text = payload.get("text", "").strip()
        calibrate = payload.get("calibrate", True)

        if not text:
            self._send_json({"error": "Empty conversation text"}, status=400)
            return

        t0 = time.time()
        try:
            res = run_async(pipeline.ingest_session(session_id, text, calibrate=calibrate))
            elapsed = time.time() - t0
            res["elapsed"] = round(elapsed, 2)
            self._send_json(res)
        except Exception as e:
            logger.error("Ingest failed: %s", e)
            self._send_json({"error": str(e)}, status=500)

    def handle_query(self, payload: Dict[str, Any]):
        query_text = payload.get("query", "").strip()
        if not query_text:
            self._send_json({"error": "Empty query"}, status=400)
            return

        t0 = time.time()
        try:
            memories, answer = run_async(pipeline.query(query_text))
            elapsed = time.time() - t0
            self._send_json({
                "query": query_text,
                "retrieved_memories": memories,
                "answer": answer,
                "elapsed": round(elapsed, 2),
            })
        except Exception as e:
            logger.error("Query failed: %s", e)
            self._send_json({"error": str(e)}, status=500)

    def handle_clear(self):
        try:
            fb = getattr(pipeline.graph_store, "_fallback", None)
            if fb:
                fb.nodes.clear()
                fb.chains.clear()
                fb.next_links.clear()
                fb.prev_links.clear()
                fb.shared_links.clear()
            elif getattr(pipeline.graph_store, "_driver", None):
                with pipeline.graph_store.session() as s:
                    s.run("MATCH (n:MemoryNode) DETACH DELETE n")
                    s.run("MATCH (c:MemoryChain) DETACH DELETE c")
            self._send_json({"success": True, "message": "Graph database cleared."})
        except Exception as e:
            self._send_json({"success": False, "error": str(e)}, status=500)

def run_server(port: int = 7860):
    server_address = ("0.0.0.0", port)
    httpd = HTTPServer(server_address, M4RequestHandler)
    print(f"🚀 M4 Interactive UI Server running at: http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7860
    run_server(port)
