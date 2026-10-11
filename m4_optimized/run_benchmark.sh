#!/usr/bin/env bash
# =============================================================================
# M4 Meta-Memory Engine — Turnkey Benchmark & Evaluation Runner
# Paper: Meta-Memory for Large Language Models (IEEE TASLPRO 2026)
# Phase 1: LoCoMo Benchmark Evaluation Suite
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$SCRIPT_DIR/m4_optimized"
if [ ! -d "$WORK_DIR" ]; then
    WORK_DIR="$SCRIPT_DIR"
fi

cd "$WORK_DIR"

echo "================================================================================"
echo "    🚀 M4 META-MEMORY ENGINE — TURNKEY BENCHMARK EVALUATOR (LoCoMo PHASE 1)    "
echo "================================================================================"
echo "Working directory: $WORK_DIR"
echo ""

# -----------------------------------------------------------------------------
# 1. Environment & Database Configuration
# -----------------------------------------------------------------------------
export NEO4J_URI="${NEO4J_URI:-neo4j+s://8d2d5bc7.databases.neo4j.io}"
export NEO4J_USER="${NEO4J_USER:-8d2d5bc7}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-bw8o9R3NXXApN2DXy72Gv7iXNI2KcVDGp_zZ7h0D5BU}"

export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
export PRIMARY_LLM="qwen2.5:7b-instruct"
export EMBEDDING_MODEL="all-minilm"

# High-throughput Ollama Environment Optimizations (Doc Page 4)
export OLLAMA_MAX_LOADED_MODELS=3
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_NUM_PARALLEL=4

echo "📋 Configuration Profile:"
echo "  • Primary LLM:        $PRIMARY_LLM"
echo "  • Embedding Model:    $EMBEDDING_MODEL"
echo "  • Neo4j Target:       $NEO4J_URI"
echo "  • Neo4j User:         $NEO4J_USER"
echo "  • Ollama Endpoint:    $OLLAMA_BASE_URL"
echo ""

# -----------------------------------------------------------------------------
# 2. Virtual Environment Setup & Verification
# -----------------------------------------------------------------------------
VENV_DIR="$WORK_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "⚙️ Creating Python virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --upgrade pip
    "$VENV_DIR/bin/pip" install -r "$WORK_DIR/requirements.txt"
fi

# Source virtualenv
source "$VENV_DIR/bin/activate"
PYTHON_BIN="$VENV_DIR/bin/python"
echo "✓ Python environment ready: $($PYTHON_BIN --version)"

# -----------------------------------------------------------------------------
# 3. Ollama Service & Models Verification
# -----------------------------------------------------------------------------
echo ""
echo "🔍 Checking Ollama Service..."
if ! curl -s "$OLLAMA_BASE_URL/api/tags" > /dev/null 2>&1; then
    echo "⚠️ Ollama daemon is not responding at $OLLAMA_BASE_URL. Attempting to start 'ollama serve'..."
    nohup ollama serve > /tmp/ollama_m4.log 2>&1 &
    sleep 3
    if ! curl -s "$OLLAMA_BASE_URL/api/tags" > /dev/null 2>&1; then
        echo "❌ Error: Could not connect to Ollama. Please start Ollama before continuing."
        exit 1
    fi
fi
echo "✓ Ollama service is active."

# Check required models
echo "Checking model weights..."
MODELS_LIST=$(curl -s "$OLLAMA_BASE_URL/api/tags")

if echo "$MODELS_LIST" | grep -q "$PRIMARY_LLM"; then
    echo "✓ Found LLM model: $PRIMARY_LLM"
else
    echo "⬇️ Pulling required LLM: $PRIMARY_LLM (this may take a few minutes)..."
    ollama pull "$PRIMARY_LLM"
fi

if echo "$MODELS_LIST" | grep -q "$EMBEDDING_MODEL"; then
    echo "✓ Found Embedding model: $EMBEDDING_MODEL"
else
    echo "⬇️ Pulling required Embedding model: $EMBEDDING_MODEL..."
    ollama pull "$EMBEDDING_MODEL"
fi

# -----------------------------------------------------------------------------
# 4. Neo4j Verification & Auto-Fallback
# -----------------------------------------------------------------------------
echo ""
echo "🔍 Verifying Neo4j Connectivity..."
echo "Testing connection to Cloud Neo4j AuraDB: $NEO4J_URI..."
if ! $PYTHON_BIN -c "
from neo4j import GraphDatabase
import os, sys
try:
    d = GraphDatabase.driver(os.environ['NEO4J_URI'], auth=(os.environ['NEO4J_USER'], os.environ['NEO4J_PASSWORD']), connection_timeout=4.0)
    d.verify_connectivity()
    d.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
"; then
    echo "⚠️ Cloud Neo4j AuraDB is dormant/unreachable (NXDOMAIN / paused tier)."
    echo "🔄 Seamlessly falling back to Local Neo4j Docker container..."
    export NEO4J_URI="bolt://localhost:7687"
    export NEO4J_USER="neo4j"
    export NEO4J_PASSWORD="StrongPassword123!"
    docker start m4_neo4j 2>/dev/null || true
    sleep 2
    echo "✓ Active Neo4j endpoint: $NEO4J_URI"
else
    echo "✓ Connected successfully to Neo4j Cloud AuraDB: $NEO4J_URI"
fi

# -----------------------------------------------------------------------------
# 5. Launch LoCoMo Evaluation Suite
# -----------------------------------------------------------------------------
echo ""
echo "================================================================================"
echo "    📊 EXECUTING LOCOMO BENCHMARK EVALUATION SUITE                              "
echo "================================================================================"
echo "Running: $PYTHON_BIN $WORK_DIR/run_locomo_eval.py $@"
echo ""

$PYTHON_BIN "$WORK_DIR/run_locomo_eval.py" "$@"

EXIT_CODE=$?

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "    ✅ BENCHMARK EVALUATION COMPLETED SUCCESSFULLY!                            "
    echo "    Detailed report saved to: $WORK_DIR/locomo_evaluation_results.json        "
else
    echo "    ❌ BENCHMARK EVALUATION ENCOUNTERED AN ERROR (Exit Code: $EXIT_CODE)        "
fi
echo "================================================================================"
exit $EXIT_CODE
