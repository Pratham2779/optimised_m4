#!/usr/bin/env bash
# =============================================================================
# M4 Meta-Memory Engine — LoCoMo Benchmark Test Runner
# Paper: Meta-Memory for Large Language Models (IEEE TASLPRO 2026)
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$SCRIPT_DIR"
if [ -d "$SCRIPT_DIR/m4_optimized" ]; then
    WORK_DIR="$SCRIPT_DIR/m4_optimized"
fi

cd "$WORK_DIR"

echo "================================================================================"
echo "    🧪 M4 META-MEMORY ENGINE — LOCOMO DATASET BENCHMARK TEST RUNNER             "
echo "================================================================================"

# Configure Database & LLM Engine
export NEO4J_URI="${NEO4J_URI:-neo4j+s://8d2d5bc7.databases.neo4j.io}"
export NEO4J_USER="${NEO4J_USER:-8d2d5bc7}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-bw8o9R3NXXApN2DXy72Gv7iXNI2KcVDGp_zZ7h0D5BU}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
export PRIMARY_LLM="qwen2.5:7b-instruct"
export EMBEDDING_MODEL="all-minilm"

# Performance Optimizations (Doc Page 4)
export OLLAMA_MAX_LOADED_MODELS=3
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_NUM_PARALLEL=4

# Detect Python Executable
PYTHON_EXEC="python3"
if [ -f "$WORK_DIR/.venv/bin/python" ]; then
    PYTHON_EXEC="$WORK_DIR/.venv/bin/python"
elif command -v python &> /dev/null; then
    PYTHON_EXEC="python"
fi

echo "Working directory: $WORK_DIR"
echo "Python binary:     $PYTHON_EXEC"
echo "Target Neo4j:      $NEO4J_URI"
echo "Target LLM:        $PRIMARY_LLM"
echo "Target Embeddings: $EMBEDDING_MODEL"
echo ""

# Ensure Ollama is running
if ! curl -s "$OLLAMA_BASE_URL/api/tags" > /dev/null 2>&1; then
    echo "⚠️ Ollama daemon not detected. Starting in background..."
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    sleep 3
fi

# Execute LoCoMo Evaluation Suite
echo "Running LoCoMo Evaluation: $PYTHON_EXEC $WORK_DIR/run_locomo_eval.py $@"
echo "--------------------------------------------------------------------------------"
$PYTHON_EXEC "$WORK_DIR/run_locomo_eval.py" "$@"

EXIT_CODE=$?
echo "--------------------------------------------------------------------------------"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Benchmark tests completed successfully!"
    echo "Results saved to: $WORK_DIR/locomo_evaluation_results.json"
else
    echo "❌ Benchmark tests exited with code $EXIT_CODE"
fi
exit $EXIT_CODE
