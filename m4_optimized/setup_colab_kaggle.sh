#!/usr/bin/env bash
# =============================================================================
# M4 Meta-Memory Engine — Turnkey Google Colab & Kaggle Setup Script
# Paper: Meta-Memory for Large Language Models (IEEE TASLPRO 2026)
# =============================================================================
set -e

echo "================================================================================"
echo "    🚀 INITIALIZING M4 BENCHMARK ENVIRONMENT (GOOGLE COLAB / KAGGLE)          "
echo "================================================================================"

# 1. GPU Verification
echo ""
echo "🔍 Checking GPU Acceleration..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || nvidia-smi
    echo "✓ GPU Acceleration Detected! LLM and Embeddings will execute on CUDA."
else
    echo "⚠️ No dedicated NVIDIA GPU found. Inference will run on CPU."
fi

# 2. Install Ollama
echo ""
echo "⚙️ [1/4] Installing Ollama Engine..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
    echo "✓ Ollama installed successfully."
else
    echo "✓ Ollama is already installed."
fi

# 3. Start Ollama Server in Background
echo ""
echo "🚀 [2/4] Starting Ollama Server..."
pkill -f "ollama serve" || true
sleep 1
nohup ollama serve > /tmp/ollama_colab.log 2>&1 &
echo "Waiting for Ollama service to become ready..."

for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✓ Ollama server online at http://localhost:11434"
        break
    fi
    sleep 1
done

# 4. Pull Both Models
echo ""
echo "⬇️ [3/4] Pulling Required Models..."
echo "  • Pulling Primary LLM: qwen2.5:7b-instruct..."
ollama pull qwen2.5:7b-instruct

echo "  • Pulling Embedding Model: all-minilm..."
ollama pull all-minilm

echo ""
echo "✓ Pre-warmed models in Ollama:"
ollama list

# 5. Install Python Dependencies
echo ""
echo "📦 [4/4] Installing Python Requirements..."
export TMPDIR="/tmp"
export PIP_CACHE_DIR="/tmp/pip_cache"

if python3 -c "import pydantic, neo4j, httpx, numpy, orjson, tenacity, dotenv, async_lru, nltk, pytest" 2>/dev/null; then
    echo "✓ All core Python packages are already installed."
else
    echo "Installing missing packages (using fast binary wheels)..."
    pip install --no-cache-dir --prefer-binary -r requirements.txt
fi

# 6. Configure Environment
export NEO4J_URI="${NEO4J_URI:-neo4j+s://8d2d5bc7.databases.neo4j.io}"
export NEO4J_USER="${NEO4J_USER:-8d2d5bc7}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-bw8o9R3NXXApN2DXy72Gv7iXNI2KcVDGp_zZ7h0D5BU}"
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_MAX_LOADED_MODELS=3
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_NUM_PARALLEL=4

echo ""
echo "================================================================================"
echo "    ✅ SETUP COMPLETE! READY TO RUN M4 BENCHMARK EVALUATION                     "
echo "================================================================================"
echo "Run the LoCoMo benchmark evaluation with:"
echo "    python run_locomo_eval.py --samples-per-cat 3 --max-sessions 4"
echo "Or run the turnkey runner:"
echo "    ./run_benchmark.sh"
echo "================================================================================"
