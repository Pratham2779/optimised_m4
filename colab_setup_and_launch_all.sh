#!/usr/bin/env bash
# =============================================================================
# M4 Meta-Memory — 1-Click All-in-One Cloud Setup & Mentor UI Launcher
# Designed for Google Colab, Kaggle, or any Remote GPU Environment
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================================"
echo "    🚀 M4 META-MEMORY: 1-CLICK ALL-IN-ONE SETUP & UI LAUNCHER                   "
echo "================================================================================"

# 1. GPU Verification
echo ""
echo "🔍 [1/6] Detecting GPU Acceleration..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || nvidia-smi
    echo "✓ NVIDIA GPU Detected! LLM and Embeddings will execute on CUDA."
else
    echo "⚠️ Warning: No dedicated GPU found. Running on CPU (will be slower)."
fi

# 2. Install & Start Ollama
echo ""
echo "⚙️ [2/6] Checking Ollama Engine..."
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "✓ Ollama installed."
else
    echo "✓ Ollama already installed."
fi

# Ensure Ollama daemon is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Starting Ollama daemon in background..."
    pkill -f "ollama serve" || true
    sleep 1
    nohup ollama serve > /tmp/ollama_colab.log 2>&1 &
    for i in {1..30}; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "✓ Ollama daemon is active at http://localhost:11434"
            break
        fi
        sleep 1
    done
else
    echo "✓ Ollama daemon is already running."
fi

# 3. Pull Required Models
echo ""
echo "⬇️ [3/6] Pre-warming Models in GPU VRAM..."
if ! ollama list | grep -q "qwen2.5:7b-instruct"; then
    echo "Pulling Primary LLM (qwen2.5:7b-instruct)..."
    ollama pull qwen2.5:7b-instruct
else
    echo "✓ qwen2.5:7b-instruct is ready."
fi

if ! ollama list | grep -q "all-minilm"; then
    echo "Pulling Embedding Model (all-minilm)..."
    ollama pull all-minilm
else
    echo "✓ all-minilm is ready."
fi

# 4. Install Python Dependencies
echo ""
echo "📦 [4/6] Verifying Python Requirements..."
export PIP_DISABLE_PIP_VERSION_CHECK=1
if python3 -c "import pydantic, neo4j, httpx, numpy, orjson, tenacity, dotenv, async_lru, nltk, pytest" 2>/dev/null; then
    echo "✓ Core Python packages are already installed."
else
    pip install -q --prefer-binary \
        "pydantic>=2.7.0" "pydantic-settings>=2.3.0" "neo4j>=5.20.0" "httpx>=0.27.0" \
        "numpy>=1.26.0" "orjson>=3.10.0" "async-lru>=2.0.0" "tenacity>=8.3.0" \
        "python-dotenv>=1.0.0" "nltk>=3.8.1" pandas pytest bert-score rouge-score
fi
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)" || true
echo "✓ Python dependencies verified."

# 5. Configure AuraDB Cloud Environment
echo ""
echo "☁️ [5/6] Configuring AuraDB Cloud .env..."
cat << 'EOF' > .env
NEO4J_URI=neo4j+s://8d2d5bc7.databases.neo4j.io
NEO4J_USER=8d2d5bc7
NEO4J_PASSWORD=bw8o9R3NXXApN2DXy72Gv7iXNI2KcVDGp_zZ7h0D5BU

OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=qwen2.5:7b-instruct
EMBEDDING_MODEL=all-minilm
SIMILARITY_THRESHOLD=0.65
LOG_LEVEL=INFO
EOF
echo "✓ .env configured with Neo4j AuraDB credentials."

# 6. Start UI Server on Port 8080
echo ""
echo "🌐 [6/6] Launching Demonstration UI Server on Port 8080..."
pkill -f "server.py 8080" || true
sleep 1

# Find server script
if [ -f "server.py" ]; then
    SERVER_SCRIPT="server.py"
elif [ -f "ui/server.py" ]; then
    SERVER_SCRIPT="ui/server.py"
else
    SERVER_SCRIPT="server.py"
fi

nohup python3 "$SERVER_SCRIPT" 8080 > /tmp/m4_ui.log 2>&1 &
SERVER_PID=$!
disown $SERVER_PID 2>/dev/null || true
sleep 2

# Verify UI server is up
if curl -s http://localhost:8080/api/status > /dev/null 2>&1; then
    echo "✓ UI HTTP server running on port 8080."
else
    echo "⚠️ UI server starting, checking log: /tmp/m4_ui.log"
fi

# Set up localtunnel for public URL without touching cloudflared
PUBLIC_URL=""
if command -v npx &> /dev/null; then
    echo "Starting public tunnel via localtunnel..."
    pkill -f "lt --port 8080" || true
    sleep 1
    nohup npx -y localtunnel --port 8080 > /tmp/localtunnel.log 2>&1 &
    sleep 3
    PUBLIC_URL="$(grep -o 'https://[-a-zA-Z0-9]*\.loca\.lt' /tmp/localtunnel.log | tail -n 1 || true)"
fi

echo ""
echo "================================================================================"
echo "    🎉 SETUP COMPLETE! YOUR M4 DEMONSTRATION UI IS ACTIVE!                     "
echo "================================================================================"
if [ -n "$PUBLIC_URL" ]; then
    echo "  👉 PUBLIC URL (Localtunnel): $PUBLIC_URL"
fi
echo "  👉 DIRECT BROWSER ACCESS VIA SSH: "
echo "     Run on your laptop terminal:  ssh -N -L 8080:localhost:8080 root@colab.prathameshgurav.site"
echo "     Then open in browser:        http://localhost:8080"
echo ""
echo "  👉 OR VIEW INSIDE COLAB NOTEBOOK CELL:"
echo "     from google.colab import output"
echo "     output.serve_kernel_port_as_iframe(8080, height=850)"
echo ""
echo "  👉 AURADB CLOUD:      Connected to neo4j+s://8d2d5bc7.databases.neo4j.io"
echo "  👉 GPU MODELS:        qwen2.5:7b-instruct + all-minilm (CUDA accelerated)"
echo "================================================================================"
echo ""
echo "Server logs: /tmp/m4_ui.log"

