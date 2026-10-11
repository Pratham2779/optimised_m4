#!/usr/bin/env bash
# =============================================================================
# M4 Quick-Launch Script for Mentor Demonstration
# Automatically checks/opens the SSH tunnel to Colab GPU and starts the Web UI
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_TARGET="${1:-root@colab.prathameshgurav.site}"
SSH_PASS="${2:-1234}"
SSH_PORT="${3:-22}"

echo "================================================================================"
echo "    🚀 STARTING M4 MENTOR DEMONSTRATION ENVIRONMENT                           "
echo "================================================================================"
echo "Remote GPU Host: $SSH_TARGET (Port $SSH_PORT)"
echo "Neo4j AuraDB:    Configured via .env"
echo "Local Web UI:    http://localhost:7860"
echo "================================================================================"

# 1. Check if Ollama is accessible locally on 11434
echo ""
echo "🔍 Checking connection to Remote GPU Ollama (port 11434)..."
if curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
    echo "✓ Port 11434 is active and responding!"
else
    echo "⚡ Opening SSH port forwarding to $SSH_TARGET..."
    pkill -f "ssh.*11434:localhost:11434" || true
    sleep 1

    if command -v sshpass &> /dev/null; then
        sshpass -p "$SSH_PASS" ssh -f -N -L 11434:localhost:11434 -p "$SSH_PORT" -o StrictHostKeyChecking=no "$SSH_TARGET"
    else
        ssh -f -N -L 11434:localhost:11434 -p "$SSH_PORT" -o StrictHostKeyChecking=no "$SSH_TARGET"
    fi

    # Wait for tunnel
    for i in {1..10}; do
        if curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
            echo "✓ SSH Tunnel established! Remote GPU Ollama is connected."
            break
        fi
        sleep 1
    done
fi

# 2. Verify Models on GPU
echo ""
echo "🤖 Active models on Remote GPU:"
curl -s http://127.0.0.1:11434/api/tags | grep -o '"name":"[^"]*"' | tr '\n' ' ' || echo "Ollama ready"
echo ""

# 3. Start Local UI Server on port 7860
echo ""
echo "🌐 Starting M4 Demonstration UI Server on http://localhost:7860..."
pkill -f "server.py 7860" || true
pkill -f "server.py 8080" || true
sleep 1

VENV_PY="$SCRIPT_DIR/m4_optimized/.venv/bin/python"
if [ ! -f "$VENV_PY" ]; then
    VENV_PY="$SCRIPT_DIR/.venv/bin/python"
fi
if [ ! -f "$VENV_PY" ]; then
    VENV_PY="python3"
fi

if [ -f "$SCRIPT_DIR/m4_optimized/ui/server.py" ]; then
    SERVER_PY="$SCRIPT_DIR/m4_optimized/ui/server.py"
elif [ -f "$SCRIPT_DIR/ui/server.py" ]; then
    SERVER_PY="$SCRIPT_DIR/ui/server.py"
else
    SERVER_PY="$SCRIPT_DIR/server.py"
fi

exec "$VENV_PY" "$SERVER_PY" 7860

