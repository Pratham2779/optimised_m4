#!/usr/bin/env bash
# Wrapper to execute run_all_paper_evaluations.py with environment variables
set -e
export OLLAMA_MAX_LOADED_MODELS=3
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_BASE_URL="http://localhost:11434"
export PRIMARY_LLM="qwen2.5:7b-instruct"
export EMBEDDING_MODEL="all-minilm"
export NEO4J_URI="${NEO4J_URI:-memory}"

python3 run_all_paper_evaluations.py "$@"
