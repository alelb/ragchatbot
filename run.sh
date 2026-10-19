#!/bin/bash

# Create necessary directories
mkdir -p docs

# Check if backend directory exists
if [ ! -d "backend" ]; then
    echo "Error: backend directory not found"
    exit 1
fi

# Read a setting from the environment, falling back to .env (same precedence as python-dotenv)
get_setting() {
    local value="${!1}"
    if [ -z "$value" ] && [ -f .env ]; then
        value=$(grep -E "^[[:space:]]*$1=" .env | tail -n 1 | cut -d= -f2- | tr -d "\"'" | xargs)
    fi
    echo "$value"
}

LLM_BASE_URL=$(get_setting LLM_BASE_URL)
ANTHROPIC_MODEL=$(get_setting ANTHROPIC_MODEL)

echo "Starting Course Materials RAG System..."

if [ -n "$LLM_BASE_URL" ]; then
    # Local model via an Anthropic-compatible server (Ollama)
    echo "Using local model '${ANTHROPIC_MODEL}' at ${LLM_BASE_URL}"

    if ! curl -sf "${LLM_BASE_URL}/api/version" > /dev/null; then
        echo "Error: Ollama is not reachable at ${LLM_BASE_URL}"
        echo "Start it by opening the Ollama app, or run 'ollama serve' in another terminal."
        exit 1
    fi

    if [ -z "$ANTHROPIC_MODEL" ]; then
        echo "Error: ANTHROPIC_MODEL is not set. Add e.g. ANTHROPIC_MODEL=gemma4 to .env"
        exit 1
    fi

    # Ollama lists models with an explicit tag, so a bare name means ':latest'
    model="$ANTHROPIC_MODEL"
    [[ "$model" != *:* ]] && model="${model}:latest"
    if ! curl -sf "${LLM_BASE_URL}/api/tags" | grep -q "\"name\":\"${model}\""; then
        echo "Error: model '${ANTHROPIC_MODEL}' is not available in Ollama."
        if [ "$ANTHROPIC_MODEL" = "gemma4-8k" ] && [ -f Modelfile ]; then
            echo "Create it with: ollama pull gemma4 && ollama create gemma4-8k -f Modelfile"
        else
            echo "Download it with: ollama pull ${ANTHROPIC_MODEL}"
        fi
        exit 1
    fi
else
    echo "Using the Anthropic API - make sure ANTHROPIC_API_KEY is set in .env"
fi

# Change to backend directory and start the server
cd backend && uv run uvicorn app:app --reload --port 8000
