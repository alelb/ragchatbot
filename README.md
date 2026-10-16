# Course Materials RAG System

A Retrieval-Augmented Generation (RAG) system designed to answer questions about course materials using semantic search and AI-powered responses.

## Overview

This application is a full-stack web application that enables users to query course materials and receive intelligent, context-aware responses. It uses ChromaDB for vector storage, Anthropic's Claude for AI generation, and provides a web interface for interaction.


## Prerequisites

- Python 3.13 or higher
- uv (Python package manager)
- An Anthropic API key (for Claude AI)
- **For Windows**: Use Git Bash to run the application commands - [Download Git for Windows](https://git-scm.com/downloads/win)

## Installation

1. **Install uv** (if not already installed)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install Python dependencies**
   ```bash
   uv sync
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```bash
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

   **Running with a local model instead (no API key):** install [Ollama](https://ollama.com), pull a model with tool-calling support (e.g. `ollama pull gemma4`) and use this `.env` instead:
   ```bash
   LLM_BASE_URL=http://localhost:11434
   ANTHROPIC_MODEL=gemma4
   ```
   The app talks to Ollama through its Anthropic-compatible API, so no code changes are needed to switch between the two.

   Minimum requirements for the local setup:
   - **Ollama 0.30.9 or newer** (required by `gemma4`; it also provides the Anthropic-compatible `/v1/messages` endpoint).
   - **A model with tool calling**: `ollama show <model>` must list `tools` under *Capabilities*, otherwise the model cannot search the course materials.
   - **Memory**: 16 GB of RAM.
   - **Disk**: the model (about 6.6 GB for `gemma4`) plus about 1 GB for the Python environment and the embedding model.
   - **GPU**: optional.

   Tested on an Apple M2 Pro with 16 GB of RAM: answers take about 5–7 seconds with `gemma4`.

## Running the Application

### Quick Start

Use the provided shell script:
```bash
chmod +x run.sh
./run.sh
```

### Manual Start

```bash
cd backend
uv run uvicorn app:app --reload --port 8000
```

The application will be available at:
- Web Interface: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`

