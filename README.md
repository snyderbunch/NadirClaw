# NadirClaw

Open-source LLM router that saves you money. Simple prompts go to cheap/local models, complex prompts go to premium models -- automatically.

NadirClaw sits between your AI tool and your LLM providers as an OpenAI-compatible proxy. It classifies every prompt in ~10ms and routes it to the right model. Works with any tool that speaks the OpenAI API: [OpenClaw](https://openclaw.dev), [Codex](https://github.com/openai/codex), Claude Code, Continue, Cursor, or plain `curl`.

```
Your AI Tool ──> NadirClaw (:8000/v1) ──> simple prompts  ──> Ollama / GPT-4o-mini (free/cheap)
                                      ──> complex prompts ──> Claude / GPT-4o (premium)
```

## Quick Start

```bash
curl -fsSL https://raw.githubusercontent.com/doramirdor/NadirClaw/main/install.sh | sh
```

Then start the router:

```bash
nadirclaw serve --verbose
```

That's it. NadirClaw starts on `http://localhost:8000` with sensible defaults (Claude Sonnet for complex, Ollama Llama 3.1 for simple).

## Prerequisites

- **Python 3.10+**
- **git**
- **At least one LLM provider:**
  - [Ollama](https://ollama.com) running locally (free, no API key needed)
  - [Anthropic API key](https://console.anthropic.com/) for Claude models
  - [OpenAI API key](https://platform.openai.com/) for GPT models
  - Or any provider supported by [LiteLLM](https://docs.litellm.ai/docs/providers)

## Install

### One-line install (recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/doramirdor/NadirClaw/main/install.sh | sh
```

This clones the repo to `~/.nadirclaw`, creates a virtual environment, installs dependencies, and adds `nadirclaw` to your PATH. Run it again to update.

### Manual install

```bash
git clone https://github.com/doramirdor/NadirClaw.git
cd NadirClaw
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### Uninstall

```bash
rm -rf ~/.nadirclaw
sudo rm -f /usr/local/bin/nadirclaw
```

## Configure

Copy the example env file and add your API keys:

```bash
cp .env.example .env
```

The two key settings are which model handles each tier:

```bash
NADIRCLAW_SIMPLE_MODEL=ollama/llama3.1:8b          # cheap/local model
NADIRCLAW_COMPLEX_MODEL=claude-sonnet-4-20250514   # premium model
```

Set the API key(s) for whichever providers you use:

```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

### Example Setups

| Setup | Simple Model | Complex Model | API Keys Needed |
|---|---|---|---|
| **Claude + Ollama** (default) | `ollama/llama3.1:8b` | `claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` |
| **Claude + Claude** | `claude-haiku-4-20250514` | `claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` |
| **OpenAI + Ollama** | `ollama/llama3.1:8b` | `gpt-4o` | `OPENAI_API_KEY` |
| **OpenAI + OpenAI** | `gpt-4o-mini` | `gpt-4o` | `OPENAI_API_KEY` |
| **Fully local** | `ollama/llama3.1:8b` | `ollama/qwen3:32b` | None |

You can use **any model** that [LiteLLM supports](https://docs.litellm.ai/docs/providers) -- the provider is auto-detected from the model name.

## Usage with Ollama

If you're running [Ollama](https://ollama.com) locally, NadirClaw works out of the box with no API keys:

```bash
# Fully local setup -- no API keys, no cost
NADIRCLAW_SIMPLE_MODEL=ollama/llama3.1:8b \
NADIRCLAW_COMPLEX_MODEL=ollama/qwen3:32b \
nadirclaw serve --verbose
```

Or mix local + cloud:

```bash
nadirclaw serve \
  --simple-model ollama/llama3.1:8b \
  --complex-model claude-sonnet-4-20250514 \
  --verbose
```

### Recommended Ollama Models

| Model | Size | Good For |
|---|---|---|
| `llama3.1:8b` | 4.7 GB | Simple tier (fast, good enough) |
| `qwen3:32b` | 19 GB | Complex tier (local, no API cost) |
| `qwen3-coder` | 19 GB | Code-heavy complex tier |
| `deepseek-r1:14b` | 9 GB | Reasoning-heavy complex tier |

## Usage with OpenClaw

[OpenClaw](https://openclaw.dev) is a personal AI assistant that bridges messaging services to AI coding agents. NadirClaw integrates as a model provider so OpenClaw's requests are automatically routed to the right model.

### Quick Setup

```bash
# Auto-configure OpenClaw to use NadirClaw
nadirclaw openclaw onboard

# Start the router
nadirclaw serve
```

This writes NadirClaw as a provider in `~/.openclaw/openclaw.json` with model `nadirclaw/auto`. If OpenClaw is already running, it will auto-reload the config -- no restart needed.

### Configure Only (Without Launching)

```bash
nadirclaw openclaw onboard
# Then start NadirClaw separately when ready:
nadirclaw serve
```

### What It Does

`nadirclaw openclaw onboard` adds this to your OpenClaw config:

```json
{
  "models": {
    "providers": {
      "nadirclaw": {
        "baseUrl": "http://localhost:8000/v1",
        "apiKey": "${NADIRCLAW_AUTH_TOKEN}",
        "api": "openai-completions",
        "models": [{ "id": "auto" }]
      }
    }
  },
  "agents": {
    "defaults": {
      "model": { "primary": "nadirclaw/auto" }
    }
  }
}
```

## Usage with Codex

[Codex](https://github.com/openai/codex) is OpenAI's CLI coding agent. NadirClaw integrates as a custom model provider.

```bash
# Auto-configure Codex
nadirclaw codex onboard

# Start the router
nadirclaw serve
```

This writes `~/.codex/config.toml`:

```toml
model_provider = "nadirclaw"

[model_providers.nadirclaw]
base_url = "http://localhost:8000/v1"
env_key = "NADIRCLAW_AUTH_TOKEN"
```

## Usage with Any OpenAI-Compatible Tool

NadirClaw exposes a standard OpenAI-compatible API. Point any tool at it:

```bash
# Base URL
http://localhost:8000/v1

# Auth
Authorization: Bearer nadir-local    # default token
# or
X-API-Key: nadir-local

# Model
model: "auto"    # or omit -- NadirClaw picks the best model
```

### Example: curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer nadir-local" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }'
```

### Example: Python (openai SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="nadir-local",
)

response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "What is 2+2?"}],
)
print(response.choices[0].message.content)
```

## CLI Reference

```bash
nadirclaw serve      # Start the router server
nadirclaw classify   # Classify a prompt (no server needed)
nadirclaw status     # Show config and check if server is running
nadirclaw codex onboard    # Configure Codex integration
nadirclaw openclaw onboard # Configure OpenClaw integration
```

### `nadirclaw serve`

```bash
nadirclaw serve [OPTIONS]

Options:
  --port INTEGER          Port to listen on (default: 8000)
  --simple-model TEXT     Model for simple prompts
  --complex-model TEXT    Model for complex prompts
  --models TEXT           Comma-separated model list (legacy)
  --token TEXT            Auth token
  --verbose               Enable debug logging
```

### `nadirclaw classify`

Classify a prompt locally without running the server. Useful for testing your setup:

```bash
$ nadirclaw classify "What is 2+2?"
Tier:       simple
Confidence: 0.2848
Score:      0.0000
Model:      ollama/llama3.1:8b

$ nadirclaw classify "Design a distributed system for real-time trading"
Tier:       complex
Confidence: 0.1843
Score:      1.0000
Model:      claude-sonnet-4-20250514
```

### `nadirclaw status`

```bash
$ nadirclaw status
NadirClaw Status
----------------------------------------
Simple model:  ollama/llama3.1:8b
Complex model: claude-sonnet-4-20250514
Tier config:   explicit (env vars)
Port:          8000
Threshold:     0.06
Log dir:       /Users/you/.nadirclaw/logs
Token:         nadir-***

Server:        RUNNING (ok)
```

## How It Works

NadirClaw uses a binary complexity classifier based on sentence embeddings:

1. **Startup**: Embeds ~170 seed prompts (simple + complex examples) using [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (~80 MB model, downloaded once) and computes a centroid vector for each class.

2. **Classification**: For each incoming prompt, computes its embedding and measures cosine similarity to both centroids. If the prompt is closer to the complex centroid, it routes to your complex model; otherwise to your simple model.

3. **Borderline handling**: When confidence is below the threshold (default 0.06), the classifier defaults to complex -- it's cheaper to over-serve a simple prompt than to under-serve a complex one.

4. **Routing**: Calls the selected model via [LiteLLM](https://docs.litellm.ai), which provides a unified interface to 100+ LLM providers.

Classification takes ~10ms on a warm encoder. The first request takes ~2-3 seconds to load the embedding model.

## API Endpoints

All endpoints require auth (`Authorization: Bearer <token>` or `X-API-Key: <token>`).

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | OpenAI-compatible completions with auto routing |
| `/v1/classify` | POST | Classify a prompt without calling an LLM |
| `/v1/classify/batch` | POST | Classify multiple prompts at once |
| `/v1/models` | GET | List available models |
| `/v1/logs` | GET | View recent request logs |
| `/health` | GET | Health check (no auth required) |

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `NADIRCLAW_SIMPLE_MODEL` | `ollama/llama3.1:8b` | Model for simple prompts |
| `NADIRCLAW_COMPLEX_MODEL` | `claude-sonnet-4-20250514` | Model for complex prompts |
| `NADIRCLAW_AUTH_TOKEN` | `nadir-local` | Bearer token for API auth |
| `ANTHROPIC_API_KEY` | -- | Anthropic API key |
| `OPENAI_API_KEY` | -- | OpenAI API key |
| `OLLAMA_API_BASE` | `http://localhost:11434` | Ollama base URL |
| `NADIRCLAW_CONFIDENCE_THRESHOLD` | `0.06` | Classification threshold (lower = more complex) |
| `NADIRCLAW_PORT` | `8000` | Server port |
| `NADIRCLAW_LOG_DIR` | `~/.nadirclaw/logs` | Log directory |
| `NADIRCLAW_MODELS` | `claude-sonnet-4-20250514,ollama/llama3.1:8b` | Legacy model list (fallback if tier vars not set) |

## Project Structure

```
nadirclaw/
  __init__.py        # Package version
  cli.py             # CLI commands (serve, classify, status, codex, openclaw)
  server.py          # FastAPI server with OpenAI-compatible API
  classifier.py      # Binary complexity classifier (sentence embeddings)
  encoder.py         # Shared SentenceTransformer singleton
  auth.py            # Bearer token / API key authentication
  settings.py        # Environment-based configuration
  models.json        # Model performance data for ranking
```

## License

MIT
