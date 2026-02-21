## TL;DR

Agentic AI doesn't have risk your privacy. Open-source models and GPU infrastructure are making it easier to deploy custom agents that help you make sense of huge amounts of data without sacrificing privacy.

A Slack bot connected to two secure AI agents:

1. **RAG agent** — automatically indexes files and zipped folders uploaded through Slack, then answers your questions using a local LLM. Your documents never leave the GPU. It also writes python scripts to analyze specific docs. 

2. **ML agent** — a Claude agent that trains and runs HuggingFace models on a GPU sandbox, without ever being exposed to your API keys.

Plus, there are no idle compute costs. Tagging the bot cold-starts a GPU quickly thanks to Modal's infrastructure.

- [Why This Exists](#why-this-exists)
- [Features](#features)
- [Architecture](#architecture)
- [Setup](#setup)
- [Demo](#demo)
  - [RAG: Wikipedia](#rag-wikipedia)
  - [ML Training](#ml-training)
  - [Debugging](#debugging)
- [Credits](#credits)

---

## Why This Exists

It is possible to protect your data and incorporate agentic AI into workflows. Advances in open-source models and GPU infrastructure is making it easier to deploy custom agents that help you make sense of huge amount of data without sending it over to a third-party service. 

This repo includes a slackbot connected to two secure AI agents. 

1. RAG bot: automatically index files and zipped folders uploaded to it from slack, and them answer your questions accuretly. It also writes python scripts to analyze specific docs. 

2. HuggingFace bot: secure claude agent that can train and/or run huggingface models and return results. Securely without being exposed to secrets.  

You also don't need to spend money on idle compute. Tagging the bot will cold-start a GPU with Modal's superfast file system. 

**How it works:** Share a file in Slack → the bot downloads it to a Modal volume → the indexer parses it into text, splits it into chunks (512 tokens each), and embeds each chunk with BGE-large on GPU → embeddings are stored in ChromaDB. When you ask a question, the ReAct agent retrieves the top-k most similar chunks, uses them as context, and generates an answer with the local LLM. Your files, embeddings, and queries never leave the GPU container.

---

## Features

### Slack Bot

Works in DMs (via the Slack [Assistant](https://api.slack.com/docs/apps/ai) protocol) and in channels (via `@mentions`). Upload files, ask questions, train models — all from Slack.

### RAG Agent — Local LLM on GPU

A fully local RAG pipeline running [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B-AWQ) (4-bit AWQ) on an A10G GPU via [vLLM](https://github.com/vllm-project/vllm). No external API calls for inference. Documents are indexed with [ChromaDB](https://www.trychroma.com/) and queried through a [LlamaIndex](https://www.llamaindex.ai/) ReAct agent with three tools:

- **`search_documents`** — semantic search over indexed documents using [BGE-large](https://huggingface.co/BAAI/bge-large-en-v1.5) embeddings
- **`execute_python`** — runs Python code for data analysis, chart generation, file processing (pandas, matplotlib, openpyxl pre-installed)
- **`list_documents`** — lists uploaded files so the agent can confirm paths before accessing them

Supports PDF, DOCX, CSV, Excel, and plain text. Indexing is incremental — only changed files are re-processed.

### ML Training Agent — Claude on GPU

A [Claude Agent SDK](https://docs.anthropic.com/en/docs/agents-and-tools/claude-agent-sdk) instance running on an A10 GPU sandbox. Writes and executes training code, installs packages, and trains HuggingFace models. The sandbox never sees the Anthropic API key — requests go through a proxy that swaps in the real key.

Training metrics sync to a [Trackio](https://huggingface.co/blog/trackio) dashboard on HuggingFace Spaces every ~30 seconds.

### Slack Commands

| Message | What happens |
|---------|-------------|
| Any text | RAG agent answers using indexed documents |
| `hf: <prompt>` | Routes to the ML training agent |
| `sources` | Lists all uploaded files with sizes |
| `reindex` | Incremental re-index of documents |
| `reindex --force` | Full rebuild of the document index |
| `status` | Index stats (source count, chunk count) |
| `clear` | Wipes index, output files, and conversation history |
| `remove <filename>` | Deletes a file from the volume |
| Share/upload files | Downloads to volume, auto-triggers reindex |

---

## Architecture

Everything deploys as a single Modal app. One `modal deploy` command and you're done.

The **Slack bot** is a FastAPI + slack-bolt server that stays warm and routes messages. Plain text and file uploads go to the RAG agent. Messages prefixed with `hf:` go to the ML training agent.

The **RAG sandbox** runs on an A10G GPU. Models load once into VRAM and stay loaded — the process is long-lived, communicating over stdin/stdout with JSON messages. vLLM serves the LLM, ChromaDB stores embeddings, and a ReAct agent orchestrates search and code execution. Documents never leave this container.

The **ML sandbox** runs on an A10 GPU. Each request launches a Claude Agent SDK session that can write code, install packages, and train models. It talks to the Anthropic API through a **proxy container** that intercepts requests and swaps the sandbox's fake key for the real one. The sandbox never sees your Anthropic API key.

The **Trackio syncer** polls a shared volume for metric databases and pushes updates to a HuggingFace Space dashboard.

Two volumes provide persistence: `sandbox-rag` holds documents, the vector store, and conversation history. `sandbox-data` holds model caches, training checkpoints, and session state.

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/DeanAvI/modal-sandbox.git
cd modal-sandbox
pip install modal
modal setup
```

### 2. Create a Slack app

Use the included [app manifest](slack-manifest.yaml) — it has all the required scopes and events pre-configured:

1. Go to [api.slack.com/apps](https://api.slack.com/apps) and click **Create New App → From a manifest**
2. Select your workspace, paste the contents of `slack-manifest.yaml`, and click **Create**
3. Under **OAuth & Permissions**, click **Install to Workspace** and authorize
4. Copy the **Bot User OAuth Token** (`xoxb-...`) from OAuth & Permissions
5. Copy the **Signing Secret** from **Basic Information → App Credentials**

### 3. Create Modal secrets

```bash
# Slack bot credentials
modal secret create slack-secret \
  SLACK_BOT_TOKEN=xoxb-... \
  SLACK_SIGNING_SECRET=...

# Anthropic API key (used by the proxy — never exposed to sandboxes)
modal secret create anthropic-secret ANTHROPIC_API_KEY=sk-ant-...

# HuggingFace token (for Trackio metric syncing)
modal secret create hf-secret HF_TOKEN=hf_...

# GitHub token (for ML agent repo access)
modal secret create github-secret GITHUB_TOKEN=ghp_...
```

### 4. Deploy

```bash
modal deploy agents/slackbot/app.py
```

This deploys the Slack bot, API proxy, and pre-builds both GPU sandbox images. Modal will print the app URL — set this as the **Request URL** in your Slack app's **Event Subscriptions** settings (the bot listens on `/`).

### 5. Upload documents (optional)

Share files directly in Slack by uploading in a channel (drag and drop, then `@rag_bot` to index them) or in a DM, or bulk upload via script:

```bash
modal run scripts/upload_test_data.py
```

---

## Demo

### RAG: Wikipedia

The [Simple English Wikipedia dump](https://dumps.wikimedia.org/simplewiki/latest/) is a clean benchmark: ~250,000 articles covering every topic, totaling around 1 GB compressed (~4 GB extracted). Too much for any context window, but exactly the kind of broad knowledge base where semantic search shines.

Upload the zip to the bot in Slack — it extracts and indexes the articles automatically.

**How indexing works:** The pipeline runs in three phases:

1. **Scan** — compares each file's mtime and size against a manifest to find only new or changed files. Already-indexed content is skipped.
2. **Embed** — files are distributed across 10 parallel GPU workers on T4s. Each worker parses the file, splits text into 512-token chunks with 64-token overlap using a sentence-aware splitter, embeds each chunk with [BGE-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5), and upserts to its own ChromaDB shard.
3. **Finalize** — worker manifests are merged into a single `manifest.json`, and ChromaDB shards are consolidated. The RAG agent reloads the index.

The full zip is 157 MB containing ~242,000 articles. Indexing took **15 minutes** across 10 parallel T4 GPUs.

![Indexing progress message in Slack](assets/rag_index.png)
*The bot reports indexing progress as it processes each batch of articles.*

Then ask questions:

> **You:** What were the main causes and consequences of the 2008 financial crisis?

![RAG answer about the 2008 financial crisis](assets/rag_query.png)
*The agent searches across thousands of articles, pulls relevant sections from multiple pages, and synthesizes a coherent answer — all on your GPU, nothing sent externally.*

> **You:** Plot a timeline of major space exploration milestones from 1957 to 2024.

![Matplotlib chart of space exploration milestones](assets/rag_chart.png)
*The agent calls `execute_python` to extract dates and events from indexed articles, builds the chart with matplotlib, and uploads it to the thread.*

### ML Training

Prefix messages with `hf:` to route to the ML training agent:

> **You:** hf: Train a model for categorizing multispectral satellite imagery

The agent asks clarifying questions about model choice, dataset, and HuggingFace username before writing any code:

![Agent suggesting models and datasets before training](assets/prompt.png)
*Agent suggests models and datasets with tradeoffs, then waits for confirmation.*

Training metrics sync to a HuggingFace Space dashboard:

![Trackio dashboard showing training metrics for swin-eurosat-multispectral](assets/trackio.png)
*Metrics sync every ~30 seconds via Trackio.*

### Debugging

Interactive REPL for testing the RAG agent directly:

```bash
modal run scripts/debug_rag.py                  # interactive mode
modal run scripts/debug_rag.py --test rag       # test document Q&A
modal run scripts/debug_rag.py --test ml        # test code execution
modal run scripts/debug_rag.py --test csv       # test CSV analysis
```

---

## Credits

- **transformers skill** — Originally by [jimmc414](https://github.com/jimmc414), from [Kosmos](https://github.com/jimmc414/Kosmos/tree/master/kosmos-claude-scientific-skills/scientific-skills/transformers). Modified to integrate Trackio.
- **hugging-face-trackio skill** — From the official [Hugging Face Skills](https://github.com/huggingface/skills) repo. Licensed under Apache 2.0.
- **Modal sandbox architecture** — Based on the [Claude Slack GIF Creator](https://modal.com/docs/examples/claude-slack-gif-creator) example from Modal's docs.
