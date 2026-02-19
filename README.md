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
  - [RAG: Enron Email Corpus](#rag-enron-email-corpus)
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

A fully local RAG pipeline running [Qwen2.5-14B](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-AWQ) (4-bit AWQ) on an A10G GPU via [vLLM](https://github.com/vllm-project/vllm). No external API calls for inference. Documents are indexed with [ChromaDB](https://www.trychroma.com/) and queried through a [LlamaIndex](https://www.llamaindex.ai/) ReAct agent with three tools:

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

### RAG: Enron Email Corpus

Imagine you're an investigative journalist with access to ~500,000 internal corporate emails from [Enron](https://www.cs.cmu.edu/~enron/) — the kind of sensitive data you'd never send to a third-party API. The collection is also far too large for a single context window. Even at 1M tokens, you'd cover maybe 1% of the data.

Upload the corpus to the volume, and the agent indexes it locally. Then ask questions from Slack:

```bash
# Download and upload the Enron corpus to the RAG volume
modal run scripts/upload_enron.py
```

> **You:** Who discussed the Raptor SPE transactions and when did awareness spread through the organization?

The agent searches across hundreds of thousands of emails, finds the relevant threads, and synthesizes an answer — all on your GPU, nothing sent externally.

> **You:** Plot email volume by month for the top 10 senders. When does communication spike?

The agent calls `execute_python` to parse timestamps across the corpus with pandas, builds a chart with matplotlib, and uploads it to the Slack thread. The kind of analysis that needs both retrieval and computation — neither a context window nor a simple search engine can do this alone.

> **You:** Find all messages between executives mentioning mark-to-market accounting. What concerns were raised?

Search + synthesis across a corpus that would take a human weeks to read through.

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
