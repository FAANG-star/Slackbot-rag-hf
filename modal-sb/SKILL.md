---
name: modal-sb
description: Build a secure Modal sandbox that runs a Claude agent without exposing credentials. Proxy-based API isolation, volume-based data sync, and Claude Agent SDK entrypoint.
---

# Modal Sandbox — Secure Agent Infrastructure

Build a secure Modal sandbox that runs a Claude agent. The agent never holds the Anthropic API key — API access goes through a proxy, and metric export goes through a volume-based syncer.

## Architecture

- **Sandbox** (GPU, `github-secret`) — runs `agent.py` with the Claude Agent SDK. Mounts two volumes: `/data` for models, datasets, and checkpoints; `/trackio` for metric `.db` files. Calls the Anthropic API through the proxy via `ANTHROPIC_BASE_URL`. Created via `modal.Sandbox.create()` from a host service (e.g. a Slack bot or CLI) that manages the multi-turn REPL over stdin/stdout with an `END_TURN` sentinel
- **Proxy** (CPU, `anthropic-secret`) — receives requests from the sandbox, swaps the sandbox ID for the real API key, and streams responses back from `api.anthropic.com`
- **Syncer** (CPU, `hf-secret`) — polls the shared `/trackio` volume for changed `.db` files and uploads them to an HF Space

## Project Layout

```
ml_agent/
  __init__.py             ← module init
  agent.py                ← runs inside sandbox, bridges stdin to Claude SDK
  proxy.py                ← Anthropic API proxy
  sandbox.py              ← sandbox image and creation
  trackio_sync.py         ← metric syncer
  .claude/
    CLAUDE.md             ← agent instructions (baked into sandbox at /app/.claude/)
    scripts/              ← utility scripts the agent can call
    skills/               ← agent skills (transformers, trackio)
```

## Reference Files

| File | Purpose |
|------|---------|
| `references/CLAUDE.md` | Agent instructions — copy into `.claude/CLAUDE.md` when building an agent |
| `references/proxy.md` | API proxy — credential isolation, streaming, header forwarding |
| `references/app_definition.md` | Modal app — shared resources, sandbox image, volumes |
| `references/entrypoint.md` | Claude Agent SDK entrypoint — config, response streaming, stdin protocol |
| `references/metric_syncer.md` | Trackio syncer — volume polling, change detection, HF Space upload |

## Building an Agent

1. Read `references/proxy.md` — adapt the proxy for your API
2. Read `references/app_definition.md` — define your app, volumes, and sandbox image
3. Read `references/entrypoint.md` — set up the Claude Agent SDK entrypoint
4. **Copy `references/CLAUDE.md` content into your project's `.claude/CLAUDE.md`** — this is the agent's system instructions. Customize it for your use case (environment, credentials, workflow).
5. Copy agent skills from `references/transformers/` and `references/hugging-face-trackio/` into your `.claude/skills/`

## Key Patterns

### Proxy — Credential Isolation

The sandbox sends its `MODAL_SANDBOX_ID` as the API key. The proxy swaps in the real key and streams the response back. The sandbox never sees the real Anthropic key. See `references/proxy.md`.

### Volume Syncer — Background Metric Export

The sandbox writes trackio `.db` files to a shared volume. The agent writes a `space_id` file to tell the syncer where to upload. A separate CPU function polls the volume and syncs changed files to an HF Space. See `references/metric_syncer.md`.

### Agent Instructions (.claude/)

The `.claude/` directory is baked into the sandbox image at `/app/.claude/`. The SDK discovers it from the agent's `cwd="/app"`. It contains:
- `CLAUDE.md` — agent instructions (environment, credentials, workflow)
- `skills/` — skills discovered by the SDK (transformers, trackio, etc.)
- `scripts/` — utility scripts the agent can call (e.g. `setup_trackio.py`)
