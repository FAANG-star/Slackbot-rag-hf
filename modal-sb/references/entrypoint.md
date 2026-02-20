# Agent Entrypoint — Claude Agent SDK Inside the Sandbox

The entrypoint runs as a long-lived process inside the sandbox. It reads messages from stdin, sends them to the Claude Agent SDK, and streams responses to stdout. The SDK maintains conversation history automatically across turns.

## IPC Protocol

```
CLI (stdin)                    Agent Process                   Claude API (via proxy)
───────────                    ─────────────                   ─────────────────────
"Train a model\n" ──stdin──→   readline()
                               client.query(message) ────────→ POST /v1/messages
                               stream new messages  ←──────── streaming response
[Bash] pip install...  ←──stdout──  print(block)
Training complete...   ←──stdout──  print(block)
---END_TURN---         ←──stdout──  print(END_TURN)
```

The `---END_TURN---` sentinel tells the CLI that the agent's turn is complete and it can prompt for the next user message.

## SDK Client Setup

```python
os.environ["ANTHROPIC_API_KEY"] = os.environ.get("MODAL_SANDBOX_ID", "")

options = ClaudeAgentOptions(
    model="claude-opus-4-6",
    system_prompt={
        "type": "preset",
        "preset": "claude_code",
        "append": "You are running inside a secure Modal GPU sandbox for ML training.",
    },
    cwd="/app",
    setting_sources=["project"],
    allowed_tools=["Read", "Write", "Bash", "Glob", "Grep"],
    permission_mode="acceptEdits",
    max_turns=100,
)
```

Key configuration:
- `ANTHROPIC_API_KEY = sandbox_id` — the proxy expects the sandbox ID as the API key and swaps in the real one
- `preset: "claude_code"` — uses the Claude Code system prompt as a base
- `append` — adds sandbox-specific context to the system prompt
- `cwd="/app"` — the SDK reads `.claude/CLAUDE.md` and `.claude/skills/` from here
- `setting_sources=["project"]` — loads project-level settings from the cwd
- `permission_mode="acceptEdits"` — auto-approves file edits (no human in the loop inside the sandbox)
- `max_turns=100` — limits agentic loops per query

## Multi-Turn History Handling

`receive_response()` only yields **new messages for the current turn** — it does not replay full history. Each call yields a `SystemMessage` (init), then new `AssistantMessage`s, then a `ResultMessage`. No history tracking is needed.

```python
async with ClaudeSDKClient(options=options) as client:
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:  # EOF — CLI sent write_eof()
            break

        await client.query(line.strip())

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text, flush=True)
                    elif isinstance(block, ToolUseBlock):
                        detail = block.input.get("command") or block.input.get("file_path") or block.input.get("pattern") or ""
                        print(f"[{block.name}] {detail}", flush=True)
                    elif isinstance(block, ToolResultBlock) and block.content:
                        text = block.content if isinstance(block.content, str) else str(block.content)
                        print(text, flush=True)
            elif isinstance(message, ResultMessage) and message.is_error:
                print(f"[error] {message.result}", flush=True)

        print(END_TURN, flush=True)
```

- `run_in_executor` — runs blocking `readline()` in a thread so the async event loop isn't blocked
- The loop exits when stdin reaches EOF (the CLI calls `process.stdin.write_eof()` on cleanup)

## Message Types

| Type | Content | Displayed As |
|------|---------|-------------|
| `AssistantMessage` > `TextBlock` | Agent's text response | Printed directly |
| `AssistantMessage` > `ToolUseBlock` | Tool invocation | `[Bash] pip install torch` |
| `AssistantMessage` > `ToolResultBlock` | Tool output | Printed directly (when non-empty) |
| `ResultMessage` (error) | SDK error | `[error] message` |

## SDK Imports

```python
from claude_agent_sdk import (
    ClaudeAgentOptions,     # configuration dataclass
    ClaudeSDKClient,        # async context manager for multi-turn sessions
    AssistantMessage,       # agent's response messages
    ResultMessage,          # end-of-turn result (includes is_error flag)
    TextBlock,              # text content block
    ToolUseBlock,           # tool invocation block
    ToolResultBlock,        # tool output block
)
```

## Full Implementation

```python
#!/usr/bin/env python3
"""Agent entrypoint — runs inside the Modal sandbox."""

import asyncio
import os
import sys

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
)

END_TURN = "---END_TURN---"


async def main():
    os.environ["ANTHROPIC_API_KEY"] = os.environ.get("MODAL_SANDBOX_ID", "")

    options = ClaudeAgentOptions(
        model="claude-opus-4-6",
        system_prompt={
            "type": "preset",
            "preset": "claude_code",
            "append": "You are running inside a secure Modal GPU sandbox for ML training.",
        },
        cwd="/app",
        setting_sources=["project"],
        allowed_tools=["Read", "Write", "Bash", "Glob", "Grep"],
        permission_mode="acceptEdits",
        max_turns=100,
    )

    loop = asyncio.get_event_loop()

    async with ClaudeSDKClient(options=options) as client:
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                break

            await client.query(line.strip())

            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            print(block.text, flush=True)
                        elif isinstance(block, ToolUseBlock):
                            detail = block.input.get("command") or block.input.get("file_path") or block.input.get("pattern") or ""
                            print(f"[{block.name}] {detail}", flush=True)
                        elif isinstance(block, ToolResultBlock) and block.content:
                            text = block.content if isinstance(block.content, str) else str(block.content)
                            print(text, flush=True)
                elif isinstance(message, ResultMessage) and message.is_error:
                    print(f"[error] {message.result}", flush=True)

            print(END_TURN, flush=True)


asyncio.run(main())
```
