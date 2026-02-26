# Agent Entrypoint — Claude Agent SDK Inside the Sandbox

The entrypoint runs as a long-lived process inside the sandbox. It reads JSON messages from stdin, sends them to the Claude Agent SDK, and streams responses to stdout. The SDK maintains conversation history automatically across turns.

## IPC Protocol

```
Host (stdin)                   Agent Process                   Claude API (via proxy)
────────────                   ─────────────                   ─────────────────────
{"message": "Train..."}\n ──→  json.loads(line)
                               client.query(message) ────────→ POST /v1/messages
                               stream new messages  ←──────── streaming response
Training complete...   ←──stdout──  print(block)
__END__                ←──stdout──  print(END_SENTINEL)
```

The `__END__` sentinel tells the host that the agent's turn is complete and it can send the next message.

## SDK Client Setup

```python
os.environ["ANTHROPIC_API_KEY"] = os.environ.get("MODAL_SANDBOX_ID", "")

options = ClaudeAgentOptions(
    model="claude-sonnet-4-6",
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
- `cwd="/app"` — the SDK reads `_claude/CLAUDE.md` and `_claude/skills/` from here
- `setting_sources=["project"]` — loads project-level settings from the cwd
- `permission_mode="acceptEdits"` — auto-approves file edits (no human in the loop inside the sandbox)
- `max_turns=100` — limits agentic loops per query

## Class-Based Agent with Lazy Init

The `Agent` class wraps a single persistent `ClaudeSDKClient`. The client is created lazily on the first message and reused across all turns. If a turn errors, the client is reset so the next turn creates a fresh one.

```python
class Agent:
    def __init__(self):
        os.environ["ANTHROPIC_API_KEY"] = os.environ.get("MODAL_SANDBOX_ID", "")
        self._client = None

    async def _ensure_client(self):
        if self._client is not None:
            return
        options = ClaudeAgentOptions(...)
        self._client = ClaudeSDKClient(options=options)
        await self._client.__aenter__()

    async def run(self, message: str):
        await self._ensure_client()
        await self._client.query(message)
        async for msg in self._client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        print(block.text, flush=True)
            elif isinstance(msg, ResultMessage):
                if msg.is_error:
                    print(f"Error: {msg.result}", flush=True)
```

## Multi-Turn Stdin Loop

The main loop reads JSON lines from stdin. Each line is a request with a `message` field. On error, the client is reset so the next turn starts fresh.

```python
async def main():
    agent = Agent()
    for line in sys.stdin:
        request = json.loads(line.strip())
        try:
            await agent.run(request["message"])
        except Exception as e:
            print(f"Error: {e}", flush=True)
            agent._client = None  # reset so next turn creates a fresh client
        print(END_SENTINEL, flush=True)

asyncio.run(main())
```

- `for line in sys.stdin` — blocks until a line is available, exits on EOF
- `json.loads` — parses the JSON request (host sends `{"message": "..."}`)
- Error recovery resets the client rather than crashing the process
- `END_SENTINEL` is always printed, even after errors, so the host knows the turn is done

## Message Types

| Type | Content | Displayed As |
|------|---------|-------------|
| `AssistantMessage` > `TextBlock` | Agent's text response | Printed directly |
| `ResultMessage` (error) | SDK error | `Error: message` |

## SDK Imports

```python
from claude_agent_sdk import (
    AssistantMessage,       # agent's response messages
    ClaudeAgentOptions,     # configuration dataclass
    ClaudeSDKClient,        # async client for multi-turn sessions
    ResultMessage,          # end-of-turn result (includes is_error flag)
    TextBlock,              # text content block
)
```

## Full Implementation

```python
"""Claude Agent SDK wrapper — runs inside the Modal sandbox.

Long-lived stdin loop: reads JSON messages, runs the Claude SDK, prints
responses, then a sentinel. One persistent ClaudeSDKClient stays alive
across all turns — avoids the subprocess teardown/recreate that crashes.
"""

import asyncio
import json
import os
import sys
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
)

END_SENTINEL = "__END__"


class Agent:
    """Wraps a single persistent ClaudeSDKClient for multi-turn conversations."""

    def __init__(self):
        os.environ["ANTHROPIC_API_KEY"] = os.environ.get("MODAL_SANDBOX_ID", "")
        self._client = None

    async def _ensure_client(self):
        """Create the client once; reuse across all turns."""
        if self._client is not None:
            return
        options = ClaudeAgentOptions(
            model="claude-sonnet-4-6",
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
        self._client = ClaudeSDKClient(options=options)
        await self._client.__aenter__()

    async def run(self, message: str):
        """Send a message on the persistent client and print responses."""
        await self._ensure_client()
        await self._client.query(message)
        # Stream response blocks to stdout — the host reads these lines
        async for msg in self._client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        print(block.text, flush=True)
            elif isinstance(msg, ResultMessage):
                if msg.is_error:
                    print(f"Error: {msg.result}", flush=True)


async def main():
    agent = Agent()
    # Stdin loop: each line is a JSON request from the host
    for line in sys.stdin:
        request = json.loads(line.strip())
        try:
            await agent.run(request["message"])
        except Exception as e:
            print(f"Error: {e}", flush=True)
            agent._client = None  # reset so next turn creates a fresh client
        # Sentinel tells the host this turn is done
        print(END_SENTINEL, flush=True)


asyncio.run(main())
```
