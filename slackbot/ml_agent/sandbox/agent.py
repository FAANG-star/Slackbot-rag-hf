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
        # Stream response blocks to stdout — ml_handler reads these lines
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
    # Stdin loop: each line is a JSON request from ml_handler
    for line in sys.stdin:
        request = json.loads(line.strip())
        try:
            await agent.run(request["message"])
        except Exception as e:
            print(f"Error: {e}", flush=True)
            agent._client = None  # reset so next turn creates a fresh client
        # Sentinel tells ml_handler this turn is done
        print(END_SENTINEL, flush=True)


asyncio.run(main())
