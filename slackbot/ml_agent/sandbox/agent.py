"""Claude Agent SDK wrapper — runs inside the Modal sandbox.

Long-lived stdin loop: reads JSON messages, runs the Claude SDK, prints
responses, then a sentinel. One persistent ClaudeSDKClient stays alive
across all turns — avoids the subprocess teardown/recreate that crashes.
"""

import asyncio
import json
import os
import sys
import traceback
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
)

END_SENTINEL = "__END__"


class Agent:
    """Persistent Claude SDK agent for multi-turn conversations."""

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._client: ClaudeSDKClient | None = None

    async def create_client(self):
        """Create (or recreate) the underlying Claude SDK client."""
        os.environ["ANTHROPIC_API_KEY"] = self._api_key
        client = ClaudeSDKClient(options=ClaudeAgentOptions(
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
        ))
        await client.__aenter__()
        self._client = client

    async def receive_prompt(self) -> str:
        """Read the next JSON request from stdin and return the message."""
        line = await asyncio.to_thread(sys.stdin.readline)
        if not line:
            raise EOFError
        return json.loads(line.strip())["message"]

    async def send_response(self, message: str):
        """Forward a message to the Claude SDK and print responses to stdout."""
        assert self._client is not None
        await self._client.query(message)
        async for msg in self._client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        print(block.text, flush=True)
            elif isinstance(msg, ResultMessage) and msg.is_error:
                print(f"Error: {msg.result}", flush=True)
        print(END_SENTINEL, flush=True)


async def main():
    api_key = os.environ.get("MODAL_SANDBOX_ID", "")
    agent = Agent(api_key)
    await agent.create_client()
    while True:
        try:
            message = await agent.receive_prompt()
            await agent.send_response(message)
        except EOFError:
            break
        except Exception:
            traceback.print_exc()
            await agent.create_client()


asyncio.run(main())
