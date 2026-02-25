"""Claude Agent SDK wrapper — runs inside the Modal sandbox.

Long-lived stdin loop: reads JSON messages, runs the Claude SDK, prints
responses, then a sentinel. Session IDs are held in memory so the SDK
can resume conversations across messages in the same thread.
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

class Agent:
    """Wraps the Claude Agent SDK with in-memory session tracking."""

    def __init__(self):
        # The proxy expects any non-empty key; sandbox ID is a convenient unique value
        os.environ["ANTHROPIC_API_KEY"] = os.environ.get("MODAL_SANDBOX_ID", "")
        self._sessions: dict[str, str] = {}

    async def run(self, message: str, session: str):
        """Send a message and print streamed responses."""
        options = ClaudeAgentOptions(
            model="claude-sonnet-4-6",
            resume=self._sessions.get(session),
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

        async with ClaudeSDKClient(options=options) as client:
            await client.query(message)
            async for msg in client.receive_response():
                self._handle(msg, session)

    def _handle(self, msg, session: str):
        """Print text responses and persist session ID."""
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    print(block.text, flush=True)
        elif isinstance(msg, ResultMessage):
            if msg.is_error:
                print(f"Error: {msg.result}", flush=True)
            self._sessions[session] = msg.session_id


agent = Agent()
END_SENTINEL = "__END__"

# Stdin loop: slackbot writes JSON requests, we print responses + sentinel
for line in sys.stdin:
    request = json.loads(line.strip())
    asyncio.run(agent.run(request["message"], request["session"]))
    print(END_SENTINEL, flush=True)
