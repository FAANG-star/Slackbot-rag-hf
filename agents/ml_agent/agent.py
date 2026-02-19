#!/usr/bin/env python3
"""Agent entrypoint — runs inside the Modal sandbox.

Each invocation handles a single user message. Conversation history is
maintained via the Claude SDK's session resume mechanism, with session
IDs persisted to a JSON file on the shared volume.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
)

SESSIONS_FILE = Path("/data/sessions.json")


def load_session_id(sandbox_name: str) -> str | None:
    """Load existing session ID for this sandbox/thread."""
    if not SESSIONS_FILE.exists():
        return None
    sessions = json.loads(SESSIONS_FILE.read_text())
    return sessions.get(sandbox_name)


def save_session_id(sandbox_name: str, session_id: str) -> None:
    """Save session ID for this sandbox/thread."""
    sessions = {}
    if SESSIONS_FILE.exists():
        sessions = json.loads(SESSIONS_FILE.read_text())
    sessions[sandbox_name] = session_id
    SESSIONS_FILE.write_text(json.dumps(sessions, indent=2))


async def main(user_msg: str, sandbox_name: str):
    os.environ["ANTHROPIC_API_KEY"] = os.environ.get("MODAL_SANDBOX_ID", "")

    session_id = load_session_id(sandbox_name)

    options = ClaudeAgentOptions(
        model="claude-sonnet-4-6",
        resume=session_id,
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
        await client.query(user_msg)

        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        print(block.text, flush=True)
                    elif isinstance(block, ToolUseBlock):
                        detail = (
                            block.input.get("command")
                            or block.input.get("file_path")
                            or block.input.get("pattern")
                            or ""
                        )
                        print(f"[{block.name}] {detail}", flush=True)
                    elif isinstance(block, ToolResultBlock) and block.content:
                        text = (
                            block.content
                            if isinstance(block.content, str)
                            else str(block.content)
                        )
                        print(text, flush=True)
            elif isinstance(msg, ResultMessage):
                save_session_id(sandbox_name, msg.session_id)
                if msg.is_error:
                    print(f"[error] {msg.result}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", type=str, required=True)
    parser.add_argument("--sandbox-name", type=str, required=True)
    args = parser.parse_args()

    asyncio.run(main(args.message, args.sandbox_name))
