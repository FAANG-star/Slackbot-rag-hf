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
