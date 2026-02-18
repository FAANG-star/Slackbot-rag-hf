#!/usr/bin/env python3
"""Interactive multi-turn CLI for the ML training agent."""

import sys
import threading

from ml_agent.infra import app, create_sandbox

msg = sys.argv[1] if len(sys.argv) > 1 else input("Message: ").strip()

with app.run():
    sb = create_sandbox()
    p = sb.exec("python", "-u", "/agent/agent.py")

    # Surface agent errors (import failures, crashes, etc.)
    threading.Thread(
        target=lambda: [print(l, end="", file=sys.stderr) for l in p.stderr],
        daemon=True,
    ).start()

    stdout = iter(p.stdout)
    while msg:
        p.stdin.write(msg + "\n")
        p.stdin.drain()
        for line in stdout:
            if "---END_TURN---" in line:
                break
            print(line, end="")
        try:
            msg = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

    p.stdin.write_eof()
    sb.terminate()
