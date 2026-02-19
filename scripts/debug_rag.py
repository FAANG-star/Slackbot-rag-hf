"""Interactive debug REPL — boots RAG sandbox, sends prompts in a loop.

Usage:
    modal run debug_rag.py                  # interactive REPL
    modal run debug_rag.py --test rag       # quick RAG test
    modal run debug_rag.py --test ml        # quick ML test
    modal run debug_rag.py --test csv       # CSV analysis via execute_python
"""
import json
import threading
import modal
modal.enable_output()

from agents.infra.shared import app, rag_vol
from agents.rag_agent.sandbox import sandbox_image, SANDBOX_NAME

END_TURN = "---END_TURN---"


def _stream_stderr(process):
    """Stream sandbox stderr to local terminal prefixed with [AGENT]."""
    for line in process.stderr:
        print(f"[AGENT] {line.rstrip()}", flush=True)

TEST_PROMPTS = {
    "rag": "What does the Attention is All You Need paper say about multi-head attention?",
    "ml": "Write a simple PyTorch script that trains a 2-layer MLP on random data for 5 epochs and prints the loss.",
    "csv": (
        "Analyze /data/rag/docs/sales_data.csv. "
        "Show total revenue by category and the top 3 products by units sold. "
        "Save a bar chart of revenue by category to /data/rag/output/revenue_by_category.png."
    ),
}


def _read_until_sentinel(stdout):
    """Read stdout lines until END_TURN, return collected text."""
    lines = []
    for line in stdout:
        if END_TURN in line:
            content = line[: line.index(END_TURN)].strip()
            if content:
                lines.append(content)
            break
        lines.append(line.rstrip())
    return "\n".join(lines)


def _send_and_print(process, stdout, prompt):
    """Send a prompt and print the response."""
    msg = json.dumps({"message": prompt, "sandbox_name": "debug-test"}) + "\n"
    process.stdin.write(msg.encode())
    process.stdin.drain()
    response = _read_until_sentinel(stdout)
    print(response)
    print()


@app.local_entrypoint()
def main(test: str = ""):
    # Kill old named sandbox if it exists
    try:
        old = modal.Sandbox.from_name(app_name=app.name, name=SANDBOX_NAME)
        print(f"Terminating old sandbox: {SANDBOX_NAME}")
        old.terminate()
    except modal.exception.NotFoundError:
        pass

    # Create fresh sandbox
    sb = modal.Sandbox.create(
        app=app,
        image=sandbox_image,
        workdir="/agent",
        volumes={"/data": rag_vol},
        gpu="A10G",
        env={"HF_HOME": "/data/hf-cache"},
        timeout=15 * 60,
        name=SANDBOX_NAME,
    )
    process = sb.exec("python", "-u", "/agent/agent.py")
    stdout = iter(process.stdout)
    threading.Thread(target=_stream_stderr, args=(process,), daemon=True).start()

    # Wait for init
    print("Loading models...")
    for line in stdout:
        text = line.rstrip()
        if text and END_TURN not in text:
            print(f"  {text}")
        if END_TURN in line:
            break
    print("Ready!\n")

    # Quick test mode — send one prompt and exit
    if test:
        prompt = TEST_PROMPTS.get(test)
        if not prompt:
            print(f"Unknown test: {test!r} (available: {', '.join(TEST_PROMPTS)})")
            sb.terminate()
            return
        print(f"[test={test}] {prompt}\n")
        _send_and_print(process, stdout, prompt)
        sb.terminate()
        return

    # Interactive loop
    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nShutting down...")
            break
        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit"):
            break

        _send_and_print(process, stdout, prompt)

    sb.terminate()
