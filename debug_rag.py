"""Quick debug script — kills old sandbox, boots agent, sends a test query."""
import json
import modal
modal.enable_output()

from agents.infra.shared import app, rag_vol
from agents.rag_agent.sandbox import sandbox_image, SANDBOX_NAME

END_TURN = "---END_TURN---"


@app.local_entrypoint()
def main():
    # Kill old named sandbox if it exists
    try:
        old = modal.Sandbox.from_name(app_name=app.name, name=SANDBOX_NAME)
        print(f"[DEBUG] Terminating old sandbox: {SANDBOX_NAME}")
        old.terminate()
    except modal.exception.NotFoundError:
        print("[DEBUG] No existing sandbox found")

    # Create fresh sandbox with current code
    sb = modal.Sandbox.create(
        app=app,
        image=sandbox_image,
        workdir="/agent",
        volumes={"/data": rag_vol},
        gpu="A10G",
        env={"HF_HOME": "/data/hf-cache"},
        timeout=10 * 60,
        name=SANDBOX_NAME,
    )
    process = sb.exec("python", "-u", "/agent/agent.py")
    stdout = iter(process.stdout)

    # Wait for init
    print("[DEBUG] Waiting for init...")
    for line in stdout:
        print(f"[STDOUT] {line.rstrip()}")
        if END_TURN in line:
            break
    print("[DEBUG] Init complete, sending test query...")

    # Send test query
    msg = json.dumps({"message": "hello", "sandbox_name": "debug-test"}) + "\n"
    process.stdin.write(msg.encode())
    process.stdin.drain()

    # Read response
    for line in stdout:
        print(f"[STDOUT] {line.rstrip()}")
        if END_TURN in line:
            break

    # Read stderr
    for line in process.stderr:
        print(f"[STDERR] {line.rstrip()}")
