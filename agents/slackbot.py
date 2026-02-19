"""Slack bot entrypoint — receives Slack events, routes to agents, posts responses."""

import json
import os
import re
import threading
import traceback
import urllib.request
from pathlib import Path

import modal

from agents.infra import app
from agents.infra.shared import rag_vol
from agents.ml_agent import get_sandbox as get_ml_sandbox
from agents.rag_agent import get_sandbox as get_rag_sandbox

SLACK_MSG_LIMIT = 3900
DOCS_DIR = Path("/data/rag/docs")
OUTPUT_FILE_PATTERN = re.compile(r"\[OUTPUT_FILE:(.+?)\]")
END_TURN = "---END_TURN---"

slack_secret = modal.Secret.from_name("slack-secret")

slack_bot_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("slack-bolt", "fastapi")
    .add_local_python_source("agents")
    .add_local_dir("agents/ml_agent/.claude", "/root/agents/ml_agent/.claude")
    .add_local_dir("agents/rag_agent/rag", "/root/agents/rag_agent/rag")
)


# ---------------------------------------------------------------------------
# Persistent RAG process management
# ---------------------------------------------------------------------------

_rag_lock = threading.Lock()
_rag_process = None
_rag_stdout = None
_rag_sandbox = None


def _reset_rag_process():
    """Reset the persistent RAG process and terminate the sandbox so retry gets a fresh one."""
    global _rag_process, _rag_stdout, _rag_sandbox
    _rag_process = None
    _rag_stdout = None
    if _rag_sandbox is not None:
        try:
            print("[RAG] Terminating sandbox...", flush=True)
            _rag_sandbox.terminate()
        except Exception as e:
            print(f"[RAG] Sandbox terminate error (ignored): {e}", flush=True)
        _rag_sandbox = None


def _ensure_rag_process(set_status=None):
    """Start the RAG server process if not already running. Returns (process, stdout_iter)."""
    global _rag_process, _rag_stdout, _rag_sandbox

    if _rag_process is not None:
        return _rag_process, _rag_stdout

    print("[RAG] Starting server process...", flush=True)
    if set_status:
        set_status(status="starting RAG server...")
    sb = get_rag_sandbox()
    _rag_sandbox = sb
    _rag_process = sb.exec("python", "-u", "/agent/agent.py")
    _rag_stdout = iter(_rag_process.stdout)

    # Wait for model loading to complete (reads until first END_TURN)
    received_sentinel = False
    for line in _rag_stdout:
        line_text = line.rstrip()
        print(f"[RAG init] {line_text}", flush=True)
        if set_status and line_text:
            set_status(status=line_text.lower())
        if END_TURN in line:
            received_sentinel = True
            break

    if not received_sentinel:
        _rag_process = None
        _rag_stdout = None
        raise RuntimeError("RAG process stdout closed before initialization sentinel — process may have crashed during startup")

    print("[RAG] Server ready.", flush=True)
    return _rag_process, _rag_stdout


def run_rag_query(message: str, sandbox_name: str, set_status=None) -> list[str]:
    """Send a message to the persistent RAG process, return response lines."""
    with _rag_lock:
        msg = json.dumps({"message": message, "sandbox_name": sandbox_name}) + "\n"
        print(f"[RAG] Sending: {msg.strip()}", flush=True)

        for attempt in range(2):
            try:
                process, stdout = _ensure_rag_process(set_status)
                process.stdin.write(msg.encode())
                process.stdin.drain()
                break
            except Exception as e:
                print(f"[RAG] Process error (attempt {attempt + 1}): {e}", flush=True)
                traceback.print_exc()
                _reset_rag_process()
                if attempt == 1:
                    raise

        print("[RAG] Message sent, reading response...", flush=True)

        lines = []
        for line in stdout:
            print(f"[RAG out] {line.rstrip()}", flush=True)
            if END_TURN in line:
                # Capture any content before END_TURN in the same chunk
                content = line[:line.index(END_TURN)].strip()
                if content:
                    lines.append(content)
                break
            lines.append(line)
        print(f"[RAG] Got {len(lines)} response lines", flush=True)
        return lines


# ---------------------------------------------------------------------------
# ML agent (one-shot exec, no local model)
# ---------------------------------------------------------------------------


def run_ml_agent_turn(sb: modal.Sandbox, user_message: str, sandbox_name: str):
    """Execute one turn in the ML sandbox. Yields response lines."""
    process = sb.exec(
        "python", "-u", "/agent/agent.py",
        "--message", user_message,
        "--sandbox-name", sandbox_name,
    )

    for line in process.stdout:
        yield {"response": line}

    exit_code = process.wait()
    print(f"ML agent process exited with status {exit_code}")

    stderr = process.stderr.read()
    if stderr:
        yield {"response": f"*** ERROR ***\n{stderr}"}


# ---------------------------------------------------------------------------
# Slack file handling
# ---------------------------------------------------------------------------


def chunk_message(text: str, limit: int = SLACK_MSG_LIMIT) -> list[str]:
    """Split a long message into chunks that fit within Slack's message limit."""
    if len(text) <= limit:
        return [text]

    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks


def _download_slack_files(files: list[dict]) -> list[str]:
    """Download Slack-shared files to volume. Returns list of saved filenames."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    saved = []
    for f in files:
        url = f.get("url_private_download") or f.get("url_private")
        if not url:
            continue
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {os.environ['SLACK_BOT_TOKEN']}")
        with urllib.request.urlopen(req, timeout=120) as resp:
            content = resp.read()

        filename = f.get("name", f["id"])
        (DOCS_DIR / filename).write_bytes(content)
        saved.append(filename)

    rag_vol.commit()
    return saved


def _list_sources() -> str:
    """List files currently on the volume."""
    rag_vol.reload()
    if not DOCS_DIR.exists():
        return "No documents uploaded yet. Share files via Slack's `/drive` integration."
    files = sorted(p.name for p in DOCS_DIR.iterdir() if p.is_file())
    if not files:
        return "No documents uploaded yet. Share files via Slack's `/drive` integration."
    lines = [f"Indexed sources ({len(files)}):"]
    for name in files:
        size = (DOCS_DIR / name).stat().st_size
        if size > 1024 * 1024:
            lines.append(f"  - {name} ({size / (1024*1024):.1f} MB)")
        else:
            lines.append(f"  - {name} ({size / 1024:.0f} KB)")
    return "\n".join(lines)


def _remove_source(filename: str) -> str:
    """Remove a file from the volume."""
    rag_vol.reload()
    path = DOCS_DIR / filename
    if not path.exists():
        return f"File `{filename}` not found. Use `sources` to see available files."
    path.unlink()
    rag_vol.commit()
    return f"Removed `{filename}`. Run `reindex` or share new files to update the search index."


def _upload_output_files(client, channel: str, thread_ts: str, file_paths: list[str]):
    """Upload generated files from the sandbox volume to Slack."""
    rag_vol.reload()
    for path_str in file_paths:
        path = Path(path_str)
        if path.exists() and path.stat().st_size > 0:
            client.files_upload_v2(
                channel=channel,
                thread_ts=thread_ts,
                file=str(path),
                filename=path.name,
                title=path.name,
            )


# ---------------------------------------------------------------------------
# RAG response helpers
# ---------------------------------------------------------------------------


def _process_rag_response(lines: list[str]) -> tuple[str, list[str]]:
    """Process RAG response lines into (display_text, output_file_paths)."""
    text = "\n".join(lines).strip()
    output_files = OUTPUT_FILE_PATTERN.findall(text)
    display_text = OUTPUT_FILE_PATTERN.sub("", text).strip()
    return display_text, output_files


def _say_text(say, text: str):
    """Post text to Slack, splitting into chunks if needed."""
    if text:
        for chunk in chunk_message(text):
            if chunk.strip():
                say(chunk)
    else:
        say("Agent returned no output.")


# ---------------------------------------------------------------------------
# Message routing
# ---------------------------------------------------------------------------


def handle_agent_message(say, set_status, user_message, thread_ts, files=None, client=None, channel=None):
    """Route and run the agent, posting response chunks back to Slack."""
    sandbox_name = f"agent-{thread_ts}".replace(".", "-")
    lower = user_message.strip().lower()

    try:
        # --- Files shared → download + reindex ---
        if files:
            set_status(status="downloading files...")
            saved = _download_slack_files(files)
            if saved:
                say(f"Saved {len(saved)} file(s): {', '.join(saved)}")
                lines = run_rag_query("reindex", sandbox_name, set_status=set_status)
                display_text, _ = _process_rag_response(lines)
                _say_text(say, display_text)
            else:
                say("No downloadable files found in the shared items.")
            return

        # --- hf: prefix → ML agent (one-shot exec) ---
        if lower.startswith("hf:"):
            set_status(status="thinking...")
            msg = user_message[3:].strip()
            sb = get_ml_sandbox()
            full_response = []
            for result in run_ml_agent_turn(sb, msg, sandbox_name):
                if result.get("response"):
                    full_response.append(result["response"])
            _say_text(say, "\n".join(full_response).strip())
            return

        # --- sources → list files on volume ---
        if lower == "sources":
            say(_list_sources())
            return

        # --- remove <filename> → delete from volume ---
        if lower.startswith("remove "):
            filename = user_message[7:].strip()
            if not filename:
                say("Usage: `remove <filename>`")
                return
            set_status(status="removing file...")
            say(_remove_source(filename))
            return

        # --- Default → RAG query (persistent process) ---
        lines = run_rag_query(user_message, sandbox_name, set_status=set_status)
        display_text, output_files = _process_rag_response(lines)
        _say_text(say, display_text)

        # Upload any generated files
        if output_files and client and channel:
            _upload_output_files(client, channel, thread_ts, output_files)

    except Exception as e:
        say(f":x: Error: {e}")


def process_channel_message(body, client, user_message):
    """Handle @mention messages in channels."""
    print(f"[DEBUG] process_channel_message called with: {user_message!r}", flush=True)
    channel = body["event"]["channel"]
    thread_ts = body["event"].get("thread_ts", body["event"]["ts"])
    sandbox_name = f"agent-{thread_ts}".replace(".", "-")
    lower = user_message.strip().lower()
    files = body["event"].get("files")
    print(f"[DEBUG] channel={channel} thread_ts={thread_ts} sandbox_name={sandbox_name} files={files}", flush=True)

    try:
        # --- Files shared → download + reindex ---
        if files:
            saved = _download_slack_files(files)
            if saved:
                client.chat_postMessage(
                    channel=channel,
                    text=f"Saved {len(saved)} file(s): {', '.join(saved)}",
                    thread_ts=thread_ts,
                )
                lines = run_rag_query("reindex", sandbox_name)
                display_text, _ = _process_rag_response(lines)
                if display_text:
                    for chunk in chunk_message(display_text):
                        if chunk.strip():
                            client.chat_postMessage(channel=channel, text=chunk, thread_ts=thread_ts)
            else:
                client.chat_postMessage(
                    channel=channel,
                    text="No downloadable files found in the shared items.",
                    thread_ts=thread_ts,
                )
            return

        if lower.startswith("hf:"):
            sb = get_ml_sandbox()
            msg = user_message[3:].strip()
            for result in run_ml_agent_turn(sb, msg, sandbox_name):
                if result.get("response"):
                    for chunk in chunk_message(result["response"]):
                        if chunk.strip():
                            client.chat_postMessage(channel=channel, text=chunk, thread_ts=thread_ts)
            return

        if lower == "sources":
            client.chat_postMessage(channel=channel, text=_list_sources(), thread_ts=thread_ts)
            return

        if lower.startswith("remove "):
            filename = user_message[7:].strip()
            client.chat_postMessage(channel=channel, text=_remove_source(filename), thread_ts=thread_ts)
            return

        # --- Default → RAG query ---
        lines = run_rag_query(user_message, sandbox_name)
        display_text, output_files = _process_rag_response(lines)

        if display_text:
            for chunk in chunk_message(display_text):
                if chunk.strip():
                    client.chat_postMessage(channel=channel, text=chunk, thread_ts=thread_ts)
        else:
            client.chat_postMessage(channel=channel, text="Agent returned no output.", thread_ts=thread_ts)

        if output_files:
            _upload_output_files(client, channel, thread_ts, output_files)

    except Exception as e:
        import traceback
        print(f"[DEBUG] process_channel_message error: {e}", flush=True)
        traceback.print_exc()
        client.chat_postMessage(channel=channel, text=f":x: Error: {e}", thread_ts=thread_ts)


# ---------------------------------------------------------------------------
# Slack app
# ---------------------------------------------------------------------------


@app.function(
    secrets=[slack_secret],
    image=slack_bot_image,
    volumes={"/data": rag_vol},
    scaledown_window=10 * 60,
    min_containers=1,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def slack_bot():
    from fastapi import FastAPI, Request
    from slack_bolt import App as SlackApp, Assistant, Say, SetStatus, SetSuggestedPrompts, SetTitle
    from slack_bolt.adapter.fastapi import SlackRequestHandler

    slack_app = SlackApp(
        token=os.environ["SLACK_BOT_TOKEN"],
        signing_secret=os.environ["SLACK_SIGNING_SECRET"],
    )

    # --- AI Assistant (DM / split-pane interface) ---
    assistant = Assistant()

    @assistant.thread_started
    def start_thread(say: Say, set_suggested_prompts: SetSuggestedPrompts):
        say("Hello! I'm an AI agent running in a Modal GPU sandbox. Share files via `/drive` to index them, then ask questions or request analysis.")
        set_suggested_prompts(
            prompts=[
                {"title": "Show sources", "message": "sources"},
                {"title": "Reindex documents", "message": "reindex"},
                {"title": "ML training", "message": "hf: Help me fine-tune a model on HuggingFace"},
            ]
        )

    @assistant.user_message
    def handle_assistant_message(say: Say, set_status: SetStatus, set_title: SetTitle, payload: dict):
        user_message = payload.get("text", "")
        thread_ts = payload.get("thread_ts", payload.get("ts", ""))
        files = payload.get("files")
        channel = payload.get("channel")

        title = user_message[:40] + ("..." if len(user_message) > 40 else "")
        set_title(title=title)

        slack_client = say.client if hasattr(say, "client") else None

        handle_agent_message(
            say, set_status, user_message, thread_ts,
            files=files, client=slack_client, channel=channel,
        )

    slack_app.use(assistant)

    # --- Channel @mention handler ---
    @slack_app.event("app_mention")
    def handle_mention(body, client, context, logger):
        print(f"[DEBUG] app_mention event received", flush=True)
        print(f"[DEBUG] body keys: {list(body.keys())}", flush=True)
        print(f"[DEBUG] event: {body.get('event', {})}", flush=True)
        user_message = body["event"]["text"]
        user_message = re.sub(r"<@[A-Z0-9]+>", "", user_message).strip()
        print(f"[DEBUG] parsed message: {user_message!r}", flush=True)
        threading.Thread(
            target=process_channel_message, args=(body, client, user_message), daemon=True
        ).start()
        print(f"[DEBUG] background thread started", flush=True)

    @slack_app.event("message")
    def handle_message(body, client, context, logger):
        print(f"[DEBUG] message event received, subtype={body.get('event', {}).get('subtype')}", flush=True)
        pass  # Assistant middleware handles DMs

    fastapi_app = FastAPI()
    handler = SlackRequestHandler(slack_app)

    @fastapi_app.post("/")
    async def root(request: Request):
        from starlette.responses import Response

        retry = request.headers.get("x-slack-retry-num")
        print(f"[DEBUG] POST / received, retry={retry}", flush=True)
        if retry:
            print(f"[DEBUG] Skipping retry #{retry}", flush=True)
            return Response(status_code=200)
        result = await handler.handle(request)
        print(f"[DEBUG] handler returned status={result.status_code}", flush=True)
        return result

    return fastapi_app
