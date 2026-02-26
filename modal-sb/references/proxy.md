# Anthropic API Proxy — Credential Isolation

The proxy keeps the real API key outside the sandbox. The sandbox sends its `MODAL_SANDBOX_ID` as the API key; the proxy swaps in the real key and streams the response back.

## How It Works

```
Sandbox                        Proxy                          Anthropic
──────                        ─────                          ─────────
POST /v1/messages              1. Extract x-api-key header
  x-api-key: <sandbox_id>     2. Swap in real ANTHROPIC_API_KEY
                               3. Stream to api.anthropic.com ──→ POST /v1/messages
                               4. Stream response back ←─────────  SSE events
```

## Modal Function Setup

```python
import os
import modal
from my_package.modal_app import app

proxy_secret = modal.Secret.from_name("anthropic-secret")
proxy_image = modal.Image.debian_slim(python_version="3.12").pip_install("httpx", "fastapi")

@app.function(image=proxy_image, secrets=[proxy_secret], min_containers=0)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def anthropic_proxy():
    ...
```

Key decorators:
- `@modal.asgi_app()` — exposes the function as a persistent web endpoint with a stable URL
- `@modal.concurrent(max_inputs=10)` — allows a single container to handle up to 10 concurrent requests (a single agent only needs 1-2)
- `min_containers=0` — scales to zero when idle (set to `1` for always-warm in production)

## Header Forwarding

```python
skip = {"host", "content-length"}
headers = {k: v for k, v in request.headers.items() if k.lower() not in skip}
headers["x-api-key"] = os.environ["ANTHROPIC_API_KEY"]
```

Two headers are dropped before forwarding — httpx sets them automatically:
- `host` — httpx sets this to the upstream host (`api.anthropic.com`)
- `content-length` — httpx recalculates from the body

The `x-api-key` is overwritten with the real API key from the secret.

## Streaming Response

```python
async def pipe():
    async with httpx.AsyncClient(timeout=300.0) as client:
        async with client.stream("POST", url, headers=headers, content=body) as resp:
            async for chunk in resp.aiter_bytes():
                yield chunk

return StreamingResponse(pipe(), media_type="text/event-stream")
```

- Uses `client.stream()` + `aiter_bytes()` to stream SSE events back to the SDK in real time
- 300s timeout accommodates long-running Claude responses
- `text/event-stream` media type for Server-Sent Events

## Full Implementation

```python
"""Anthropic API proxy — injects the real API key and streams responses.

The sandbox can't hold secrets directly, so this proxy sits between the
Claude Agent SDK and api.anthropic.com, swapping in ANTHROPIC_API_KEY.
"""

import os

import modal

from my_package.modal_app import app

proxy_image = modal.Image.debian_slim(python_version="3.12").pip_install("httpx", "fastapi")
proxy_secret = modal.Secret.from_name("anthropic-secret")


@app.function(image=proxy_image, secrets=[proxy_secret], min_containers=0)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def anthropic_proxy():
    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse

    proxy = FastAPI()

    # Catch-all route: {path:path} matches any path including slashes (e.g. "v1/messages")
    @proxy.api_route("/{path:path}", methods=["POST"])
    async def forward(request: Request, path: str):
        body = await request.body()

        # Drop headers that httpx sets automatically, swap in the real API key
        skip = {"host", "content-length"}
        headers = {k: v for k, v in request.headers.items() if k.lower() not in skip}
        headers["x-api-key"] = os.environ["ANTHROPIC_API_KEY"]
        url = f"https://api.anthropic.com/{path}"

        # Stream the response so the SDK gets SSE events in real time
        async def pipe():
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream("POST", url, headers=headers, content=body) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk

        return StreamingResponse(pipe(), media_type="text/event-stream")

    return proxy
```
