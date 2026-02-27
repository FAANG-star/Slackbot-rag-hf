"""Anthropic API proxy — injects the real API key and streams responses.

The sandbox can't hold secrets directly, so this proxy sits between the
Claude Agent SDK and api.anthropic.com, swapping in ANTHROPIC_API_KEY.
"""

import os

import modal

from slackbot.modal_app import app

proxy_image = modal.Image.debian_slim(python_version="3.12").pip_install("httpx", "fastapi")
proxy_secret = modal.Secret.from_name("anthropic-secret")


@app.function(image=proxy_image, secrets=[proxy_secret], min_containers=0)
@modal.concurrent(max_inputs=100)
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
                    async for chunk in resp.aiter_bytes(): # Stream instead of buffer 
                        yield chunk

        return StreamingResponse(pipe(), media_type="text/event-stream")

    return proxy
