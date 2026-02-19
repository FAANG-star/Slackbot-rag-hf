"""Anthropic API proxy — swaps in the real API key and forwards to Anthropic."""

import os

import modal

from .shared import app

proxy_secret = modal.Secret.from_name("anthropic-secret")
proxy_image = modal.Image.debian_slim(python_version="3.12").pip_install("httpx", "fastapi")


def run_proxy():
    """Build the FastAPI app that forwards requests to Anthropic."""
    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse

    app = FastAPI()

    @app.api_route("/{path:path}", methods=["POST"])
    async def forward(request: Request, path: str):
        body = await request.body()

        # Forward all headers, but swap in the real API key
        skip = {"host", "content-length"}
        headers = {k: v for k, v in request.headers.items() if k.lower() not in skip}
        headers["x-api-key"] = os.environ["ANTHROPIC_API_KEY"]

        url = f"https://api.anthropic.com/{path}"

        # Stream the response back so the SDK gets SSE events in real time
        async def pipe():
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream("POST", url, headers=headers, content=body) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk

        return StreamingResponse(pipe(), media_type="text/event-stream")

    return app


@app.function(image=proxy_image, secrets=[proxy_secret], min_containers=0)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def anthropic_proxy():
    return run_proxy()
