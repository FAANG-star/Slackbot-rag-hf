"""Quick test: run the SlackBot's create_fastapi_app and inspect routes."""
import os
os.environ.setdefault("SLACK_BOT_TOKEN", "xoxb-test")
os.environ.setdefault("SLACK_SIGNING_SECRET", "test123")

from fastapi.testclient import TestClient
from agents.slackbot.bot import SlackBot
from agents.slackbot.services.container import ServiceContainer

app = SlackBot(ServiceContainer()).create_fastapi_app()

print("Routes:")
for route in app.routes:
    print(f"  {getattr(route, 'methods', 'N/A')} {getattr(route, 'path', route)}")

client = TestClient(app)
resp = client.post("/", json={"type": "url_verification", "challenge": "test123"})
print(f"\nStatus: {resp.status_code}")
print(f"Body: {resp.text[:500]}")
