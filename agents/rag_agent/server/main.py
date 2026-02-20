"""RAG agent server entrypoint — create container and start stdin loop."""

from .container import ServiceContainer
from .handler import END_TURN, MessageHandler


def main():
    container = ServiceContainer()
    print("Models loaded. Ready for queries.", flush=True)
    print(END_TURN, flush=True)
    MessageHandler(container).run()


if __name__ == "__main__":
    main()
