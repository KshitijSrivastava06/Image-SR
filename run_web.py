"""Serve the standalone super-resolution webpage on this computer."""

import argparse
from threading import Event, Thread
from urllib.request import urlopen
import webbrowser

import uvicorn


def studio_is_ready(url: str) -> bool:
    """Identify the Studio before opening a browser or reusing a running server."""
    try:
        with urlopen(url, timeout=0.5) as response:
            return b'id="settings-form"' in response.read()
    except (OSError, ValueError):
        return False


def open_when_ready(url: str, stopped: Event) -> None:
    """Wait for the page to respond before asking Windows to open it."""
    while not stopped.is_set():
        if studio_is_ready(url):
            if not stopped.is_set():
                webbrowser.open(url)
            return
        stopped.wait(0.3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    host = "127.0.0.1"
    url = f"http://{host}:{args.port}/"
    stopped = Event()
    if args.open_browser:
        if studio_is_ready(url):
            print(f"ImageSR Studio is already running at {url}")
            webbrowser.open(url)
            return
        Thread(target=open_when_ready, args=(url, stopped), daemon=True).start()

    # Uvicorn serves the FastAPI object on this computer's loopback address.
    try:
        uvicorn.run("app.web_api:app", host=host, port=args.port)
    finally:
        stopped.set()


if __name__ == "__main__":
    main()
