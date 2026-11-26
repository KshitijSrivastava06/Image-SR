"""Launch the Streamlit application from any working directory."""

from pathlib import Path
import subprocess
import sys


def main() -> int:
    root = Path(__file__).resolve().parent
    app = root / "app" / "streamlit_app.py"
    try:
        return subprocess.call(
            [sys.executable, "-m", "streamlit", "run", str(app), *sys.argv[1:]],
            cwd=root,
        )
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
