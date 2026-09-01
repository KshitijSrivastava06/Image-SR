import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Streamlit web application."""
    app_path = Path(__file__).parent / "app" / "streamlit_app.py"
    
    if not app_path.exists():
        print(f"Error: Could not find {app_path}")
        sys.exit(1)
        
    print(f"Starting Streamlit app: {app_path}")
    print("Press Ctrl+C to stop the server.")
    
    try:
        # Run streamlit as a subprocess
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path)],
            check=True
        )
    except KeyboardInterrupt:
        print("\nShutting down Streamlit server...")
    except subprocess.CalledProcessError as e:
        print(f"\nError running Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
