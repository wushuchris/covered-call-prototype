# main.py — launches both microservices as subprocesses
# FastHTML UI server (port 8008) and FastAPI inference server (port 8009)
# run concurrently. on keyboard interrupt both are terminated gracefully.

import subprocess
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")


def main():
    """Spawn the UI and inference servers as parallel Popen processes.

    Both processes inherit stdout/stderr so logs are visible in the
    terminal. SIGINT (Ctrl+C) terminates both gracefully.
    CWD is project root so `src` package resolves correctly.
    """
    procs = []
    try:
        # FastAPI inference service — port 8009
        inference_proc = subprocess.Popen(
            [sys.executable, os.path.join(SRC_DIR, "inference", "app.py")],
            cwd=PROJECT_ROOT,
        )
        procs.append(inference_proc)
        print("[main] Inference server starting on http://localhost:8009")

        # FastHTML UI service — port 8008
        ui_proc = subprocess.Popen(
            [sys.executable, os.path.join(SRC_DIR, "ui", "app.py")],
            cwd=PROJECT_ROOT,
        )
        procs.append(ui_proc)
        print("[main] UI server starting on http://localhost:8008")

        # wait for both — blocks until one exits or interrupted
        for proc in procs:
            proc.wait()

    except KeyboardInterrupt:
        print("\n[main] Shutting down...")
        for proc in procs:
            proc.terminate()
        for proc in procs:
            proc.wait()
        print("[main] All services stopped.")


if __name__ == "__main__":
    main()
