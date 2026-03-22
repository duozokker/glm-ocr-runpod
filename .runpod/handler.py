"""
Hub-specific entrypoint shim.

Runpod Hub can prioritize files under .runpod/, so this forwards execution
to the repository-root handler used by the Docker image.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from handler import main


if __name__ == "__main__":
    main()
