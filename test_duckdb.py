"""Print one filtered Arctic month. Run: python3 test_duckdb.py"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from osi.loaders.reddit_duckdb import load_month  # noqa: E402

def main() -> None:
    frame = load_month(2012, 8, ["programming", "science"])
    print(len(frame))
    print(frame.head(5))


if __name__ == "__main__":
    main()
