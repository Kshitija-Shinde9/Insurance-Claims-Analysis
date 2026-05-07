"""Download the raw freMTPL2 CSV files used by the final project."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fremtpl2_utils import FREQ_FILE, FREQ_URL, SEV_FILE, SEV_URL, ensure_directories


def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"Already exists: {out_path} ({out_path.stat().st_size:,} bytes)")
        return

    print(f"Downloading {url}")
    print(f"       to {out_path}")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        with out_path.open("wb") as fh:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
    print(f"Saved {out_path} ({out_path.stat().st_size:,} bytes)")


def main() -> None:
    ensure_directories()
    download(FREQ_URL, FREQ_FILE)
    download(SEV_URL, SEV_FILE)
    print("\nDone. Next run:")
    print("python scripts/build_fremtpl2_modeling_file.py")


if __name__ == "__main__":
    main()
