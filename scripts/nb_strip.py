"""Run nbstripout over every committed notebook, passing extra flags through; succeed when there is none."""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    # NUL-delimited so unusual filenames survive; git C-quotes non-ASCII paths otherwise.
    listing = subprocess.run(["git", "ls-files", "-z", "*.ipynb"], capture_output=True, text=True, check=True)
    notebooks = [path for path in listing.stdout.split("\0") if path]
    if not notebooks:
        return 0
    return subprocess.run(["nbstripout", *sys.argv[1:], *notebooks]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
