from __future__ import annotations

import argparse
from pathlib import Path

from mcollapse.experiment import run_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cells", type=int)
    args = parser.parse_args(argv)
    output = args.output or Path("mcollapse/results") / args.manifest.stem
    result = run_manifest(
        args.manifest,
        output,
        resume=args.resume,
        max_cells=args.max_cells,
    )
    print(
        f"manifest={result.manifest_hash} "
        f"cells={result.completed_cells}/{result.total_cells} output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
