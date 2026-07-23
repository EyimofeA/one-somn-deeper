"""Materialize a self-contained carry-scan competition submission.

Use this only when promoting a researched configuration to an experiment's
frozen ``submission.py`` or a remote upload. Do not hand-edit the output.
"""

from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "solving" / "research" / "carry_scan.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-prototypes", type=int, default=0)
    parser.add_argument(
        "--prototype-mode",
        choices=("hard", "soft"),
        default="hard",
    )
    args = parser.parse_args()
    if args.num_prototypes < 0:
        raise ValueError("--num-prototypes must be non-negative")
    source = SOURCE.read_text()
    frozen = (
        "# Frozen generated carry-scan submission; source: "
        "solving/research/carry_scan.py.\n\n"
        + source
        + "\n\nSUBMISSION = build_submission(\n"
        + "    CarryScanSettings(\n"
        + f"        num_prototypes={args.num_prototypes},\n"
        + f"        prototype_mode={args.prototype_mode!r},\n"
        + "    )\n"
        + ")\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(frozen)


if __name__ == "__main__":
    main()
