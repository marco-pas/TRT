#!/usr/bin/env python3
"""Fail CI when a forward benchmark did not produce finite converged records.

This deliberately checks correctness only, not wall time or a required Krylov
speedup.  Performance varies across GitHub-hosted runners and belongs in the
archived benchmark data rather than in a brittle merge gate.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    args = parser.parse_args()

    failures: list[str] = []
    for directory in args.directories:
        metrics_path = directory / "metrics.csv"
        if not metrics_path.is_file():
            failures.append(f"{directory}: missing metrics.csv")
            continue
        with metrics_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            failures.append(f"{directory}: metrics.csv contains no time steps")
            continue
        for row in rows:
            step = row.get("step", "?")
            if not as_bool(row.get("converged", "false")):
                failures.append(f"{directory}: step {step} is not converged")
            for name, value in row.items():
                if name in {"converged", "reason"} or value in {"", None}:
                    continue
                try:
                    numeric = float(value)
                except ValueError:
                    continue
                if not math.isfinite(numeric):
                    failures.append(f"{directory}: step {step} has non-finite {name}={value}")
        print(f"PASS {directory}: {len(rows)} converged, finite step records")

    if failures:
        raise SystemExit("CI result validation failed:\n- " + "\n- ".join(failures))


if __name__ == "__main__":
    main()
