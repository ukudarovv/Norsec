"""Entry point: `python -m bullying_ai` — smoke import or optional webcam demo."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="bullying_ai phase-1 (detection / tracking / pose)")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="import package and optional heavy deps check",
    )
    args = p.parse_args(argv)
    import bullying_ai  # noqa: F401

    print("bullying_ai import OK — PersonDetector, PeopleTracker, PoseEstimator")
    if args.smoke:
        try:
            import ultralytics  # noqa: F401
            import supervision  # noqa: F401

            print("optional: ultralytics, supervision OK (see requirements-bullying.txt)")
        except ImportError as e:
            print("optional deps missing:", e, file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
