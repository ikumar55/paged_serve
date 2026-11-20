"""Run the router in front of engine replicas.

    python -m router.server --replica http://10.0.0.1:8000 --replica http://10.0.0.2:8000 \\
        --policy precise-prefix --port 9000

The replicas are ``engine.server`` processes (real or ``--backend mock``). The router speaks the
same API, so the load generator can be pointed at it unchanged.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import uvicorn

from router.app import create_router_app
from router.policies import DEFAULT_MAX_ENTRIES, POLICIES, make_policy


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument(
        "--replica",
        action="append",
        required=True,
        help="base URL of an engine replica, for example http://host:8000; repeat for each",
    )
    parser.add_argument("--policy", choices=POLICIES, default="least-outstanding")
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=16,
        help="approx-prefix: tokens per remembered chunk (match the replicas' block size)",
    )
    parser.add_argument(
        "--load-cap",
        type=int,
        default=16,
        help="prefix policies: requests in flight on a replica above which it is not preferred",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=DEFAULT_MAX_ENTRIES,
        help="approx-prefix: how many prompt chunks the router remembers (its memory limit)",
    )
    parser.add_argument("--poll-interval", type=float, default=0.5, help="precise-prefix: seconds")
    parser.add_argument(
        "--cooldown", type=float, default=5.0, help="seconds a failed replica rests"
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: local only)")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args(argv)

    for url in args.replica:
        if not url.startswith(("http://", "https://")):
            parser.error(f"replica {url!r} must start with http:// or https://")
    if args.max_entries <= 0:
        parser.error("--max-entries must be positive")
    policy = make_policy(
        args.policy,
        chunk_tokens=args.chunk_tokens,
        load_cap=args.load_cap,
        max_entries=args.max_entries,
    )
    app = create_router_app(
        args.replica, policy, poll_interval_s=args.poll_interval, cooldown_s=args.cooldown
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
