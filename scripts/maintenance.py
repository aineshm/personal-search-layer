"""SQLite maintenance utilities (vacuum, integrity check, backup)."""

from __future__ import annotations

import argparse
from pathlib import Path

from personal_search_layer.config import DB_PATH
from personal_search_layer.storage import connect, initialize_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SQLite maintenance tasks")
    parser.add_argument(
        "--vacuum", action="store_true", help="Run VACUUM to compact the database"
    )
    parser.add_argument(
        "--integrity-check",
        action="store_true",
        help="Run PRAGMA integrity_check",
    )
    parser.add_argument(
        "--backup", type=Path, help="Write a backup copy to the given path"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (args.vacuum or args.integrity_check or args.backup):
        print(
            "No maintenance actions requested. Use --vacuum, --integrity-check, or --backup."
        )
        return

    with connect(DB_PATH) as conn:
        initialize_schema(conn)
        if args.integrity_check:
            result = conn.execute("PRAGMA integrity_check").fetchone()
            print("Integrity check:", result[0] if result else "unknown")
        if args.vacuum:
            conn.execute("VACUUM")
            print("VACUUM completed.")
        if args.backup:
            args.backup.parent.mkdir(parents=True, exist_ok=True)
            with connect(args.backup) as backup_conn:
                conn.backup(backup_conn)
                backup_conn.commit()
            print(f"Backup written to {args.backup}")


if __name__ == "__main__":
    main()
