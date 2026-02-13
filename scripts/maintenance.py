"""SQLite maintenance utilities (vacuum, integrity check, backup)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    from personal_search_layer.config import DB_PATH
    from personal_search_layer.storage import connect, migrate_schema, require_schema
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from personal_search_layer.config import DB_PATH  # type: ignore[reportMissingImports]
    from personal_search_layer.storage import (  # type: ignore[reportMissingImports]
        connect,
        migrate_schema,
        require_schema,
    )


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
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Apply schema migrations to the current database",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (args.vacuum or args.integrity_check or args.backup or args.migrate):
        print(
            "No maintenance actions requested. Use --migrate, --vacuum, --integrity-check, or --backup."
        )
        return

    with connect(DB_PATH) as conn:
        if args.migrate:
            migrate_schema(conn)
            conn.commit()
            print("Schema migration completed.")
        else:
            require_schema(conn)
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
