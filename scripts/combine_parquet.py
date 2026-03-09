"""Combine parquet files in a directory into a single file."""

import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq


def combine_parquet(directory: Path, output: Path | None = None) -> Path:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        print(f"No parquet files found in {directory}")
        sys.exit(1)

    if output is None:
        output = directory.parent / f"{directory.name}.parquet"

    print(f"Combining {len(files)} files from {directory} -> {output}")
    table = pq.read_table(files[0])
    tables = [table]
    for f in files[1:]:
        tables.append(pq.read_table(f))

    import pyarrow as pa
    combined = pa.concat_tables(tables)
    pq.write_table(combined, output)
    print(f"Done: {len(combined)} rows, {output.stat().st_size / 1024 / 1024:.1f} MB")
    return output


def main():
    parser = argparse.ArgumentParser(description="Combine parquet files in a directory")
    parser.add_argument("directories", nargs="*", help="Directories to combine (default: all 2026-03-09 dirs)")
    parser.add_argument("--date", default="2026-03-09", help="Date folder to combine")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent / "src" / "data"

    if args.directories:
        dirs = [Path(d) for d in args.directories]
    else:
        dirs = [
            base / "coinbase" / "orderbook" / args.date,
            base / "coinbase" / "trade" / args.date,
            base / "kalshi" / "orderbook" / args.date,
            base / "kalshi" / "trade" / args.date,
            base / "kalshi" / "kalshi_market" / args.date,
        ]

    for d in dirs:
        if d.exists():
            combine_parquet(d)
        else:
            print(f"Skipping {d} (not found)")


if __name__ == "__main__":
    main()
