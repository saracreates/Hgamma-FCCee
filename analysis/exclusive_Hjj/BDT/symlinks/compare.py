#!/usr/bin/env python3
import re
import sys
from pathlib import Path

# Regex to extract the numeric ID before ".root" or ".root.gz"
PATTERN = re.compile(r"/events_(\d+)\.root$")


def extract_ids(list_file: Path) -> set[str]:
    """Extract numeric IDs from lines like .../events/12345.root"""
    ids = set()
    with list_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            m = PATTERN.search(line)
            if m:
                ids.add(m.group(1))
    return ids

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {Path(sys.argv[0]).name} <file1.txt> <file2.txt>", file=sys.stderr)
        sys.exit(1)

    f1 = Path(sys.argv[1])
    f2 = Path(sys.argv[2])

    ids1 = extract_ids(f1)
    ids2 = extract_ids(f2)

    both   = ids1 & ids2
    only_1 = ids1 - ids2
    only_2 = ids2 - ids1

    print(f"# Summary")
    print(f"File 1: {f1} -> {len(ids1)} IDs")
    print(f"File 2: {f2} -> {len(ids2)} IDs")
    print(f"Common IDs: {len(both)}")
    print(f"Only in file 1: {len(only_1)}")
    print(f"Only in file 2: {len(only_2)}\n")

    # Print IDs extracted from each file
    print(f"# IDs from {f1}:")
    print(", ".join(sorted(ids1, key=int)) if ids1 else "(none)")
    print()
    print(f"# IDs from {f2}:")
    print(", ".join(sorted(ids2, key=int)) if ids2 else "(none)")
    print()

    # Print common IDs
    print("# Common IDs:")
    if both:
        print(", ".join(sorted(both, key=int)))
    else:
        print("(none)")

if __name__ == "__main__":
    main()

