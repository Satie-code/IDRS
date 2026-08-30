"""Turn the raw acquisition log into a tracked provenance artifact.

The acquisition step fetches Git LFS objects whose oid *is* the SHA-256 of the
content, so the resulting manifest doubles as an integrity record: it proves
which upstream object each local file came from and that it verified.

Usage::

    python scripts/build_acquisition_manifest.py \\
        --log data/metadata/_acquisition_raw.json \\
        --pointer-root data/IO-VNBD_dataset/IO-VNBD-master \\
        --output data/metadata/io_vnbd
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from idr.dataset.acquire import DEFAULT_LFS_REPO
from idr.dataset.lfs import read_pointer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--pointer-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    entries = json.loads(args.log.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    for entry in entries:
        relative = entry["relative_path"]
        pointer = read_pointer(args.pointer_root / relative)
        rows.append(
            {
                "relative_path": relative,
                "lfs_oid_sha256": entry["oid"] or (pointer.oid if pointer else ""),
                "expected_size_bytes": entry["expected_size"],
                "downloaded_bytes": entry["downloaded_bytes"],
                "sha256_verified": entry["verified"],
                "skipped_existing": entry["skipped_existing"],
                "error": entry["error"] or "",
            }
        )
    rows.sort(key=lambda row: str(row["relative_path"]))

    manifest_path = args.output / "acquisition_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    verified = sum(1 for row in rows if row["sha256_verified"])
    summary = {
        "source_repository": DEFAULT_LFS_REPO,
        "source_kind": "Git LFS batch API (official upstream repository)",
        "objects_requested": len(rows),
        "objects_sha256_verified": verified,
        "objects_failed": sum(1 for row in rows if row["error"]),
        "total_bytes": sum(int(row["expected_size_bytes"]) for row in rows),
        "verification_method": (
            "each object's Git LFS oid is the SHA-256 of its content; every "
            "downloaded file was hashed and compared before being moved into place"
        ),
        "note": (
            "The pointer tree under data/IO-VNBD_dataset/ was left untouched and "
            "remains the provenance record; real content was written to a separate "
            "destination."
        ),
    }
    (args.output / "acquisition_report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"wrote {manifest_path} ({len(rows)} rows, {verified} verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
