"""Materialize Git LFS objects referenced by pointer files.

The local IO-VNBD copy under ``data/IO-VNBD_dataset/`` is a ZIP export of the
upstream GitHub repository, so every CSV/JPG in it is an LFS *pointer* with
no actual content. This module fetches the real objects from the upstream
LFS endpoint using the batch API, verifying each download against the SHA-256
recorded in its pointer (an LFS oid *is* the content hash, so verification is
free and non-negotiable).

Acquisition writes into a separate destination tree and never modifies the
pointer tree, which is preserved as the provenance record of what was
originally shipped.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from idr.dataset.lfs import LfsPointer, read_pointer
from idr.logging import get_logger

logger = get_logger(__name__)

#: Official upstream source of IO-VNBD (matches the oids in the local pointers).
DEFAULT_LFS_REPO = "https://github.com/onyekpeu/IO-VNBD.git"

_LFS_CONTENT_TYPE = "application/vnd.git-lfs+json"
_BATCH_LIMIT = 100


class AcquisitionError(RuntimeError):
    """Raised when an LFS object cannot be fetched or fails verification."""


@dataclass(frozen=True)
class AcquisitionResult:
    """Outcome of materializing one pointer."""

    relative_path: str
    oid: str
    expected_size: int
    downloaded_bytes: int
    verified: bool
    skipped_existing: bool
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and (self.verified or self.skipped_existing)


def _batch_request(repo_url: str, pointers: list[LfsPointer], timeout: float) -> dict[str, str]:
    """Ask the LFS server for download URLs. Returns ``{oid: href}``."""
    payload = json.dumps(
        {
            "operation": "download",
            "transfers": ["basic"],
            "objects": [{"oid": p.oid, "size": p.size} for p in pointers],
        }
    ).encode()
    request = urllib.request.Request(
        f"{repo_url}/info/lfs/objects/batch",
        data=payload,
        headers={"Accept": _LFS_CONTENT_TYPE, "Content-Type": _LFS_CONTENT_TYPE},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.load(response)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise AcquisitionError(f"LFS batch request to {repo_url} failed: {exc}") from exc

    hrefs: dict[str, str] = {}
    for obj in body.get("objects", []):
        oid = obj.get("oid", "")
        action = obj.get("actions", {}).get("download")
        if action is None:
            reason = obj.get("error", {}).get("message", "no download action returned")
            logger.warning("LFS object %s unavailable: %s", oid, reason)
            continue
        hrefs[oid] = action["href"]
    return hrefs


def _download_one(
    href: str, pointer: LfsPointer, destination: Path, timeout: float
) -> tuple[int, bool]:
    """Download to a temp file, verify SHA-256, then move into place.

    Writing through a ``.part`` file means an interrupted or corrupt download
    can never be mistaken for a complete one on a later run.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")

    digest = hashlib.sha256()
    total = 0
    with urllib.request.urlopen(href, timeout=timeout) as response, temporary.open("wb") as handle:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            handle.write(chunk)
            total += len(chunk)

    verified = digest.hexdigest() == pointer.oid and total == pointer.size
    if not verified:
        temporary.unlink(missing_ok=True)
        return total, False

    temporary.replace(destination)
    return total, True


def acquire_pointers(
    pointer_root: Path,
    destination_root: Path,
    *,
    patterns: tuple[str, ...] = ("*.csv",),
    repo_url: str = DEFAULT_LFS_REPO,
    max_workers: int = 8,
    timeout: float = 300.0,
    copy_non_pointers: bool = True,
) -> list[AcquisitionResult]:
    """Materialize every LFS pointer under ``pointer_root`` into ``destination_root``.

    The destination mirrors the source layout. Files that are already present
    and the right size are skipped, so the operation is resumable. Real
    (non-pointer) files matching ``patterns`` are copied verbatim when
    ``copy_non_pointers`` is set, so the destination is a complete tree.

    ``patterns`` limits which files are considered; the default fetches only
    CSVs, since the dataset's JPGs are scene photographs irrelevant to sensor
    forensics (their pointers are still inventoried).
    """
    if not pointer_root.is_dir():
        raise AcquisitionError(f"Pointer root does not exist or is not a directory: {pointer_root}")

    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(sorted(pointer_root.rglob(pattern)))

    jobs: list[tuple[Path, LfsPointer]] = []
    results: list[AcquisitionResult] = []

    for path in candidates:
        relative = path.relative_to(pointer_root)
        target = destination_root / relative
        pointer = read_pointer(path)

        if pointer is None:
            if copy_non_pointers and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
            results.append(
                AcquisitionResult(
                    relative_path=relative.as_posix(),
                    oid="",
                    expected_size=path.stat().st_size,
                    downloaded_bytes=0,
                    verified=False,
                    skipped_existing=True,
                )
            )
            continue

        if target.exists() and target.stat().st_size == pointer.size:
            results.append(
                AcquisitionResult(
                    relative_path=relative.as_posix(),
                    oid=pointer.oid,
                    expected_size=pointer.size,
                    downloaded_bytes=0,
                    verified=False,
                    skipped_existing=True,
                )
            )
            continue

        jobs.append((target, pointer))

    if not jobs:
        logger.info("Nothing to acquire; %d file(s) already present", len(results))
        return results

    total_bytes = sum(p.size for _, p in jobs)
    logger.info(
        "Acquiring %d LFS object(s), %.2f GB from %s", len(jobs), total_bytes / 1e9, repo_url
    )

    # Resolve download URLs in batches (they are short-lived signed URLs).
    for start in range(0, len(jobs), _BATCH_LIMIT):
        window = jobs[start : start + _BATCH_LIMIT]
        hrefs = _batch_request(repo_url, [p for _, p in window], timeout)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for target, pointer in window:
                href = hrefs.get(pointer.oid)
                relative_name = target.relative_to(destination_root).as_posix()
                if href is None:
                    results.append(
                        AcquisitionResult(
                            relative_path=relative_name,
                            oid=pointer.oid,
                            expected_size=pointer.size,
                            downloaded_bytes=0,
                            verified=False,
                            skipped_existing=False,
                            error="LFS server returned no download URL (object missing?)",
                        )
                    )
                    continue
                futures[pool.submit(_download_one, href, pointer, target, timeout)] = (
                    relative_name,
                    pointer,
                )

            for future in as_completed(futures):
                relative_name, pointer = futures[future]
                try:
                    downloaded, verified = future.result()
                except (urllib.error.URLError, OSError) as exc:
                    results.append(
                        AcquisitionResult(
                            relative_path=relative_name,
                            oid=pointer.oid,
                            expected_size=pointer.size,
                            downloaded_bytes=0,
                            verified=False,
                            skipped_existing=False,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
                    continue
                results.append(
                    AcquisitionResult(
                        relative_path=relative_name,
                        oid=pointer.oid,
                        expected_size=pointer.size,
                        downloaded_bytes=downloaded,
                        verified=verified,
                        skipped_existing=False,
                        error=None if verified else "SHA-256 / size verification failed",
                    )
                )

        done = len([r for r in results if r.downloaded_bytes])
        logger.info("Acquired %d/%d object(s)", done, len(jobs))

    return results
