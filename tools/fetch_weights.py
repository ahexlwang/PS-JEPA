"""Put the released weights where the code expects them, and verify what arrives.

The code and the weights are distributed separately: a source tree should not carry a
126 MB binary, and a decoder checkpoint is an order of magnitude larger again. This program
is the bridge. It reads `weights/MANIFEST.tsv`, which is tracked and lists every released
file with its size and sha256, and for each one either verifies a copy already present or
fetches it.

    python tools/fetch_weights.py                     # fetch what is missing, verify the rest
    python tools/fetch_weights.py --verify-only       # check what is there, fetch nothing
    python tools/fetch_weights.py --from /shared/psjepa-weights
    python tools/fetch_weights.py --from-release v1.2.3
    python tools/fetch_weights.py --base-url https://example.org/psjepa/v1.2.3

Where files come from, in the order tried:

1. ``--from DIR``, a directory holding the files (a shared filesystem, a mounted release, an
   unpacked archive). Files are copied, not linked, so a later change on the other side
   cannot alter what this checkout runs.
2. ``--from-release TAG``, a GitHub release, fetched with ``gh release download``. This is
   the route that works while the repository is private: a plain download has no credential,
   but ``gh`` uses the one the collaborator already has. Asset names are the manifest paths
   with ``/`` replaced by ``-``, because release assets have no directories.
3. ``--base-url URL`` or ``PSJEPA_WEIGHTS_URL``, a location each file's path is appended to.
   Use this once the files are somewhere unauthenticated.
4. the ``url`` column of the manifest, where one is recorded.

Everything is checked against the manifest's sha256 after it arrives, and a file that does
not match is deleted rather than left in place. A partially-correct weights directory is the
failure that produces plausible wrong numbers instead of an error.
"""
from __future__ import annotations

from psjepa.environment import getenv

import argparse
import csv
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from psjepa.paths import file_sha256, project_root, weights_root

MANIFEST = project_root() / "weights" / "MANIFEST.tsv"


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"no manifest at {path}")
    with path.open() as handle:
        rows = [
            row
            for row in csv.DictReader(
                (line for line in handle if not line.startswith("#")), delimiter="\t"
            )
            if row.get("path")
        ]
    if not rows:
        raise SystemExit(f"{path} lists no files")
    return rows


def verify(destination: Path, expected_sha: str, expected_bytes: int) -> str | None:
    """Return None when the file is correct, or a sentence saying what is wrong."""
    if not destination.is_file():
        return "absent"
    actual_bytes = destination.stat().st_size
    if actual_bytes != expected_bytes:
        return f"{actual_bytes} bytes, expected {expected_bytes}"
    actual_sha = file_sha256(destination)
    if actual_sha != expected_sha:
        return f"sha256 {actual_sha[:16]}..., expected {expected_sha[:16]}..."
    return None


def asset_name(path: str) -> str:
    """Release assets have no directories, so `encoder/x.pt` is uploaded as `encoder-x.pt`."""
    return path.replace("/", "-")


def fetch_from_release(row: dict[str, str], destination: Path, tag: str, repo: str | None) -> bool:
    name = asset_name(row["path"])
    command = ["gh", "release", "download", tag, "--pattern", name,
               "--output", str(destination), "--clobber"]
    if repo:
        command += ["--repo", repo]
    print(f"    gh release download {tag} --pattern {name}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    {result.stderr.strip() or 'gh release download failed'}", file=sys.stderr)
        destination.unlink(missing_ok=True)
        return False
    return True


def fetch(row: dict[str, str], destination: Path, source_dir: Path | None, base_url: str | None,
          release: str | None = None, repo: str | None = None) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if release:
        if fetch_from_release(row, destination, release, repo):
            return True
    if source_dir is not None:
        candidate = source_dir / row["path"]
        if not candidate.is_file():
            candidate = source_dir / Path(row["path"]).name
        if candidate.is_file():
            print(f"    copying from {candidate}")
            shutil.copyfile(candidate, destination)
            return True
    url = row.get("url") or ""
    if base_url:
        url = f"{base_url.rstrip('/')}/{row['path']}"
    if url:
        print(f"    downloading {url}")
        try:
            with urllib.request.urlopen(url) as response, destination.open("wb") as out:
                shutil.copyfileobj(response, out)
            return True
        except (urllib.error.URLError, OSError) as error:
            print(f"    download failed: {error}", file=sys.stderr)
            destination.unlink(missing_ok=True)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="source_dir", type=Path, help="a directory holding the files")
    parser.add_argument("--from-release", dest="release",
                        help="a GitHub release tag, fetched with gh (works while private)")
    parser.add_argument("--repo", help="owner/name, when gh cannot infer it from the remote")
    parser.add_argument("--base-url", help="a URL each manifest path is appended to")
    parser.add_argument("--verify-only", action="store_true", help="check, fetch nothing")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()

    import os

    base_url = args.base_url or getenv("PSJEPA_WEIGHTS_URL")
    rows = read_manifest(args.manifest)
    root = weights_root()
    print(f"weights root: {root}")

    missing: list[str] = []
    for row in rows:
        destination = root / row["path"]
        expected_bytes = int(row["bytes"])
        problem = verify(destination, row["sha256"], expected_bytes)
        if problem is None:
            print(f"  ok       {row['path']}")
            continue
        print(f"  {problem:<40} {row['path']}")
        if args.verify_only:
            missing.append(row["path"])
            continue
        if problem != "absent":
            print("    removing the file that does not match")
            destination.unlink()
        if not fetch(row, destination, args.source_dir, base_url, args.release, args.repo):
            missing.append(row["path"])
            continue
        problem = verify(destination, row["sha256"], expected_bytes)
        if problem is not None:
            print(f"    what arrived does not match the manifest: {problem}", file=sys.stderr)
            destination.unlink(missing_ok=True)
            missing.append(row["path"])
        else:
            print("    verified")

    if not missing:
        print(f"\nall {len(rows)} released files are present and verified")
        return 0

    print(f"\n{len(missing)} file(s) still missing:", file=sys.stderr)
    for path in missing:
        print(f"  {path}", file=sys.stderr)
    if not (args.source_dir or base_url or args.release):
        print(
            "\nNo source was given. One of:\n"
            "  --from DIR          a directory holding the files\n"
            "  --from-release TAG  a GitHub release, via gh (this is the route while the\n"
            "                      repository is private)\n"
            "  --base-url URL      or PSJEPA_WEIGHTS_URL, once they are somewhere public\n"
            "If you already have them elsewhere, point PSJEPA_WEIGHTS at that directory\n"
            "instead and run with --verify-only. See README.md.",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
