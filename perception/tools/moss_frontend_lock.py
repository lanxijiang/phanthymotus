"""Export and verify the exact conda-forge ARM frontend artifact set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from urllib.parse import urlsplit


FORBIDDEN = {"torch", "pytorch", "libtorch", "numpy", "tensorrt"}
PINNED = {"pynini": ("2.1.7", "py310h0992a49_2"), "openfst": ("1.8.4", None)}


def artifact(url: str, digest: str) -> tuple[str, str, str]:
    parsed = urlsplit(url)
    parts = parsed.path.split("/")
    if (parsed.scheme != "https" or parsed.netloc != "conda.anaconda.org"
            or parsed.query or parsed.fragment or len(parts) != 4
            or parts[1] != "conda-forge" or parts[2] not in {"linux-aarch64", "noarch"}):
        raise ValueError(f"Unexpected frontend artifact URL: {url}")
    if not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError(f"Missing SHA256 for frontend artifact: {url}")
    filename = parts[3]
    for suffix in (".tar.bz2", ".conda"):
        if filename.endswith(suffix):
            fields = filename[:-len(suffix)].rsplit("-", 2)
            break
    else:
        raise ValueError(f"Not a conda package: {filename}")
    if len(fields) != 3 or not all(fields):
        raise ValueError(f"Malformed frontend artifact filename: {filename}")
    name, version, build = fields
    if name in FORBIDDEN or name.startswith(("cuda-", "cudnn", "pytorch-", "tensorrt-")):
        raise ValueError(f"Frontend must not install the inference stack: {name}")
    return name, version, build


def validate_set(records: dict[str, tuple[str, str]]) -> None:
    for name, (version, build) in PINNED.items():
        if name not in records:
            raise ValueError(f"Missing required frontend package: {name}")
        actual = artifact(*records[name])
        if actual[1] != version or (build is not None and actual[2] != build):
            raise ValueError(f"Unvalidated frontend artifact: {actual}")
    if "python" not in records or not artifact(*records["python"])[1].startswith("3.10."):
        raise ValueError("Frontend requires the CPython 3.10 package set")
    if "importlib_resources" not in records:
        raise ValueError("Missing importlib_resources frontend dependency")


def read_lock(path: Path) -> dict[str, tuple[str, str]]:
    lines = [line.strip() for line in path.read_text().splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if not lines or lines.pop(0) != "@EXPLICIT":
        raise ValueError("Frontend lock must begin with @EXPLICIT")
    records = {}
    for line in lines:
        url, marker, digest = line.partition("#sha256:")
        if not marker:
            raise ValueError("Every frontend artifact requires #sha256:<64 hex digits>")
        name, _, _ = artifact(url, digest)
        if name in records:
            raise ValueError(f"Duplicate frontend package: {name}")
        records[name] = (url, digest)
    validate_set(records)
    return records


def installed_records(metadata: Path) -> dict[str, tuple[str, str]]:
    records = {}
    for path in sorted(metadata.glob("*.json")):
        record = json.loads(path.read_text())
        url, digest = record.get("url", ""), record.get("sha256", "")
        name, version, build = artifact(url, digest)
        if (record.get("name"), record.get("version"), record.get("build")) != (name, version, build):
            raise ValueError(f"Frontend metadata disagrees with its artifact: {path}")
        if name in records:
            raise ValueError(f"Duplicate installed frontend package: {name}")
        records[name] = (url, digest)
    validate_set(records)
    return records


def export_lock(metadata: Path, output: Path) -> None:
    records = installed_records(metadata)
    content = "# Verified Orin frontend; platform: linux-aarch64\n@EXPLICIT\n"
    content += "".join(f"{url}#sha256:{digest}\n" for _, (url, digest) in sorted(records.items()))
    staged = output.with_name(output.name + ".part")
    staged.write_text(content)
    read_lock(staged)
    staged.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "check", "export"))
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "export":
        if args.metadata is None or args.output is None:
            parser.error("export requires --metadata and --output")
        export_lock(args.metadata, args.output)
    else:
        if args.lock is None:
            parser.error("validate/check requires --lock")
        expected = read_lock(args.lock)
        if args.action == "check":
            if args.metadata is None:
                parser.error("check requires --metadata")
            actual = installed_records(args.metadata)
            if actual != expected:
                changed = sorted(name for name in actual.keys() | expected.keys()
                                 if actual.get(name) != expected.get(name))
                raise ValueError(f"Installed frontend does not match the lock: {changed}")
    print("MOSS frontend artifact lock verified")


if __name__ == "__main__":
    main()
