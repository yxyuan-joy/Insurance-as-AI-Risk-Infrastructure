#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from validate_data import REQUIRED_FILES, validate_data


DEFAULT_REPO_ID = "yxyuan-joy/Insurance-as-AI-Risk-Infrastructure-Data"


def download_file(repo_id: str, revision: str, relative_path: str, destination: Path, token: str) -> None:
    encoded_revision = urllib.parse.quote(revision, safe="")
    encoded_path = "/".join(urllib.parse.quote(part, safe="") for part in relative_path.split("/"))
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/{encoded_revision}/{encoded_path}?download=true"
    headers = {"User-Agent": "insurance-ai-risk-infrastructure-data/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        temporary.replace(destination)
    except urllib.error.HTTPError as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download {relative_path}: HTTP {exc.code} from {url}") from exc
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the companion Hugging Face input dataset.")
    parser.add_argument("--repo-id", default=os.environ.get("HF_DATASET_REPO", DEFAULT_REPO_ID))
    parser.add_argument("--revision", default=os.environ.get("HF_DATASET_REVISION", "main"))
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = args.output_root.resolve()
    token = os.environ.get("HF_TOKEN", "")
    for relative_path in REQUIRED_FILES:
        destination = root / relative_path
        if destination.is_file() and not args.force:
            print(f"Keeping existing {relative_path}")
            continue
        print(f"Downloading {relative_path}")
        download_file(args.repo_id, args.revision, relative_path, destination, token)

    summary = validate_data(root)
    print(
        "Download complete: "
        f"{summary['files']} files, {summary['firms']} firms, "
        f"{summary['firm_update_rows']:,} firm-update rows."
    )


if __name__ == "__main__":
    main()
