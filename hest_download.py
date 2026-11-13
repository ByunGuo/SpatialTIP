"""Download explicitly selected HEST samples without embedding credentials."""

import argparse
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--sample-id", nargs="+", help="One or more HEST sample IDs, e.g. MISC1 TENX13")
    selection.add_argument("--all", action="store_true", help="Explicitly request the entire HEST repository; may be very large")
    parser.add_argument("--revision", default="main", help="Repository revision or commit for reproducible downloads")
    args = parser.parse_args()
    if args.sample_id and any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", sample) for sample in args.sample_id):
        parser.error("--sample-id accepts alphanumeric HEST identifiers only")
    from huggingface_hub import snapshot_download
    patterns = None if args.all else [f"*{sample}[_.]*" for sample in dict.fromkeys(args.sample_id)]
    location = snapshot_download(repo_id="MahmoodLab/hest", repo_type="dataset",
                                 revision=args.revision, local_dir=str(Path(args.local_dir).expanduser()),
                                 allow_patterns=patterns)
    print(f"Downloaded to {location}")
    print("Authentication, when required, uses HF_TOKEN or your existing Hugging Face login.")


if __name__ == "__main__":
    main()
