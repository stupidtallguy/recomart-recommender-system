"""Publish the sealed serving snapshot without loading any final-test labels."""
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.serving.database import publish_snapshot
from src.serving.store import ROOT, load_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ROOT)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if args.env_file:
        load_dotenv(args.env_file, override=False)
    dsn = os.environ.get("RECOMART_DATABASE_URL")
    if not dsn:
        parser.error("Set RECOMART_DATABASE_URL or supply --env-file")
    print(json.dumps(publish_snapshot(dsn, load_snapshot(args.artifact_root)), indent=2))


if __name__ == "__main__":
    main()
