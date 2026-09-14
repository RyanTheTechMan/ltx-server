"""Explicitly download or verify the exact LTX 2.5 pack; never runs on server startup."""

import argparse
import hashlib
import json
from pathlib import Path

from ltx_server.config import Settings
from ltx_server.inference.models import CHECKPOINTS, MODEL_REPO, MODEL_REVISION, ModelInventory


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="Print pinned files; no network/writes")
    parser.add_argument(
        "--verify-only", action="store_true", help="Hash existing files; no network"
    )
    args = parser.parse_args()
    settings = Settings()
    inventory = ModelInventory(settings)
    if args.list:
        print(
            json.dumps(
                {
                    "repo": MODEL_REPO,
                    "revision": MODEL_REVISION,
                    "size_gib": sum(item.size for item in CHECKPOINTS) / 1024**3,
                    "files": [item.filename for item in CHECKPOINTS],
                },
                indent=2,
            )
        )
        return
    if inventory.overrides and not args.verify_only:
        raise SystemExit("Download uses the standard layout; unset overrides or use --verify-only")
    assert settings.model_dir is not None
    destination = settings.model_dir / "ltx-2.5"
    for checkpoint in CHECKPOINTS:
        path = inventory.paths[checkpoint.component]
        valid = (
            path.is_file()
            and path.stat().st_size == checkpoint.size
            and sha256(path) == checkpoint.sha256
        )
        if not valid and not args.verify_only:
            from huggingface_hub import hf_hub_download

            print(f"Downloading {checkpoint.component} from pinned revision {MODEL_REVISION}")
            path = Path(
                hf_hub_download(
                    repo_id=MODEL_REPO,
                    revision=MODEL_REVISION,
                    filename=checkpoint.filename,
                    local_dir=destination,
                    token=settings.hf_token.get_secret_value() or None,
                    force_download=path.exists(),
                )
            )
            valid = path.stat().st_size == checkpoint.size and sha256(path) == checkpoint.sha256
        if not valid:
            raise SystemExit(f"Missing or failed SHA-256 verification: {checkpoint.component}")
        print(f"Verified {checkpoint.component}")
    inventory.validate()
    print("All five model components match the pinned sizes and SHA-256 hashes.")


if __name__ == "__main__":
    main()
