"""Push the serving stack to a Hugging Face Space (Docker SDK).

Usage:
    uv run python scripts/deploy_hf_space.py <username>/allium-cepa [--private]

The script creates the Space if it does not exist, then uploads:
  - Dockerfile.space  →  Dockerfile  (HF builds this)
  - space/README.md   →  README.md   (Space card with YAML front matter)
  - entrypoint.sh
  - pyproject.toml + uv.lock
  - src/              (allium_cepa_classifier + ui)
  - app/              (FastAPI app)

Required Space secrets (set manually in HF Space settings):
  ALLIUM_STORAGE, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, ALLIUM_BUCKET
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, create_repo

ROOT = Path(__file__).resolve().parent.parent


def _dir_ops(src_dir: Path, dest_prefix: str) -> list[CommitOperationAdd]:
    ops = []
    for f in sorted(src_dir.rglob("*")):
        if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc":
            ops.append(
                CommitOperationAdd(
                    path_in_repo=f"{dest_prefix}/{f.relative_to(src_dir)}",
                    path_or_fileobj=str(f),
                )
            )
    return ops


def deploy(space_id: str, *, private: bool = False) -> None:
    api = HfApi()

    print(f"Creating Space {space_id} (if not exists)…")
    create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk="docker",
        private=private,
        exist_ok=True,
    )

    ops: list[CommitOperationAdd] = [
        CommitOperationAdd("Dockerfile", (ROOT / "Dockerfile.space").read_bytes()),
        CommitOperationAdd("README.md", (ROOT / "space/README.md").read_bytes()),
        CommitOperationAdd("entrypoint.sh", (ROOT / "entrypoint.sh").read_bytes()),
        CommitOperationAdd("pyproject.toml", (ROOT / "pyproject.toml").read_bytes()),
        CommitOperationAdd("uv.lock", (ROOT / "uv.lock").read_bytes()),
    ]
    ops += _dir_ops(ROOT / "src", "src")
    ops += _dir_ops(ROOT / "app", "app")

    print(f"Uploading {len(ops)} files…")
    api.create_commit(
        repo_id=space_id,
        repo_type="space",
        operations=ops,
        commit_message="chore: deploy allium-cepa serving stack",
    )

    url = f"https://huggingface.co/spaces/{space_id}"
    print(f"\nDeployed → {url}")
    print(
        "Remember to set Space secrets: ALLIUM_STORAGE, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, ALLIUM_BUCKET"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("space_id", help="HF Space ID, e.g. Nictauro98/allium-cepa")
    parser.add_argument("--private", action="store_true", help="Create a private Space")
    args = parser.parse_args()

    deploy(args.space_id, private=args.private)
