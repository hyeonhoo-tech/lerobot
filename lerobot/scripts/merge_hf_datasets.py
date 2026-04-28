#!/usr/bin/env python3
"""Merge multiple HuggingFace dataset repos into one and preserve extra files.

Usage examples:
  HF_TOKEN=xxx python merge_hf_datasets.py \
    --repo-base ykorkmaz/play_robot_demo_ \
    --start 0 --end 20 \
    --output-repo ykorkmaz/play_robot_demo_merged

Or pass comma-separated repos:
  python merge_hf_datasets.py --repos ykorkmaz/play_robot_demo_0,ykorkmaz/play_robot_demo_1 \
    --output-repo ykorkmaz/play_robot_demo_merged --token <HF_TOKEN>
"""

import argparse
import os
import tempfile
import shutil
import time
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download, upload_folder
from datasets import load_dataset, DatasetDict, concatenate_datasets


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-base", type=str, help="Base repo id prefix (e.g. user/play_robot_demo_)")
    p.add_argument("--start", type=int, default=0, help="start index (inclusive)")
    p.add_argument("--end", type=int, default=0, help="end index (inclusive)")
    p.add_argument("--repos", type=str, help="Comma-separated explicit repo ids to merge")
    p.add_argument("--output-repo", required=True, help="Destination dataset repo id (user/repo)")
    p.add_argument("--token", type=str, default=None, help="Hugging Face token or set HF_TOKEN env var")
    p.add_argument("--private", action="store_true", help="Create private repo when pushing")
    p.add_argument("--dry-run", action="store_true", help="Only load and report; do not push or upload files")
    p.add_argument("--skip-large-mb", type=int, default=100, help="Skip files larger than this MB when uploading sources (0=disabled)")
    p.add_argument("--max-retries", type=int, default=5, help="Max retries for upload operations on 429")
    return p.parse_args()


def main():
    args = parse_args()
    token = args.token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    # Build list of source repo ids
    repos = []
    if args.repos:
        repos = [r.strip() for r in args.repos.split(",") if r.strip()]
    elif args.repo_base is not None:
        repos = [f"{args.repo_base}{i}" for i in range(args.start, args.end + 1)]
    else:
        raise SystemExit("Either --repos or --repo-base must be provided")

    print(f"Will merge {len(repos)} repos -> {args.output_repo}")

    # Load and collect splits
    merged = {}
    succeeded = []
    failed = []
    first_features = None
    for repo in repos:
        print(f"Loading {repo} ...")
        try:
            ds = load_dataset(repo)
        except Exception as e:
            err_str = str(e)
            # If we have features from a previous successful repo, try loading parquet files directly
            if first_features is not None:
                print(f"  Initial load failed for {repo}, attempting parquet fallback with override features...")
                try:
                    files = api.list_repo_files(repo_id=repo, repo_type="dataset")
                except Exception as e_list:
                    print(f"    Could not list files for fallback: {e_list}")
                    failed.append((repo, err_str))
                    continue

                parquet_files = [f for f in files if f.startswith("data/") and f.endswith(".parquet")]
                local_paths = []
                for pf in parquet_files:
                    try:
                        local = hf_hub_download(repo_id=repo, filename=pf, repo_type="dataset", token=token)
                        local_paths.append(local)
                    except Exception as e_dl:
                        print(f"    Failed to download parquet {pf}: {e_dl}")

                if not local_paths:
                    print(f"    No parquet files downloaded for {repo}; cannot fallback")
                    failed.append((repo, err_str))
                    continue

                try:
                    ds = load_dataset("parquet", data_files=local_paths, features=first_features)
                    print(f"    Loaded {repo} from parquet with override features")
                except Exception as e_parq:
                    print(f"    Failed to load {repo} from parquet with override: {e_parq}")
                    failed.append((repo, str(e_parq)))
                    continue
            else:
                print(f"  Failed to load {repo}: {e}")
                failed.append((repo, err_str))
                continue

        succeeded.append(repo)

        # Capture features from the first successfully loaded repo to use as an override
        if first_features is None:
            try:
                if isinstance(ds, DatasetDict):
                    # use features from the first split
                    first_split = next(iter(ds.values()))
                    first_features = first_split.features
                else:
                    first_features = ds.features
                print(f"  Captured features from {repo} to use as override for others.")
            except Exception:
                first_features = None

        if isinstance(ds, DatasetDict):
            for split_name, split_ds in ds.items():
                merged.setdefault(split_name, []).append(split_ds)
        else:
            merged.setdefault("train", []).append(ds)

    # Concatenate
    final = {}
    for split_name, list_ds in merged.items():
        if len(list_ds) == 1:
            final[split_name] = list_ds[0]
        else:
            final[split_name] = concatenate_datasets(list_ds)
        print(f"Merged split {split_name}: {len(final[split_name])} examples")

    final_dd = DatasetDict(final)

    print("\nLoad summary:")
    print(f"  succeeded: {len(succeeded)} repos")
    for s in succeeded:
        print(f"    - {s}")
    if failed:
        print(f"  failed: {len(failed)} repos")
        for r, err in failed:
            print(f"    - {r}: {err}")

    if args.dry_run:
        print("Dry run enabled; skipping repo creation, push, and file uploads.")
    else:
        # Create destination repo (if not exists)
        try:
            api.create_repo(repo_id=args.output_repo, repo_type="dataset", private=args.private)
            print(f"Created repo {args.output_repo}")
        except Exception:
            print(f"Repo {args.output_repo} may already exist or could not be created; continuing")

        # Push merged dataset with simple retry/backoff on 429
        print(f"Pushing merged dataset to {args.output_repo} ...")
        push_retries = 0
        while True:
            try:
                final_dd.push_to_hub(args.output_repo, private=args.private, token=token)
                print("Merged dataset pushed.")
                break
            except Exception as e:
                msg = str(e)
                push_retries += 1
                if "429" in msg and push_retries <= args.max_retries:
                    wait = 60 * push_retries
                    print(f"Push hit rate limit, retrying after {wait}s ({push_retries}/{args.max_retries})")
                    time.sleep(wait)
                    continue
                raise

    # Preserve and upload extra files from each source repo into a 'sources/' folder in the new repo
    print("Collecting and uploading extra files from sources (folder uploads)...")
    for repo in repos:
        print(f"Processing source repo: {repo}")
        try:
            files = api.list_repo_files(repo_id=repo, repo_type="dataset")
        except Exception as e:
            print(f"  Could not list files for {repo}: {e}")
            continue

        # Prepare local temp folder for this repo's files
        tmpdir = Path(tempfile.mkdtemp(prefix="merge_src_"))
        large_files = []
        downloaded = 0
        for fname in files:
            if fname.endswith(".arrow") or fname.endswith(".parquet") or fname.startswith(".git"):
                continue
            local_target = tmpdir / fname
            local_target.parent.mkdir(parents=True, exist_ok=True)
            try:
                local = hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset", token=token)
            except Exception as e:
                print(f"    Failed download {repo}/{fname}: {e}")
                continue

            # move downloaded file into structure
            try:
                shutil.copy(local, str(local_target))
            except Exception as e:
                print(f"    Failed copying {local} -> {local_target}: {e}")
                continue

            downloaded += 1
            # check size and optionally skip large files
            if args.skip_large_mb > 0:
                try:
                    size_mb = local_target.stat().st_size / (1024 * 1024)
                    if size_mb > args.skip_large_mb:
                        large_files.append((str(local_target), size_mb))
                        local_target.unlink()
                except Exception:
                    pass

        if downloaded == 0 and not large_files:
            print(f"  No source files to upload for {repo}")
            shutil.rmtree(tmpdir)
            continue

        if large_files:
            print(f"  Skipped {len(large_files)} large files for {repo} (>{args.skip_large_mb} MB):")
            for p, sz in large_files:
                print(f"    - {p} ({sz:.1f} MB)")

        if args.dry_run:
            print(f"  Dry run: would upload folder {tmpdir} -> sources/{repo.replace('/', '_')}")
            shutil.rmtree(tmpdir)
            continue

        # upload folder with retries/backoff
        upload_retries = 0
        while True:
            try:
                upload_folder(
                    folder_path=str(tmpdir),
                    path_in_repo=f"sources/{repo.replace('/', '_')}",
                    repo_id=args.output_repo,
                    repo_type="dataset",
                    token=token,
                )
                print(f"  Uploaded folder for {repo}")
                break
            except Exception as e:
                upload_retries += 1
                msg = str(e)
                if "429" in msg and upload_retries <= args.max_retries:
                    wait = 60 * upload_retries
                    print(f"  Upload hit rate limit, retrying after {wait}s ({upload_retries}/{args.max_retries})")
                    time.sleep(wait)
                    continue
                print(f"  Failed folder upload for {repo}: {e}")
                break

        shutil.rmtree(tmpdir)

    print("Done. Verify the merged dataset and attached source files on the Hub.")


if __name__ == "__main__":
    main()
