import argparse
import json
import os
import sys
from pathlib import Path, PurePosixPath

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "alfa"
MANIFEST_FILE = PROJECT_ROOT / "dataset_manifest.json"

SUPPORTED_SECTIONS = ["processed", "raw", "telemetry", "dataflash"]


def load_manifest():
    # Loading the tracked dataset manifest
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(f"Manifest file not found : {MANIFEST_FILE}")

    manifest = json.loads(MANIFEST_FILE.read_text())

    # Checking that the required manifest fields are present
    required_keys = ["dataset_name", "storage_backend", "bucket_name", "bucket_prefix", "sections"]
    for key in required_keys:
        if key not in manifest:
            raise ValueError(f"Missing manifest field : {key}")

    return manifest


def create_s3_client():
    # Loading local AWS environment variables
    load_dotenv()

    # Checking that the required AWS credentials are available locally.
    required_env_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"]
    missing_env_vars = [name for name in required_env_vars if not os.getenv(name)]

    if missing_env_vars:
        raise ValueError(
            f"Missing AWS environment variables : {', '.join(missing_env_vars)}"
        )

    # Creating the S3 client from the local AWS configuration
    return boto3.client("s3")


def ensure_local_root():
    # Creating the local dataset root when needed
    DATA_ROOT.mkdir(parents = True, exist_ok = True)


def iter_object_keys(s3_client, bucket_name, prefix):
    paginator = s3_client.get_paginator("list_objects_v2")
    found_any_object = False

    # Listing matching S3 objects page by page
    for page in paginator.paginate(Bucket = bucket_name, Prefix = prefix):
        for item in page.get("Contents", []):
            object_key = item["Key"]

            if object_key.endswith("/"):
                continue

            found_any_object = True
            yield object_key

    # Failing when the expected S3 prefix does not contain files
    if not found_any_object:
        raise RuntimeError(f"No files found in S3 for prefix : {prefix}")


def download_section(s3_client, bucket_name, bucket_prefix, section_name, section_prefix):
    downloaded_files = 0
    skipped_files = 0

    # Downloading one dataset section into the standard local path
    for object_key in iter_object_keys(s3_client, bucket_name, section_prefix):
        relative_key = PurePosixPath(object_key).relative_to(bucket_prefix)
        destination = DATA_ROOT / Path(*relative_key.parts)

        destination.parent.mkdir(parents = True, exist_ok = True)

        # Reusing the local file when it already exists
        if destination.exists():
            skipped_files += 1
            continue

        s3_client.download_file(bucket_name, object_key, str(destination))
        downloaded_files += 1

    print(
        f"Downloaded section '{section_name}' with {downloaded_files} new files "
        f"and skipped {skipped_files} existing files."
    )


def count_files(directory, pattern):
    # Counting matching files recursively
    return len(list(directory.rglob(pattern)))


def count_directories(directory):
    # Counting direct child directories
    return len([path for path in directory.iterdir() if path.is_dir()])


def validate_section(section_name, expected):
    section_directory = DATA_ROOT / section_name

    # Checking that the downloaded section exists locally
    if not section_directory.exists():
        raise FileNotFoundError(f"Local section not found: {section_directory}")

    if section_name == "processed":
        actual_directories = count_directories(section_directory)
        actual_csv_files = count_files(section_directory, "*.csv")
        actual_mat_files = count_files(section_directory, "*.mat")
        actual_bag_files = count_files(section_directory, "*.bag")

        if actual_directories != expected["directories"]:
            raise RuntimeError(
                f"Unexpected processed directory count : expected {expected['directories']}, found {actual_directories}"
            )

        if actual_csv_files != expected["csv_files"]:
            raise RuntimeError(
                f"Unexpected processed csv count : expected {expected['csv_files']}, found {actual_csv_files}"
            )

        if actual_mat_files != expected["mat_files"]:
            raise RuntimeError(
                f"Unexpected processed mat count : expected {expected['mat_files']}, found {actual_mat_files}"
            )

        if actual_bag_files != expected["bag_files"]:
            raise RuntimeError(
                f"Unexpected processed bag count : expected {expected['bag_files']}, found {actual_bag_files}"
            )

    elif section_name == "raw":
        actual_bag_files = count_files(section_directory, "*.bag")

        if actual_bag_files != expected["bag_files"]:
            raise RuntimeError(
                f"Unexpected raw bag count : expected {expected['bag_files']}, found {actual_bag_files}"
            )

    elif section_name == "telemetry":
        actual_tlog_files = count_files(section_directory, "*.tlog")
        actual_parm_files = count_files(section_directory, "*.parm")

        if actual_tlog_files != expected["tlog_files"]:
            raise RuntimeError(
                f"Unexpected telemetry tlog count : expected {expected['tlog_files']}, found {actual_tlog_files}"
            )

        if actual_parm_files != expected["parm_files"]:
            raise RuntimeError(
                f"Unexpected telemetry parm count : expected {expected['parm_files']}, found {actual_parm_files}"
            )

    elif section_name == "dataflash":
        actual_bin_files = count_files(section_directory, "*.bin")

        if actual_bin_files != expected["bin_files"]:
            raise RuntimeError(
                f"Unexpected dataflash bin count : expected {expected['bin_files']}, found {actual_bin_files}"
            )

    print(f"Validated section '{section_name}' successfully.")


def parse_args():
    parser = argparse.ArgumentParser()

    # Defaulting to the section that the ML pipeline will use first
    parser.add_argument(
        "--sections",
        nargs = "+",
        default = ["processed"],
        choices = SUPPORTED_SECTIONS
    )

    return parser.parse_args()


def main():
    args = parse_args()
    manifest = load_manifest()

    if manifest["storage_backend"] != "s3":
        raise ValueError("Only 's3' storage_backend is supported in this script.")

    s3_client = create_s3_client()

    bucket_name = manifest["bucket_name"]
    bucket_prefix = manifest["bucket_prefix"]

    ensure_local_root()

    # Downloading and validating only the requested dataset sections
    for section_name in args.sections:
        section_config = manifest["sections"][section_name]

        download_section(
            s3_client = s3_client,
            bucket_name = bucket_name,
            bucket_prefix = bucket_prefix,
            section_name = section_name,
            section_prefix = section_config["s3_prefix"]
        )

        validate_section(section_name, section_config["expected"])

    print("ALFA dataset ingestion from AWS S3 completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except (BotoCoreError, ClientError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ALFA dataset ingestion from AWS S3 failed : {exc}")
        sys.exit(1)