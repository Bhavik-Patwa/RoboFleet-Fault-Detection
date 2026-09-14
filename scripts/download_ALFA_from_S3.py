import argparse
import json
import os
import sys
from pathlib import Path, PurePosixPath

import base64
import hashlib
from uuid import uuid4

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "alfa"
MANIFEST_FILE = PROJECT_ROOT / "dataset_manifest.json"
ACQUISITION_MANIFEST_FILE = (
    PROJECT_ROOT / "dataset_acquisition_manifest.json"
)

SUPPORTED_SECTIONS = ["processed", "raw", "telemetry", "dataflash"]

REMOTE_IDENTITY_FIELDS = (
    "object_key",
    "size_bytes",
    "etag",
    "last_modified_utc"
)


# Checking whether two records identify the same immutable S3 object
def remote_identities_match(previous_record: dict, remote_identity: dict) -> bool:
    return all(
        previous_record.get(field_name)
        == remote_identity.get(field_name)
        for field_name in REMOTE_IDENTITY_FIELDS
    )


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
        raise ValueError(f"Missing AWS environment variables : {', '.join(missing_env_vars)}")

    # Creating the S3 client from the local AWS configuration
    return boto3.client("s3")


def ensure_local_root():
    # Creating the local dataset root when needed
    DATA_ROOT.mkdir(parents = True, exist_ok = True)


# Calculating a stable content digest for one local object
def calculate_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


# Calculating the deterministic revision of acquired object records
def calculate_acquisition_revision(object_index: dict, section_name: str | None = None) -> str:
    object_records = [
        object_index[object_key]
        for object_key in sorted(object_index)
        if (
            section_name is None
            or object_index[object_key].get("section")
            == section_name
        )
    ]

    revision_payload = json.dumps(
        object_records,
        sort_keys = True,
        separators = (",", ":")
    ).encode("utf-8")

    return hashlib.sha256(
        revision_payload
    ).hexdigest()


# Loading previously verified local and remote object identities
def load_acquisition_manifest(dataset_manifest: dict) -> dict:
    empty_manifest = {
        "manifest_schema_version": 1,
        "dataset_name": dataset_manifest["dataset_name"],
        "bucket_name": dataset_manifest["bucket_name"],
        "bucket_prefix": dataset_manifest["bucket_prefix"],
        "objects": {}
    }

    if not ACQUISITION_MANIFEST_FILE.exists():
        return empty_manifest

    acquisition_manifest = json.loads(
        ACQUISITION_MANIFEST_FILE.read_text()
    )

    expected_identity = {
        "dataset_name": dataset_manifest["dataset_name"],
        "bucket_name": dataset_manifest["bucket_name"],
        "bucket_prefix": dataset_manifest["bucket_prefix"]
    }

    actual_identity = {
        key: acquisition_manifest.get(key)
        for key in expected_identity
    }

    if actual_identity != expected_identity:
        raise RuntimeError("Dataset acquisition manifest identifies a different dataset.")

    if not isinstance(
        acquisition_manifest.get("objects"),
        dict
    ):
        raise RuntimeError("Dataset acquisition manifest has an invalid object index.")

    if (
        acquisition_manifest.get(
            "manifest_schema_version"
        )
        != 1
    ):
        raise RuntimeError("Dataset acquisition manifest has an unsupported schema version.")

    expected_revision = (
        calculate_acquisition_revision(
            acquisition_manifest["objects"]
        )
    )

    if (
        acquisition_manifest.get(
            "dataset_revision_sha256"
        )
        != expected_revision
    ):
        raise RuntimeError("Dataset acquisition manifest revision is invalid.")

    return acquisition_manifest


# Writing acquisition provenance only after successful validation
def save_acquisition_manifest(acquisition_manifest: dict) -> None:
    section_names = sorted(
        {
            record["section"]
            for record
            in acquisition_manifest["objects"].values()
        }
    )

    acquisition_manifest[
        "section_revision_sha256"
    ] = {
        section_name: calculate_acquisition_revision(
            acquisition_manifest["objects"],
            section_name = section_name
        )
        for section_name in section_names
    }

    acquisition_manifest[
        "dataset_revision_sha256"
    ] = calculate_acquisition_revision(
        acquisition_manifest["objects"]
    )

    temporary_path = (
        ACQUISITION_MANIFEST_FILE.parent
        / (
            f".{ACQUISITION_MANIFEST_FILE.name}."
            f"{uuid4().hex}.tmp"
        )
    )

    ACQUISITION_MANIFEST_FILE.parent.mkdir(
        parents = True,
        exist_ok = True
    )

    try:
        temporary_path.write_text(
            json.dumps(
                acquisition_manifest,
                indent = 2,
                sort_keys = True
            ) + "\n"
        )

        temporary_path.replace(
            ACQUISITION_MANIFEST_FILE
        )

    finally:
        temporary_path.unlink(
            missing_ok = True
        )


# Listing matching S3 objects page by page
def iter_s3_objects(s3_client, bucket_name: str, prefix: str):
    paginator = s3_client.get_paginator(
        "list_objects_v2"
    )

    found_any_object = False

    for page in paginator.paginate(
        Bucket = bucket_name,
        Prefix = prefix
    ):
        for item in page.get("Contents", []):
            object_key = item["Key"]

            if object_key.endswith("/"):
                continue

            found_any_object = True
            yield item

    if not found_any_object:
        raise RuntimeError(f"No files found in S3 for prefix : {prefix}")


# Building object identity from the authorized S3 listing response
def build_listed_object_identity(object_item: dict) -> dict:
    return {
        "object_key": object_item["Key"],
        "size_bytes": int(
            object_item["Size"]
        ),
        "etag": object_item["ETag"].strip('"'),
        "last_modified_utc": (
            object_item["LastModified"].isoformat()
        ),
        "version_id": None,
        "checksum_type": object_item.get(
            "ChecksumType"
        ),
        "checksum_sha256": None
    }


# Confirming that a previously acquired local object remains valid
def can_reuse_local_object(destination: Path, previous_record: dict | None, remote_identity: dict) -> bool:
    if previous_record is None:
        return False

    if not remote_identities_match(
        previous_record = previous_record,
        remote_identity = remote_identity
    ):
        return False

    if (
        not destination.is_file()
        or destination.stat().st_size
        != remote_identity["size_bytes"]
    ):
        return False

    return (
        calculate_file_sha256(destination)
        == previous_record.get("local_sha256")
    )


# Downloading one object without exposing an incomplete destination
def download_verified_object(s3_client, bucket_name: str,
                             remote_identity: dict, destination: Path,
                             expected_existing_sha256: str | None = None
) -> str:
    temporary_path = (
        destination.parent
        / f".{destination.name}.{uuid4().hex}.part"
    )

    request = {
        "Bucket": bucket_name,
        "Key": remote_identity["object_key"]
    }

    if remote_identity.get("version_id"):
        request["VersionId"] = (
            remote_identity["version_id"]
        )

    try:
        try:
            response = s3_client.get_object(
                **request,
                ChecksumMode = "ENABLED"
            )

        except ClientError as exc:
            status_code = exc.response.get(
                "ResponseMetadata",
                {}
            ).get(
                "HTTPStatusCode"
            )

            if status_code != 403:
                raise

            response = s3_client.get_object(
                **request
            )

        response_identity = {
            "object_key": remote_identity[
                "object_key"
            ],
            "size_bytes": int(
                response["ContentLength"]
            ),
            "etag": response["ETag"].strip('"'),
            "last_modified_utc": (
                response["LastModified"].isoformat()
            )
        }

        if not remote_identities_match(
            previous_record = remote_identity,
            remote_identity = response_identity
        ):
            raise RuntimeError(
                "S3 object changed between listing and download : "
                f"{remote_identity['object_key']}"
            )

        remote_identity["version_id"] = response.get(
            "VersionId"
        )

        remote_identity["checksum_type"] = response.get(
            "ChecksumType"
        )

        remote_identity["checksum_sha256"] = response.get(
            "ChecksumSHA256"
        )

        response_body = response["Body"]

        try:
            with temporary_path.open("wb") as handle:
                while True:
                    chunk = response_body.read(
                        1024 * 1024
                    )

                    if not chunk:
                        break

                    handle.write(chunk)

        finally:
            response_body.close()

        actual_size = temporary_path.stat().st_size

        if actual_size != remote_identity["size_bytes"]:
            raise RuntimeError(
                "Downloaded object size does not match S3 metadata : "
                f"{remote_identity['object_key']}"
            )

        local_sha256 = calculate_file_sha256(
            temporary_path
        )

        if (
            expected_existing_sha256 is not None
            and local_sha256
            != expected_existing_sha256
        ):
            raise RuntimeError(
                "Existing local data differs from the current S3 object : "
                f"{remote_identity['object_key']}"
            )

        if (
            remote_identity["checksum_type"]
            == "FULL_OBJECT"
            and remote_identity["checksum_sha256"]
        ):
            encoded_sha256 = base64.b64encode(
                bytes.fromhex(local_sha256)
            ).decode("ascii")

            if (
                encoded_sha256
                != remote_identity["checksum_sha256"]
            ):
                raise RuntimeError(
                    "Downloaded object SHA-256 does not match S3 metadata : "
                    f"{remote_identity['object_key']}"
                )

        temporary_path.replace(
            destination
        )

        return local_sha256

    finally:
        temporary_path.unlink(
            missing_ok = True
        )


# Downloading and verifying one complete dataset section
def download_section(s3_client, bucket_name: str, bucket_prefix: str,
                     section_name: str, section_prefix: str, acquisition_manifest: dict
) -> None:
    downloaded_files = 0
    skipped_files = 0
    observed_object_keys = set()

    normalized_section_prefix = (
        f"{section_prefix.rstrip('/')}/"
    )

    object_items = list(
        iter_s3_objects(
            s3_client = s3_client,
            bucket_name = bucket_name,
            prefix = normalized_section_prefix
        )
    )

    observed_object_keys = {
        item["Key"]
        for item in object_items
    }

    previously_recorded_keys = {
        object_key
        for object_key, record
        in acquisition_manifest["objects"].items()
        if record.get("section") == section_name
    }

    if (
        previously_recorded_keys
        and observed_object_keys
        != previously_recorded_keys
    ):
        raise RuntimeError(f"The S3 object inventory changed for previously acquired section '{section_name}'.")

    for object_item in object_items:
        object_key = object_item["Key"]
        observed_object_keys.add(object_key)

        relative_key = PurePosixPath(
            object_key
        ).relative_to(bucket_prefix)

        destination = (
            DATA_ROOT
            / Path(*relative_key.parts)
        )

        destination = destination.resolve()

        if not destination.is_relative_to(
            DATA_ROOT.resolve()
        ):
            raise RuntimeError(f"S3 object resolves outside the dataset directory : {object_key}")

        destination.parent.mkdir(
            parents = True,
            exist_ok = True
        )

        remote_identity = (
            build_listed_object_identity(
                object_item
            )
        )

        previous_record = (
            acquisition_manifest["objects"].get(
                object_key
            )
        )

        # Reusing a recorded version identifier when repairing a trusted object
        if (
            previous_record is not None
            and previous_record.get("version_id")
        ):
            remote_identity["version_id"] = (
                previous_record["version_id"]
            )

        if (
            previous_record is not None
            and not remote_identities_match(
                previous_record = previous_record,
                remote_identity = remote_identity
            )
        ):
            raise RuntimeError(f"A previously acquired S3 object changed identity : {object_key}")

        if can_reuse_local_object(
            destination = destination,
            previous_record = previous_record,
            remote_identity = remote_identity
        ):
            skipped_files += 1
            continue

        expected_existing_sha256 = None

        # Requiring repaired content to match the previously trusted digest
        if previous_record is not None:
            expected_existing_sha256 = (
                previous_record.get(
                    "local_sha256"
                )
            )

            if (
                not isinstance(
                    expected_existing_sha256,
                    str
                )
                or len(expected_existing_sha256) != 64
            ):
                raise RuntimeError(f"Previously acquired object has no valid SHA-256 : {object_key}")

        # Establishing provenance without silently changing pre-existing data
        elif destination.exists():
            if not destination.is_file():
                raise RuntimeError(f"Dataset destination is not a regular file : {destination}")

            expected_existing_sha256 = (
                calculate_file_sha256(
                    destination
                )
            )

        local_sha256 = download_verified_object(
            s3_client = s3_client,
            bucket_name = bucket_name,
            remote_identity = remote_identity,
            destination = destination,
            expected_existing_sha256 = (
                expected_existing_sha256
            )
        )

        acquisition_manifest["objects"][
            object_key
        ] = {
            **remote_identity,
            "section": section_name,
            "relative_path": str(
                destination.relative_to(
                    DATA_ROOT
                )
            ),
            "local_sha256": local_sha256
        }

        downloaded_files += 1

    print(
        f"Downloaded section '{section_name}' with "
        f"{downloaded_files} verified downloads and "
        f"{skipped_files} verified existing files."
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

    acquisition_manifest = (
        load_acquisition_manifest(
            dataset_manifest = manifest
        )
    )


    # Downloading and validating only the requested dataset sections
    for section_name in args.sections:
        section_config = manifest["sections"][section_name]

        download_section(
            s3_client = s3_client,
            bucket_name = bucket_name,
            bucket_prefix = bucket_prefix,
            section_name = section_name,
            section_prefix = section_config[
                "s3_prefix"
            ],
            acquisition_manifest = (
                acquisition_manifest
            )
        )

        validate_section(section_name, section_config["expected"])

        save_acquisition_manifest(acquisition_manifest)

    print("ALFA dataset ingestion from AWS S3 completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except (BotoCoreError, ClientError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ALFA dataset ingestion from AWS S3 failed : {exc}")
        sys.exit(1)