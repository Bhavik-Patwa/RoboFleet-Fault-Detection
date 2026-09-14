import hashlib
import tempfile
import unittest
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path

from scripts import download_ALFA_from_S3 as ingestion


# Simulating an S3 response stream that fails after its first chunk
class InterruptedBody:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.read_count = 0

    def read(self, size: int = -1) -> bytes:
        self.read_count += 1

        if self.read_count == 1:
            return self.content[:2]

        raise RuntimeError("Simulated interrupted download.")

    def close(self) -> None:
        pass


# Providing deterministic S3 responses without making network requests
class FakeS3Client:
    def __init__(
        self,
        content: bytes,
        fail_download: bool = False
    ) -> None:
        self.content = content
        self.fail_download = fail_download

    def get_object(self, **request) -> dict:
        body = (
            InterruptedBody(self.content)
            if self.fail_download
            else BytesIO(self.content)
        )

        return {
            "Body": body,
            "ContentLength": len(self.content),
            "ETag": '"verified-etag"',
            "LastModified": datetime(
                2026,
                1,
                1,
                tzinfo = timezone.utc
            )
        }


# Calculating the expected SHA-256 value for in-memory test content
def calculate_bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


# Building the S3 identity expected by the ingestion functions
def build_remote_identity(content: bytes) -> dict:
    return {
        "object_key": (
            "alfa/processed/flight/telemetry.csv"
        ),
        "size_bytes": len(content),
        "etag": "verified-etag",
        "last_modified_utc": (
            "2026-01-01T00:00:00+00:00"
        ),
        "version_id": None,
        "checksum_type": None,
        "checksum_sha256": None
    }


# Verifying integrity checks and atomic download behavior
class DatasetIngestionTests(unittest.TestCase):
    # Rejecting a local file whose content differs from its trusted digest
    def test_corrupted_existing_file_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = (
                Path(directory) / "telemetry.csv"
            )

            destination.write_bytes(b"bad-data")

            remote_content = b"new-data"
            remote_identity = build_remote_identity(
                remote_content
            )

            previous_record = {
                **remote_identity,
                "local_sha256": (
                    calculate_bytes_sha256(
                        remote_content
                    )
                )
            }

            reusable = ingestion.can_reuse_local_object(
                destination = destination,
                previous_record = previous_record,
                remote_identity = remote_identity
            )

            self.assertFalse(reusable)

    # Ensuring an interrupted download cannot replace existing data
    def test_interrupted_download_preserves_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = (
                Path(directory) / "telemetry.csv"
            )

            existing_content = b"existing-data"
            remote_content = b"replacement-data"

            destination.write_bytes(
                existing_content
            )

            with self.assertRaises(RuntimeError):
                ingestion.download_verified_object(
                    s3_client = FakeS3Client(
                        content = remote_content,
                        fail_download = True
                    ),
                    bucket_name = "dataset-bucket",
                    remote_identity = (
                        build_remote_identity(
                            remote_content
                        )
                    ),
                    destination = destination
                )

            self.assertEqual(
                destination.read_bytes(),
                existing_content
            )

            self.assertEqual(
                list(destination.parent.glob("*.part")),
                []
            )

    # Preserving existing data when downloaded content has a different digest
    def test_different_remote_content_preserves_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = (
                Path(directory) / "telemetry.csv"
            )

            existing_content = b"existing-data"
            remote_content = b"replacement-data"

            destination.write_bytes(
                existing_content
            )

            with self.assertRaises(RuntimeError):
                ingestion.download_verified_object(
                    s3_client = FakeS3Client(
                        content = remote_content
                    ),
                    bucket_name = "dataset-bucket",
                    remote_identity = (
                        build_remote_identity(
                            remote_content
                        )
                    ),
                    destination = destination,
                    expected_existing_sha256 = (
                        calculate_bytes_sha256(
                            existing_content
                        )
                    )
                )

            self.assertEqual(
                destination.read_bytes(),
                existing_content
            )

    # Publishing a completed download only after successful verification
    def test_verified_download_is_published_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = (
                Path(directory) / "telemetry.csv"
            )

            remote_content = b"verified-data"

            local_sha256 = (
                ingestion.download_verified_object(
                    s3_client = FakeS3Client(
                        content = remote_content
                    ),
                    bucket_name = "dataset-bucket",
                    remote_identity = (
                        build_remote_identity(
                            remote_content
                        )
                    ),
                    destination = destination
                )
            )

            self.assertEqual(
                destination.read_bytes(),
                remote_content
            )

            self.assertEqual(
                local_sha256,
                calculate_bytes_sha256(
                    remote_content
                )
            )

    # Reusing an unchanged local file with matching identity and content
    def test_verified_existing_file_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = (
                Path(directory) / "telemetry.csv"
            )

            content = b"verified-data"

            destination.write_bytes(
                content
            )

            remote_identity = build_remote_identity(
                content
            )

            previous_record = {
                **remote_identity,
                "local_sha256": (
                    calculate_bytes_sha256(
                        content
                    )
                )
            }

            reusable = ingestion.can_reuse_local_object(
                destination = destination,
                previous_record = previous_record,
                remote_identity = remote_identity
            )

            self.assertTrue(reusable)

    # Detecting an S3 object whose ETag changed
    def test_changed_remote_identity_does_not_match(self) -> None:
        content = b"verified-data"

        previous_record = {
            **build_remote_identity(content),
            "local_sha256": (
                calculate_bytes_sha256(
                    content
                )
            )
        }

        changed_identity = {
            **build_remote_identity(content),
            "etag": "changed-etag"
        }

        self.assertFalse(
            ingestion.remote_identities_match(
                previous_record = previous_record,
                remote_identity = changed_identity
            )
        )

    # Detecting an S3 object whose modification time changed
    def test_changed_remote_modification_time_does_not_match(self) -> None:
        content = b"verified-data"

        previous_record = {
            **build_remote_identity(content),
            "local_sha256": (
                calculate_bytes_sha256(
                    content
                )
            )
        }

        changed_identity = {
            **build_remote_identity(content),
            "last_modified_utc": (
                "2026-01-02T00:00:00+00:00"
            )
        }

        self.assertFalse(
            ingestion.remote_identities_match(
                previous_record = previous_record,
                remote_identity = changed_identity
            )
        )


if __name__ == "__main__":
    unittest.main()