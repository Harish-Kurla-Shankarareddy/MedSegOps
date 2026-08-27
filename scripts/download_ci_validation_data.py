from __future__ import annotations

import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CI_DATA_ROOT = PROJECT_ROOT / ".ci_data"
TAR_PATH = CI_DATA_ROOT / "Task09_Spleen.tar"

DATA_URL = (
    "https://msd-for-monai.s3-us-west-2.amazonaws.com/"
    "Task09_Spleen.tar"
)

EXPECTED_MD5 = "410d4a301da4e5b2f6f86ec3ddba524e"

VALIDATION_CASES = [
    "10",
    "12",
    "13",
    "14",
    "16",
]


def md5_file(path: Path) -> str:
    digest = hashlib.md5()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def download_with_progress() -> None:
    CI_DATA_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if TAR_PATH.exists():
        print("Archive already exists:")
        print(TAR_PATH)

        print("Checking MD5...")

        checksum = md5_file(TAR_PATH)

        if checksum != EXPECTED_MD5:
            raise RuntimeError(
                "Existing archive has incorrect MD5.\n"
                f"Expected: {EXPECTED_MD5}\n"
                f"Actual:   {checksum}"
            )

        print("MD5 verification passed.")
        return

    print()
    print("=" * 70)
    print("Downloading Task09_Spleen.tar")
    print("=" * 70)
    print(f"URL: {DATA_URL}")
    print(f"Output: {TAR_PATH}")
    print("=" * 70)
    print()

    def progress(
        block_count: int,
        block_size: int,
        total_size: int,
    ) -> None:
        downloaded = (
            block_count * block_size
        )

        if total_size > 0:
            downloaded = min(
                downloaded,
                total_size,
            )

            percent = (
                downloaded
                / total_size
                * 100
            )

            downloaded_gb = (
                downloaded
                / (1024 ** 3)
            )

            total_gb = (
                total_size
                / (1024 ** 3)
            )

            print(
                f"\rDownloaded: "
                f"{downloaded_gb:.2f} / "
                f"{total_gb:.2f} GB "
                f"({percent:6.2f}%)",
                end="",
                flush=True,
            )

        else:
            downloaded_mb = (
                downloaded
                / (1024 ** 2)
            )

            print(
                f"\rDownloaded: "
                f"{downloaded_mb:.1f} MB",
                end="",
                flush=True,
            )

    urllib.request.urlretrieve(
        DATA_URL,
        TAR_PATH,
        reporthook=progress,
    )

    print()
    print()
    print("Download complete.")
    print("Checking MD5...")

    checksum = md5_file(TAR_PATH)

    if checksum != EXPECTED_MD5:
        TAR_PATH.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            "Downloaded archive has incorrect MD5.\n"
            f"Expected: {EXPECTED_MD5}\n"
            f"Actual:   {checksum}"
        )

    print("MD5 verification passed.")


def extract_validation_cases() -> None:
    output_root = (
        CI_DATA_ROOT / "Task09_Spleen"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    required_files: list[str] = []

    for case_id in VALIDATION_CASES:
        required_files.extend(
            [
                (
                    "Task09_Spleen/imagesTr/"
                    f"spleen_{case_id}.nii.gz"
                ),
                (
                    "Task09_Spleen/labelsTr/"
                    f"spleen_{case_id}.nii.gz"
                ),
            ]
        )

    print()
    print("Extracting validation cases...")

    with tarfile.open(
        TAR_PATH,
        mode="r",
    ) as archive:

        members = {
            member.name: member
            for member in archive.getmembers()
        }

        missing = [
            path
            for path in required_files
            if path not in members
        ]

        if missing:
            raise RuntimeError(
                "Required files were not found "
                "in the archive:\n"
                + "\n".join(missing)
            )

        for relative_path in required_files:
            member = members[relative_path]

            destination = (
                output_root
                / relative_path.removeprefix(
                    "Task09_Spleen/"
                )
            )

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            extracted = archive.extractfile(
                member
            )

            if extracted is None:
                raise RuntimeError(
                    f"Could not extract {relative_path}"
                )

            with destination.open("wb") as output:
                shutil.copyfileobj(
                    extracted,
                    output,
                )

            print(
                f"  extracted: {destination}"
            )

    print("Validation data ready.")


def main() -> None:
    download_with_progress()
    extract_validation_cases()

    print()
    print("=" * 70)
    print("CI MODEL VALIDATION DATA READY")
    print("=" * 70)

    for case_id in VALIDATION_CASES:
        print(
            f"spleen_{case_id}: "
            "image + ground truth"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()