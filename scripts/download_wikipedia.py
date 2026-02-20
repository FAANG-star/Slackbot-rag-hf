"""Download Simple English Wikipedia and package into ~900MB zip files for Slack upload.

Downloads via HuggingFace datasets (cached locally). Splits into zip files when
the compressed size hits 900MB. Run locally, then upload zips from Slack.

Usage:
    python scripts/download_wikipedia.py
    python scripts/download_wikipedia.py --output-dir ~/Downloads
"""

import argparse
import os
import zipfile

TARGET_BYTES = 900 * 1024 * 1024  # 900MB per zip


def sanitize(s: str) -> str:
    return s.replace("/", "_").replace("\\", "_").replace("\0", "")[:80]


def main(output_dir: str):
    from datasets import load_dataset

    os.makedirs(output_dir, exist_ok=True)

    print("Loading wikimedia/wikipedia (20231101.simple)...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")
    print(f"  {len(ds):,} articles")

    zip_idx = 1
    zf = None
    zip_path = None

    def open_zip():
        nonlocal zf, zip_path, zip_idx
        zip_path = os.path.join(output_dir, f"wikipedia_simple_{zip_idx:03d}.zip")
        zf = zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6)
        print(f"\nWriting {zip_path}...")
        zip_idx += 1

    def close_zip():
        zf.close()
        size_mb = os.path.getsize(zip_path) / 1024 / 1024
        print(f"  Closed — {size_mb:.0f} MB")

    open_zip()
    total = 0

    for article in ds:
        text = f"# {article['title']}\n\nURL: {article['url']}\n\n{article['text']}\n"
        filename = f"{sanitize(article['title'])}.txt"
        zf.writestr(filename, text)
        total += 1

        if total % 5_000 == 0:
            size_mb = os.path.getsize(zip_path) / 1024 / 1024
            print(f"  {total:,} articles — {size_mb:.0f} MB", flush=True)
            if os.path.getsize(zip_path) >= TARGET_BYTES:
                close_zip()
                open_zip()

    close_zip()
    print(f"\nDone — {total:,} articles across {zip_idx - 1} zip file(s) in {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=".", help="Directory to write zip files")
    args = parser.parse_args()
    main(args.output_dir)
