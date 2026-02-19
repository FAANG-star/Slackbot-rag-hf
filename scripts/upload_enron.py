"""Download the Enron email corpus and upload to the RAG volume.

Source: https://www.cs.cmu.edu/~enron/ (May 7, 2015 version)
~500K emails, ~1.7GB uncompressed.
"""

import os
import tarfile
import tempfile
import urllib.request

import modal

modal.enable_output()

from agents.infra.shared import app, rag_vol

ENRON_URL = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"


@app.local_entrypoint()
def main():
    with tempfile.TemporaryDirectory() as tmp:
        tarball = os.path.join(tmp, "enron.tar.gz")
        extract_dir = os.path.join(tmp, "enron")

        print(f"Downloading Enron corpus from {ENRON_URL}...")
        urllib.request.urlretrieve(ENRON_URL, tarball)

        print("Extracting...")
        with tarfile.open(tarball, "r:gz") as tar:
            tar.extractall(extract_dir)

        # The tarball extracts to maildir/ with per-user folders
        maildir = os.path.join(extract_dir, "maildir")
        if not os.path.isdir(maildir):
            # Handle case where extraction structure differs
            for entry in os.listdir(extract_dir):
                candidate = os.path.join(extract_dir, entry)
                if os.path.isdir(candidate):
                    maildir = candidate
                    break

        # Collect all email files
        email_files = []
        for root, _, files in os.walk(maildir):
            for f in files:
                if f.startswith("."):
                    continue
                email_files.append(os.path.join(root, f))

        print(f"Found {len(email_files):,} emails. Uploading to volume...")

        with rag_vol.batch_upload(force=True) as batch:
            for filepath in email_files:
                # Flatten into /rag/docs/ so the indexer picks them up
                # (Manifest.scan only does a shallow iterdir on DOCS_DIR)
                rel = os.path.relpath(filepath, maildir)
                flat_name = rel.replace(os.sep, "_")
                batch.put_file(filepath, f"/rag/docs/{flat_name}")

        print(f"Uploaded {len(email_files):,} emails to sandbox-rag volume at /rag/docs/")
        print("(maps to /data/rag/docs/ inside the sandbox)")
