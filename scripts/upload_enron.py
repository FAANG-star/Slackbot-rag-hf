"""Download the Enron email corpus and create per-user zip files on the volume.

First run: downloads the 1.7GB tarball to /data/rag/enron.tar.gz on the volume.
Subsequent runs: skips the download and extracts directly from the cached tarball.

Source: https://www.cs.cmu.edu/~enron/ (May 7, 2015 version)
~500K emails across 150 users, ~1.7GB compressed.

Usage:
    modal run scripts/upload_enron.py                              # all users
    modal run scripts/upload_enron.py --users lay-k,skilling-j    # subset
    modal run scripts/upload_enron.py --redownload                 # force re-download
"""

import modal

modal.enable_output()

app = modal.App("ml-agent")
rag_vol = modal.Volume.from_name("sandbox-rag")

ENRON_URL = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"
TARBALL = "/data/rag/enron.tar.gz"

_image = modal.Image.debian_slim(python_version="3.12")


@app.function(
    image=_image,
    volumes={"/data": rag_vol},
    timeout=60 * 60,
    memory=2048,
)
def download_and_extract(users: list[str], redownload: bool = False):
    import os
    import shutil
    import tarfile
    import urllib.request
    import zipfile

    # --- Download tarball to volume (once) ---
    if redownload or not os.path.exists(TARBALL):
        print(f"Downloading {ENRON_URL}...", flush=True)
        os.makedirs(os.path.dirname(TARBALL), exist_ok=True)
        urllib.request.urlretrieve(ENRON_URL, TARBALL)
        rag_vol.commit()
        print(f"Saved to {TARBALL}.", flush=True)
    else:
        size_mb = os.path.getsize(TARBALL) / 1024 / 1024
        print(f"Using cached {TARBALL} ({size_mb:.0f} MB).", flush=True)

    # --- Extract per-user zip files ---
    docs_dir = "/data/rag/docs"
    if os.path.exists(docs_dir):
        shutil.rmtree(docs_dir)
    os.makedirs(docs_dir)

    user_set = set(users) if users else None
    filter_desc = f"users={','.join(sorted(user_set))}" if user_set else "all users"
    print(f"Extracting ({filter_desc})...", flush=True)

    open_zips: dict[str, zipfile.ZipFile] = {}
    counts: dict[str, int] = {}
    total = 0

    try:
        with tarfile.open(TARBALL, "r:gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                parts = member.name.split("/")
                if len(parts) < 2:
                    continue
                user = parts[1]
                if user_set and user not in user_set:
                    continue
                filename = os.path.basename(member.name)
                if filename.startswith("."):
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue

                inner_path = "/".join(parts[1:])
                data = f.read()

                if user not in open_zips:
                    zip_path = os.path.join(docs_dir, f"{user}.zip")
                    open_zips[user] = zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED)
                    counts[user] = 0

                open_zips[user].writestr(inner_path, data)
                counts[user] += 1
                total += 1

                if total % 50_000 == 0:
                    print(f"  {total:,} emails written ({len(open_zips)} users)...", flush=True)
    finally:
        for zf in open_zips.values():
            zf.close()

    print(f"Wrote {total:,} emails across {len(counts)} users:", flush=True)
    for user in sorted(counts):
        print(f"  {user}.zip — {counts[user]:,} emails", flush=True)

    rag_vol.commit()
    print("Volume committed.", flush=True)


@app.local_entrypoint()
def main(users: str = "", redownload: bool = False):
    user_list = [u.strip() for u in users.split(",") if u.strip()] if users else []
    desc = f"users={user_list}" if user_list else "all users"
    print(f"Uploading Enron mailboxes ({desc})")
    download_and_extract.remote(user_list, redownload)
