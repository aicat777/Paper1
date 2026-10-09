#!/usr/bin/env python3
"""Download and verify the original-format DIOR mirror, without changing splits."""
import concurrent.futures
import fcntl
import hashlib
import json
import threading
import time
import urllib.parse
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "DIOR"
DOWNLOADS = DATA / "downloads"
LOGS = ROOT / "logs"
REPO = "ObjEarth/ObjEarth-Data"
REVISION = "f58d220a2b74b610badebaefa122d1b61628bf73"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}"
TRANSFER_BASE = f"https://hf-mirror.com/datasets/{REPO}/resolve/{REVISION}"
TRANSFER_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
CHUNK_SIZE = 64 * 1024 * 1024
WORKERS = 8
REQUEST_LOCK = threading.Lock()
NEXT_REQUEST = 0.0


def wait_transfer_slot():
    global NEXT_REQUEST
    with REQUEST_LOCK:
        delay = max(0, NEXT_REQUEST - time.monotonic())
        if delay:
            time.sleep(delay)
        NEXT_REQUEST = time.monotonic() + 2


def pause_mirror(seconds):
    global NEXT_REQUEST
    with REQUEST_LOCK:
        NEXT_REQUEST = max(NEXT_REQUEST, time.monotonic() + seconds)


def read_url(url):
    req = urllib.request.Request(url, headers={"User-Agent": "DIOR-research-download/1.0"})
    with urllib.request.urlopen(req, timeout=45) as response:
        return response.read()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_target(item):
    return DOWNLOADS / Path(item["path"]).name


def download_chunk(item, index):
    target = archive_target(item)
    directory = DOWNLOADS / "parts" / target.name
    directory.mkdir(parents=True, exist_ok=True)
    start = index * CHUNK_SIZE
    end = min(start + CHUNK_SIZE, item["size"]) - 1
    expected = end - start + 1
    part = directory / f"{index:04d}.part"
    if part.exists() and part.stat().st_size > expected:
        raise RuntimeError(f"Oversized chunk: {part}")
    suffix = "/" + urllib.parse.quote(item["path"], safe="/") + "?download=true"
    use_original = False
    for attempt in range(8):
        present = part.stat().st_size if part.exists() else 0
        if present == expected:
            return
        absolute_start = start + present
        if not use_original:
            wait_transfer_slot()
        url = (BASE if use_original else TRANSFER_BASE) + suffix
        request = urllib.request.Request(url, headers={
            "User-Agent": "DIOR-research-download/2.0",
            "Range": f"bytes={absolute_start}-{end}",
        })
        try:
            open_request = urllib.request.urlopen if use_original else TRANSFER_OPENER.open
            with open_request(request, timeout=45) as response:
                content_range = f"bytes {absolute_start}-{end}/{item['size']}"
                if response.status != 206 or response.headers.get("Content-Range") != content_range:
                    raise RuntimeError(f"Server returned an incorrect range: {response.status}, {response.headers.get('Content-Range')}")
                remaining = expected - present
                with part.open("ab") as output:
                    while remaining:
                        block = response.read(min(1024 * 1024, remaining))
                        if not block:
                            raise RuntimeError("Incomplete range response")
                        output.write(block)
                        remaining -= len(block)
            return
        except Exception as error:
            print(f"Retry {target.name} chunk {index}, attempt {attempt + 1}: {error}", flush=True)
            if isinstance(error, urllib.error.HTTPError) and error.code == 429 and not use_original:
                retry_after = error.headers.get("Retry-After", "60")
                pause_mirror(float(retry_after) if retry_after.isdigit() else 60)
                use_original = True
            if attempt == 7:
                raise
            time.sleep(min(2 ** attempt, 30))


def verify_existing(item):
    target = DOWNLOADS / Path(item["path"]).name
    expected_hash = item["lfs"]["oid"]
    if target.exists():
        if target.stat().st_size == item["size"] and sha256(target) == expected_hash:
            print(f"Already verified: {target.name}", flush=True)
            return True
        raise RuntimeError(f"Existing archive failed verification: {target}")
    return False


def assemble_archive(item):
    target = archive_target(item)
    part = target.with_suffix(target.suffix + ".part")
    digest = hashlib.sha256()
    with part.open("wb") as output:
        for index in range((item["size"] + CHUNK_SIZE - 1) // CHUNK_SIZE):
            chunk = DOWNLOADS / "parts" / target.name / f"{index:04d}.part"
            with chunk.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
                    output.write(block)
    if part.stat().st_size != item["size"]:
        raise RuntimeError(f"Size mismatch: {part.name}")
    if digest.hexdigest() != item["lfs"]["oid"]:
        raise RuntimeError(f"SHA256 mismatch: {part.name}")
    part.rename(target)
    print(f"SHA256 verified: {target.name}", flush=True)
    directory = DOWNLOADS / "parts" / target.name
    for chunk in directory.glob("*.part"):
        chunk.unlink()
    directory.rmdir()
    previous = target.with_suffix(target.suffix + ".previous.part")
    if previous.exists():
        previous.unlink()
    return target


def extract_archive(path):
    marker = DOWNLOADS / (path.name + ".extracted")
    if marker.exists():
        return
    with zipfile.ZipFile(path) as archive:
        files = [entry for entry in archive.infolist() if not entry.is_dir()]
        if path.name == "Annotations.zip":
            destination_root = DATA / "Annotations" / "Horizontal Bounding Boxes"
        elif all("/" not in entry.filename for entry in files):
            destination_root = DATA / path.stem
        else:
            destination_root = DATA
        destination_root.mkdir(parents=True, exist_ok=True)
        # Validate all destinations before extracting the untrusted archive.
        for entry in archive.infolist():
            destination = (destination_root / entry.filename).resolve()
            if not destination.is_relative_to(destination_root.resolve()):
                raise RuntimeError(f"Unsafe archive path: {entry.filename}")
        print(f"Extracting {path.name}: {len(files)} files", flush=True)
        archive.extractall(destination_root)
    marker.write_text(sha256(path) + "\n")


def main():
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    lock = (DOWNLOADS / "download.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another DIOR download is already running")
    api = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}"
    items = json.loads(read_url(api + "/DIOR?recursive=false&expand=false"))
    archives = [entry for entry in items if entry["type"] == "file" and entry["path"].endswith(".zip")]
    splits = json.loads(read_url(api + "/DIOR/ImageSets?recursive=true&expand=false"))
    manifest = {
        "dataset": "DIOR horizontal-box detection, original-format mirror",
        "original_homepage": "https://gcheng-nwpu.github.io/#Datasets",
        "mirror": f"https://huggingface.co/datasets/{REPO}/tree/{REVISION}/DIOR",
        "revision": REVISION,
        "image_transfer_endpoint": TRANSFER_BASE,
        "license": "CC BY-NC 4.0",
        "archives": archives,
        "split_files": splits,
    }
    (DATA / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for entry in splits:
        if entry["type"] != "file":
            continue
        content = read_url(BASE + "/" + urllib.parse.quote(entry["path"], safe="/"))
        git_oid = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if len(content) != entry["size"] or git_oid != entry["oid"]:
            raise RuntimeError(f"Split verification failed: {entry['path']}")
        relative = Path(entry["path"]).relative_to("DIOR")
        destination = DATA / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        print(f"Verified split {relative}: {len(content.splitlines())} images", flush=True)
    needed = [item for item in archives if not verify_existing(item)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        jobs = [(item, index) for index in range(max((item["size"] + CHUNK_SIZE - 1) // CHUNK_SIZE for item in needed))
                for item in needed if index * CHUNK_SIZE < item["size"]] if needed else []
        pending = {pool.submit(download_chunk, item, index): (item, index) for item, index in jobs}
        while pending:
            done, _ = concurrent.futures.wait(pending, timeout=20, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                future.result()
                del pending[future]
            if pending:
                progress = []
                for item in needed:
                    directory = DOWNLOADS / "parts" / archive_target(item).name
                    size = sum(path.stat().st_size for path in directory.glob("*.part"))
                    progress.append(f"{Path(item['path']).name}: {size / item['size']:.1%}")
                print("Progress: " + "; ".join(progress), flush=True)
    for item in needed:
        assemble_archive(item)
    for item in archives:
        extract_archive(archive_target(item))
    print("DIOR download, checksum verification and extraction complete.", flush=True)


if __name__ == "__main__":
    main()
