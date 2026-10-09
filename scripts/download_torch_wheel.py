#!/usr/bin/env python3
"""Verified, resumable download of the official PyTorch CUDA 11.8 wheel."""
import concurrent.futures
import fcntl
import hashlib
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / '.download-cache'
NAME = 'torch-2.0.1+cu118-cp310-cp310-linux_x86_64.whl'
URL = 'https://download.pytorch.org/whl/cu118/torch-2.0.1%2Bcu118-cp310-cp310-linux_x86_64.whl'
SIZE = 2267321259
SHA256 = 'a7a49d459bf4862f64f7bc1a68beccf8881c2fa9f3e0569608e16ba6f85ebf7b'
CHUNK = 64 * 1024 * 1024


def download_chunk(index):
    path = DIRECTORY / f'torch-{index:03d}.part'
    start = index * CHUNK
    end = min(SIZE, start + CHUNK) - 1
    expected = end - start + 1
    for attempt in range(8):
        present = path.stat().st_size if path.exists() else 0
        if present == expected:
            return
        assert present < expected
        request = urllib.request.Request(URL, headers={'Range': f'bytes={start + present}-{end}'})
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=45) as response:
                assert response.status == 206
                assert response.headers.get('Content-Range') == f'bytes {start + present}-{end}/{SIZE}'
                remaining = expected - present
                with path.open('ab') as output:
                    while remaining:
                        block = response.read(min(1024 * 1024, remaining))
                        if not block:
                            raise RuntimeError('Incomplete response')
                        output.write(block)
                        remaining -= len(block)
            return
        except Exception as error:
            print(f'Retry chunk {index}: {error}', flush=True)
            if attempt == 7:
                raise
            time.sleep(min(2 ** attempt, 30))


def main():
    DIRECTORY.mkdir(exist_ok=True)
    lock = (DIRECTORY / 'torch-download.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    target = DIRECTORY / NAME
    if target.exists():
        digest = hashlib.sha256()
        with target.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        assert target.stat().st_size == SIZE and digest.hexdigest() == SHA256
        print('Existing wheel verified', flush=True)
        return
    count = (SIZE + CHUNK - 1) // CHUNK
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        pending = {pool.submit(download_chunk, index) for index in range(count)}
        while pending:
            done, pending = concurrent.futures.wait(pending, timeout=20,
                return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                future.result()
            present = sum(path.stat().st_size for path in DIRECTORY.glob('torch-*.part'))
            print(f'PyTorch wheel download: {present / SIZE:.1%}', flush=True)
    digest = hashlib.sha256()
    with target.open('wb') as output:
        for index in range(count):
            with (DIRECTORY / f'torch-{index:03d}.part').open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(block)
                    output.write(block)
    assert target.stat().st_size == SIZE and digest.hexdigest() == SHA256
    for path in DIRECTORY.glob('torch-*.part'):
        path.unlink()
    print(f'SHA256 verified: {target}', flush=True)


if __name__ == '__main__':
    main()
