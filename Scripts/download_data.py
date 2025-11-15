#!/usr/bin/env python3
import os
import sys
import argparse
import tempfile
import zipfile
import urllib.request

DEST = "DATA/PROTEINS"
os.makedirs(DEST, exist_ok=True)

def try_pyg():
    try:
        from torch_geometric.datasets import TUDataset
    except Exception:
        return False
    try:
        print("Trying torch_geometric.datasets.TUDataset(name='PROTEINS_full') ...")
        TUDataset(root=DEST, name="PROTEINS_full")
        print("Downloaded via PyG into", DEST)
        return True
    except Exception as e:
        print("PyG TUDataset(PROTEINS_full) failed:", e)
        return False

def download_zip(url):
    tmp = os.path.join(tempfile.gettempdir(), "PROTEINS_full.zip")
    print("Downloading from:", url)
    urllib.request.urlretrieve(url, tmp)
    print("Extracting to", DEST)
    with zipfile.ZipFile(tmp, "r") as z:
        z.extractall(DEST)
    try:
        os.remove(tmp)
    except OSError:
        pass
    print("Done. Files are in", DEST)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="Direct URL to PROTEINS_full zip (optional fallback)")
    args = parser.parse_args()

    if try_pyg():
        return

    if not args.url:
        print("\nPyG fallback failed. Open this page in a browser and copy the direct link to PROTEINS_full.zip:")
        print("https://chrsmrrs.github.io/datasets/docs/datasets/\n")
        print("Then rerun this script with --url <direct-zip-link>\n")
        sys.exit(1)

    download_zip(args.url)

if __name__ == "__main__":
    main()