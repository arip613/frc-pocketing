"""Download FRC vendor CAD (STEP/DXF, or zips of them) from a list of direct links.

The link lists come from browsing vendor docs and product pages (WCP, REV, SDS, Thrifty
Bot, VEXpro). Every download goes through PoliteSession, so robots.txt is honoured (Google
Drive's download endpoints disallow bots and are skipped) and each file is logged with its
vendor and product page. Zips are unpacked to their STEP/DXF members.

Not used: Onshape (API terms forbid automated export), GrabCAD (bot-blocked, login and
terms forbid automated download), AndyMark (robots.txt ai-train=no).

    python -I -m frcpocket.scrape.vendors links.json [more.json ...] --out data/raw/vendors
"""

import argparse
import json
import re
import zipfile
from pathlib import Path
from urllib.parse import urlparse

from frcpocket.scrape.polite import PoliteSession

cadExtensions = (".step", ".stp", ".dxf")
maxBytes = 60 * 1024 * 1024
skipHosts = ("onshape.com", "grabcad.com", "andymark.com", "chiefdelphi.com")


def safeName(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)[:150]


def unpackZip(zipPath, outDir):
    """CAD members of a zip, written next to it; the zip itself is removed."""
    members = []
    try:
        with zipfile.ZipFile(zipPath) as archive:
            for info in archive.infolist():
                if info.is_dir() or not info.filename.lower().endswith(cadExtensions) or info.file_size > maxBytes:
                    continue
                destPath = outDir / safeName(f"{zipPath.stem}__{info.filename.replace('/', '__')}")
                destPath.write_bytes(archive.read(info))
                members.append((destPath, info.filename))
    except zipfile.BadZipFile:
        pass
    zipPath.unlink(missing_ok=True)
    return members


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lists", nargs="+")
    parser.add_argument("--out", default="data/raw/vendors")
    parser.add_argument("--log", default="data/raw/vendors/sources.jsonl")
    args = parser.parse_args()

    client = PoliteSession(args.log, minInterval=1.0)
    links = [link for path in args.lists if Path(path).exists() for link in json.loads(Path(path).read_text())]
    seenUrls = set()
    nFiles = nSkipped = 0
    for link in links:
        url = link["url"]
        host = urlparse(url).netloc.lower()
        if url in seenUrls or any(host.endswith(skip) for skip in skipHosts) or link.get("kind") == "gdrive":
            nSkipped += 1
            continue
        seenUrls.add(url)
        vendorDir = Path(args.out) / safeName(link.get("vendor", "unknown"))
        name = safeName(f"{link.get('product', '')}__{Path(urlparse(url).path).name}") or "file"
        destPath = vendorDir / name
        try:
            if not client.isAllowed(url):
                print(f"robots.txt disallows {url}", flush=True)
                nSkipped += 1
                continue
            result = client.download(url, destPath, maxBytes)
        except Exception as error:
            print(f"failed {url}: {error}", flush=True)
            continue
        if result is None:
            print(f"no file (status or size) {url}", flush=True)
            continue
        path, sha256, nBytes = result
        if path.suffix.lower() == ".zip" or zipfile.is_zipfile(path):
            files = unpackZip(path, vendorDir)
        elif path.suffix.lower() in cadExtensions:
            files = [(path, path.name)]
        else:
            head = path.read_bytes()[:200]
            if b"ISO-10303" in head:
                files = [(path.rename(path.with_suffix(".step")), path.name)]
            else:
                path.unlink()
                files = []
        for filePath, member in files:
            client.logSource(source="vendor", vendor=link.get("vendor"), product=link.get("product"),
                             pageUrl=link.get("pageUrl"), url=url, member=member, localPath=str(filePath),
                             sha256=sha256, bytes=nBytes, license="vendor CAD, published for teams",
                             trainingAllowed=client.trainingAllowed(link.get("pageUrl") or url),
                             notes="vendor-published product CAD; keep local and don't redistribute")
            nFiles += 1
        print(f"{link.get('vendor')} {link.get('product')}: {len(files)} CAD files", flush=True)
    print(f"done: {nFiles} CAD files from {len(seenUrls)} links ({nSkipped} skipped)", flush=True)


if __name__ == "__main__":
    main()
