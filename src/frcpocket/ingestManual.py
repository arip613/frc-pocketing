"""Copy plates you exported by hand (Onshape, SolidWorks, vendor sites) into the dataset.

Hand-picked exports are the curated "looks great" tier: you chose them, so they are
tagged origin=manual and treated as positive examples once they pass extraction.

Each file lands in its own folder under data/raw/manual/ and gets a source-log entry.
STEP, STP, DXF, and ZIP files (Onshape zips multi-part exports) are accepted. Naming
files with the team number first ("1690 elevator plate.step") lets previews show it.

Usage:
    python -I -m frcpocket.ingestManual --inbox ~/Downloads/frcPlates
"""

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

from frcpocket.scrape.polite import PoliteSession

cadExtensions = (".step", ".stp", ".dxf")
maxFileBytes = 200 * 1024 * 1024


def safeName(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)[:120]


def sha256Of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def teamFromName(name):
    match = re.match(r"\s*(?:frc\s*|team\s*)?(\d{1,5})\b", name, re.I)
    return int(match.group(1)) if match else None


def cadFilesIn(inbox):
    for path in sorted(Path(inbox).expanduser().iterdir()):
        if path.is_file() and path.suffix.lower() in cadExtensions + (".zip",):
            yield path


def extractZip(zipPath, destDir):
    """Unpack only CAD files, flattened, so nothing can land outside destDir."""
    extracted = []
    with zipfile.ZipFile(zipPath) as archive:
        for member in archive.infolist():
            memberName = Path(member.filename).name
            if member.is_dir() or Path(memberName).suffix.lower() not in cadExtensions:
                continue
            if member.file_size > maxFileBytes:
                continue
            target = destDir / safeName(memberName)
            with archive.open(member) as source, open(target, "wb") as sink:
                shutil.copyfileobj(source, sink)
            extracted.append(target)
    return extracted


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inbox", default="~/Downloads/frcPlates")
    parser.add_argument("--out", default="data/raw/manual")
    parser.add_argument("--log", default="data/sources.jsonl")
    args = parser.parse_args()

    outDir = Path(args.out)
    outDir.mkdir(parents=True, exist_ok=True)
    logger = PoliteSession(args.log)
    known = set()
    if Path(args.log).exists():
        for line in Path(args.log).read_text().splitlines():
            entry = json.loads(line)
            known.update(entry[key] for key in ("sha256", "containerSha256") if entry.get(key))
    nAdded = 0
    for path in cadFilesIn(args.inbox):
        if path.stat().st_size > maxFileBytes:
            print(f"skip (too big): {path.name}")
            continue
        fileHash = sha256Of(path)
        if fileHash in known:
            print(f"already ingested: {path.name}")
            continue
        destDir = outDir / f"{safeName(path.stem)}_{fileHash[:8]}"
        destDir.mkdir(parents=True, exist_ok=False)
        if path.suffix.lower() == ".zip":
            copies = extractZip(path, destDir)
        else:
            copies = [destDir / safeName(path.name)]
            shutil.copy2(path, copies[0])
        for copy in copies:
            logger.logSource(
                source="manual-export",
                originalName=path.name,
                localPath=str(copy),
                sha256=sha256Of(copy),
                containerSha256=fileHash,
                team=teamFromName(path.name),
                license="exported by the user from a public document; keep local",
                trainingAllowed=True,
                notes="hand-picked by the user as aesthetically strong pocketing",
            )
        known.add(fileHash)
        nAdded += 1
        print(f"added: {path.name} -> {len(copies)} CAD file(s)")
    print(f"done: {nAdded} new file(s) from {Path(args.inbox).expanduser()}")


if __name__ == "__main__":
    main()
