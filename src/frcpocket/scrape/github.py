"""Find FRC team CAD repos on GitHub and download plate-like STEP/DXF files.

Uses the public REST API (repo search + git trees) and raw.githubusercontent.com for
file contents. Unauthenticated limits are 10 searches/min and 60 API calls/hour, so a
full crawl is slow; set GITHUB_TOKEN to raise the limit. Tree listings are cached so
re-runs don't spend quota.

Usage:
    python -I -m frcpocket.scrape.github --out data/raw/github --max-repos 60
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

from frcpocket.scrape.polite import PoliteSession

apiRoot = "https://api.github.com"
searchQueries = [
    "frc cad",
    "frc in:name cad",
    "topic:frc cad",
    "frc robot cad",
    "frc dxf",
    "frc step files",
    "frc solidworks",
    "frc onshape export",
    "frc inventor cad",
    "frc mechanical design",
    "frc swerve cad",
    "frc gearbox cad",
    "frc in:description cad robot",
    "frc 2024 cad",
    "frc 2025 cad",
    "frc 2023 cad",
    "frc 2022 cad",
    "frc swerve module step",
    "frc elevator cad",
    "frc turret cad",
    "frc gearbox step",
    "frc robot step files",
    "frc offseason cad",
    "frc cad library",
    "frc 2019 cad",
    "frc 2020 cad",
    "frc 2021 cad",
    "frc 2026 cad",
    "frc robot 2024",
    "frc robot 2025",
    "first robotics cad",
    "first robotics competition cad",
    "frc step",
    "frc stp",
    "frc team cad",
    "frc offseason robot",
    "frc mechanical",
    "frc onshape",
    "frc fusion 360",
    "frc solidworks robot",
    "frc inventor robot",
    "frc sheet metal",
    "frc router plates",
    "frc waterjet",
    "frc cnc plates",
    "frc dxf plates",
    "frc drivetrain cad",
    "frc intake cad",
    "frc climber cad",
    "frc shooter cad",
    "frc robot design",
    "frc robot model",
    "frc 3d model",
    "frc cad files",
]
repoKeepWords = ("cad", "step", "dxf", "solidworks", "onshape", "inventor", "fusion", "mechanical", "design")
repoSkipWords = ("frcnn", "kicad", "rag", "synthesis", "electronics", "koans", "harness", "joystick", "academy")
cadExtensions = (".step", ".stp", ".dxf")
plateNameHints = (
    "plate", "gusset", "side", "frame", "bracket", "mount", "panel", "arm", "wall",
    "rail", "belly", "pan", "tube", "bearing", "spacer", "flat", "sheet", "cut", "router",
)
wholeAssemblyHints = ("robot", "full", "field", "assembly", "assy", "asm", "complete", "entire", "whole")
maxStepBytes = 8 * 1024 * 1024
maxDxfBytes = 3 * 1024 * 1024
lfsPointerMaxBytes = 200
maxFilesPerRepo = 40
maxBigStepBytes = 60 * 1024 * 1024
bigFileHints = (
    "swerve", "gearbox", "drive", "chassis", "chasis", "frame", "elevator", "arm", "intake", "telescope",
    "robot", "module", "shooter", "climber", "estructura", "base", "track", "ri3d", "everybot", "reduction",
    "pto", "superstructure", "launcher", "wrist", "turret",
)
bigFileSkipHints = (
    "roborio", "rev-11", "217-424", "talon", "motor", "falcon", "wheel", "field", "colson", "maxplanetary",
    "regulator", "pneumatic", "nmrv", "pit", "versaplanetary", "rev-21-2099", "advantagescope/arm",
)


def safeName(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)[:180]


def authHeaders():
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def searchRepos(client, queries, perQuery=100, pages=1):
    reposByName = {}
    for query in queries:
        for page in range(1, pages + 1):
            url = f"{apiRoot}/search/repositories?q={quote(query)}&per_page={perQuery}&sort=stars&page={page}"
            response = client.get(url, headers=authHeaders())
            time.sleep(2.2 if authHeaders() else 7)  # search allows 30/min with a token, 10/min without
            if response.status_code != 200:
                print(f"search '{query}' page {page} failed: {response.status_code}", flush=True)
                break
            items = response.json().get("items", [])
            for item in items:
                reposByName.setdefault(item["full_name"], item)
            if len(items) < perQuery:
                break
        print(f"search '{query}': {len(reposByName)} unique repos so far", flush=True)
    return list(reposByName.values())


def looksLikeCadRepo(repo):
    text = f"{repo['full_name']} {repo.get('description') or ''}".lower()
    if any(word in text for word in repoSkipWords):
        return False
    return any(word in text for word in repoKeepWords) and repo.get("size", 0) > 0


licenseGuesses = (("mit license", "MIT"), ("apache license", "Apache-2.0"), ("gnu general public", "GPL"),
                  ("bsd", "BSD"), ("creative commons", "CC"), ("unlicense", "Unlicense"), ("mozilla public", "MPL-2.0"))


def gitTree(fullName, timeoutSeconds=120):
    """Recursive file listing (path, size) from a shallow, blob-less clone, for when the REST API is unavailable."""
    with tempfile.TemporaryDirectory() as work:
        url = f"https://github.com/{fullName}.git"
        try:
            subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--no-checkout", url, work],
                           check=True, timeout=timeoutSeconds, capture_output=True,
                           env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
            branch = subprocess.run(["git", "-C", work, "rev-parse", "--abbrev-ref", "HEAD"], check=True,
                                    capture_output=True, text=True).stdout.strip()
            # no -l: asking for sizes makes a blob-less clone fetch every blob. Sizes stay unknown (0,
            # handled like an LFS pointer) and the streaming download enforces the byte cap instead.
            listing = subprocess.run(["git", "-C", work, "ls-tree", "-r", "HEAD"], check=True,
                                     capture_output=True, text=True, timeout=timeoutSeconds,
                                     env={**os.environ, "GIT_NO_LAZY_FETCH": "1"}).stdout
        except (subprocess.SubprocessError, OSError) as error:
            print(f"  git listing failed for {fullName}: {type(error).__name__}", flush=True)
            return None
    tree = []
    for line in listing.splitlines():
        meta, path = line.split("\t", 1)
        if meta.split()[1] == "blob":
            tree.append({"path": path, "type": "blob", "size": 0})
    return {"tree": tree, "defaultBranch": branch}


def repoFromName(client, fullName):
    """Repo dict like the search API's, built from git alone, with a license guessed from its LICENSE file."""
    tree = gitTree(fullName)
    if tree is None:
        return None, None
    branch = tree["defaultBranch"]
    license = "none-declared"
    for entry in tree["tree"]:
        if "/" not in entry["path"] and entry["path"].lower().split(".")[0] in ("license", "licence", "copying"):
            try:
                text = client.get(f"https://raw.githubusercontent.com/{fullName}/{quote(branch)}/{quote(entry['path'])}").text[:3000].lower()
                license = next((spdx for phrase, spdx in licenseGuesses if phrase in text), "other")
            except Exception:
                license = "other"
            break
    repo = {"full_name": fullName, "default_branch": branch, "html_url": f"https://github.com/{fullName}",
            "license": {"spdx_id": license}, "stargazers_count": 0}
    return repo, tree


def listCadFiles(client, repo, metaDir):
    cachePath = metaDir / f"{safeName(repo['full_name'])}.json"
    if cachePath.exists():
        tree = json.loads(cachePath.read_text())
    else:
        url = f"{apiRoot}/repos/{repo['full_name']}/git/trees/{quote(repo['default_branch'])}?recursive=1"
        response = client.get(url, headers=authHeaders())
        if response.status_code != 200:
            print(f"  tree failed for {repo['full_name']}: {response.status_code}", flush=True)
            return []
        tree = response.json()
        cachePath.write_text(json.dumps(tree))
    cadFiles = [
        entry for entry in tree.get("tree", [])
        if entry.get("type") == "blob" and entry["path"].lower().endswith(cadExtensions)
    ]
    return cadFiles


def filePriority(entry):
    name = entry["path"].lower()
    baseName = name.rsplit("/", 1)[-1]
    score = sum(hint in baseName for hint in plateNameHints) * 2
    score += sum(hint in name for hint in plateNameHints)
    score -= sum(hint in baseName for hint in wholeAssemblyHints) * 5
    if name.endswith(".dxf"):
        score += 3  # 2D router/waterjet exports are almost always plates
    return score


def chooseFiles(cadFiles, maxCount=maxFilesPerRepo):
    candidates = []
    for entry in cadFiles:
        isDxf = entry["path"].lower().endswith(".dxf")
        size = entry.get("size", 0)
        isLfsPointer = size < lfsPointerMaxBytes
        limit = maxDxfBytes if isDxf else maxStepBytes
        if not isLfsPointer and size > limit:
            continue
        if filePriority(entry) < 0:
            continue
        candidates.append(entry)
    candidates.sort(key=lambda entry: (-filePriority(entry), entry.get("size", 0)))
    return candidates[:maxCount]


def downloadFile(client, repo, entry, outDir):
    fullName = repo["full_name"]
    branch = repo["default_branch"]
    encodedPath = quote(entry["path"])
    rawUrl = f"https://raw.githubusercontent.com/{fullName}/{quote(branch)}/{encodedPath}"
    repoDir = outDir / safeName(fullName.replace("/", "__"))
    destPath = repoDir / safeName(entry["path"].replace("/", "__"))
    if destPath.exists():
        return None
    limit = maxDxfBytes if entry["path"].lower().endswith(".dxf") else maxStepBytes
    result = client.download(rawUrl, destPath, limit)
    sourceUrl = rawUrl
    if result and result[2] < lfsPointerMaxBytes and destPath.read_bytes().startswith(b"version https://git-lfs"):
        destPath.unlink()
        sourceUrl = f"https://media.githubusercontent.com/media/{fullName}/{quote(branch)}/{encodedPath}"
        result = client.download(sourceUrl, destPath, limit)
    if result is None:
        return None
    path, sha256, nBytes = result
    client.logSource(
        source="github",
        repo=fullName,
        repoUrl=repo["html_url"],
        filePath=entry["path"],
        url=sourceUrl,
        localPath=str(path),
        sha256=sha256,
        bytes=nBytes,
        license=(repo.get("license") or {}).get("spdx_id") or "none-declared",
        stars=repo.get("stargazers_count", 0),
        trainingAllowed=True,
        notes="public GitHub repo; no license means all rights reserved, so keep local and don't redistribute",
    )
    return path


def chooseBigFiles(repos, metaDir, maxCount):
    """Mechanism/robot assemblies between the normal cap and maxBigStepBytes, from cached trees only."""
    picked = []
    seenKeys = set()
    for repo in repos:
        cachePath = metaDir / f"{safeName(repo['full_name'])}.json"
        if not cachePath.exists():
            continue
        for entry in json.loads(cachePath.read_text()).get("tree", []):
            path = entry["path"].lower()
            size = entry.get("size", 0)
            if entry.get("type") != "blob" or not path.endswith((".step", ".stp")):
                continue
            if not (maxStepBytes < size <= maxBigStepBytes):
                continue
            if any(hint in path for hint in bigFileSkipHints) or not any(hint in path for hint in bigFileHints):
                continue
            key = (size, path.rsplit("/", 1)[-1])
            if key in seenKeys:
                continue
            seenKeys.add(key)
            picked.append((repo, entry))
    picked.sort(key=lambda item: item[1]["size"])
    return picked[:maxCount]


def crawlRepoList(client, args, outDir, metaDir):
    names = [item["repo"] if isinstance(item, dict) else item for item in json.loads(Path(args.repo_list).read_text())]
    shardIndex, shardCount = map(int, args.shard.split("/"))
    nDownloaded = 0
    for fullName in names[shardIndex::shardCount]:
        if shutil.disk_usage(outDir).free < args.min_free_gb * 1e9:
            print("stopping: low disk space", flush=True)
            break
        cachePath = metaDir / f"{safeName(fullName)}.json"
        if cachePath.exists():
            cached = json.loads(cachePath.read_text())
            repo, tree = cached.get("repo"), cached
        else:
            repo, tree = repoFromName(client, fullName)
            if repo is None:
                continue
            cachePath.write_text(json.dumps({**tree, "repo": repo}))
        if repo is None:
            continue
        cadFiles = [entry for entry in tree["tree"] if entry["path"].lower().endswith(cadExtensions)]
        chosen = chooseFiles(cadFiles, args.files_per_repo)
        print(f"{fullName}: {len(cadFiles)} CAD files, fetching {len(chosen)}", flush=True)
        for entry in chosen:
            if nDownloaded >= args.max_files:
                break
            try:
                if downloadFile(client, repo, entry, outDir):
                    nDownloaded += 1
            except Exception as error:
                print(f"  failed {entry['path']}: {error}", flush=True)
        if nDownloaded >= args.max_files:
            break
    print(f"done: {nDownloaded} files downloaded", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/raw/github")
    parser.add_argument("--log", default="data/sources.jsonl")
    parser.add_argument("--max-repos", type=int, default=60)
    parser.add_argument("--max-files", type=int, default=600)
    parser.add_argument("--pages", type=int, default=1, help="search result pages (100 repos each) per query")
    parser.add_argument("--shard", default="0/1", help="k/N: crawl every Nth CAD repo starting at k, for parallel runs")
    parser.add_argument("--search-only", action="store_true", help="refresh the repo list and stop")
    parser.add_argument("--min-free-gb", type=float, default=6.0, help="stop downloading below this much free disk")
    parser.add_argument("--files-per-repo", type=int, default=maxFilesPerRepo)
    parser.add_argument("--repo-list", help="JSON list of owner/name (or {repo: ...}) to crawl via git instead of API search")
    parser.add_argument("--big-assemblies", type=int, default=0,
                        help="instead of crawling, fetch up to N large mechanism/robot STEPs from cached trees")
    args = parser.parse_args()

    outDir = Path(args.out)
    metaDir = outDir / "_meta"
    metaDir.mkdir(parents=True, exist_ok=True)
    client = PoliteSession(args.log, minInterval=1.5)

    if args.repo_list:
        crawlRepoList(client, args, outDir, metaDir)
        return

    reposPath = metaDir / "_repos.json"
    cached = json.loads(reposPath.read_text()) if reposPath.exists() else {"queries": [], "repos": []}
    if isinstance(cached, list):  # first-version cache: a bare list from the original queries
        cached = {"queries": searchQueries[:17], "repos": cached}
    newQueries = [query for query in searchQueries if query not in cached["queries"]]
    if newQueries:
        knownNames = {repo["full_name"] for repo in cached["repos"]}
        cached["repos"] += [repo for repo in searchRepos(client, newQueries, pages=args.pages) if repo["full_name"] not in knownNames]
        cached["queries"] += newQueries
        reposPath.write_text(json.dumps(cached))
    repos = cached["repos"]
    cadRepos = [repo for repo in repos if looksLikeCadRepo(repo)]
    cadRepos.sort(key=lambda repo: (-repo.get("stargazers_count", 0), -repo.get("size", 0)))
    print(f"{len(cadRepos)} CAD-looking repos of {len(repos)} found", flush=True)
    if args.search_only:
        return
    shardIndex, shardCount = map(int, args.shard.split("/"))

    if args.big_assemblies:
        global maxStepBytes
        picked = chooseBigFiles(cadRepos, metaDir, args.big_assemblies)
        maxStepBytes = maxBigStepBytes
        for repo, entry in picked:
            print(f"{repo['full_name']}: {entry['path']} ({entry['size'] / 1e6:.1f} MB)", flush=True)
            try:
                downloadFile(client, repo, entry, outDir / "_big")
            except Exception as error:
                print(f"  failed: {error}", flush=True)
        return

    nDownloaded = 0
    for repo in cadRepos[: args.max_repos][shardIndex::shardCount]:
        if shutil.disk_usage(outDir).free < args.min_free_gb * 1e9:
            print("stopping: low disk space", flush=True)
            break
        cadFiles = listCadFiles(client, repo, metaDir)
        chosen = chooseFiles(cadFiles)
        print(f"{repo['full_name']}: {len(cadFiles)} CAD files, fetching {len(chosen)}", flush=True)
        for entry in chosen:
            if nDownloaded >= args.max_files:
                break
            try:
                if downloadFile(client, repo, entry, outDir):
                    nDownloaded += 1
            except Exception as error:  # keep crawling past one bad file
                print(f"  failed {entry['path']}: {error}", flush=True)
        if nDownloaded >= args.max_files:
            break
    print(f"done: {nDownloaded} files downloaded", flush=True)


if __name__ == "__main__":
    main()
