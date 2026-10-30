"""Collect Chief Delphi pocketing discussions through the Discourse JSON API.

Chief Delphi's robots.txt carries 'Content-Signal: ai-train=no, use=reference', so
everything collected here is logged with trainingAllowed=False. It feeds the
human-readable design notes and the reference gallery only, never model training.

Usage:
    python -I -m frcpocket.scrape.chiefdelphi --out data/raw/chiefdelphi
"""

import argparse
import html
import json
import re
from pathlib import Path
from urllib.parse import quote, urlparse

from frcpocket.scrape.polite import PoliteSession

siteRoot = "https://www.chiefdelphi.com"
searchQueries = [
    "pocketing in:title",
    "pocket in:title",
    "pocketing order:likes",
    "pocketing pattern",
    "pocketing plates",
    "pocketing gusset",
    "pocketing tips",
    "pocketing strength",
    "pocketing featurescript",
    "lightening pattern",
    "lightening holes in:title",
    "truss pocketing",
    "triangle pocketing",
    "weight reduction pocketing",
    "pocketing router",
    "rib thickness pocket",
]
relevantWords = ("pocket", "lighten", "lightening", "weight reduction", "truss")
maxTopics = 25
maxPosts = 120
maxImages = 40
imageExtensions = (".png", ".jpg", ".jpeg", ".webp")


def cleanCooked(cookedHtml):
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", cookedHtml, flags=re.S)
    text = re.sub(r"<br\s*/?>|</p>|</li>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n", text)).strip()


def imageUrls(cookedHtml):
    urls = re.findall(r'<img[^>]+src="([^"]+)"', cookedHtml)
    keep = []
    for url in urls:
        if url.startswith("//"):
            url = "https:" + url
        if "emoji" in url or "avatar" in url or "user_avatar" in url:
            continue
        if urlparse(url).path.lower().endswith(imageExtensions):
            keep.append(url)
    return keep


releaseQueries = [
    "CAD release in:title",
    "CAD and code release in:title",
    "CAD release onshape order:likes",
    "technical binder CAD release",
    "open alliance CAD onshape",
    "WCP product release in:title",
    "Thrifty Bot product release",
    "REV Robotics product release",
    "Swerve Drive Specialties release",
    "gearbox release onshape",
]
onshapeDocPattern = re.compile(r"https?://cad\.onshape\.com/documents/([0-9a-f]{24})[^\s\"'<>)]*")


def harvestCadReleaseLinks(client, outDir, maxTopics=60):
    """Public Onshape document links from CAD/product release threads.

    Only the links are kept, as a reading list for manual export. No post text is
    stored, which stays within Chief Delphi's use=reference signal.
    """
    topicsById = {}
    for query in releaseQueries:
        response = client.get(f"{siteRoot}/search.json?q={quote(query)}")
        if response.status_code != 200:
            continue
        for topic in response.json().get("topics", []):
            topicsById.setdefault(topic["id"], topic)
        print(f"search '{query}': {len(topicsById)} topics", flush=True)
    ranked = sorted(topicsById.values(), key=lambda topic: -topic.get("like_count", 0))[:maxTopics]
    releases = []
    for topic in ranked:
        response = client.get(f"{siteRoot}/t/{topic['id']}.json")
        if response.status_code != 200:
            continue
        data = response.json()
        posts = data.get("post_stream", {}).get("posts", [])[:3]  # releases link their CAD up top
        links = {}
        for post in posts:
            for match in onshapeDocPattern.finditer(post.get("cooked", "")):
                links.setdefault(match.group(1), html.unescape(match.group(0)))
        topicUrl = f"{siteRoot}/t/{data.get('slug', topic['id'])}/{topic['id']}"
        if links:
            releases.append({
                "title": data.get("title"),
                "topicUrl": topicUrl,
                "likes": data.get("like_count", topic.get("like_count", 0)),
                "onshapeUrls": list(links.values()),
            })
        client.logSource(
            source="chiefdelphi-links",
            url=topicUrl,
            title=data.get("title"),
            nOnshapeLinks=len(links),
            license="Chief Delphi terms; robots.txt content-signal ai-train=no, use=reference",
            trainingAllowed=False,
            notes="only Onshape links kept, as a manual-export reading list",
        )
        print(f"  {data.get('title')}: {len(links)} Onshape links", flush=True)
    releases.sort(key=lambda release: -release["likes"])
    (outDir / "onshapeReleaseLinks.json").write_text(json.dumps(releases, indent=1))
    return releases


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/raw/chiefdelphi")
    parser.add_argument("--log", default="data/sources.jsonl")
    parser.add_argument("--max-topics", type=int, default=maxTopics)
    parser.add_argument("--max-images", type=int, default=maxImages)
    parser.add_argument("--cad-release-links", action="store_true",
                        help="only collect public Onshape links from CAD/product release threads")
    args = parser.parse_args()

    outDir = Path(args.out)
    if args.cad_release_links:
        outDir.mkdir(parents=True, exist_ok=True)
        releases = harvestCadReleaseLinks(PoliteSession(args.log, minInterval=3.0), outDir)
        print(f"done: {len(releases)} release threads with Onshape links", flush=True)
        return
    (outDir / "topics").mkdir(parents=True, exist_ok=True)
    (outDir / "images").mkdir(parents=True, exist_ok=True)
    client = PoliteSession(args.log, minInterval=3.0)
    trainingAllowed = client.trainingAllowed(siteRoot + "/")
    print(f"robots.txt content signals: {client.contentSignals.get(urlparse(siteRoot).netloc)}", flush=True)

    topicsById = {}
    postHits = {}
    for query in searchQueries:
        response = client.get(f"{siteRoot}/search.json?q={quote(query)}")
        if response.status_code != 200:
            print(f"search '{query}' failed: {response.status_code}", flush=True)
            continue
        result = response.json()
        for topic in result.get("topics", []):
            topicsById.setdefault(topic["id"], topic)
        for post in result.get("posts", []):
            if any(word in (post.get("blurb") or "").lower() for word in relevantWords):
                postHits.setdefault(post["id"], post)
        print(f"search '{query}': {len(topicsById)} topics, {len(postHits)} matching posts", flush=True)

    hitsPerTopic = {}
    for post in postHits.values():
        hitsPerTopic[post["topic_id"]] = hitsPerTopic.get(post["topic_id"], 0) + 1

    def topicRelevance(topic):
        titleHit = any(word in (topic.get("title") or "").lower() for word in relevantWords)
        return (titleHit, hitsPerTopic.get(topic["id"], 0), topic.get("like_count", 0))

    rankedTopics = [
        topic for topic in sorted(topicsById.values(), key=topicRelevance, reverse=True)
        if topicRelevance(topic)[0] or topicRelevance(topic)[1] >= 2
    ][: args.max_topics]
    fullTopicIds = {topic["id"] for topic in rankedTopics}

    corpus = []
    pendingImages = []

    def addPost(post, title, topicUrl):
        text = cleanCooked(post.get("cooked", ""))
        images = imageUrls(post.get("cooked", ""))
        isRelevant = any(word in text.lower() for word in relevantWords)
        corpus.append({
            "topicId": post.get("topic_id"),
            "title": title,
            "postNumber": post.get("post_number"),
            "likes": sum(
                action.get("count", 0) for action in post.get("actions_summary", []) if action.get("id") == 2
            ) or post.get("like_count", 0),
            "relevant": isRelevant,
            "text": text,
            "images": images,
            "url": f"{topicUrl}/{post.get('post_number')}",
        })
        if isRelevant:
            pendingImages.extend((url, post.get("topic_id"), post.get("post_number")) for url in images)

    for topic in rankedTopics:
        url = f"{siteRoot}/t/{topic['id']}.json"
        response = client.get(url)
        if response.status_code != 200:
            continue
        data = response.json()
        (outDir / "topics" / f"{topic['id']}.json").write_text(json.dumps(data))
        posts = data.get("post_stream", {}).get("posts", [])
        topicUrl = f"{siteRoot}/t/{data.get('slug', topic['id'])}/{topic['id']}"
        client.logSource(
            source="chiefdelphi",
            url=topicUrl,
            apiUrl=url,
            title=data.get("title"),
            localPath=str(outDir / "topics" / f"{topic['id']}.json"),
            nPosts=len(posts),
            license="Chief Delphi terms; robots.txt content-signal ai-train=no, use=reference",
            trainingAllowed=trainingAllowed,
        )
        for post in posts:
            addPost(post, data.get("title"), topicUrl)
        print(f"topic {topic['id']} '{data.get('title')}': {len(posts)} posts", flush=True)

    # Individual matching posts from threads that are mostly about something else.
    strayPosts = sorted(
        (post for post in postHits.values() if post["topic_id"] not in fullTopicIds),
        key=lambda post: -post.get("like_count", 0),
    )[:maxPosts]
    for hit in strayPosts:
        url = f"{siteRoot}/posts/{hit['id']}.json"
        response = client.get(url)
        if response.status_code != 200:
            continue
        post = response.json()
        topic = topicsById.get(hit["topic_id"], {})
        topicUrl = f"{siteRoot}/t/{topic.get('slug', hit['topic_id'])}/{hit['topic_id']}"
        client.logSource(
            source="chiefdelphi",
            url=f"{topicUrl}/{post.get('post_number')}",
            apiUrl=url,
            title=topic.get("title"),
            license="Chief Delphi terms; robots.txt content-signal ai-train=no, use=reference",
            trainingAllowed=trainingAllowed,
        )
        addPost(post, topic.get("title"), topicUrl)
    print(f"{len(strayPosts)} individual matching posts", flush=True)

    (outDir / "corpus.json").write_text(json.dumps(corpus, indent=1))

    # A small reference gallery for side-by-side previews (not for training).
    nImages = 0
    for imageUrl, topicId, postNumber in pendingImages:
        if nImages >= args.max_images:
            break
        suffix = Path(urlparse(imageUrl).path).suffix.lower()
        destPath = outDir / "images" / f"{topicId}_{postNumber}_{nImages}{suffix}"
        try:
            result = client.download(imageUrl, destPath, maxBytes=4 * 1024 * 1024)
        except PermissionError:
            continue
        if result is None:
            continue
        client.logSource(
            source="chiefdelphi-image",
            url=imageUrl,
            postUrl=f"{siteRoot}/t/{topicId}/{postNumber}",
            localPath=str(destPath),
            sha256=result[1],
            bytes=result[2],
            license="Chief Delphi terms; robots.txt content-signal ai-train=no, use=reference",
            trainingAllowed=trainingAllowed,
        )
        nImages += 1
    print(f"done: {len(rankedTopics)} topics, {len(corpus)} posts, {nImages} images", flush=True)


if __name__ == "__main__":
    main()
