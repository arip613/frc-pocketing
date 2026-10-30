"""HTTP client that checks robots.txt, rate-limits per host, and logs every source.

Every fetch that produces data goes into a JSONL source log so the dataset can be
audited later (where it came from, its license, and whether training on it is allowed).
"""

import hashlib
import json
import time
import urllib.robotparser
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

userAgent = "frcpocket-research/0.1 (local educational FRC tool; rate-limited)"
defaultMinInterval = 2.0  # seconds between requests to the same host
defaultTimeout = 60


def utcNow():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parseContentSignals(robotsText):
    """Read Cloudflare-style 'Content-Signal' lines for the '*' user agent.

    Returns e.g. {'search': 'yes', 'ai-train': 'no', 'use': 'reference'}.
    """
    signals = {}
    currentAgents = []
    for rawLine in robotsText.splitlines():
        line = rawLine.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        if key.lower() == "user-agent":
            currentAgents = [value]
        elif key.lower() == "content-signal" and "*" in currentAgents:
            for pair in value.split(","):
                if "=" in pair:
                    name, setting = pair.split("=", 1)
                    signals[name.strip().lower()] = setting.strip().lower()
    return signals


class PoliteSession:
    def __init__(self, logPath, minInterval=defaultMinInterval):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = userAgent
        self.minInterval = minInterval
        self.lastRequestTime = {}
        self.robotParsers = {}
        self.contentSignals = {}
        self.logPath = Path(logPath)
        self.logPath.parent.mkdir(parents=True, exist_ok=True)

    def _loadRobots(self, host, scheme):
        if host in self.robotParsers:
            return
        parser = urllib.robotparser.RobotFileParser()
        robotsUrl = f"{scheme}://{host}/robots.txt"
        self._waitTurn(host)
        try:
            response = self.session.get(robotsUrl, timeout=defaultTimeout)
            if response.status_code == 200 and "text/plain" in response.headers.get("Content-Type", ""):
                parser.parse(response.text.splitlines())
                self.contentSignals[host] = parseContentSignals(response.text)
            else:
                parser.parse([])  # no robots.txt means no restrictions
                self.contentSignals[host] = {}
        except requests.RequestException:
            parser.parse([])
            self.contentSignals[host] = {}
        self.robotParsers[host] = parser

    def _waitTurn(self, host):
        elapsed = time.monotonic() - self.lastRequestTime.get(host, 0.0)
        if elapsed < self.minInterval:
            time.sleep(self.minInterval - elapsed)
        self.lastRequestTime[host] = time.monotonic()

    def isAllowed(self, url):
        parts = urlparse(url)
        self._loadRobots(parts.netloc, parts.scheme)
        return self.robotParsers[parts.netloc].can_fetch(userAgent, url)

    def trainingAllowed(self, url):
        """False when the site's robots.txt says ai-train=no."""
        host = urlparse(url).netloc
        self._loadRobots(host, urlparse(url).scheme)
        return self.contentSignals.get(host, {}).get("ai-train") != "no"

    def get(self, url, maxRetries=4, **kwargs):
        if not self.isAllowed(url):
            raise PermissionError(f"robots.txt disallows {url}")
        host = urlparse(url).netloc
        for attempt in range(maxRetries):
            self._waitTurn(host)
            response = self.session.get(url, timeout=defaultTimeout, **kwargs)
            if response.status_code in (429, 502, 503) or (
                response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0"
            ):
                retryAfter = response.headers.get("Retry-After")
                resetAt = response.headers.get("X-RateLimit-Reset")
                if retryAfter:
                    pause = float(retryAfter)
                elif resetAt:
                    pause = max(5.0, float(resetAt) - time.time() + 2)
                else:
                    pause = 30.0 * (attempt + 1)
                print(f"  rate limited by {host}; sleeping {pause:.0f}s", flush=True)
                time.sleep(min(pause, 3700))
                continue
            return response
        return response

    def download(self, url, destPath, maxBytes):
        """Stream url to destPath. Returns (path, sha256, nBytes) or None if too big/failed."""
        if not self.isAllowed(url):
            raise PermissionError(f"robots.txt disallows {url}")
        host = urlparse(url).netloc
        self._waitTurn(host)
        destPath = Path(destPath)
        destPath.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        nBytes = 0
        with self.session.get(url, timeout=defaultTimeout, stream=True) as response:
            if response.status_code != 200:
                return None
            with open(destPath, "wb") as handle:
                for chunk in response.iter_content(1 << 16):
                    nBytes += len(chunk)
                    if nBytes > maxBytes:
                        handle.close()
                        destPath.unlink(missing_ok=True)
                        return None
                    digest.update(chunk)
                    handle.write(chunk)
        return destPath, digest.hexdigest(), nBytes

    def logSource(self, **record):
        record.setdefault("timestamp", utcNow())
        with open(self.logPath, "a") as handle:
            handle.write(json.dumps(record) + "\n")
