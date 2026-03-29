"""
Scraper for Kubeflow documentation from the kubeflow/website GitHub repo.

Uses GitHub API to fetch markdown files from content/en/docs/.
Handles pagination and rate limiting with exponential backoff.
"""
import os
import json
import time
import base64
import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
REPO_OWNER = "kubeflow"
REPO_NAME = "website"
DOCS_PATH = "content/en/docs"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

MAX_RETRIES = 5
BASE_DELAY = 1.0


def _headers() -> dict:
    h = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"token {GITHUB_TOKEN}"
    return h


def _request_with_backoff(url: str, params: dict = None) -> requests.Response:
    """GET with exponential backoff on rate limits and transient errors."""
    for attempt in range(MAX_RETRIES):
        resp = requests.get(url, headers=_headers(), params=params, timeout=30)

        if resp.status_code == 200:
            return resp

        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            # check reset header for precise wait
            reset_ts = int(resp.headers.get("X-RateLimit-Reset", 0))
            wait = max(reset_ts - int(time.time()), BASE_DELAY * (2 ** attempt))
            logger.warning("Rate limited. Waiting %.1fs (attempt %d/%d)", wait, attempt + 1, MAX_RETRIES)
            time.sleep(wait)
            continue

        if resp.status_code >= 500:
            wait = BASE_DELAY * (2 ** attempt)
            logger.warning("Server error %d. Retrying in %.1fs", resp.status_code, wait)
            time.sleep(wait)
            continue

        resp.raise_for_status()

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {url}")


def _fetch_tree(path: str) -> list[dict]:
    """Recursively fetch all markdown files under a directory path."""
    url = f"{GITHUB_API}/repos/{REPO_OWNER}/{REPO_NAME}/contents/{path}"
    resp = _request_with_backoff(url)
    items = resp.json()

    if not isinstance(items, list):
        logger.warning("Expected list from %s, got %s", path, type(items).__name__)
        return []

    results = []
    for item in items:
        if item["type"] == "dir":
            results.extend(_fetch_tree(item["path"]))
        elif item["type"] == "file" and item["name"].endswith((".md", ".html")):
            results.append(item)

    return results


def _fetch_content(file_info: dict) -> str:
    """Download and decode a file's content from GitHub."""
    # prefer download_url for raw content (no base64 overhead)
    download_url = file_info.get("download_url")
    if download_url:
        resp = _request_with_backoff(download_url)
        return resp.text

    # fallback: use API endpoint and decode base64
    url = file_info["url"]
    resp = _request_with_backoff(url)
    data = resp.json()
    if data.get("encoding") == "base64":
        return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    return data.get("content", "")


def scrape_docs(output_path: str = "data/raw_docs.jsonl") -> str:
    """
    Scrape kubeflow.org docs from GitHub and write to JSONL.

    Returns the output file path.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching file tree from %s/%s/%s", REPO_OWNER, REPO_NAME, DOCS_PATH)
    files = _fetch_tree(DOCS_PATH)
    logger.info("Found %d documentation files", len(files))

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for file_info in files:
            try:
                content = _fetch_content(file_info)
                if not content.strip():
                    continue

                record = {
                    "path": file_info["path"],
                    "file_name": file_info["name"],
                    "sha": file_info["sha"],
                    "content": content,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

                if count % 25 == 0:
                    logger.info("Scraped %d/%d files", count, len(files))

            except Exception as e:
                logger.error("Failed to fetch %s: %s", file_info.get("path", "?"), e)

    logger.info("Scraping complete: %d files written to %s", count, output_path)
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    scrape_docs()
