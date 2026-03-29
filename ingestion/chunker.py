"""
Text chunker with content cleaning for Kubeflow documentation.

Strips Hugo frontmatter, template syntax, HTML tags, and navigation artifacts.
Same cleaning logic as pipelines/kubeflow-pipeline.py but extracted to a reusable util.
"""
import os
import re
import json
import logging
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))


def clean_content(text: str) -> str:
    """
    Clean raw markdown/HTML from Kubeflow docs.

    Removes Hugo frontmatter, shortcodes, HTML tags, excessive whitespace,
    and navigation boilerplate.
    """
    # strip Hugo YAML frontmatter (--- delimited)
    text = re.sub(r'^---\s*\n.*?\n---\s*\n', '', text, flags=re.DOTALL)

    # strip Hugo TOML frontmatter (+++ delimited)
    text = re.sub(r'^\+\+\+\s*\n.*?\n\+\+\+\s*\n', '', text, flags=re.DOTALL)

    # remove Hugo shortcodes: {{< ... >}}, {{% ... %}}
    text = re.sub(r'\{\{[<%].*?[>%]\}\}', '', text, flags=re.DOTALL)

    # remove HTML tags but keep content
    text = re.sub(r'<[^>]+>', ' ', text)

    # remove HTML entities
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)

    # remove navigation artifacts (breadcrumbs, sidebar refs)
    text = re.sub(r'\[.*?\]\(#.*?\)', '', text)

    # collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()


def build_citation_url(file_path: str, base_url: str = "https://www.kubeflow.org/docs/") -> str:
    """
    Convert a GitHub file path to a kubeflow.org documentation URL.

    Example: content/en/docs/started/installing.md -> https://www.kubeflow.org/docs/started/installing/
    """
    # strip the content/en/docs/ prefix
    path = re.sub(r'^content/en/docs/?', '', file_path)
    # strip file extension
    path = re.sub(r'\.(md|html)$', '', path)
    # remove _index suffix (Hugo section pages)
    path = re.sub(r'/_index$', '', path)
    # ensure trailing slash
    if path and not path.endswith('/'):
        path += '/'
    return base_url + path


def chunk_documents(input_path: str, output_path: str = "data/chunked_docs.jsonl") -> str:
    """
    Read raw JSONL docs, clean content, split into chunks, and write to JSONL.

    Each output record includes: file_path, file_unique_id, content_text,
    citation_url, chunk_index, source_type.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""],
    )

    total_chunks = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            if not line.strip():
                continue

            doc = json.loads(line)
            content = clean_content(doc["content"])

            if len(content) < 50:
                continue

            citation_url = build_citation_url(doc["path"])
            chunks = splitter.split_text(content)

            for idx, chunk_text in enumerate(chunks):
                if len(chunk_text.strip()) < 20:
                    continue

                file_unique_id = f"{REPO_OWNER}/{REPO_NAME}:{doc['path']}:{idx}"
                record = {
                    "file_unique_id": file_unique_id,
                    "file_path": doc["path"],
                    "citation_url": citation_url,
                    "content_text": chunk_text,
                    "chunk_index": idx,
                    "source_type": "docs",
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

    logger.info("Chunking complete: %d chunks written to %s", total_chunks, output_path)
    return output_path


REPO_OWNER = "kubeflow"
REPO_NAME = "website"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    chunk_documents("data/raw_docs.jsonl")
