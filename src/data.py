"""NFCorpus (BEIR) download and loading. The dataset is never stored in the repo."""
import csv
import json
import urllib.request
import zipfile
from pathlib import Path

from . import config


def download_nfcorpus(data_dir: Path = config.DATA_DIR) -> Path:
    """Download and unzip NFCorpus into ``data_dir``."""
    data_dir = Path(data_dir)
    target = data_dir / "nfcorpus"

    # If NFCorpus is already extracted, use it.
    if (target / "corpus.jsonl").exists():
        print(f"Using existing NFCorpus directory: {target}")
        return target

    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "nfcorpus.zip"

    # If the ZIP was downloaded manually, use it instead of downloading again.
    if zip_path.exists():
        print(f"Using existing NFCorpus archive: {zip_path}")
    else:
        print(f"Downloading NFCorpus from {config.NFCORPUS_URL}")
        urllib.request.urlretrieve(config.NFCORPUS_URL, zip_path)

    # Extract the dataset.
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)

    # Remove the ZIP after successful extraction.
    zip_path.unlink()

    return target


def load_nfcorpus(data_dir: Path = config.DATA_DIR, split: str = "test"):
    """Return ``(corpus, queries, qrels)`` for one split.

    corpus  : {doc_id: {"title": str, "text": str}}   (corpus.jsonl order)
    queries : {query_id: str}   only queries with at least one relevant document
    qrels   : {query_id: {doc_id: grade}}   grade 2 = highly relevant, 1 = relevant
    """
    path = download_nfcorpus(data_dir)

    corpus = {}
    with open(path / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            corpus[row["_id"]] = {"title": row.get("title", ""), "text": row.get("text", "")}

    qrels = {}
    with open(path / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header: query-id, corpus-id, score
        for qid, did, score in reader:
            if int(score) > 0 and did in corpus:
                qrels.setdefault(qid, {})[did] = int(score)

    queries = {}
    with open(path / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["_id"] in qrels:
                queries[row["_id"]] = row["text"]

    return corpus, queries, qrels


def build_doc_texts(corpus):
    """Each document is represented as ``title + ". " + text``."""
    doc_ids = list(corpus.keys())
    doc_texts = [f"{corpus[d]['title']}. {corpus[d]['text']}" for d in doc_ids]
    return doc_ids, doc_texts


def load_rag_qa(path: Path = config.RAG_QA_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
