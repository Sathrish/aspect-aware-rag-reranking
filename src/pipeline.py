"""Shared setup: seeding, model/index loading and per-query ranking for every system."""
import functools
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from . import config
from .aspects import extract_aspects
from .data import build_doc_texts, load_nfcorpus
from .rerankers import aspect_aware_rerank, cross_encoder_rerank, cross_encoder_scores
from .retrievers import BM25Retriever, DenseRetriever, encode_documents

FIRST_STAGES = ("BM25", "Dense")
RERANKERS = ("CrossEncoder", "Aspect-Aware")


def system_names(first_stages=FIRST_STAGES):
    names = []
    for stage in first_stages:
        names += [stage] + [f"{stage} + {r}" for r in RERANKERS]
    return names


def set_seed(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class Resources:
    doc_ids: list
    doc_texts: list
    queries: dict
    qrels: dict
    doc_embs: np.ndarray
    bi_encoder: object
    cross_encoder: object
    bm25: BM25Retriever
    dense: DenseRetriever
    cache_dir: Path
    _ce_cache: dict = field(default_factory=dict)

    @functools.cached_property
    def doc_index(self):
        return {doc_id: i for i, doc_id in enumerate(self.doc_ids)}

    # ---- cross-encoder score cache (avoids re-scoring between experiments) ----
    @property
    def _ce_cache_path(self):
        return self.cache_dir / "ce_scores.json"

    def load_ce_cache(self):
        if self._ce_cache_path.exists():
            self._ce_cache = json.loads(self._ce_cache_path.read_text())

    def save_ce_cache(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._ce_cache_path.write_text(json.dumps(self._ce_cache))

    def ce_scores(self, query, cand_idx):
        key = query + "|" + ",".join(str(int(i)) for i in cand_idx)
        if key not in self._ce_cache:
            texts = [self.doc_texts[i] for i in cand_idx]
            self._ce_cache[key] = cross_encoder_scores(self.cross_encoder, query, texts).tolist()
        return np.asarray(self._ce_cache[key], dtype=np.float32)

    # ---- encoding ----
    def encode_query_and_aspects(self, query, max_aspects=config.MAX_ASPECTS):
        """One bi-encoder call for the query and its aspects (all L2-normalised)."""
        aspects = extract_aspects(query, max_aspects=max_aspects)
        embs = self.bi_encoder.encode(
            [query] + aspects, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        ).astype(np.float32)
        return embs[0], aspects, embs[1:]

    def first_stage(self, stage, query, query_emb, k):
        if stage == "BM25":
            return self.bm25.search(query, k)
        if stage == "Dense":
            return self.dense.search(query_emb, k)
        raise ValueError(f"Unknown first stage: {stage}")


def build_resources(data_dir=config.DATA_DIR, cache_dir=config.CACHE_DIR, device=None):
    """Load NFCorpus, both encoders and both first-stage indexes."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    corpus, queries, qrels = load_nfcorpus(data_dir, split="test")
    doc_ids, doc_texts = build_doc_texts(corpus)
    print(f"NFCorpus test split: {len(doc_ids)} documents, {len(queries)} queries | device: {device}")

    bi_encoder = SentenceTransformer(config.BI_ENCODER, device=device)
    cross_encoder = CrossEncoder(config.CROSS_ENCODER, max_length=512, device=device)
    doc_embs = encode_documents(bi_encoder, doc_texts, Path(cache_dir) / "doc_embeddings.npy")

    res = Resources(
        doc_ids=doc_ids, doc_texts=doc_texts, queries=queries, qrels=qrels,
        doc_embs=doc_embs, bi_encoder=bi_encoder, cross_encoder=cross_encoder,
        bm25=BM25Retriever(doc_texts), dense=DenseRetriever(doc_embs), cache_dir=Path(cache_dir),
    )
    res.load_ce_cache()
    return res


def rank_all_systems(
    res: Resources,
    query: str,
    top_k=config.TOP_K,
    n_candidates=config.N_CANDIDATES,
    lambda_param=config.LAMBDA,
    threshold=config.ASPECT_SIM_THRESHOLD,
    first_stages=FIRST_STAGES,
):
    """Rank one query with every system.

    For each first stage (BM25, Dense) the same top-``n_candidates`` pool is
    (a) truncated to ``top_k``, (b) re-ranked by the cross-encoder and
    (c) re-ranked by the Aspect-Aware re-ranker, so the two re-rankers are
    always compared on identical candidates.

    Returns ``(rankings, aspects, aspect_embs)`` where ``rankings`` maps a
    system name to a list of corpus indices.
    """
    query_emb, aspects, aspect_embs = res.encode_query_and_aspects(query)
    rankings = {}
    for stage in first_stages:
        cand = res.first_stage(stage, query, query_emb, n_candidates)
        ce = res.ce_scores(query, cand)
        rankings[stage] = [int(i) for i in cand[:top_k]]
        rankings[f"{stage} + CrossEncoder"] = [int(cand[p]) for p in cross_encoder_rerank(ce, top_k)]
        aa = aspect_aware_rerank(ce, res.doc_embs[cand], aspect_embs, top_k, lambda_param, threshold)
        rankings[f"{stage} + Aspect-Aware"] = [int(cand[p]) for p in aa]
    return rankings, aspects, aspect_embs


def select_queries(res: Resources, num_queries=None, seed=config.SEED):
    """All test queries with relevance labels, or a seeded random sample of them."""
    qids = [q for q in res.queries if res.qrels.get(q)]
    if num_queries is not None and num_queries < len(qids):
        rng = np.random.default_rng(seed)
        keep = set(rng.choice(len(qids), size=num_queries, replace=False).tolist())
        qids = [q for i, q in enumerate(qids) if i in keep]
    return qids
