"""The three experiments. Each returns pandas DataFrames; the scripts save them."""
import numpy as np
import pandas as pd

from . import config
from .metrics import (
    aspect_coverage_score, diversity_score, ndcg_at_k, paired_randomization_test,
    precision_at_k, reciprocal_rank,
)
from .pipeline import Resources, rank_all_systems, select_queries, system_names
from .rerankers import aspect_aware_rerank

METRICS = ["NDCG@5", "P@5", "MRR", "ACS", "Diversity"]


def _relevance(res: Resources, qid):
    """Relevance grades of one query, keyed by corpus index."""
    idx = res.doc_index
    return {idx[d]: g for d, g in res.qrels[qid].items() if d in idx}


def _score(res, ranked, relevant, aspect_embs, top_k, threshold):
    embs = res.doc_embs[ranked]
    return {
        f"NDCG@{top_k}": ndcg_at_k(ranked, relevant, top_k),
        f"P@{top_k}": precision_at_k(ranked, relevant, top_k),
        "MRR": reciprocal_rank(ranked, relevant),
        "ACS": aspect_coverage_score(aspect_embs, embs, threshold),
        "Diversity": diversity_score(embs),
    }


def run_retrieval_eval(
    res: Resources, num_queries=None, top_k=config.TOP_K, n_candidates=config.N_CANDIDATES,
    lambda_param=config.LAMBDA, threshold=config.ASPECT_SIM_THRESHOLD, seed=config.SEED,
):
    """Experiment 1: six systems (2 first stages x {none, CrossEncoder, Aspect-Aware}).

    Returns ``(summary, per_query, significance)``.
    """
    qids = select_queries(res, num_queries, seed)
    rows = []
    for n, qid in enumerate(qids, start=1):
        query = res.queries[qid]
        rankings, aspects, aspect_embs = rank_all_systems(
            res, query, top_k, n_candidates, lambda_param, threshold)
        relevant = _relevance(res, qid)
        for system, ranked in rankings.items():
            rows.append({"query_id": qid, "system": system, "n_aspects": len(aspects),
                         **_score(res, ranked, relevant, aspect_embs, top_k, threshold)})
        if n % 25 == 0 or n == len(qids):
            print(f"  evaluated {n}/{len(qids)} queries", flush=True)
    per_query = pd.DataFrame(rows)
    print(f"Queries evaluated: {per_query['query_id'].nunique()}")

    metric_cols = [c for c in per_query.columns if c not in ("query_id", "system", "n_aspects")]
    summary = (per_query.groupby("system")[metric_cols].mean()
               .reindex(system_names()).reset_index().rename(columns={"system": "System"}))
    summary.insert(1, "Queries", per_query["query_id"].nunique())

    # Paired tests: Aspect-Aware vs CrossEncoder and vs its first stage, same pool.
    sig = []
    wide = per_query.set_index(["query_id", "system"])
    for stage in ("BM25", "Dense"):
        a_name = f"{stage} + Aspect-Aware"
        for b_name in (f"{stage} + CrossEncoder", stage):
            for metric in metric_cols:
                a = wide.xs(a_name, level="system")[metric]
                b = wide.xs(b_name, level="system")[metric].reindex(a.index)
                diff, p = paired_randomization_test(a.values, b.values, seed=seed)
                sig.append({"System A": a_name, "System B": b_name, "Metric": metric,
                            "Mean diff (A - B)": diff, "p-value": p})
    return summary, per_query, pd.DataFrame(sig)


def run_lambda_sweep(
    res: Resources, lambdas=config.LAMBDA_GRID, num_queries=None, first_stage="Dense",
    top_k=config.TOP_K, n_candidates=config.N_CANDIDATES,
    threshold=config.ASPECT_SIM_THRESHOLD, seed=config.SEED,
):
    """Experiment 2: sweep the relevance weight of the Aspect-Aware re-ranker."""
    qids = select_queries(res, num_queries, seed)
    collected = {lam: [] for lam in lambdas}
    for qid in qids:
        query = res.queries[qid]
        query_emb, _, aspect_embs = res.encode_query_and_aspects(query)
        cand = res.first_stage(first_stage, query, query_emb, n_candidates)
        ce = res.ce_scores(query, cand)
        relevant = _relevance(res, qid)
        for lam in lambdas:
            pos = aspect_aware_rerank(ce, res.doc_embs[cand], aspect_embs, top_k, lam, threshold)
            ranked = [int(cand[p]) for p in pos]
            collected[lam].append(_score(res, ranked, relevant, aspect_embs, top_k, threshold))
    print(f"Queries evaluated: {len(qids)}")
    rows = [{"lambda": lam, "Queries": len(qids), **pd.DataFrame(v).mean().to_dict()}
            for lam, v in collected.items()]
    return pd.DataFrame(rows)


RAG_SYSTEMS = ["BM25", "Dense", "Dense + CrossEncoder", "Dense + Aspect-Aware"]


def run_rag_eval(
    res: Resources, generator, qa_pairs, top_k=config.TOP_K, n_candidates=config.N_CANDIDATES,
    lambda_param=config.LAMBDA, threshold=config.ASPECT_SIM_THRESHOLD,
):
    """Experiment 3: end-to-end RAG answers scored with ROUGE-L F1.

    Returns ``(scores, answers, retrieved)``. ``scores`` and ``answers`` have one
    row per query and one column per system; ``retrieved`` has one row per
    (query, system) with the document ids given to the generator, the prompt
    length in tokens, and whether the documents match the cross-encoder's.
    """
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    score_rows, answer_rows, doc_rows = [], [], []
    for n, item in enumerate(qa_pairs, start=1):
        query, reference = item["query"], item["reference"]
        rankings, aspects, _ = rank_all_systems(
            res, query, top_k, n_candidates, lambda_param, threshold)
        ce_ranking = rankings["Dense + CrossEncoder"]
        s_row, a_row = {"Query": query}, {"Query": query, "Reference": reference}
        for system in RAG_SYSTEMS:
            ranked = rankings[system]
            answer = generator.generate(query, [res.doc_texts[i] for i in ranked])
            s_row[system] = scorer.score(reference, answer)["rougeL"].fmeasure
            a_row[system] = answer
            doc_rows.append({
                "Query": query, "System": system, "Aspects": "; ".join(aspects),
                "Doc ids (ranked)": " ".join(res.doc_ids[i] for i in ranked),
                "Prompt tokens": generator.last_prompt_tokens,
                "Same order as CrossEncoder": ranked == ce_ranking,
                "Same set as CrossEncoder": set(ranked) == set(ce_ranking),
            })
        score_rows.append(s_row)
        answer_rows.append(a_row)
        print(f"  generated answers for query {n}/{len(qa_pairs)}", flush=True)
    retrieved = pd.DataFrame(doc_rows)
    print(f"Queries evaluated: {len(score_rows)} | longest prompt: "
          f"{retrieved['Prompt tokens'].max()} tokens (limit {generator.max_input_tokens})")
    return pd.DataFrame(score_rows), pd.DataFrame(answer_rows), retrieved
