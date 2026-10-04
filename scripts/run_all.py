"""Run all three experiments in one process and record the package versions used."""
import platform
from importlib.metadata import version

import pandas as pd
from _common import base_parser, config

from src.data import load_rag_qa
from src.experiments import run_lambda_sweep, run_rag_eval, run_retrieval_eval
from src.generation import AnswerGenerator
from src.pipeline import build_resources, set_seed
from src.plots import plot_lambda, plot_rag, plot_retrieval

PACKAGES = ["torch", "sentence-transformers", "transformers", "faiss-cpu", "rank-bm25",
            "rouge-score", "numpy", "pandas", "matplotlib"]


def main():
    args = base_parser(__doc__).parse_args()
    out, figs = args.results_dir, args.results_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    res = build_resources(args.data_dir, args.data_dir / "cache")

    print("\n[1/3] Retrieval evaluation")
    summary, per_query, significance = run_retrieval_eval(
        res, None, args.top_k, args.candidates, config.LAMBDA, args.threshold, args.seed)
    res.save_ce_cache()
    summary.round(4).to_csv(out / "reranking_results.csv", index=False)
    per_query.round(4).to_csv(out / "reranking_per_query.csv", index=False)
    significance.round(4).to_csv(out / "reranking_significance.csv", index=False)
    plot_retrieval(summary, figs / "retrieval_comparison.png")
    print(summary.round(4).to_string(index=False))

    print("\n[2/3] Lambda sweep")
    sweep = run_lambda_sweep(res, config.LAMBDA_GRID, None, "Dense", args.top_k,
                             args.candidates, args.threshold, args.seed)
    sweep.round(4).to_csv(out / "lambda_analysis.csv", index=False)
    plot_lambda(sweep, figs / "lambda_sensitivity.png")
    print(sweep.round(4).to_string(index=False))

    print("\n[3/3] End-to-end RAG evaluation")
    set_seed(args.seed)
    scores, answers, retrieved = run_rag_eval(res, AnswerGenerator(), load_rag_qa(), args.top_k,
                                   args.candidates, config.LAMBDA, args.threshold)
    res.save_ce_cache()
    plot_rag(scores, figs / "rag_comparison.png")
    average = scores.drop(columns="Query").mean().to_dict()
    table = pd.concat([scores, pd.DataFrame([{"Query": "AVERAGE", **average}])], ignore_index=True)
    table.round(4).to_csv(out / "rag_comparison.csv", index=False)
    answers.to_csv(out / "rag_answers.csv", index=False)
    retrieved.to_csv(out / "rag_retrieved_docs.csv", index=False)
    print(table.round(4).to_string(index=False))

    lines = [f"python=={platform.python_version()}", f"platform=={platform.platform()}",
             f"device=={res.bi_encoder.device}", f"seed=={args.seed}"]
    lines += [f"{p}=={version(p)}" for p in PACKAGES]
    (out / "environment.txt").write_text("\n".join(lines) + "\n")
    print(f"\nAll results written to {out}")


if __name__ == "__main__":
    main()
