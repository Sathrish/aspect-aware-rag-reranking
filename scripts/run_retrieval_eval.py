"""Experiment 1: retrieval metrics and ACS for all six systems."""
from _common import base_parser, config

from src.experiments import run_retrieval_eval
from src.pipeline import build_resources, set_seed
from src.plots import plot_retrieval


def main():
    p = base_parser(__doc__)
    p.add_argument("--num-queries", type=int, default=None,
                   help="seeded random sample size (default: every test query)")
    p.add_argument("--lambda", dest="lam", type=float, default=config.LAMBDA)
    args = p.parse_args()

    set_seed(args.seed)
    res = build_resources(args.data_dir, args.data_dir / "cache")
    summary, per_query, significance = run_retrieval_eval(
        res, args.num_queries, args.top_k, args.candidates, args.lam, args.threshold, args.seed)
    res.save_ce_cache()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary.round(4).to_csv(args.results_dir / "reranking_results.csv", index=False)
    per_query.round(4).to_csv(args.results_dir / "reranking_per_query.csv", index=False)
    significance.round(4).to_csv(args.results_dir / "reranking_significance.csv", index=False)
    plot_retrieval(summary, args.results_dir / "figures" / "retrieval_comparison.png")
    print(summary.round(4).to_string(index=False))
    print(significance.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
