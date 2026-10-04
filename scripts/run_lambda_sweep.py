"""Experiment 2: lambda sensitivity of the Aspect-Aware re-ranker."""
from _common import base_parser, config

from src.experiments import run_lambda_sweep
from src.pipeline import build_resources, set_seed
from src.plots import plot_lambda


def main():
    p = base_parser(__doc__)
    p.add_argument("--num-queries", type=int, default=None,
                   help="seeded random sample size (default: every test query)")
    p.add_argument("--first-stage", choices=["BM25", "Dense"], default="Dense")
    p.add_argument("--lambdas", type=float, nargs="+", default=list(config.LAMBDA_GRID))
    args = p.parse_args()

    set_seed(args.seed)
    res = build_resources(args.data_dir, args.data_dir / "cache")
    sweep = run_lambda_sweep(res, args.lambdas, args.num_queries, args.first_stage,
                             args.top_k, args.candidates, args.threshold, args.seed)
    res.save_ce_cache()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    sweep.round(4).to_csv(args.results_dir / "lambda_analysis.csv", index=False)
    plot_lambda(sweep, args.results_dir / "figures" / "lambda_sensitivity.png")
    print(sweep.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
