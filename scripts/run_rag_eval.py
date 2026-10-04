"""Experiment 3: end-to-end RAG answer quality (ROUGE-L) with flan-t5-base."""
import pandas as pd
from _common import base_parser, config

from src.data import load_rag_qa
from src.experiments import run_rag_eval
from src.generation import AnswerGenerator
from src.pipeline import build_resources, set_seed
from src.plots import plot_rag


def main():
    p = base_parser(__doc__)
    p.add_argument("--lambda", dest="lam", type=float, default=config.LAMBDA)
    p.add_argument("--max-input-tokens", type=int, default=config.MAX_INPUT_TOKENS)
    args = p.parse_args()

    set_seed(args.seed)
    res = build_resources(args.data_dir, args.data_dir / "cache")
    generator = AnswerGenerator(max_input_tokens=args.max_input_tokens)
    scores, answers, retrieved = run_rag_eval(res, generator, load_rag_qa(), args.top_k,
                                   args.candidates, args.lam, args.threshold)
    res.save_ce_cache()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    plot_rag(scores, args.results_dir / "figures" / "rag_comparison.png")
    average = scores.drop(columns="Query").mean().to_dict()
    table = pd.concat([scores, pd.DataFrame([{"Query": "AVERAGE", **average}])], ignore_index=True)
    table.round(4).to_csv(args.results_dir / "rag_comparison.csv", index=False)
    answers.to_csv(args.results_dir / "rag_answers.csv", index=False)
    retrieved.to_csv(args.results_dir / "rag_retrieved_docs.csv", index=False)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
