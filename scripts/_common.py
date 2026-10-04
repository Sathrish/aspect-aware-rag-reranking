"""Shared command-line options for the experiment scripts."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402


def base_parser(description):
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--top-k", type=int, default=config.TOP_K)
    p.add_argument("--candidates", type=int, default=config.N_CANDIDATES,
                   help="first-stage candidates given to the re-rankers")
    p.add_argument("--threshold", type=float, default=config.ASPECT_SIM_THRESHOLD,
                   help="cosine threshold shared by the Aspect-Aware re-ranker and ACS")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    p.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    return p
