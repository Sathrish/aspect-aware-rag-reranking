"""Central configuration: models, hyper-parameters and paths.

Every script reads its defaults from here, so the Aspect-Aware re-ranker and
the Aspect Coverage Score always share the same similarity threshold.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RAG_QA_PATH = DATA_DIR / "rag_qa.json"

NFCORPUS_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"
)

BI_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GENERATOR = "google/flan-t5-base"

SEED = 42
TOP_K = 5                     # documents kept after re-ranking
N_CANDIDATES = 20             # first-stage candidates handed to the re-rankers
LAMBDA = 0.7                  # relevance weight in the Aspect-Aware score
ASPECT_SIM_THRESHOLD = 0.25   # cosine threshold: "document covers aspect"
MAX_ASPECTS = 4
LAMBDA_GRID = (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

# Generator settings (end-to-end RAG experiment)
MAX_INPUT_TOKENS = 512       # flan-t5 input limit; shared equally by the retrieved documents
MAX_NEW_TOKENS = 100
NUM_BEAMS = 2
