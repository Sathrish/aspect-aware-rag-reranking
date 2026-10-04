"""Offline unit tests for the pure-Python parts (no model downloads needed).

Run with:  python -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.aspects import extract_aspects  # noqa: E402
from src.metrics import (  # noqa: E402
    aspect_coverage_score, diversity_score, ndcg_at_k, paired_randomization_test,
    precision_at_k, reciprocal_rank,
)
from src.prompting import build_prompt  # noqa: E402
from src.rerankers import aspect_aware_rerank, cross_encoder_rerank  # noqa: E402


class TestAspects(unittest.TestCase):
    def test_punctuation_removed(self):
        self.assertEqual(extract_aspects("Do Cholesterol Statin Drugs Cause Breast Cancer?"),
                         ["cholesterol statin", "drugs cause", "breast cancer"])

    def test_and_split(self):
        self.assertEqual(extract_aspects("Heart disease and diabetes?"),
                         ["heart disease", "diabetes"])

    def test_hyphen_kept_and_cap(self):
        self.assertEqual(extract_aspects("What are the effects of COVID-19 on the heart?"),
                         ["effects covid-19", "heart"])
        self.assertLessEqual(len(extract_aspects("alpha bravo charlie delta echoes foxtrot "
                                                 "golfing hotel india juliet", max_aspects=4)), 4)

    def test_empty(self):
        self.assertEqual(extract_aspects("?"), [])


class TestMetrics(unittest.TestCase):
    def test_notebook_example(self):
        retrieved, relevant = [0, 3, 1, 5, 2], {0: 2, 1: 1, 4: 2}
        self.assertAlmostEqual(precision_at_k(retrieved, relevant, 5), 0.4)
        self.assertAlmostEqual(reciprocal_rank(retrieved, relevant), 1.0)
        self.assertAlmostEqual(ndcg_at_k(retrieved, relevant, 5), 0.6646, places=4)

    def test_acs_and_diversity(self):
        aspects = np.eye(3, dtype=np.float32)
        docs = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        self.assertAlmostEqual(aspect_coverage_score(aspects, docs, 0.25), 2 / 3)
        self.assertEqual(aspect_coverage_score(aspects[:0], docs, 0.25), 0.0)
        self.assertAlmostEqual(diversity_score(docs), 1.0)
        self.assertAlmostEqual(diversity_score(np.vstack([docs[0], docs[0]])), 0.0)

    def test_randomization(self):
        diff, p = paired_randomization_test([1.0] * 20, [0.0] * 20)
        self.assertEqual(diff, 1.0)
        self.assertLess(p, 0.001)
        self.assertEqual(paired_randomization_test([1, 2], [1, 2]), (0.0, 1.0))


class TestRerankers(unittest.TestCase):
    def setUp(self):
        self.aspects = np.eye(3, dtype=np.float32)[:2]            # two aspects
        self.cands = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        self.ce = np.array([3.0, 2.9, 1.0])

    def test_lambda_one_equals_cross_encoder(self):
        self.assertEqual(aspect_aware_rerank(self.ce, self.cands, self.aspects, 3, 1.0, 0.25),
                         cross_encoder_rerank(self.ce, 3))

    def test_coverage_promotes_new_aspect(self):
        # The redundant second document is skipped for the one covering a new aspect.
        self.assertEqual(aspect_aware_rerank(self.ce, self.cands, self.aspects, 2, 0.3, 0.25),
                         [0, 2])

    def test_no_aspects_falls_back_to_relevance(self):
        self.assertEqual(aspect_aware_rerank(self.ce, self.cands, self.aspects[:0], 3, 0.7, 0.25),
                         [0, 1, 2])


class _WordTokenizer:
    """Whitespace tokenizer with the same call signature as a Hugging Face tokenizer."""

    def __call__(self, text, add_special_tokens=False, truncation=True, max_length=None):
        ids = text.split() + (["</s>"] if add_special_tokens else [])
        return {"input_ids": ids[:max_length] if truncation else ids}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(i for i in ids if i != "</s>")


class TestPrompt(unittest.TestCase):
    def test_prompt_respects_limit_and_keeps_every_document(self):
        docs = [f"doc{i} " + "word " * 700 for i in range(5)]   # far longer than the window
        prompt, n_tokens = build_prompt(_WordTokenizer(), "Does it fit?", docs, 512)
        self.assertLessEqual(n_tokens, 512)
        self.assertGreater(n_tokens, 400)                        # the window is actually used
        for i in range(5):
            self.assertIn(f"[{i + 1}] doc{i}", prompt)
        self.assertTrue(prompt.endswith("Question: Does it fit?\nAnswer:"))

    def test_short_documents_are_not_truncated(self):
        prompt, n_tokens = build_prompt(_WordTokenizer(), "Q?", ["alpha beta", "gamma"], 512)
        self.assertIn("[1] alpha beta\n[2] gamma", prompt)
        self.assertLess(n_tokens, 30)


if __name__ == "__main__":
    unittest.main()
