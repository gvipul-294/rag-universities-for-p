import unittest

from phd_rag.decisions import assess_retrieval_evidence


class EvidenceDecisionTests(unittest.TestCase):
    def test_approves_strong_results_from_multiple_universities(self) -> None:
        decision = assess_retrieval_evidence(
            [
                {"university": "University A", "score": 0.72},
                {"university": "University B", "score": 0.58},
            ]
        )

        self.assertTrue(decision.approved)
        self.assertEqual(decision.unique_universities, 2)
        self.assertEqual(decision.top_score, 0.72)

    def test_rejects_a_low_similarity_result(self) -> None:
        decision = assess_retrieval_evidence(
            [{"university": "University A", "score": 0.12}]
        )

        self.assertFalse(decision.approved)
        self.assertIn("below the configured threshold", decision.reason)

    def test_rejects_results_without_source_diversity(self) -> None:
        decision = assess_retrieval_evidence(
            [
                {"university": "University A", "score": 0.86},
                {"university": "University A", "score": 0.64},
            ]
        )

        self.assertFalse(decision.approved)
        self.assertIn("unique university", decision.reason)


if __name__ == "__main__":
    unittest.main()
