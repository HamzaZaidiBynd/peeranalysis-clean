import unittest

from rerank_candidate_pool import build_union_rerank_payload


class FakePeerData:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def peers(self, *, cin: str, limit: int, scoring_method: str, **_kwargs):
        self.calls.append((scoring_method, limit))
        rows = []
        prefix = "P" if scoring_method == "product_max_sim" else "E"
        for index in range(limit):
            peer_cin = "OVERLAP" if index == 0 else f"{prefix}{index:02d}"
            rows.append(
                {
                    "cin": peer_cin,
                    "name": peer_cin,
                    "base_similarity_score": 1 - (index / 100),
                    "final_score": 1 - (index / 100),
                    "product_coverage_score": 1 - (index / 100) if scoring_method == "product_max_sim" else None,
                    "company_embedding_score": 1 - (index / 100) if scoring_method == "company_embedding" else None,
                }
            )
        return {"target": {"cin": cin, "name": "Target"}, "peers": rows, "filters": {}}

    def product_coverage_scores(self, _cin: str):
        return {}

    def company_embedding_scores(self, _cin: str):
        return {}


class RerankCandidatePoolTests(unittest.TestCase):
    def test_union_uses_top_40_from_each_source_and_dedupes(self) -> None:
        data = FakePeerData()
        payload = build_union_rerank_payload(
            data,
            "TARGET",
            exclude_flagged=False,
            same_value_chain=False,
            same_customer_type=False,
            use_revenue=False,
            use_enum_weighting=False,
            state_filter="",
            min_revenue=None,
            max_revenue=None,
            min_score=0,
        )

        self.assertEqual(data.calls, [("product_max_sim", 40), ("company_embedding", 40)])
        self.assertEqual(len(payload["peers"]), 79)
        self.assertEqual(payload["candidate_source_counts"], {"product_max_sim": 40, "company_embedding": 40})
        overlap = next(peer for peer in payload["peers"] if peer["cin"] == "OVERLAP")
        self.assertEqual(overlap["rerank_candidate_sources"], ["product_max_sim", "company_embedding"])
        self.assertIn("top 40", payload["method"])


if __name__ == "__main__":
    unittest.main()
