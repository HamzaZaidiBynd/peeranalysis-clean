import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.index import app


def fake_union_payload() -> dict:
    peers = [
        {
            "cin": f"PEER{i:02d}",
            "name": f"PEER {i}",
            "core_products": [f"Product {i}"],
            "secondary_products": [],
            "value_chain_primary": "IT Services",
            "value_chain_secondary": "",
            "customer_type": "B2B",
            "revenue_crore": i * 100,
            "rerank_candidate_sources": ["product_max_sim"],
            "product_candidate_score": 1 - (i / 100),
            "company_candidate_score": None,
        }
        for i in range(1, 64)
    ]
    return {
        "target": {
            "cin": "TARGET",
            "name": "TARGET LIMITED",
            "core_products": ["Target product"],
            "secondary_products": [],
            "value_chain_primary": "IT Services",
            "value_chain_secondary": "",
            "customer_type": "B2B",
            "revenue_crore": 1000,
        },
        "peers": peers,
        "method": "product max-sim top 40 + company embedding top 40 union",
        "filters": {},
    }


class ApiRerankModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    @patch("api.index.select_final_peers_with_openai")
    @patch("api.index.build_union_rerank_payload")
    @patch("api.index.get_peer_data")
    def test_default_rerank_sends_full_union_directly_to_openai(self, mock_data, mock_union, mock_openai) -> None:
        payload = fake_union_payload()
        mock_data.return_value = SimpleNamespace(rows_by_cin={"TARGET": {}})
        mock_union.return_value = payload
        mock_openai.return_value = (
            payload["peers"][:20],
            {
                "provider": "azure_openai",
                "used": True,
                "fallback": False,
                "candidate_count": 63,
                "returned_count": 20,
                "selected_numbers": list(range(1, 21)),
            },
        )

        response = self.client.get("/api/peers?cin=TARGET&limit=20")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["rerank"]["mode"], "openai_direct")
        self.assertEqual(body["rerank"]["openai_candidate_count"], 63)
        self.assertEqual(body["rerank"]["requested_count"], 20)
        self.assertEqual(len(body["rerank_trace"]["openai_candidates"]), 63)
        self.assertNotIn("cohere_prompt", body["rerank_trace"])
        mock_openai.assert_called_once()
        self.assertEqual(len(mock_openai.call_args.kwargs["candidates"]), 63)
        self.assertEqual(mock_openai.call_args.kwargs["final_count"], 20)

    @patch("api.index.rerank_peers")
    @patch("api.index.select_final_peers_with_openai")
    @patch("api.index.build_union_rerank_payload")
    @patch("api.index.get_peer_data")
    def test_cohere_openai_mode_keeps_old_shortlist_flow(self, mock_data, mock_union, mock_openai, mock_cohere) -> None:
        payload = fake_union_payload()
        mock_data.return_value = SimpleNamespace(rows_by_cin={"TARGET": {}})
        mock_union.return_value = payload
        mock_cohere.return_value = (
            payload["peers"][:25],
            {
                "provider": "cohere",
                "used": True,
                "fallback": False,
                "candidate_count": 63,
                "returned_count": 25,
                "derived_categories": False,
            },
        )
        mock_openai.return_value = (
            payload["peers"][:5],
            {
                "provider": "azure_openai",
                "used": True,
                "fallback": False,
                "candidate_count": 25,
                "returned_count": 5,
                "selected_numbers": list(range(1, 6)),
            },
        )

        response = self.client.get("/api/peers?cin=TARGET&rerank=true&rerank_mode=cohere_openai&k=5")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["rerank"]["mode"], "cohere_openai")
        self.assertEqual(body["rerank"]["openai_candidate_count"], 25)
        self.assertEqual(body["rerank"]["requested_count"], 5)
        self.assertIn("cohere_prompt", body["rerank_trace"])
        self.assertEqual(len(body["rerank_trace"]["cohere_shortlist"]), 25)
        mock_cohere.assert_called_once()
        mock_openai.assert_called_once()
        self.assertEqual(len(mock_openai.call_args.kwargs["candidates"]), 25)
        self.assertEqual(mock_openai.call_args.kwargs["final_count"], 5)

    @patch("api.index.select_final_peers_with_openai")
    @patch("api.index.build_union_rerank_payload")
    @patch("api.index.get_peer_data")
    def test_peers_can_resolve_exact_company_name(self, mock_data, mock_union, mock_openai) -> None:
        data = SimpleNamespace(
            rows_by_cin={"TARGET": {}},
            search=lambda **_kwargs: [{"cin": "TARGET", "name": "TARGET LIMITED"}],
        )
        payload = fake_union_payload()
        mock_data.return_value = data
        mock_union.return_value = payload
        mock_openai.return_value = (
            payload["peers"][:5],
            {
                "provider": "azure_openai",
                "used": True,
                "fallback": False,
                "candidate_count": 63,
                "returned_count": 5,
                "selected_numbers": list(range(1, 6)),
            },
        )

        response = self.client.get("/api/peers?name=TARGET%20LIMITED&rerank=true&limit=5")

        self.assertEqual(response.status_code, 200)
        mock_union.assert_called_once()
        self.assertEqual(mock_union.call_args.args[1], "TARGET")

    @patch("api.index.select_final_peers_with_openai")
    @patch("api.index.build_union_rerank_payload")
    @patch("api.index.get_peer_data")
    def test_peers_can_resolve_company_alias(self, mock_data, mock_union, mock_openai) -> None:
        data = SimpleNamespace(
            rows_by_cin={"TARGET": {}},
            search=lambda **_kwargs: [{"cin": "TARGET", "name": "TARGET LIMITED"}],
        )
        payload = fake_union_payload()
        mock_data.return_value = data
        mock_union.return_value = payload
        mock_openai.return_value = (
            payload["peers"][:5],
            {
                "provider": "azure_openai",
                "used": True,
                "fallback": False,
                "candidate_count": 63,
                "returned_count": 5,
                "selected_numbers": list(range(1, 6)),
            },
        )

        response = self.client.get("/api/peers?company=TARGET%20LIMITED&limit=5")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["rerank"]["mode"], "openai_direct")
        mock_union.assert_called_once()
        self.assertEqual(mock_union.call_args.args[1], "TARGET")

    @patch("api.index.get_peer_data")
    def test_peers_returns_409_for_ambiguous_company_name(self, mock_data) -> None:
        data = SimpleNamespace(
            rows_by_cin={},
            search=lambda **_kwargs: [
                {"cin": "ONE", "name": "ABC LIMITED", "state_name": "Maharashtra", "is_enriched": True, "revenue_crore": 100},
                {"cin": "TWO", "name": "ABC INDIA LIMITED", "state_name": "Karnataka", "is_enriched": True, "revenue_crore": 200},
            ],
        )
        mock_data.return_value = data

        response = self.client.get("/api/peers?name=ABC&rerank=true&limit=5")

        self.assertEqual(response.status_code, 409)
        detail = response.json()["detail"]
        self.assertIn("Multiple companies matched", detail["message"])
        self.assertEqual(len(detail["matches"]), 2)


if __name__ == "__main__":
    unittest.main()
