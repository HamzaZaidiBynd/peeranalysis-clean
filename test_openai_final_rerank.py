import unittest

from openai_final_rerank import build_openai_final_prompt, parse_openai_rank_numbers


class OpenAIFinalRerankTests(unittest.TestCase):
    def test_parse_valid_comma_separated_numbers(self) -> None:
        self.assertEqual(
            parse_openai_rank_numbers("3, 7, 1, 12, 4, 9, 2, 18, 6, 10", candidate_count=25),
            [3, 7, 1, 12, 4, 9, 2, 18, 6, 10],
        )

    def test_parse_rejects_duplicates(self) -> None:
        with self.assertRaises(ValueError):
            parse_openai_rank_numbers("1,2,3,4,5,6,7,8,9,9", candidate_count=25)

    def test_parse_rejects_out_of_range_numbers(self) -> None:
        with self.assertRaises(ValueError):
            parse_openai_rank_numbers("1,2,3,4,5,6,7,8,9,26", candidate_count=25)

    def test_parse_rejects_wrong_count(self) -> None:
        with self.assertRaises(ValueError):
            parse_openai_rank_numbers("1,2,3,4,5,6,7,8,9", candidate_count=25)

    def test_parse_rejects_prose(self) -> None:
        with self.assertRaises(ValueError):
            parse_openai_rank_numbers("The answer is 1,2,3,4,5,6,7,8,9,10", candidate_count=25)

    def test_prompt_uses_allowed_candidate_fields(self) -> None:
        target = {
            "name": "TARGET LIMITED",
            "core_products": ["Paints"],
            "secondary_products": ["Waterproofing"],
            "value_chain_primary": "Manufacturer",
            "value_chain_secondary": "Unknown",
            "customer_type": "B2C",
            "revenue_crore": 1000,
        }
        candidates = [
            {
                "name": "PEER LIMITED",
                "cin": "SECRET",
                "business_description": "Should not be sent",
                "revenue_crore": 123,
                "core_products": ["Decorative paints"],
                "secondary_products": ["Industrial coatings"],
                "value_chain_primary": "Manufacturer",
                "value_chain_secondary": "Trader",
                "customer_type": "B2C",
            }
        ]
        prompt = build_openai_final_prompt(target, candidates)
        self.assertIn("investment banking comparable-company peers", prompt)
        self.assertIn("1. PEER LIMITED", prompt)
        self.assertIn("Core products/services: Decorative paints", prompt)
        self.assertIn("Secondary products/services: Industrial coatings", prompt)
        self.assertIn("Value chain: Manufacturer / Trader", prompt)
        self.assertIn("Customer type: B2C", prompt)
        self.assertIn("Revenue: INR 123.00 Cr", prompt)
        self.assertNotIn("SECRET", prompt)
        self.assertNotIn("Should not be sent", prompt)


if __name__ == "__main__":
    unittest.main()
