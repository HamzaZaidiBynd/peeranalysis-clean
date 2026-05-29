"""Compact data loader and peer search logic for Vercel."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from cohere_rerank import derive_normalized_categories


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "vercel_data"

INDIA_STATE_BY_CIN = {
    "AN": "Andaman and Nicobar Islands",
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CH": "Chandigarh",
    "CT": "Chhattisgarh",
    "DN": "Dadra and Nagar Haveli and Daman and Diu",
    "DD": "Dadra and Nagar Haveli and Daman and Diu",
    "DL": "Delhi",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HR": "Haryana",
    "HP": "Himachal Pradesh",
    "JK": "Jammu and Kashmir",
    "JH": "Jharkhand",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LD": "Lakshadweep",
    "MP": "Madhya Pradesh",
    "MH": "Maharashtra",
    "MN": "Manipur",
    "ML": "Meghalaya",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OR": "Odisha",
    "OD": "Odisha",
    "PY": "Puducherry",
    "PB": "Punjab",
    "PN": "Maharashtra",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TZ": "Tamil Nadu",
    "TG": "Telangana",
    "TS": "Telangana",
    "TR": "Tripura",
    "UP": "Uttar Pradesh",
    "UT": "Uttarakhand",
    "UA": "Uttarakhand",
    "UR": "Uttarakhand",
    "WB": "West Bengal",
}


def norm_text(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def split_pipe(value: str) -> list[str]:
    return [part.strip() for part in (value or "").split("|") if part.strip()]


def state_code_from_cin(cin: str) -> str:
    cin = (cin or "").strip().upper()
    return cin[6:8] if len(cin) >= 8 else ""


def parse_revenue(value: str) -> float | None:
    value = str(value or "").strip().replace(",", "")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def jaccard(left: list[str], right: list[str]) -> float:
    left_set = {norm_text(item) for item in left if norm_text(item)}
    right_set = {norm_text(item) for item in right if norm_text(item)}
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def revenue_similarity(left: float | None, right: float | None) -> float | None:
    if left is None or right is None or left <= 0 or right <= 0:
        return None
    ratio = max(left, right) / min(left, right)
    return max(0.0, 1.0 - (np.log10(ratio) / 2.0))


class PeerData:
    def __init__(self) -> None:
        with (DATA_DIR / "peer_metadata.json").open(encoding="utf-8") as f:
            metadata = json.load(f)
        arrays = np.load(DATA_DIR / "peer_vectors.npz")

        self.rows_by_cin: dict[str, dict[str, str]] = metadata["rows_by_cin"]
        self.master_by_cin: dict[str, dict[str, str]] = metadata["master_by_cin"]
        self.flags_by_cin: dict[str, dict[str, str]] = metadata["flags_by_cin"]
        self.company_cins: list[str] = metadata["company_cins"]
        self.company_index_by_cin = {cin: index for index, cin in enumerate(self.company_cins)}
        self.company_embeddings = arrays["company_embeddings"].astype(np.float32)
        self.product_target_cins: list[str] = metadata["product_target_cins"]
        self.product_candidate_cins: list[str] = metadata["product_candidate_cins"]
        self.product_cins: list[str] = metadata["product_cins"]
        self.search_cins: list[str] = metadata["search_cins"]
        self.product_target_vectors = arrays["product_target_vectors"].astype(np.float32)
        self.product_candidate_vectors = arrays["product_candidate_vectors"].astype(np.float32)
        self.product_candidate_weights = arrays.get("product_candidate_weights")
        if self.product_candidate_weights is not None:
            self.product_candidate_weights = self.product_candidate_weights.astype(np.float32)
        self.target_offsets = arrays["target_offsets"]
        self.candidate_offsets = arrays["candidate_offsets"]
        self.target_index_by_cin = {cin: index for index, cin in enumerate(self.product_target_cins)}
        self.candidate_index_by_cin = {cin: index for index, cin in enumerate(self.product_candidate_cins)}
        self.product_cin_set = set(self.product_cins)
        self.search_index = [
            (
                cin,
                norm_text(
                    " ".join(
                        [
                            self.company_name(cin),
                            cin,
                            self.company_website(cin),
                            self.rows_by_cin.get(cin, {}).get("core_products", ""),
                        ]
                    )
                ),
            )
            for cin in self.search_cins
        ]

    def company_name(self, cin: str) -> str:
        cin = cin.upper()
        if cin in self.rows_by_cin:
            return self.rows_by_cin[cin].get("name", "")
        return self.master_by_cin.get(cin, {}).get("name", "")

    def company_website(self, cin: str) -> str:
        cin = cin.upper()
        if cin in self.rows_by_cin and self.rows_by_cin[cin].get("website"):
            return self.rows_by_cin[cin].get("website", "")
        return self.master_by_cin.get(cin, {}).get("website", "")

    def quality(self, cin: str) -> dict[str, str]:
        return self.flags_by_cin.get(cin.upper(), {"severity": "", "flags": ""})

    def serialize_company(self, cin: str) -> dict[str, Any]:
        cin = cin.upper()
        row = self.rows_by_cin.get(cin, {})
        master = self.master_by_cin.get(cin, {})
        quality = self.quality(cin)
        state_code = state_code_from_cin(cin)
        revenue = parse_revenue(master.get("revenue", ""))
        item = {
            "cin": cin,
            "name": self.company_name(cin),
            "website": self.company_website(cin),
            "is_enriched": cin in self.rows_by_cin,
            "is_product_peerable": cin in self.product_cin_set,
            "has_company_embedding": cin in self.company_index_by_cin,
            "revenue": revenue,
            "revenue_crore": round(revenue / 10_000_000, 2) if revenue is not None else None,
            "revenue_financial_year": master.get("revenue_financial_year", ""),
            "state_code": state_code,
            "state_name": INDIA_STATE_BY_CIN.get(state_code, state_code or "Unknown"),
            "business_description": row.get("business_description", ""),
            "products_or_services": split_pipe(row.get("products_or_services", "")),
            "core_products": split_pipe(row.get("core_products", "")),
            "secondary_products": split_pipe(row.get("secondary_products", "")),
            "minor_products": split_pipe(row.get("minor_products", "")),
            "end_markets": split_pipe(row.get("end_markets", "")),
            "value_chain_primary": row.get("value_chain_primary", ""),
            "value_chain_secondary": row.get("value_chain_secondary", ""),
            "customer_type": row.get("customer_type", ""),
            "geographic_signals": row.get("geographic_signals", ""),
            "quality_severity": quality["severity"],
            "quality_flags": quality["flags"],
        }
        categories = derive_normalized_categories(item)
        item["derived_primary_categories"] = categories["primary_categories"]
        item["derived_secondary_categories"] = categories["secondary_categories"]
        item["derived_category"] = " | ".join(categories["primary_categories"])
        return item

    def company_passes_filters(
        self,
        cin: str,
        state_filter: str,
        min_revenue: float | None,
        max_revenue: float | None,
    ) -> bool:
        if state_filter:
            code = state_code_from_cin(cin)
            state_name = INDIA_STATE_BY_CIN.get(code, code)
            state_filter_norm = state_filter.strip().lower()
            if code.lower() != state_filter_norm and state_name.lower() != state_filter_norm:
                return False
        revenue = parse_revenue(self.master_by_cin.get(cin, {}).get("revenue", ""))
        if min_revenue is not None and (revenue is None or revenue < min_revenue):
            return False
        if max_revenue is not None and (revenue is None or revenue > max_revenue):
            return False
        return True

    def search(
        self,
        query: str,
        limit: int,
        include_flagged: bool,
        state_filter: str,
        min_revenue: float | None,
        max_revenue: float | None,
    ) -> list[dict[str, Any]]:
        query_norm = norm_text(query)
        terms = query_norm.split()
        matches: list[tuple[int, str]] = []
        for cin, haystack in self.search_index:
            if not include_flagged and self.quality(cin)["flags"]:
                continue
            if not self.company_passes_filters(cin, state_filter, min_revenue, max_revenue):
                continue
            if not terms or all(term in haystack for term in terms):
                name = self.company_name(cin)
                starts = 0 if query_norm and norm_text(name).startswith(query_norm) else 1
                enriched_rank = 0 if cin in self.rows_by_cin else 1
                matches.append((starts, enriched_rank, cin))
        matches.sort(key=lambda item: (item[0], item[1], self.company_name(item[2])))
        return [self.serialize_company(cin) for _, _enriched_rank, cin in matches[:limit]]

    def product_coverage_scores(self, cin: str) -> dict[str, float]:
        t_i = self.target_index_by_cin.get(cin)
        if t_i is None:
            return {}
        t_start, t_end = int(self.target_offsets[t_i]), int(self.target_offsets[t_i + 1])
        target_vectors = self.product_target_vectors[t_start:t_end]
        if target_vectors.size == 0:
            return {}
        scores = target_vectors @ self.product_candidate_vectors.T
        coverage_by_cin: dict[str, float] = {}
        for c_i, peer_cin in enumerate(self.product_candidate_cins):
            if peer_cin == cin:
                continue
            c_start, c_end = int(self.candidate_offsets[c_i]), int(self.candidate_offsets[c_i + 1])
            peer_scores = scores[:, c_start:c_end]
            if peer_scores.size:
                if self.product_candidate_weights is not None:
                    peer_scores = peer_scores * self.product_candidate_weights[c_start:c_end]
                coverage_by_cin[peer_cin] = float(np.mean(np.max(peer_scores, axis=1)))
        return coverage_by_cin

    def company_embedding_scores(self, cin: str) -> dict[str, float]:
        target_index = self.company_index_by_cin.get(cin)
        if target_index is None:
            return {}
        similarities = self.company_embeddings @ self.company_embeddings[target_index]
        return {
            peer_cin: float(similarities[index])
            for index, peer_cin in enumerate(self.company_cins)
            if peer_cin != cin
        }

    def peers(
        self,
        cin: str,
        limit: int,
        exclude_flagged: bool,
        same_value_chain: bool,
        same_customer_type: bool,
        use_revenue: bool,
        use_enum_weighting: bool,
        state_filter: str,
        min_revenue: float | None,
        max_revenue: float | None,
        min_score: float,
        scoring_method: str,
    ) -> dict[str, Any]:
        cin = cin.strip().upper()
        if cin not in self.rows_by_cin:
            raise KeyError(cin)
        if scoring_method == "company_embedding":
            base_scores = self.company_embedding_scores(cin)
            method_label = "original company embedding cosine"
            score_kind = "company_embedding"
            if not base_scores:
                raise KeyError(f"{cin} has no company embedding")
        else:
            scoring_method = "product_max_sim"
            base_scores = self.product_coverage_scores(cin)
            method_label = "product max-sim: target core vs peer core+secondary"
            score_kind = "product_max_sim"
            if not base_scores:
                raise KeyError(f"{cin} has no core product embeddings")

        target = self.rows_by_cin[cin]
        target_vc = target.get("value_chain_primary", "")
        target_customer = target.get("customer_type", "")
        target_geo = target.get("geographic_signals", "")
        target_end_markets = split_pipe(target.get("end_markets", ""))
        target_revenue = parse_revenue(self.master_by_cin.get(cin, {}).get("revenue", ""))

        candidates: list[dict[str, Any]] = []
        for peer_cin, score in base_scores.items():
            score_value = float(score)
            if score_value < min_score:
                continue
            peer = self.rows_by_cin.get(peer_cin)
            if not peer:
                continue
            quality = self.quality(peer_cin)
            if exclude_flagged and quality["flags"]:
                continue
            if same_value_chain and target_vc and peer.get("value_chain_primary", "") != target_vc:
                continue
            if same_customer_type and target_customer and peer.get("customer_type", "") != target_customer:
                continue
            if not self.company_passes_filters(peer_cin, state_filter, min_revenue, max_revenue):
                continue
            enum_adjustment = 0.0
            enum_reasons: list[str] = []
            if use_enum_weighting:
                if target_vc and peer.get("value_chain_primary", "") == target_vc:
                    enum_adjustment += 0.025
                    enum_reasons.append("value chain")
                if target_customer and peer.get("customer_type", "") == target_customer:
                    enum_adjustment += 0.015
                    enum_reasons.append("customer type")
                end_market_overlap = jaccard(target_end_markets, split_pipe(peer.get("end_markets", "")))
                if end_market_overlap:
                    enum_adjustment += 0.020 * end_market_overlap
                    enum_reasons.append(f"end markets {end_market_overlap:.2f}")
                if target_geo and peer.get("geographic_signals", "") == target_geo:
                    enum_adjustment += 0.005
                    enum_reasons.append("geography")

            revenue_adjustment = 0.0
            revenue_sim = None
            if use_revenue:
                peer_revenue = parse_revenue(self.master_by_cin.get(peer_cin, {}).get("revenue", ""))
                revenue_sim = revenue_similarity(target_revenue, peer_revenue)
                if revenue_sim is not None:
                    revenue_adjustment = 0.05 * (revenue_sim - 0.5)

            final_score = score_value + enum_adjustment + revenue_adjustment
            candidates.append(
                {
                    "cin": peer_cin,
                    "cosine_similarity": round(score_value, 6),
                    "base_similarity_score": round(score_value, 6),
                    "product_coverage_score": round(score_value, 6) if score_kind == "product_max_sim" else None,
                    "company_embedding_score": round(score_value, 6) if score_kind == "company_embedding" else None,
                    "enum_adjustment": round(enum_adjustment, 6),
                    "enum_reasons": enum_reasons,
                    "revenue_similarity": round(revenue_sim, 6) if revenue_sim is not None else None,
                    "revenue_adjustment": round(revenue_adjustment, 6),
                    "final_score": round(final_score, 6),
                }
            )

        candidates.sort(key=lambda item: item["final_score"], reverse=True)
        peers = []
        for candidate in candidates[:limit]:
            item = self.serialize_company(candidate["cin"])
            item.update(candidate)
            peers.append(item)
        return {
            "target": self.serialize_company(cin),
            "peers": peers,
            "method": method_label
            + (" + enum weighting" if use_enum_weighting else "")
            + (" + revenue weighting" if use_revenue else ""),
            "scoring_method": scoring_method,
            "filters": {
                "exclude_flagged": exclude_flagged,
                "same_value_chain": same_value_chain,
                "same_customer_type": same_customer_type,
                "use_revenue": use_revenue,
                "use_enum_weighting": use_enum_weighting,
                "state": state_filter,
                "min_revenue": min_revenue,
                "max_revenue": max_revenue,
                "min_score": min_score,
                "scoring_method": scoring_method,
            },
        }


@lru_cache(maxsize=1)
def get_peer_data() -> PeerData:
    return PeerData()
