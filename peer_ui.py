"""Local peer-search UI backed by company and product embeddings.

Run:
    .venv/bin/python peer_ui.py

Then open:
    http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import mimetypes
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from cohere_rerank import (
    derive_normalized_categories,
    format_company_for_rerank,
    rerank_query,
    rerank_peers,
)
from openai_final_rerank import (
    DEFAULT_OPENAI_CANDIDATE_COUNT,
    build_openai_final_prompt,
    select_final_peers_with_openai,
    summarize_openai_candidate,
)
from rerank_candidate_pool import build_union_rerank_payload


DEFAULT_CANONICAL = Path("reports/company_enrichment_canonical_with_primacy.csv")
DEFAULT_COMPANY_EMBEDDINGS = Path("embedding_artifacts_pgvector_company_current/embeddings.jsonl")
DEFAULT_PRODUCT_EMBEDDINGS = Path("embedding_artifacts_pgvector_product_items_current/product_item_embeddings.jsonl")
DEFAULT_FLAGS = Path("reports/enrichment_quality_flags_current.csv")
DEFAULT_FULL_LIST = Path("full_list.csv")
DESIGN_SYSTEM_ASSETS = Path("/Users/hamzazaidi/Documents/Bynd/Bynd Design System/assets")
DESIGN_SYSTEM_FONTS = Path("/Users/hamzazaidi/Documents/Bynd/Bynd Design System/fonts")

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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


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
    # 1.0 means same scale; 0.0 means 100x+ apart.
    return max(0.0, 1.0 - (np.log10(ratio) / 2.0))


def summarize_trace_company(company: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "rank": index,
        "name": company.get("name", ""),
        "cin": company.get("cin", ""),
        "sources": company.get("rerank_candidate_sources") or [],
        "product_candidate_score": company.get("product_candidate_score"),
        "company_candidate_score": company.get("company_candidate_score"),
        "cohere_rerank_score": company.get("cohere_rerank_score"),
        "openai_final_rank": company.get("openai_final_rank"),
        "pre_openai_rank": company.get("pre_openai_rank"),
        "core_products": company.get("core_products") or [],
        "secondary_products": company.get("secondary_products") or [],
        "value_chain": " / ".join(
            part
            for part in [
                str(company.get("value_chain_primary") or "").strip(),
                str(company.get("value_chain_secondary") or "").strip(),
            ]
            if part
        ),
        "customer_type": company.get("customer_type", ""),
        "revenue_crore": company.get("revenue_crore"),
    }


class PeerData:
    def __init__(
        self,
        canonical_path: Path,
        company_embeddings_path: Path,
        product_embeddings_path: Path,
        flags_path: Path,
        full_list_path: Path,
    ) -> None:
        self.canonical_path = canonical_path
        self.company_embeddings_path = company_embeddings_path
        self.product_embeddings_path = product_embeddings_path
        self.flags_path = flags_path
        self.full_list_path = full_list_path
        self.rows_by_cin: dict[str, dict[str, str]] = {}
        self.flags_by_cin: dict[str, dict[str, str]] = {}
        self.master_by_cin: dict[str, dict[str, str]] = {}
        self.cins: list[str] = []
        self.search_cins: list[str] = []
        self.company_cins: list[str] = []
        self.company_index_by_cin: dict[str, int] = {}
        self.company_embeddings: np.ndarray
        self.product_cins: list[str] = []
        self.target_core_vectors_by_cin: dict[str, np.ndarray] = {}
        self.candidate_product_vectors: np.ndarray
        self.candidate_slices_by_cin: dict[str, slice] = {}
        self.search_index: list[tuple[str, str]]
        self.load()

    def load(self) -> None:
        canonical_rows = read_csv_rows(self.canonical_path)
        self.rows_by_cin = {
            row["cin"].strip().upper(): row
            for row in canonical_rows
            if row.get("cin") and row.get("name")
        }

        self.master_by_cin = {}
        if self.full_list_path.exists():
            for row in read_csv_rows(self.full_list_path):
                cin = row.get("identifier_value", "").strip().upper()
                if cin:
                    self.master_by_cin[cin] = row

        self.flags_by_cin = {}
        if self.flags_path.exists():
            for row in read_csv_rows(self.flags_path):
                cin = row.get("cin", "").strip().upper()
                if cin:
                    self.flags_by_cin[cin] = {
                        "severity": row.get("severity", ""),
                        "flags": row.get("flags", ""),
                    }

        self.load_company_embeddings()
        self.load_product_embeddings()

        self.cins = sorted(
            set(self.company_cins) | set(self.product_cins),
            key=lambda cin: self.rows_by_cin[cin].get("name", ""),
        )
        self.search_cins = sorted(
            set(self.master_by_cin) | set(self.cins),
            key=lambda cin: self.company_name(cin),
        )
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

    def load_company_embeddings(self) -> None:
        cins: list[str] = []
        vectors: list[list[float]] = []
        with self.company_embeddings_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                cin = str(item.get("cin") or "").strip().upper()
                vector = item.get("embedding")
                if cin in self.rows_by_cin and vector:
                    cins.append(cin)
                    vectors.append(vector)

        if not vectors:
            raise RuntimeError(f"No usable company embeddings found in {self.company_embeddings_path}")

        self.company_cins = cins
        self.company_index_by_cin = {cin: index for index, cin in enumerate(cins)}
        self.company_embeddings = self.normalize_matrix(np.asarray(vectors, dtype=np.float32))

    def load_product_embeddings(self) -> None:
        product_rows: list[tuple[str, str, str, list[float]]] = []
        with self.product_embeddings_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                cin = str(item.get("cin") or "").strip().upper()
                vector = item.get("embedding")
                if cin in self.rows_by_cin and vector:
                    product_text = str(item.get("product_text") or "").strip()
                    primacy = self.product_primacy(cin, product_text)
                    if primacy:
                        product_rows.append((cin, product_text, primacy, vector))

        if not product_rows:
            raise RuntimeError(f"No usable product embeddings found in {self.product_embeddings_path}")

        target_vectors_by_cin: dict[str, list[list[float]]] = {}
        candidate_rows: list[tuple[str, list[float], float]] = []
        for cin, _product_text, primacy, vector in product_rows:
            if primacy == "core":
                target_vectors_by_cin.setdefault(cin, []).append(vector)
            if primacy in {"core", "secondary"}:
                weight = 1.0 if primacy == "core" else 0.55
                candidate_rows.append((cin, vector, weight))

        self.target_core_vectors_by_cin = {
            cin: self.normalize_matrix(np.asarray(vectors, dtype=np.float32))
            for cin, vectors in target_vectors_by_cin.items()
        }
        candidate_rows.sort(key=lambda item: item[0])
        candidate_vectors = [vector for _cin, vector, _weight in candidate_rows]
        self.candidate_product_vectors = self.normalize_matrix(np.asarray(candidate_vectors, dtype=np.float32))
        self.candidate_product_weights = np.asarray(
            [weight for _cin, _vector, weight in candidate_rows],
            dtype=np.float32,
        )
        self.candidate_slices_by_cin = {}
        start = 0
        while start < len(candidate_rows):
            cin = candidate_rows[start][0]
            end = start + 1
            while end < len(candidate_rows) and candidate_rows[end][0] == cin:
                end += 1
            self.candidate_slices_by_cin[cin] = slice(start, end)
            start = end

        self.product_cins = sorted(
            set(self.target_core_vectors_by_cin) & set(self.candidate_slices_by_cin),
            key=lambda cin: self.rows_by_cin[cin].get("name", ""),
        )

    def normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            return matrix.reshape(0, 0).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def product_primacy(self, cin: str, product_text: str) -> str:
        row = self.rows_by_cin[cin]
        product_norm = norm_text(product_text)
        if not product_norm:
            return ""
        for primacy, field in (
            ("core", "core_products"),
            ("secondary", "secondary_products"),
            ("minor", "minor_products"),
        ):
            values = {norm_text(item) for item in split_pipe(row.get(field, ""))}
            if product_norm in values:
                return primacy
        return ""

    def product_coverage_scores(self, cin: str) -> dict[str, float]:
        target_vectors = self.target_core_vectors_by_cin.get(cin)
        if target_vectors is None or target_vectors.size == 0:
            return {}
        scores = target_vectors @ self.candidate_product_vectors.T
        coverage_by_cin: dict[str, float] = {}
        for peer_cin, product_slice in self.candidate_slices_by_cin.items():
            if peer_cin == cin:
                continue
            peer_scores = scores[:, product_slice]
            if peer_scores.size == 0:
                continue
            peer_scores = peer_scores * self.candidate_product_weights[product_slice]
            coverage_by_cin[peer_cin] = float(np.mean(np.max(peer_scores, axis=1)))
        return coverage_by_cin

    def company_embedding_scores(self, cin: str) -> dict[str, float]:
        target_index = self.company_index_by_cin.get(cin)
        if target_index is None:
            return {}
        similarities = self.company_embeddings @ self.company_embeddings[target_index]
        scores: dict[str, float] = {}
        for index, peer_cin in enumerate(self.company_cins):
            if peer_cin != cin:
                scores[peer_cin] = float(similarities[index])
        return scores

    def quality(self, cin: str) -> dict[str, str]:
        return self.flags_by_cin.get(cin.upper(), {"severity": "", "flags": ""})

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

    def serialize_company(self, cin: str) -> dict[str, Any]:
        cin = cin.upper()
        row = self.rows_by_cin.get(cin, {})
        master = self.master_by_cin.get(cin, {})
        quality = self.quality(cin)
        state_code = state_code_from_cin(cin)
        revenue = parse_revenue(master.get("revenue", ""))
        enriched = cin in self.rows_by_cin
        item = {
            "cin": cin,
            "name": self.company_name(cin),
            "website": self.company_website(cin),
            "is_enriched": enriched,
            "is_product_peerable": cin in self.product_cins,
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
                matches.append((starts, cin))
        matches.sort(key=lambda item: (item[0], self.company_name(item[1])))
        return [self.serialize_company(cin) for _, cin in matches[:limit]]

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
            peer = self.rows_by_cin[peer_cin]
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
                    # Small symmetric scale adjustment: similar revenue gets a boost;
                    # very different revenue gets a mild penalty.
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


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Peer explorer</title>
  <style>
    @font-face { font-family: "Inter"; src: url("/design-fonts/Inter_18pt-Regular.ttf") format("truetype"); font-weight: 400; font-style: normal; font-display: swap; }
    @font-face { font-family: "Inter"; src: url("/design-fonts/Inter_18pt-Medium.ttf") format("truetype"); font-weight: 500; font-style: normal; font-display: swap; }
    @font-face { font-family: "Inter"; src: url("/design-fonts/Inter_18pt-SemiBold.ttf") format("truetype"); font-weight: 600; font-style: normal; font-display: swap; }
    @font-face { font-family: "Inter"; src: url("/design-fonts/Inter_18pt-Bold.ttf") format("truetype"); font-weight: 700; font-style: normal; font-display: swap; }
    :root {
      color-scheme: light;
      --bynd-primary: #001742;
      --bynd-brand: #1A5BE7;
      --bynd-brand-hover: #1548C4;
      --bynd-brand-active: #103A9E;
      --bynd-brand-light: rgba(26, 91, 231, 0.10);
      --bynd-warm-active: rgba(91, 77, 47, 0.07);
      --bynd-warm-hover: rgba(91, 77, 47, 0.05);
      --bynd-warm-subtle: rgba(91, 77, 47, 0.03);
      --bynd-warm-border: rgba(91, 77, 47, 0.15);
      --bynd-bg-page: #FAFAF9;
      --bynd-bg-panel: #FFFFFF;
      --bynd-bg-sidebar: rgba(91, 77, 47, 0.05);
      --bynd-bg-sunken: #F5F5F4;
      --bynd-border-light: #F1F1F0;
      --bynd-border-medium: #E5E5E3;
      --bynd-border-strong: #E2E2DE;
      --bynd-text-primary: #000000;
      --bynd-text-navy: #001742;
      --bynd-text-secondary: #4E5971;
      --bynd-text-tertiary: #999999;
      --bynd-success: #22C55E;
      --bynd-warning: #F59E0B;
      --bynd-warning-light: rgba(245, 158, 11, 0.10);
      --bynd-error: #EF4444;
      --bynd-error-light: rgba(239, 68, 68, 0.10);
      --bynd-shadow-soft: -1px -1px 16px 0 rgba(91,77,47,0.05), 1px 4px 24px 0 rgba(91,77,47,0.05);
      --bynd-shadow-card: 0 1px 3px rgba(0,0,0,0.08);
      --bynd-shadow-focus: 0 0 0 3px rgba(26, 91, 231, 0.10);
      --bynd-font-sans: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: var(--bynd-font-sans);
      background: var(--bynd-bg-page);
      color: var(--bynd-text-primary);
      letter-spacing: 0;
      -webkit-font-smoothing: antialiased;
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      background: var(--bynd-warm-subtle);
      pointer-events: none;
    }
    .app-shell {
      position: relative;
      display: grid;
      grid-template-columns: 76px minmax(0, 1fr);
      gap: 0;
      min-height: 100vh;
      padding: 12px 12px 12px 0;
    }
    .rail {
      background: var(--bynd-bg-sidebar);
      padding: 16px 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 24px;
      color: var(--bynd-text-secondary);
    }
    .rail-brand {
      display: grid;
      justify-items: center;
      gap: 6px;
      color: var(--bynd-text-navy);
      font-size: 12px;
      font-weight: 500;
      letter-spacing: 0.02em;
    }
    .rail-brand img { width: 22px; height: 28px; object-fit: contain; }
    .rail-nav {
      display: grid;
      gap: 8px;
      width: 100%;
      justify-items: center;
    }
    .rail-item {
      width: 56px;
      border-radius: 4px;
      padding: 6px 4px;
      display: grid;
      justify-items: center;
      gap: 3px;
      color: var(--bynd-text-secondary);
      font-size: 11px;
      font-weight: 500;
    }
    .rail-item.active {
      background: var(--bynd-warm-active);
      color: var(--bynd-text-primary);
    }
    .rail-item svg { width: 22px; height: 22px; stroke-width: 1.5; }
    .app-panel {
      background: var(--bynd-bg-panel);
      border-radius: 6px;
      box-shadow: var(--bynd-shadow-soft);
      overflow: hidden;
      min-height: calc(100vh - 24px);
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
    }
    header {
      padding: 24px 24px 0;
      border-bottom: 1px solid var(--bynd-border-light);
      background: var(--bynd-bg-panel);
    }
    h1 {
      margin: 0 0 6px;
      font-size: 20px;
      font-weight: 600;
      line-height: 1.3;
    }
    header p {
      margin: 0;
      color: var(--bynd-text-secondary);
      font-size: 13px;
      line-height: 1.4;
    }
    .tabs {
      display: flex;
      gap: 0;
      margin-top: 18px;
    }
    .tab {
      padding: 10px 16px;
      margin-bottom: -1px;
      border-bottom: 2px solid transparent;
      color: var(--bynd-text-secondary);
      font-size: 14px;
      font-weight: 500;
    }
    .tab.active {
      color: var(--bynd-text-primary);
      border-bottom-color: var(--bynd-brand);
    }
    main {
      display: grid;
      grid-template-columns: 344px minmax(0, 1fr);
      min-height: 0;
    }
    aside {
      background: rgba(91, 77, 47, 0.012);
      border-right: 1px solid var(--bynd-border-light);
      padding: 16px;
      overflow: auto;
    }
    section {
      padding: 16px;
      overflow: auto;
      min-width: 0;
    }
    label {
      display: block;
      font-size: 12px;
      color: var(--bynd-text-secondary);
      margin-bottom: 6px;
      font-weight: 500;
    }
    input[type="search"], input[type="number"], select {
      width: 100%;
      border: 1px solid var(--bynd-border-strong);
      border-radius: 6px;
      padding: 9px 10px;
      font-size: 14px;
      background: #fff;
      color: var(--bynd-text-primary);
      font-family: var(--bynd-font-sans);
      transition: border-color 150ms ease, box-shadow 150ms ease;
    }
    input[type="search"]:focus, input[type="number"]:focus, select:focus {
      outline: none;
      border-color: var(--bynd-brand);
      box-shadow: var(--bynd-shadow-focus);
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 12px 0;
    }
    .check {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-height: 28px;
      border: 1px solid var(--bynd-border-strong);
      border-radius: 4px;
      padding: 3px 8px;
      background: white;
      font-size: 12px;
      color: var(--bynd-text-secondary);
      font-weight: 500;
      cursor: pointer;
      transition: background 100ms ease, color 100ms ease, border-color 100ms ease;
    }
    .check:hover {
      background: var(--bynd-warm-subtle);
      color: var(--bynd-text-primary);
    }
    .check:has(input:checked) {
      background: var(--bynd-warm-active);
      border-color: var(--bynd-warm-border);
      color: var(--bynd-text-primary);
    }
    .check input { width: 14px; height: 14px; accent-color: var(--bynd-brand); }
    .tip {
      position: relative;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 16px;
      height: 16px;
      margin-left: 4px;
      border: 1px solid var(--bynd-border-strong);
      border-radius: 50%;
      color: var(--bynd-text-secondary);
      background: #fff;
      font-size: 11px;
      font-weight: 700;
      line-height: 1;
      cursor: help;
      vertical-align: text-bottom;
    }
    .tip::after {
      content: attr(data-tip);
      position: absolute;
      left: 50%;
      bottom: calc(100% + 8px);
      z-index: 10;
      width: min(260px, 70vw);
      transform: translateX(-50%);
      border: 1px solid var(--bynd-border-strong);
      border-radius: 6px;
      padding: 8px 9px;
      background: var(--bynd-text-primary);
      color: #fff;
      box-shadow: var(--bynd-shadow-card);
      font-size: 12px;
      font-weight: 500;
      line-height: 1.35;
      white-space: normal;
      opacity: 0;
      pointer-events: none;
      transition: opacity 100ms ease;
    }
    .tip::before {
      content: "";
      position: absolute;
      left: 50%;
      bottom: calc(100% + 3px);
      z-index: 11;
      transform: translateX(-50%);
      border: 5px solid transparent;
      border-top-color: var(--bynd-text-primary);
      opacity: 0;
      pointer-events: none;
      transition: opacity 100ms ease;
    }
    .tip:hover::after,
    .tip:hover::before,
    .tip:focus::after,
    .tip:focus::before {
      opacity: 1;
    }
    .company-list {
      display: grid;
      gap: 6px;
      margin-top: 10px;
    }
    button.company {
      width: 100%;
      text-align: left;
      border: 1px solid var(--bynd-border-light);
      border-radius: 6px;
      background: #fff;
      padding: 10px;
      cursor: pointer;
      color: var(--bynd-text-primary);
      font-family: var(--bynd-font-sans);
      transition: background 100ms ease, border-color 100ms ease, box-shadow 100ms ease;
    }
    button.company:hover, button.company.active {
      border-color: var(--bynd-border-strong);
      background: var(--bynd-warm-subtle);
    }
    button.company.not-enriched {
      cursor: default;
      opacity: 0.78;
    }
    button.company.not-enriched:hover {
      border-color: var(--bynd-border-light);
      background: #fff;
    }
    button.company.active {
      border-color: var(--bynd-brand);
      box-shadow: inset 3px 0 0 var(--bynd-brand);
    }
    .company-name {
      font-weight: 500;
      font-size: 13px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }
    .meta {
      margin-top: 4px;
      color: var(--bynd-text-secondary);
      font-size: 12px;
      line-height: 1.4;
      overflow-wrap: anywhere;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      border-radius: 4px;
      border: 1px solid var(--bynd-border-light);
      padding: 2px 6px;
      font-size: 11px;
      font-weight: 500;
      background: var(--bynd-warm-hover);
      color: var(--bynd-text-secondary);
      margin: 4px 6px 0 0;
      max-width: 100%;
    }
    .badge.bad { background: var(--bynd-error-light); color: #D92D20; border-color: rgba(239, 68, 68, 0.18); }
    .badge.warn { background: var(--bynd-warning-light); color: #B45309; border-color: rgba(245, 158, 11, 0.18); }
    .summary {
      display: grid;
      grid-template-columns: minmax(0, 1.5fr) minmax(260px, 1fr);
      gap: 12px;
      margin-bottom: 12px;
    }
    .panel {
      background: var(--bynd-bg-panel);
      border: 1px solid var(--bynd-border-light);
      border-radius: 6px;
      padding: 14px;
      box-shadow: var(--bynd-shadow-card);
    }
    h2, h3 {
      margin: 0 0 8px;
      font-size: 16px;
      font-weight: 600;
    }
    h3 {
      font-size: 13px;
      color: var(--bynd-text-secondary);
      font-weight: 500;
    }
    .description {
      margin: 8px 0 10px;
      font-size: 14px;
      line-height: 1.5;
      color: var(--bynd-text-primary);
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
    }
    .chip {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 3px 8px;
      border-radius: 4px;
      border: 1px solid var(--bynd-border-light);
      background: var(--bynd-warm-hover);
      color: var(--bynd-text-secondary);
      font-size: 12px;
      font-weight: 500;
    }
    .empty {
      color: var(--bynd-text-secondary);
      font-size: 14px;
      padding: 24px;
      text-align: center;
      border: 1px dashed var(--bynd-border-strong);
      border-radius: 6px;
      background: #fff;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      background: #fff;
      border: 1px solid var(--bynd-border-light);
      border-radius: 6px;
      overflow: hidden;
      font-size: 13px;
    }
    th, td {
      border-bottom: 1px solid var(--bynd-border-light);
      border-right: 1px solid var(--bynd-border-light);
      padding: 8px;
      text-align: left;
      vertical-align: top;
    }
    th:last-child, td:last-child { border-right: 0; }
    th {
      background: rgba(91, 77, 47, 0.031);
      color: var(--bynd-text-primary);
      font-size: 13px;
      font-weight: 500;
      text-transform: none;
      letter-spacing: 0;
      height: 34px;
      position: sticky;
      top: 0;
      z-index: 1;
    }
    td { color: var(--bynd-text-secondary); line-height: 1.45; }
    tbody tr:hover td { background: var(--bynd-warm-subtle); }
    tr:last-child td { border-bottom: 0; }
    .score {
      font-variant-numeric: tabular-nums;
      font-family: "SF Mono", Monaco, "Cascadia Code", "Courier New", monospace;
      font-weight: 600;
      color: var(--bynd-brand);
      white-space: nowrap;
    }
    .products {
      max-width: 420px;
      line-height: 1.35;
    }
    .toolbar {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }
    .toolbar-fields {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: end;
    }
    .toolbar-settings {
      flex-basis: 100%;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 6px;
      padding-top: 4px;
    }
    .settings-label {
      color: var(--bynd-text-secondary);
      font-size: 12px;
      font-weight: 600;
      margin-right: 4px;
    }
    .field-small { width: 110px; }
    .field-medium { width: 168px; }
    .field-revenue {
      width: 260px;
    }
    .range-values {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-bottom: 6px;
    }
    .range-values input {
      height: 31px;
      font-size: 12px;
    }
    .range-slider {
      position: relative;
      height: 26px;
    }
    .range-track {
      position: absolute;
      left: 0;
      right: 0;
      top: 11px;
      height: 4px;
      border-radius: 999px;
      background: var(--bynd-border-light);
    }
    .range-fill {
      position: absolute;
      top: 11px;
      height: 4px;
      border-radius: 999px;
      background: var(--bynd-brand);
    }
    .range-slider input[type="range"] {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 26px;
      padding: 0;
      border: 0;
      background: transparent;
      pointer-events: none;
      appearance: none;
      -webkit-appearance: none;
    }
    .range-slider input[type="range"]::-webkit-slider-thumb {
      width: 16px;
      height: 16px;
      border: 2px solid var(--bynd-brand);
      border-radius: 50%;
      background: #fff;
      box-shadow: var(--bynd-shadow-soft);
      cursor: pointer;
      pointer-events: auto;
      appearance: none;
      -webkit-appearance: none;
    }
    .range-slider input[type="range"]::-moz-range-thumb {
      width: 14px;
      height: 14px;
      border: 2px solid var(--bynd-brand);
      border-radius: 50%;
      background: #fff;
      box-shadow: var(--bynd-shadow-soft);
      cursor: pointer;
      pointer-events: auto;
    }
    .range-caption {
      color: var(--bynd-text-secondary);
      font-size: 11px;
      line-height: 1.2;
    }
    .action-button {
      height: 39px;
      border: 1px solid var(--bynd-brand);
      border-radius: 4px;
      padding: 0 12px;
      background: var(--bynd-brand);
      color: white;
      font-family: var(--bynd-font-sans);
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      white-space: nowrap;
    }
    .action-button:hover { background: var(--bynd-brand-hover); }
    .trace-button {
      margin: 0 0 10px;
      height: 32px;
      border: 1px solid var(--bynd-border-strong);
      border-radius: 4px;
      padding: 0 10px;
      background: #fff;
      color: var(--bynd-text-primary);
      font-family: var(--bynd-font-sans);
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
    }
    .trace-panel {
      display: none;
      margin-bottom: 12px;
    }
    .trace-panel.open { display: block; }
    .trace-section {
      margin-bottom: 12px;
      border: 1px solid var(--bynd-border-light);
      border-radius: 6px;
      background: #fff;
      overflow: hidden;
    }
    .trace-section h3 {
      margin: 0;
      padding: 8px 10px;
      border-bottom: 1px solid var(--bynd-border-light);
      background: rgba(91, 77, 47, 0.031);
      color: var(--bynd-text-primary);
    }
    .trace-section pre {
      margin: 0;
      padding: 10px;
      max-height: 340px;
      overflow: auto;
      white-space: pre-wrap;
      font-size: 11px;
      line-height: 1.45;
      color: var(--bynd-text-primary);
      background: #fff;
    }
    .trace-list {
      padding: 8px 10px;
      max-height: 340px;
      overflow: auto;
    }
    .trace-item {
      padding: 6px 0;
      border-bottom: 1px solid var(--bynd-border-light);
      font-size: 12px;
      line-height: 1.35;
    }
    .trace-item:last-child { border-bottom: 0; }
    .status {
      color: var(--bynd-text-secondary);
      font-size: 12px;
    }
    @media (max-width: 860px) {
      .app-shell { grid-template-columns: 1fr; padding: 0; }
      .rail { display: none; }
      .app-panel { min-height: 100vh; border-radius: 0; }
      main { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--bynd-border-light); max-height: 46vh; }
      .summary { grid-template-columns: 1fr; }
      table { min-width: 780px; }
    }
  </style>
</head>
<body>
  <div class="app-shell">
    <nav class="rail" aria-label="App navigation">
      <div class="rail-brand">
        <img src="/design-assets/ByndLogoFavicon.svg" alt="Bynd">
        <span>INTEL</span>
      </div>
      <div class="rail-nav">
        <div class="rail-item active" title="Peers">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><path d="M3 21V7a2 2 0 0 1 2-2h6v16"/><path d="M13 9h6a2 2 0 0 1 2 2v10"/><path d="M7 9h1"/><path d="M7 13h1"/><path d="M16 13h1"/><path d="M16 17h1"/></svg>
          <span>Peers</span>
        </div>
      </div>
    </nav>
    <div class="app-panel">
      <header>
        <h1>Peer explorer</h1>
        <p>Compare product max-sim with the original company embedding approach.</p>
        <div class="tabs" role="tablist">
          <div class="tab active" role="tab">Companies</div>
        </div>
      </header>
      <main>
        <aside>
          <label for="search">Company</label>
          <input id="search" type="search" placeholder="Search name, CIN, product..." autocomplete="off">
          <div class="controls">
            <label class="check"><input id="includeFlaggedCompanies" type="checkbox" checked> show flagged <span class="tip" tabindex="0" data-tip="Include companies with enrichment quality flags in the selectable company list.">?</span></label>
          </div>
          <div id="companyStatus" class="status">Loading companies...</div>
          <div id="companyList" class="company-list"></div>
        </aside>
        <section>
          <div class="toolbar">
            <div>
              <h2 id="pageTitle">Pick a company</h2>
              <div id="method" class="status"></div>
            </div>
            <div class="toolbar-fields">
              <div class="field-medium">
                <label for="scoringMethod">Scoring <span class="tip" tabindex="0" data-tip="Choose the first-stage scorer that generates candidate peers before optional Cohere reranking.">?</span></label>
                <select id="scoringMethod">
                  <option value="product_max_sim">Product max-sim</option>
                  <option value="company_embedding">Original company embedding</option>
                </select>
              </div>
              <div class="field-small">
                <label for="topK">Peers <span class="tip" tabindex="0" data-tip="Number of peers to display for the first-stage scorer. The rerank button uses a 40+40 union, Cohere top 25, then OpenAI top 10.">?</span></label>
                <input id="topK" type="number" min="1" max="50" value="10">
              </div>
              <div class="field-small">
                <label for="minScore">Min score <span class="tip" tabindex="0" data-tip="Hide candidates below this first-stage similarity score before displaying or reranking.">?</span></label>
                <input id="minScore" type="number" min="0" max="1" step="0.01" value="0">
              </div>
              <div class="field-medium">
                <label for="stateFilter">Registered state <span class="tip" tabindex="0" data-tip="Limit both the company list and peer candidates by the registered state inferred from CIN.">?</span></label>
                <select id="stateFilter">
                  <option value="">All states</option>
                  <option value="Andhra Pradesh">Andhra Pradesh</option>
                  <option value="Assam">Assam</option>
                  <option value="Bihar">Bihar</option>
                  <option value="Chandigarh">Chandigarh</option>
                  <option value="Chhattisgarh">Chhattisgarh</option>
                  <option value="Dadra and Nagar Haveli and Daman and Diu">Dadra and Nagar Haveli and Daman and Diu</option>
                  <option value="Delhi">Delhi</option>
                  <option value="Goa">Goa</option>
                  <option value="Gujarat">Gujarat</option>
                  <option value="Haryana">Haryana</option>
                  <option value="Himachal Pradesh">Himachal Pradesh</option>
                  <option value="Jammu and Kashmir">Jammu and Kashmir</option>
                  <option value="Jharkhand">Jharkhand</option>
                  <option value="Karnataka">Karnataka</option>
                  <option value="Kerala">Kerala</option>
                  <option value="Madhya Pradesh">Madhya Pradesh</option>
                  <option value="Maharashtra">Maharashtra</option>
                  <option value="Odisha">Odisha</option>
                  <option value="Punjab">Punjab</option>
                  <option value="Rajasthan">Rajasthan</option>
                  <option value="Tamil Nadu">Tamil Nadu</option>
                  <option value="Telangana">Telangana</option>
                  <option value="Uttar Pradesh">Uttar Pradesh</option>
                  <option value="Uttarakhand">Uttarakhand</option>
                  <option value="West Bengal">West Bengal</option>
                </select>
              </div>
              <div class="field-revenue">
                <label for="minRevenueRange">Revenue range, Cr <span class="tip" tabindex="0" data-tip="Filter companies and peer candidates by annual revenue in INR crore. Drag the handles or type exact values.">?</span></label>
                <div class="range-values">
                  <input id="minRevenue" type="number" min="0" max="250000" step="100" placeholder="Min">
                  <input id="maxRevenue" type="number" min="0" max="250000" step="100" placeholder="Max">
                </div>
                <div class="range-slider">
                  <div class="range-track"></div>
                  <div id="revenueRangeFill" class="range-fill"></div>
                  <input id="minRevenueRange" type="range" min="0" max="250000" step="100" value="0" aria-label="Minimum revenue">
                  <input id="maxRevenueRange" type="range" min="0" max="250000" step="100" value="250000" aria-label="Maximum revenue">
                </div>
                <div id="revenueRangeCaption" class="range-caption">Any revenue</div>
              </div>
              <button id="rerankButton" class="action-button" type="button">Rerank 40+40 -> 10 <span class="tip" tabindex="0" data-tip="Send the top 40 product max-sim candidates plus top 40 company-embedding candidates to Cohere, then ask OpenAI to select the final top 10 investment-banking peers from Cohere's top 25.">?</span></button>
              <div class="toolbar-settings">
                <span class="settings-label">Peer result settings</span>
                <label class="check"><input id="excludeFlaggedPeers" type="checkbox" checked> hide flagged peers <span class="tip" tabindex="0" data-tip="Exclude peer candidates that have enrichment quality flags.">?</span></label>
                <label class="check"><input id="sameValueChain" type="checkbox"> same value chain <span class="tip" tabindex="0" data-tip="Only keep peers with the same primary value-chain role as the selected company.">?</span></label>
                <label class="check"><input id="sameCustomerType" type="checkbox"> same customer type <span class="tip" tabindex="0" data-tip="Only keep peers with the same customer type classification, such as B2B, B2C, or mixed.">?</span></label>
                <label class="check"><input id="useRevenue" type="checkbox"> use revenue <span class="tip" tabindex="0" data-tip="Add a small score adjustment favoring companies with revenue scale closer to the target.">?</span></label>
                <label class="check"><input id="useEnumWeighting" type="checkbox"> use enum weighting <span class="tip" tabindex="0" data-tip="Add small boosts for matching structured fields like value chain, customer type, end markets, and geography.">?</span></label>
              </div>
            </div>
          </div>
          <div id="targetArea" class="empty">Select a company on the left to see peers.</div>
          <div id="peerArea"></div>
        </section>
      </main>
    </div>
  </div>
  <script>
    const state = { companies: [], selectedCin: null, searchTimer: null };
    const $ = (id) => document.getElementById(id);

    function esc(value) {
      return String(value ?? "").replace(/[&<>"']/g, (c) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[c]));
    }

    function chips(items) {
      if (!items || !items.length) return '<span class="meta">None listed</span>';
      return `<div class="chips">${items.map((x) => `<span class="chip">${esc(x)}</span>`).join("")}</div>`;
    }

    function qualityBadge(company, prefix = "") {
      if (!company.quality_flags) return "";
      const cls = company.quality_severity === "high" ? "bad" : "warn";
      return `<span class="badge ${cls}">${esc(prefix + company.quality_severity)}: ${esc(company.quality_flags)}</span>`;
    }

    function revenueLabel(company) {
      if (company.revenue_crore === null || company.revenue_crore === undefined) return "Revenue unavailable";
      const year = company.revenue_financial_year ? `, FY ${company.revenue_financial_year.slice(0, 4)}` : "";
      return `Revenue: INR ${Number(company.revenue_crore).toLocaleString("en-IN")} Cr${year}`;
    }

    function revenueCroreToRaw(value) {
      if (String(value ?? "").trim() === "") return "";
      const parsed = Number(value);
      if (!Number.isFinite(parsed) || parsed < 0) return "";
      return String(parsed * 10000000);
    }

    function revenueInputValue(id) {
      const value = $(id).value;
      return String(value ?? "").trim() === "" ? "" : value;
    }

    function formatCrore(value) {
      return Number(value).toLocaleString("en-IN");
    }

    function syncRevenueRange(source) {
      const maxBound = Number($("maxRevenueRange").max);
      let minValue = source === "minRange"
        ? Number($("minRevenueRange").value)
        : (source === "minInput" && $("minRevenue").value === "" ? 0 : Number($("minRevenue").value || $("minRevenueRange").value || 0));
      let maxValue = source === "maxRange"
        ? Number($("maxRevenueRange").value)
        : (source === "maxInput" && $("maxRevenue").value === "" ? maxBound : Number($("maxRevenue").value || $("maxRevenueRange").value || maxBound));
      if (!Number.isFinite(minValue)) minValue = 0;
      if (!Number.isFinite(maxValue)) maxValue = maxBound;
      minValue = Math.max(0, Math.min(maxBound, minValue));
      maxValue = Math.max(0, Math.min(maxBound, maxValue));
      if (minValue > maxValue) {
        if (source === "maxRange" || source === "maxInput") minValue = maxValue;
        else maxValue = minValue;
      }
      $("minRevenueRange").value = String(minValue);
      $("maxRevenueRange").value = String(maxValue);
      $("minRevenue").value = minValue === 0 ? "" : String(minValue);
      $("maxRevenue").value = maxValue === maxBound ? "" : String(maxValue);
      const left = (minValue / maxBound) * 100;
      const right = 100 - ((maxValue / maxBound) * 100);
      $("revenueRangeFill").style.left = `${left}%`;
      $("revenueRangeFill").style.right = `${right}%`;
      const minLabel = minValue === 0 ? "0" : formatCrore(minValue);
      const maxLabel = maxValue === maxBound ? `${formatCrore(maxBound)}+` : formatCrore(maxValue);
      $("revenueRangeCaption").textContent = minValue === 0 && maxValue === maxBound
        ? "Any revenue"
        : `INR ${minLabel} Cr to ${maxLabel} Cr`;
    }

    async function api(path) {
      const res = await fetch(path);
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    }

    async function searchCompanies() {
      const q = $("search").value.trim();
      const includeFlagged = $("includeFlaggedCompanies").checked ? "true" : "false";
      const stateCode = $("stateFilter") ? $("stateFilter").value : "";
      const minRevenue = $("minRevenue") ? revenueCroreToRaw(revenueInputValue("minRevenue")) : "";
      const maxRevenue = $("maxRevenue") ? revenueCroreToRaw(revenueInputValue("maxRevenue")) : "";
      $("companyStatus").textContent = "Searching...";
      const data = await api(`/api/companies?q=${encodeURIComponent(q)}&limit=80&include_flagged=${includeFlagged}&state=${encodeURIComponent(stateCode)}&min_revenue=${encodeURIComponent(minRevenue)}&max_revenue=${encodeURIComponent(maxRevenue)}`);
      state.companies = data.companies;
      $("companyStatus").textContent = `${data.companies.length} shown of ${data.total_companies.toLocaleString()} companies`;
      renderCompanies();
      if (!state.selectedCin && data.companies.length) {
        const firstEnriched = data.companies.find((company) => company.is_enriched);
        if (firstEnriched) selectCompany(firstEnriched.cin);
      }
    }

    function renderCompanies() {
      $("companyList").innerHTML = state.companies.map((c) => `
        <button class="company ${c.cin === state.selectedCin ? "active" : ""} ${c.is_enriched ? "" : "not-enriched"}" data-cin="${esc(c.cin)}">
          <div class="company-name">${esc(c.name)}</div>
          <div class="meta">${esc(c.cin)} - ${esc(c.state_name || "Unknown")} - ${esc(c.is_enriched ? (c.value_chain_primary || "Unknown") : "not enriched yet")}</div>
          <div class="meta">${esc(revenueLabel(c))}</div>
          ${c.is_enriched ? "" : '<span class="badge warn">not enriched yet</span>'}
          ${qualityBadge(c)}
        </button>
      `).join("");
      document.querySelectorAll("button.company").forEach((btn) => {
        btn.addEventListener("click", () => {
          const company = state.companies.find((item) => item.cin === btn.dataset.cin);
          if (company && !company.is_enriched) {
            state.selectedCin = null;
            renderCompanies();
            $("pageTitle").textContent = company.name;
            $("method").textContent = "Not enriched yet";
            $("targetArea").className = "empty";
            $("targetArea").textContent = "This company is in the master list, but it has not been enriched yet.";
            $("peerArea").innerHTML = "";
            return;
          }
          selectCompany(btn.dataset.cin);
        });
      });
    }

    async function selectCompany(cin, rerank = false) {
      state.selectedCin = cin;
      renderCompanies();
      $("targetArea").className = "empty";
      $("targetArea").textContent = rerank ? "Running Cohere + OpenAI investment-banking rerank..." : "Loading peers...";
      $("peerArea").innerHTML = "";
      const params = new URLSearchParams({
        cin,
        k: rerank ? "10" : ($("topK").value || "10"),
        exclude_flagged: $("excludeFlaggedPeers").checked ? "true" : "false",
        same_value_chain: $("sameValueChain").checked ? "true" : "false",
        same_customer_type: $("sameCustomerType").checked ? "true" : "false",
        use_revenue: $("useRevenue").checked ? "true" : "false",
        use_enum_weighting: $("useEnumWeighting").checked ? "true" : "false",
        state: $("stateFilter").value || "",
        min_revenue: revenueCroreToRaw(revenueInputValue("minRevenue")),
        max_revenue: revenueCroreToRaw(revenueInputValue("maxRevenue")),
        min_score: $("minScore").value || "0",
        scoring_method: $("scoringMethod").value || "product_max_sim",
      });
      if (rerank) params.set("rerank", "true");
      const data = await api(`/api/peers?${params.toString()}`);
      renderPeers(data);
    }

    function renderTarget(target) {
      $("pageTitle").textContent = target.name;
      $("targetArea").className = "summary";
      $("targetArea").innerHTML = `
        <div class="panel">
          <h2>${esc(target.name)}</h2>
          <div class="meta">${esc(target.cin)} - ${esc(target.website)}</div>
          ${qualityBadge(target, "target ")}
          <p class="description">${esc(target.business_description)}</p>
          <h3>Core products</h3>
          ${chips(target.core_products)}
          <h3 style="margin-top:12px;">Secondary products</h3>
          ${chips(target.secondary_products)}
        </div>
        <div class="panel">
          <h3>Classification</h3>
          <div class="meta">Value chain: <strong>${esc(target.value_chain_primary || "Unknown")}</strong></div>
          <div class="meta">Customer type: <strong>${esc(target.customer_type || "Unknown")}</strong></div>
          <div class="meta">Geography: <strong>${esc(target.geographic_signals || "Unknown")}</strong></div>
          <div class="meta">Registered state: <strong>${esc(target.state_name || "Unknown")}</strong></div>
          <div class="meta">Derived category: <strong>${esc((target.derived_primary_categories || []).join(" | ") || "None")}</strong></div>
          <div class="meta"><strong>${esc(revenueLabel(target))}</strong></div>
          <h3 style="margin-top:12px;">End markets</h3>
          ${chips(target.end_markets)}
        </div>
      `;
    }

    function scoreMeta(item) {
      const parts = [];
      if (item.sources && item.sources.length) parts.push(item.sources.join("+"));
      if (item.product_candidate_score !== null && item.product_candidate_score !== undefined) parts.push(`product ${Number(item.product_candidate_score).toFixed(6)}`);
      if (item.company_candidate_score !== null && item.company_candidate_score !== undefined) parts.push(`embedding ${Number(item.company_candidate_score).toFixed(6)}`);
      if (item.cohere_rerank_score !== null && item.cohere_rerank_score !== undefined) parts.push(`cohere ${Number(item.cohere_rerank_score).toFixed(6)}`);
      if (item.pre_openai_rank) parts.push(`OpenAI input #${item.pre_openai_rank}`);
      return parts.join(" | ");
    }

    function traceList(items) {
      if (!items || !items.length) return '<div class="trace-list"><span class="meta">None</span></div>';
      return `<div class="trace-list">${items.map((item) => `
        <div class="trace-item">
          <strong>${esc(item.rank)}. ${esc(item.name)}</strong>
          <div class="meta">${esc(item.cin || "")}</div>
          ${scoreMeta(item) ? `<div class="meta">${esc(scoreMeta(item))}</div>` : ""}
          ${item.value_chain ? `<div class="meta">Value chain: ${esc(item.value_chain)}</div>` : ""}
          ${item.customer_type ? `<div class="meta">Customer: ${esc(item.customer_type)}</div>` : ""}
          ${item.revenue ? `<div class="meta">Revenue: ${esc(item.revenue)}</div>` : ""}
          ${item.revenue_crore !== null && item.revenue_crore !== undefined ? `<div class="meta">Revenue: INR ${Number(item.revenue_crore).toLocaleString("en-IN")} Cr</div>` : ""}
          ${(item.core_products || []).length ? `<div class="meta">Core: ${esc(item.core_products.join(" | "))}</div>` : ""}
          ${(item.secondary_products || []).length ? `<div class="meta">Secondary: ${esc(item.secondary_products.join(" | "))}</div>` : ""}
        </div>
      `).join("")}</div>`;
    }

    function renderTrace(trace) {
      if (!trace) return "";
      return `
        <button id="traceToggle" class="trace-button" type="button">Show rerank trace</button>
        <div id="tracePanel" class="trace-panel">
          <div class="trace-section">
            <h3>Initial 40+40 union (${(trace.union_candidates || []).length})</h3>
            ${traceList(trace.union_candidates)}
          </div>
          <div class="trace-section">
            <h3>Cohere prompt</h3>
            <pre>${esc(trace.cohere_prompt || "")}</pre>
          </div>
          <div class="trace-section">
            <h3>Cohere top 25</h3>
            ${traceList(trace.cohere_shortlist)}
          </div>
          <div class="trace-section">
            <h3>OpenAI prompt</h3>
            <pre>${esc(trace.openai_prompt || "")}</pre>
          </div>
          <div class="trace-section">
            <h3>OpenAI final 10</h3>
            ${traceList(trace.openai_final)}
          </div>
        </div>
      `;
    }

    function renderPeers(data) {
      renderTarget(data.target);
      $("method").textContent = `${data.method} - ${data.peers.length} peers shown`;
      if (!data.peers.length) {
        $("peerArea").innerHTML = '<div class="empty">No peers matched the current filters.</div>';
        return;
      }
      $("peerArea").innerHTML = `
        ${renderTrace(data.rerank_trace)}
        <table>
          <thead>
            <tr>
              <th>Rank</th>
              <th>Peer</th>
              <th>Score</th>
              <th>Core products</th>
              <th>Value chain</th>
              <th>Customer</th>
              <th>State / revenue</th>
              <th>Quality</th>
            </tr>
          </thead>
          <tbody>
            ${data.peers.map((p, i) => `
              <tr>
                <td>${i + 1}</td>
                <td>
                  <strong>${esc(p.name)}</strong>
                  <div class="meta">${esc(p.cin)}</div>
                  <div class="meta">${esc(p.business_description)}</div>
                </td>
                <td class="score">
                  ${p.final_score.toFixed(6)}
                  <div class="meta">${p.cohere_rerank_score !== null && p.cohere_rerank_score !== undefined ? `cohere ${p.cohere_rerank_score.toFixed(6)}` : (p.product_coverage_score !== null && p.product_coverage_score !== undefined ? `max-sim ${p.product_coverage_score.toFixed(6)}` : `cos ${p.company_embedding_score.toFixed(6)}`)}</div>
                  ${p.pre_rerank_rank ? `<div class="meta">pre #${p.pre_rerank_rank} ${Number(p.pre_rerank_score || 0).toFixed(6)}</div>` : ""}
                  ${(p.enum_adjustment || p.revenue_adjustment) ? `<div class="meta">adj ${Number((p.enum_adjustment || 0) + (p.revenue_adjustment || 0)).toFixed(4)}</div>` : ""}
                </td>
                <td class="products">${esc((p.core_products || []).join(" | "))}</td>
                <td>${esc(p.value_chain_primary || "Unknown")}</td>
                <td>${esc(p.customer_type || "Unknown")}</td>
                <td>
                  <strong>${esc(p.state_name || "Unknown")}</strong>
                  <div class="meta">${esc(revenueLabel(p))}</div>
                </td>
                <td>${qualityBadge(p, "") || '<span class="badge">ok</span>'}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      `;
      const traceToggle = $("traceToggle");
      if (traceToggle) {
        traceToggle.addEventListener("click", () => {
          const panel = $("tracePanel");
          const open = panel.classList.toggle("open");
          traceToggle.textContent = open ? "Hide rerank trace" : "Show rerank trace";
        });
      }
    }

    function scheduleSearch(resetSelected = true) {
      clearTimeout(state.searchTimer);
      state.searchTimer = setTimeout(() => {
        if (resetSelected) state.selectedCin = null;
        searchCompanies().catch((err) => {
          $("companyStatus").textContent = err.message;
        });
      }, 180);
    }

    ["search", "includeFlaggedCompanies"].forEach((id) => $(id).addEventListener("input", () => scheduleSearch(true)));
    $("stateFilter").addEventListener("input", () => {
      scheduleSearch(false);
      if (state.selectedCin) selectCompany(state.selectedCin);
    });
    Object.entries({
      minRevenue: "minInput",
      maxRevenue: "maxInput",
      minRevenueRange: "minRange",
      maxRevenueRange: "maxRange",
    }).forEach(([id, source]) => {
      $(id).addEventListener("input", () => {
        syncRevenueRange(source);
        scheduleSearch(false);
        if (state.selectedCin) selectCompany(state.selectedCin);
      });
    });
    ["excludeFlaggedPeers", "sameValueChain", "sameCustomerType", "useRevenue", "useEnumWeighting", "scoringMethod", "topK", "minScore"].forEach((id) => {
      $(id).addEventListener("input", () => {
        if (state.selectedCin) selectCompany(state.selectedCin);
      });
    });
    $("rerankButton").addEventListener("click", () => {
      if (state.selectedCin) {
        selectCompany(state.selectedCin, true).catch((err) => {
          $("targetArea").className = "empty";
          $("targetArea").textContent = err.message;
        });
      }
    });
    syncRevenueRange();
    searchCompanies().catch((err) => {
      $("companyStatus").textContent = err.message;
      $("targetArea").textContent = err.message;
    });
  </script>
</body>
</html>
"""


def parse_bool(params: dict[str, list[str]], key: str, default: bool) -> bool:
    raw = params.get(key, [str(default).lower()])[0].strip().lower()
    return raw in {"1", "true", "yes", "on"}


def parse_int(params: dict[str, list[str]], key: str, default: int, min_value: int, max_value: int) -> int:
    try:
        value = int(params.get(key, [str(default)])[0])
    except ValueError:
        value = default
    return max(min_value, min(max_value, value))


def parse_float(params: dict[str, list[str]], key: str, default: float, min_value: float, max_value: float) -> float:
    try:
        value = float(params.get(key, [str(default)])[0])
    except ValueError:
        value = default
    return max(min_value, min(max_value, value))


def parse_optional_float(params: dict[str, list[str]], key: str) -> float | None:
    raw = params.get(key, [""])[0].strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


class PeerHandler(BaseHTTPRequestHandler):
    data: PeerData

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format % args))

    def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, text: str, content_type: str = "text/html; charset=utf-8") -> None:
        body = text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        body = HTML_PAGE.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                self.send_text(HTML_PAGE)
            elif parsed.path.startswith("/design-assets/") or parsed.path.startswith("/design-fonts/"):
                asset_name = Path(parsed.path).name
                base_path = DESIGN_SYSTEM_FONTS if parsed.path.startswith("/design-fonts/") else DESIGN_SYSTEM_ASSETS
                asset_path = base_path / asset_name
                if not asset_path.exists() or not asset_path.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                    return
                content_type = mimetypes.guess_type(str(asset_path))[0] or "application/octet-stream"
                body = asset_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif parsed.path == "/api/companies":
                query = params.get("q", [""])[0]
                limit = parse_int(params, "limit", 80, 1, 500)
                include_flagged = parse_bool(params, "include_flagged", True)
                state_filter = params.get("state", [""])[0].strip()
                companies = self.data.search(
                    query=query,
                    limit=limit,
                    include_flagged=include_flagged,
                    state_filter=state_filter,
                    min_revenue=parse_optional_float(params, "min_revenue"),
                    max_revenue=parse_optional_float(params, "max_revenue"),
                )
                self.send_json(
                    {
                        "companies": companies,
                        "shown": len(companies),
                        "total_companies": len(self.data.search_cins),
                        "total_enriched_companies": len(self.data.cins),
                        "total_company_embedding_companies": len(self.data.company_cins),
                        "total_product_peerable_companies": len(self.data.product_cins),
                        "quality_flagged_companies": len(self.data.flags_by_cin),
                    }
                )
            elif parsed.path == "/api/company":
                cin = params.get("cin", [""])[0].strip().upper()
                self.send_json({"company": self.data.serialize_company(cin)})
            elif parsed.path == "/api/peers":
                cin = params.get("cin", [""])[0].strip().upper()
                use_rerank = parse_bool(params, "rerank", False)
                requested_limit = parse_int(params, "k", 10, 1, 50)
                filters = {
                    "exclude_flagged": parse_bool(params, "exclude_flagged", True),
                    "same_value_chain": parse_bool(params, "same_value_chain", False),
                    "same_customer_type": parse_bool(params, "same_customer_type", False),
                    "use_revenue": parse_bool(params, "use_revenue", False),
                    "use_enum_weighting": parse_bool(params, "use_enum_weighting", False),
                    "state_filter": params.get("state", [""])[0].strip(),
                    "min_revenue": parse_optional_float(params, "min_revenue"),
                    "max_revenue": parse_optional_float(params, "max_revenue"),
                    "min_score": parse_float(params, "min_score", 0.0, 0.0, 1.0),
                }
                if use_rerank:
                    payload = build_union_rerank_payload(self.data, cin, **filters)
                    rerank_limit = 10
                    union_candidate_count = len(payload["peers"])
                    try:
                        cohere_peers, cohere_metadata = rerank_peers(
                            target=payload["target"],
                            peers=payload["peers"],
                            top_n=DEFAULT_OPENAI_CANDIDATE_COUNT,
                            use_derived_categories=False,
                        )
                    except Exception as exc:
                        cohere_peers = [dict(peer) for peer in payload["peers"][:DEFAULT_OPENAI_CANDIDATE_COUNT]]
                        cohere_metadata = {
                            "provider": "cohere",
                            "used": False,
                            "fallback": True,
                            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                            "candidate_count": len(payload["peers"]),
                            "returned_count": len(cohere_peers),
                            "derived_categories": False,
                        }
                    final_peers, openai_metadata = select_final_peers_with_openai(
                        target=payload["target"],
                        candidates=cohere_peers,
                        final_count=rerank_limit,
                    )
                    cohere_prompt = rerank_query(payload["target"], use_derived_categories=False)
                    openai_prompt = build_openai_final_prompt(payload["target"], cohere_peers)
                    payload["rerank_trace"] = {
                        "union_candidates": [
                            summarize_trace_company(peer, index)
                            for index, peer in enumerate(payload["peers"], start=1)
                        ],
                        "cohere_prompt": cohere_prompt,
                        "cohere_documents": [
                            {
                                "rank": index,
                                "name": peer.get("name", ""),
                                "cin": peer.get("cin", ""),
                                "document": format_company_for_rerank(
                                    peer,
                                    target=payload["target"],
                                    use_derived_categories=False,
                                    include_match_evidence=False,
                                ),
                            }
                            for index, peer in enumerate(payload["peers"], start=1)
                        ],
                        "cohere_shortlist": [
                            summarize_trace_company(peer, index)
                            for index, peer in enumerate(cohere_peers, start=1)
                        ],
                        "openai_prompt": openai_prompt,
                        "openai_candidates": [
                            summarize_openai_candidate(peer, index)
                            for index, peer in enumerate(cohere_peers, start=1)
                        ],
                        "openai_final": [
                            summarize_trace_company(peer, index)
                            for index, peer in enumerate(final_peers, start=1)
                        ],
                        "openai_metadata": openai_metadata,
                        "cohere_metadata": cohere_metadata,
                    }
                    payload["peers"] = final_peers
                    payload["method"] = (
                        f"{payload['method']} + "
                        f"{'Cohere fallback' if cohere_metadata.get('fallback') else 'Cohere rerank'} "
                        f"{cohere_metadata['candidate_count']} -> "
                        f"{len(cohere_peers)} + OpenAI final {len(cohere_peers)} -> {len(final_peers)}"
                    )
                    payload["rerank"] = {
                        "cohere": cohere_metadata,
                        "openai": openai_metadata,
                        "candidate_count": cohere_metadata["candidate_count"],
                        "returned_count": len(final_peers),
                        "candidate_count_before_filter": union_candidate_count,
                        "candidate_count_after_filter": union_candidate_count,
                        "union_candidate_count": cohere_metadata["candidate_count"],
                        "cohere_candidate_count": cohere_metadata["candidate_count"],
                        "cohere_returned_count": len(cohere_peers),
                        "cohere_fallback": cohere_metadata.get("fallback", False),
                        "openai_candidate_count": openai_metadata["candidate_count"],
                        "openai_returned_count": openai_metadata["returned_count"],
                        "openai_fallback": openai_metadata["fallback"],
                    }
                    payload["filters"]["rerank"] = True
                    payload["filters"]["use_derived_categories"] = False
                    payload["filters"]["rerank_candidate_count"] = cohere_metadata["candidate_count"]
                else:
                    payload = self.data.peers(
                        cin=cin,
                        limit=requested_limit,
                        **filters,
                        scoring_method=params.get("scoring_method", ["product_max_sim"])[0].strip(),
                    )
                self.send_json(payload)
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except KeyError as exc:
            self.send_json({"error": f"Company not found: {html.escape(str(exc))}"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:  # Keep local UI errors visible in the browser.
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local company peer explorer UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--company-embeddings", type=Path, default=DEFAULT_COMPANY_EMBEDDINGS)
    parser.add_argument(
        "--product-embeddings",
        dest="product_embeddings",
        type=Path,
        default=DEFAULT_PRODUCT_EMBEDDINGS,
        help="Product-item embeddings JSONL.",
    )
    parser.add_argument("--flags", type=Path, default=DEFAULT_FLAGS)
    parser.add_argument("--full-list", type=Path, default=DEFAULT_FULL_LIST)
    args = parser.parse_args()

    data = PeerData(args.canonical, args.company_embeddings, args.product_embeddings, args.flags, args.full_list)
    PeerHandler.data = data
    server = ThreadingHTTPServer((args.host, args.port), PeerHandler)
    print(
        f"Loaded {len(data.cins):,} companies "
        f"({len(data.product_cins):,} product-peerable, {len(data.company_cins):,} company embeddings)"
    )
    print(f"Open http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
