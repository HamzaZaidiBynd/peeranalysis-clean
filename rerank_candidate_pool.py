from __future__ import annotations

from typing import Any


def build_union_rerank_payload(
    data: Any,
    cin: str,
    *,
    exclude_flagged: bool,
    same_value_chain: bool,
    same_customer_type: bool,
    use_revenue: bool,
    use_enum_weighting: bool,
    state_filter: str,
    min_revenue: float | None,
    max_revenue: float | None,
    min_score: float,
) -> dict[str, Any]:
    common_args = {
        "cin": cin,
        "limit": 40,
        "exclude_flagged": exclude_flagged,
        "same_value_chain": same_value_chain,
        "same_customer_type": same_customer_type,
        "use_revenue": use_revenue,
        "use_enum_weighting": use_enum_weighting,
        "state_filter": state_filter,
        "min_revenue": min_revenue,
        "max_revenue": max_revenue,
        "min_score": min_score,
    }
    source_payloads: list[tuple[str, dict[str, Any]]] = []
    errors: list[str] = []
    for scoring_method in ("product_max_sim", "company_embedding"):
        try:
            source_payloads.append(
                (
                    scoring_method,
                    data.peers(
                        **common_args,
                        scoring_method=scoring_method,
                    ),
                )
            )
        except KeyError as exc:
            errors.append(str(exc))

    if not source_payloads:
        raise KeyError("; ".join(errors) or cin)

    by_cin: dict[str, dict[str, Any]] = {}
    for source, payload in source_payloads:
        for peer in payload["peers"]:
            peer_cin = peer.get("cin", "")
            if not peer_cin:
                continue
            item = by_cin.get(peer_cin)
            if item is None:
                item = dict(peer)
                item["rerank_candidate_sources"] = []
                item["product_candidate_score"] = None
                item["company_candidate_score"] = None
                by_cin[peer_cin] = item
            if source not in item["rerank_candidate_sources"]:
                item["rerank_candidate_sources"].append(source)

            score = peer.get("base_similarity_score")
            if source == "product_max_sim":
                item["product_candidate_score"] = score
                item["product_coverage_score"] = peer.get("product_coverage_score")
            else:
                item["company_candidate_score"] = score
                item["company_embedding_score"] = peer.get("company_embedding_score")

            item["final_score"] = max(
                float(item.get("final_score") or 0.0),
                float(peer.get("final_score") or 0.0),
            )
            item["base_similarity_score"] = max(
                float(item.get("base_similarity_score") or 0.0),
                float(score or 0.0),
            )

    if hasattr(data, "product_coverage_scores"):
        for peer_cin, score in data.product_coverage_scores(cin).items():
            if peer_cin in by_cin:
                by_cin[peer_cin]["product_candidate_score"] = round(float(score), 6)
                by_cin[peer_cin]["product_coverage_score"] = round(float(score), 6)
    if hasattr(data, "company_embedding_scores"):
        for peer_cin, score in data.company_embedding_scores(cin).items():
            if peer_cin in by_cin:
                by_cin[peer_cin]["company_candidate_score"] = round(float(score), 6)
                by_cin[peer_cin]["company_embedding_score"] = round(float(score), 6)

    peers = list(by_cin.values())
    return {
        "target": source_payloads[0][1]["target"],
        "peers": peers,
        "method": "product max-sim top 40 + company embedding top 40 union",
        "scoring_method": "product_embedding_union",
        "filters": {
            **source_payloads[0][1]["filters"],
            "scoring_method": "product_embedding_union",
        },
        "candidate_source_counts": {
            source: len(payload["peers"])
            for source, payload in source_payloads
        },
    }
