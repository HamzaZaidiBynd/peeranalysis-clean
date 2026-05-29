from __future__ import annotations

import json
import mimetypes
from http import HTTPStatus
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file

from cohere_rerank import format_company_for_rerank, rerank_peers, rerank_query
from openai_final_rerank import (
    DEFAULT_OPENAI_CANDIDATE_COUNT,
    build_openai_final_prompt,
    select_final_peers_with_openai,
    summarize_openai_candidate,
)
from peer_ui import HTML_PAGE
from rerank_candidate_pool import build_union_rerank_payload
from vercel_peer_data import get_peer_data


app = Flask(__name__)
ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "public"


def parse_bool(name: str, default: bool) -> bool:
    raw = request.args.get(name, str(default).lower()).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def parse_int(name: str, default: int, min_value: int, max_value: int) -> int:
    try:
        value = int(request.args.get(name, str(default)))
    except ValueError:
        value = default
    return max(min_value, min(max_value, value))


def parse_float(name: str, default: float, min_value: float, max_value: float) -> float:
    try:
        value = float(request.args.get(name, str(default)))
    except ValueError:
        value = default
    return max(min_value, min(max_value, value))


def parse_optional_float(name: str) -> float | None:
    raw = request.args.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def json_response(payload: dict, status: HTTPStatus = HTTPStatus.OK) -> Response:
    return Response(
        json.dumps(payload, ensure_ascii=False),
        status=int(status),
        content_type="application/json; charset=utf-8",
    )


def summarize_trace_company(company: dict, index: int) -> dict:
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


@app.get("/")
def index() -> Response:
    return Response(HTML_PAGE, content_type="text/html; charset=utf-8")


@app.get("/design-assets/<path:name>")
def design_assets(name: str):
    asset_path = STATIC_DIR / "design-assets" / Path(name).name
    if not asset_path.exists():
        return json_response({"error": "Not found"}, HTTPStatus.NOT_FOUND)
    return send_file(asset_path, mimetype=mimetypes.guess_type(str(asset_path))[0])


@app.get("/design-fonts/<path:name>")
def design_fonts(name: str):
    font_path = STATIC_DIR / "design-fonts" / Path(name).name
    if not font_path.exists():
        return json_response({"error": "Not found"}, HTTPStatus.NOT_FOUND)
    return send_file(font_path, mimetype=mimetypes.guess_type(str(font_path))[0])


@app.get("/api/companies")
def companies() -> Response:
    data = get_peer_data()
    results = data.search(
        query=request.args.get("q", ""),
        limit=parse_int("limit", 80, 1, 500),
        include_flagged=parse_bool("include_flagged", True),
        state_filter=request.args.get("state", "").strip(),
        min_revenue=parse_optional_float("min_revenue"),
        max_revenue=parse_optional_float("max_revenue"),
    )
    return jsonify(
        {
            "companies": results,
            "shown": len(results),
            "total_companies": len(data.search_cins),
            "total_enriched_companies": len(data.company_cins),
            "total_company_embedding_companies": len(data.company_cins),
            "total_product_peerable_companies": len(data.product_cins),
            "quality_flagged_companies": len(data.flags_by_cin),
        }
    )


@app.get("/api/company")
def company() -> Response:
    data = get_peer_data()
    cin = request.args.get("cin", "").strip().upper()
    try:
        return jsonify({"company": data.serialize_company(cin)})
    except KeyError:
        return json_response({"error": f"Company not found: {cin}"}, HTTPStatus.NOT_FOUND)


@app.get("/api/peers")
def peers() -> Response:
    data = get_peer_data()
    cin = request.args.get("cin", "").strip().upper()
    requested_limit = parse_int("k", 10, 1, 50)
    use_rerank = parse_bool("rerank", False)
    filters = {
        "exclude_flagged": parse_bool("exclude_flagged", True),
        "same_value_chain": parse_bool("same_value_chain", False),
        "same_customer_type": parse_bool("same_customer_type", False),
        "use_revenue": parse_bool("use_revenue", False),
        "use_enum_weighting": parse_bool("use_enum_weighting", False),
        "state_filter": request.args.get("state", "").strip(),
        "min_revenue": parse_optional_float("min_revenue"),
        "max_revenue": parse_optional_float("max_revenue"),
        "min_score": parse_float("min_score", 0.0, 0.0, 1.0),
    }
    try:
        if use_rerank:
            payload = build_union_rerank_payload(data, cin, **filters)
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
            payload = data.peers(
                cin=cin,
                limit=requested_limit,
                **filters,
                scoring_method=request.args.get("scoring_method", "product_max_sim").strip(),
            )
        return jsonify(payload)
    except KeyError as exc:
        return json_response({"error": f"Company not found: {exc}"}, HTTPStatus.NOT_FOUND)
    except Exception as exc:
        return json_response({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


@app.get("/healthz")
def healthz() -> Response:
    data = get_peer_data()
    return jsonify(
        {
            "ok": True,
            "searchable_companies": len(data.search_cins),
            "enriched_companies": len(data.company_cins),
            "product_peerable_companies": len(data.product_cins),
        }
    )
