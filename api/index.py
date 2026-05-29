from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

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


app = FastAPI(title="Peeranalysis API", version="1.0.0")
ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "public"


def parse_bool_value(value: str | bool | None, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_optional_float_value(value: str | None) -> float | None:
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def parse_float_value(value: str | None, default: float, min_value: float, max_value: float) -> float:
    try:
        parsed = float(value if value is not None else default)
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(max_value, parsed))


def peer_limit(k: int | None, limit: int | None, *, default: int = 10) -> int:
    value = limit if limit is not None else k
    if value is None:
        value = default
    return max(1, min(40, int(value)))


def resolve_company_cin(data: Any, cin: str | None, query: str | None) -> str:
    normalized_cin = (cin or "").strip().upper()
    if normalized_cin:
        if normalized_cin not in data.rows_by_cin:
            raise HTTPException(status_code=404, detail=f"Company not found: {normalized_cin}")
        return normalized_cin

    raw_query = (query or "").strip()
    if not raw_query:
        raise HTTPException(status_code=400, detail="Provide either cin or name/q")

    matches = data.search(
        query=raw_query,
        limit=10,
        include_flagged=True,
        state_filter="",
        min_revenue=None,
        max_revenue=None,
    )
    exact_matches = [
        company
        for company in matches
        if company.get("name", "").strip().lower() == raw_query.lower()
        or company.get("cin", "").strip().upper() == raw_query.upper()
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]["cin"].strip().upper()
    if len(matches) == 1:
        return matches[0]["cin"].strip().upper()
    if matches:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Multiple companies matched. Retry with cin or exact name.",
                "matches": [
                    {
                        "cin": company.get("cin", ""),
                        "name": company.get("name", ""),
                        "state_name": company.get("state_name", ""),
                        "is_enriched": company.get("is_enriched"),
                        "revenue_crore": company.get("revenue_crore"),
                    }
                    for company in matches
                ],
            },
        )
    raise HTTPException(status_code=404, detail=f"Company not found: {raw_query}")


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


def cohere_openai_rerank(payload: dict[str, Any], rerank_limit: int, union_candidate_count: int) -> dict[str, Any]:
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
    openai_prompt = build_openai_final_prompt(payload["target"], cohere_peers, final_count=rerank_limit)
    payload["rerank_trace"] = {
        "mode": "cohere_openai",
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
        "mode": "cohere_openai",
        "cohere": cohere_metadata,
        "openai": openai_metadata,
        "candidate_count": cohere_metadata["candidate_count"],
        "returned_count": len(final_peers),
        "requested_count": rerank_limit,
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
    payload["filters"]["rerank_mode"] = "cohere_openai"
    payload["filters"]["use_derived_categories"] = False
    payload["filters"]["rerank_candidate_count"] = cohere_metadata["candidate_count"]
    payload["filters"]["requested_peer_count"] = rerank_limit
    return payload


def direct_openai_rerank(payload: dict[str, Any], rerank_limit: int, union_candidate_count: int) -> dict[str, Any]:
    openai_candidates = [dict(peer) for peer in payload["peers"]]
    final_peers, openai_metadata = select_final_peers_with_openai(
        target=payload["target"],
        candidates=openai_candidates,
        final_count=rerank_limit,
    )
    openai_prompt = build_openai_final_prompt(payload["target"], openai_candidates, final_count=rerank_limit)
    payload["rerank_trace"] = {
        "mode": "openai_direct",
        "union_candidates": [
            summarize_trace_company(peer, index)
            for index, peer in enumerate(payload["peers"], start=1)
        ],
        "openai_prompt": openai_prompt,
        "openai_candidates": [
            summarize_openai_candidate(peer, index)
            for index, peer in enumerate(openai_candidates, start=1)
        ],
        "openai_final": [
            summarize_trace_company(peer, index)
            for index, peer in enumerate(final_peers, start=1)
        ],
        "openai_metadata": openai_metadata,
    }
    payload["peers"] = final_peers
    payload["method"] = (
        f"{payload['method']} + OpenAI direct final "
        f"{len(openai_candidates)} -> {len(final_peers)}"
    )
    payload["rerank"] = {
        "mode": "openai_direct",
        "openai": openai_metadata,
        "candidate_count": len(openai_candidates),
        "returned_count": len(final_peers),
        "requested_count": rerank_limit,
        "candidate_count_before_filter": union_candidate_count,
        "candidate_count_after_filter": union_candidate_count,
        "union_candidate_count": len(openai_candidates),
        "openai_candidate_count": openai_metadata["candidate_count"],
        "openai_returned_count": openai_metadata["returned_count"],
        "openai_fallback": openai_metadata["fallback"],
    }
    payload["filters"]["rerank_mode"] = "openai_direct"
    payload["filters"]["rerank_candidate_count"] = len(openai_candidates)
    payload["filters"]["requested_peer_count"] = rerank_limit
    return payload


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(HTML_PAGE)


@app.get("/design-assets/{name:path}")
def design_assets(name: str) -> FileResponse:
    asset_path = STATIC_DIR / "design-assets" / Path(name).name
    if not asset_path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(asset_path, media_type=mimetypes.guess_type(str(asset_path))[0])


@app.get("/design-fonts/{name:path}")
def design_fonts(name: str) -> FileResponse:
    font_path = STATIC_DIR / "design-fonts" / Path(name).name
    if not font_path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(font_path, media_type=mimetypes.guess_type(str(font_path))[0])


@app.get("/api/companies")
def companies(
    q: str = "",
    limit: int = Query(80, ge=1, le=500),
    include_flagged: str | bool = True,
    state: str = "",
    min_revenue: str | None = None,
    max_revenue: str | None = None,
) -> dict[str, Any]:
    data = get_peer_data()
    results = data.search(
        query=q,
        limit=limit,
        include_flagged=parse_bool_value(include_flagged, True),
        state_filter=state.strip(),
        min_revenue=parse_optional_float_value(min_revenue),
        max_revenue=parse_optional_float_value(max_revenue),
    )
    return {
        "companies": results,
        "shown": len(results),
        "total_companies": len(data.search_cins),
        "total_enriched_companies": len(data.company_cins),
        "total_company_embedding_companies": len(data.company_cins),
        "total_product_peerable_companies": len(data.product_cins),
        "quality_flagged_companies": len(data.flags_by_cin),
    }


@app.get("/api/company")
def company(cin: str) -> dict[str, Any]:
    data = get_peer_data()
    normalized_cin = cin.strip().upper()
    try:
        return {"company": data.serialize_company(normalized_cin)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Company not found: {normalized_cin}") from exc


@app.get("/api/peers", response_model=None)
def peers(
    cin: str | None = None,
    name: str | None = None,
    q: str | None = None,
    k: int | None = Query(None, ge=1, le=40),
    limit: int | None = Query(None, ge=1, le=40),
    rerank: str | bool = False,
    rerank_mode: str = "openai_direct",
    exclude_flagged: str | bool = True,
    same_value_chain: str | bool = False,
    same_customer_type: str | bool = False,
    use_revenue: str | bool = False,
    use_enum_weighting: str | bool = False,
    state: str = "",
    min_revenue: str | None = None,
    max_revenue: str | None = None,
    min_score: str | None = "0",
    scoring_method: str = "product_max_sim",
):
    data = get_peer_data()
    normalized_cin = resolve_company_cin(data, cin, name or q)
    requested_limit = peer_limit(k, limit)
    filters = {
        "exclude_flagged": parse_bool_value(exclude_flagged, True),
        "same_value_chain": parse_bool_value(same_value_chain, False),
        "same_customer_type": parse_bool_value(same_customer_type, False),
        "use_revenue": parse_bool_value(use_revenue, False),
        "use_enum_weighting": parse_bool_value(use_enum_weighting, False),
        "state_filter": state.strip(),
        "min_revenue": parse_optional_float_value(min_revenue),
        "max_revenue": parse_optional_float_value(max_revenue),
        "min_score": parse_float_value(min_score, 0.0, 0.0, 1.0),
    }
    try:
        if parse_bool_value(rerank, False):
            payload = build_union_rerank_payload(data, normalized_cin, **filters)
            union_candidate_count = len(payload["peers"])
            mode = rerank_mode.strip().lower() or "openai_direct"
            if mode == "cohere_openai":
                payload = cohere_openai_rerank(payload, requested_limit, union_candidate_count)
            else:
                payload = direct_openai_rerank(payload, requested_limit, union_candidate_count)
            payload["filters"]["rerank"] = True
        else:
            payload = data.peers(
                cin=normalized_cin,
                limit=requested_limit,
                **filters,
                scoring_method=scoring_method.strip(),
            )
            payload["filters"]["requested_peer_count"] = requested_limit
        return payload
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Company not found: {exc}") from exc
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    data = get_peer_data()
    return {
        "ok": True,
        "searchable_companies": len(data.search_cins),
        "enriched_companies": len(data.company_cins),
        "product_peerable_companies": len(data.product_cins),
    }
