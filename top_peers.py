from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from cohere_rerank import rerank_peers
from openai_final_rerank import DEFAULT_OPENAI_CANDIDATE_COUNT, select_final_peers_with_openai
from rerank_candidate_pool import build_union_rerank_payload
from vercel_peer_data import get_peer_data


DEFAULT_FILTERS = {
    "exclude_flagged": True,
    "same_value_chain": False,
    "same_customer_type": False,
    "use_revenue": False,
    "use_enum_weighting": False,
    "state_filter": "",
    "min_revenue": None,
    "max_revenue": None,
    "min_score": 0.0,
}


def load_local_env(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip("'\"")


def find_company(data: Any, query: str) -> dict[str, Any]:
    query = query.strip()
    query_upper = query.upper()
    if query_upper in data.rows_by_cin:
        return data.serialize_company(query_upper)

    matches = data.search(
        query=query,
        limit=10,
        include_flagged=True,
        state_filter="",
        min_revenue=None,
        max_revenue=None,
    )
    if not matches:
        raise SystemExit(f"No company found for: {query}")

    exact_matches = [
        company
        for company in matches
        if company["cin"].upper() == query_upper or company["name"].strip().lower() == query.lower()
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(matches) == 1:
        return matches[0]

    print("\nPick a company:")
    for index, company in enumerate(matches, start=1):
        revenue = company.get("revenue_crore")
        revenue_label = f"INR {revenue:,.2f} Cr" if revenue is not None else "Revenue unknown"
        enriched = "enriched" if company.get("is_enriched") else "not enriched"
        print(f"{index:>2}. {company['name']} | {company['cin']} | {company.get('state_name', '')} | {revenue_label} | {enriched}")

    while True:
        raw = input("Company number: ").strip()
        try:
            choice = int(raw)
        except ValueError:
            print("Enter a number from the list.")
            continue
        if 1 <= choice <= len(matches):
            return matches[choice - 1]
        print("Enter a number from the list.")


def rank_top_peers(data: Any, cin: str, filters: dict[str, Any], final_count: int) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    payload = build_union_rerank_payload(data, cin, **filters)
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
        final_count=max(final_count, 10),
    )
    metadata = {
        "method": (
            f"{payload['method']} + "
            f"{'Cohere fallback' if cohere_metadata.get('fallback') else 'Cohere rerank'} "
            f"{cohere_metadata['candidate_count']} -> {len(cohere_peers)} + "
            f"OpenAI final {len(cohere_peers)} -> {len(final_peers)}"
        ),
        "union_candidate_count": len(payload["peers"]),
        "cohere": cohere_metadata,
        "openai": openai_metadata,
    }
    return payload["target"], final_peers[:final_count], metadata


def print_text(target: dict[str, Any], peers: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    print(f"\nTarget: {target['name']} ({target['cin']})")
    print(f"Method: {metadata['method']}")
    print(
        "Fallbacks: "
        f"Cohere={bool(metadata['cohere'].get('fallback'))}, "
        f"OpenAI={bool(metadata['openai'].get('fallback'))}"
    )
    if metadata["cohere"].get("error"):
        print(f"Cohere note: {metadata['cohere']['error']}")
    if metadata["openai"].get("error"):
        print(f"OpenAI note: {metadata['openai']['error']}")
    print()

    for index, peer in enumerate(peers, start=1):
        revenue = peer.get("revenue_crore")
        revenue_label = f"INR {revenue:,.2f} Cr" if revenue is not None else "Revenue unknown"
        products = " | ".join(peer.get("core_products") or []) or "Core products not listed"
        print(f"{index}. {peer['name']}")
        print(f"   CIN: {peer['cin']}")
        print(f"   Score: {float(peer.get('final_score') or 0):.6f}")
        print(f"   Value chain: {peer.get('value_chain_primary') or 'Unknown'}")
        print(f"   Customer: {peer.get('customer_type') or 'Unknown'}")
        print(f"   Revenue: {revenue_label}")
        print(f"   Core: {products}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Return top peers using the current 40+40 -> Cohere -> OpenAI methodology.")
    parser.add_argument("company", nargs="?", help="Company name or CIN. If omitted, you will be prompted.")
    parser.add_argument("--top", type=int, default=5, help="Number of final peers to print. Defaults to 5.")
    parser.add_argument("--use-revenue", action="store_true", help="Apply revenue weighting during candidate generation.")
    parser.add_argument("--use-enum-weighting", action="store_true", help="Apply enum weighting during candidate generation.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of text.")
    args = parser.parse_args()

    load_local_env()
    data = get_peer_data()
    query = args.company or input("Company name or CIN: ").strip()
    target = find_company(data, query)

    filters = dict(DEFAULT_FILTERS)
    filters["use_revenue"] = args.use_revenue
    filters["use_enum_weighting"] = args.use_enum_weighting

    target, peers, metadata = rank_top_peers(data, target["cin"], filters, final_count=max(1, args.top))
    if args.json:
        print(json.dumps({"target": target, "peers": peers, "metadata": metadata}, indent=2, ensure_ascii=False))
    else:
        print_text(target, peers, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
