from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
from pathlib import Path
from typing import Any

from openai import AzureOpenAI

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


def shorten(value: Any, max_chars: int) -> str:
    text = str(value or "").replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


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


def rank_top_peers(
    data: Any,
    cin: str,
    filters: dict[str, Any],
    final_count: int,
    include_retrieval_evidence: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    payload = build_union_rerank_payload(data, cin, **filters)
    try:
        cohere_peers, cohere_metadata = rerank_peers(
            target=payload["target"],
            peers=payload["peers"],
            top_n=DEFAULT_OPENAI_CANDIDATE_COUNT,
            use_derived_categories=False,
            include_retrieval_evidence=include_retrieval_evidence,
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
            "retrieval_evidence": include_retrieval_evidence,
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
        "cohere_retrieval_evidence": include_retrieval_evidence,
    }
    return payload["target"], final_peers[:final_count], metadata


def compact_company_for_universe(company: dict[str, Any], row_id: int, field_chars: int) -> dict[str, Any]:
    value_chain = " / ".join(
        part
        for part in [
            str(company.get("value_chain_primary") or "").strip(),
            str(company.get("value_chain_secondary") or "").strip(),
        ]
        if part
    )
    return {
        "row_id": row_id,
        "cin": company.get("cin", ""),
        "name": shorten(company.get("name", ""), field_chars),
        "core_products": shorten(" | ".join(company.get("core_products") or []), field_chars),
        "secondary_products": shorten(" | ".join(company.get("secondary_products") or []), field_chars),
        "value_chain": shorten(value_chain, 80),
        "customer_type": shorten(company.get("customer_type", ""), 80),
        "revenue_crore": company.get("revenue_crore") if company.get("revenue_crore") is not None else "",
    }


def enriched_universe_csv(
    data: Any,
    target_cin: str,
    *,
    max_rows: int | None,
    field_chars: int,
) -> tuple[str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for cin in sorted(data.rows_by_cin, key=data.company_name):
        if cin == target_cin:
            continue
        company = data.serialize_company(cin)
        rows.append(compact_company_for_universe(company, len(rows) + 1, field_chars))
        if max_rows is not None and len(rows) >= max_rows:
            break

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "row_id",
            "cin",
            "name",
            "core_products",
            "secondary_products",
            "value_chain",
            "customer_type",
            "revenue_crore",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue(), rows


def build_universe_openai_prompt(target: dict[str, Any], universe_csv: str, row_count: int) -> str:
    target_core = " | ".join(target.get("core_products") or []) or "Not listed"
    target_secondary = " | ".join(target.get("secondary_products") or []) or "Not listed"
    target_value_chain = " / ".join(
        part
        for part in [
            str(target.get("value_chain_primary") or "").strip(),
            str(target.get("value_chain_secondary") or "").strip(),
        ]
        if part
    ) or "Not listed"
    return "\n\n".join(
        [
            "You are selecting investment banking comparable-company peers.",
            (
                "Choose the 10 companies from the CSV universe that would make the strongest "
                "peer set for valuation or benchmarking of the target company. Prioritize "
                "direct business-model and core-product/service similarity over broad thematic "
                "or adjacent similarity. Demote suppliers, customers, distributors, loose "
                "semantic matches, and diversified side-business matches. Use revenue only as "
                "a scale check and tie-breaker after business similarity."
            ),
            (
                f"Return only 10 row_id numbers between 1 and {row_count}, comma-separated. "
                "No names, no CINs, no prose, no bullets. Example format:\n\n"
                "3, 7, 1, 12, 4, 9, 2, 18, 6, 10"
            ),
            "Target company:",
            f"Name: {target.get('name', '')}",
            f"CIN: {target.get('cin', '')}",
            f"Core products/services: {target_core}",
            f"Secondary products/services: {target_secondary}",
            f"Value chain: {target_value_chain}",
            f"Customer type: {target.get('customer_type') or 'Not listed'}",
            f"Revenue crore: {target.get('revenue_crore') if target.get('revenue_crore') is not None else 'Not listed'}",
            "CSV universe:",
            universe_csv,
        ]
    )


def parse_universe_row_numbers(raw: str, row_count: int, expected_count: int = 10) -> list[int]:
    text = (raw or "").strip()
    if not re.fullmatch(r"\d+(?:\s*,\s*\d+)*", text):
        raise ValueError("OpenAI universe response must contain only comma-separated row numbers")
    numbers = [int(part.strip()) for part in text.split(",")]
    if len(numbers) != expected_count:
        raise ValueError(f"OpenAI universe response must contain exactly {expected_count} row numbers")
    if len(set(numbers)) != len(numbers):
        raise ValueError("OpenAI universe response contains duplicate row numbers")
    if any(number < 1 or number > row_count for number in numbers):
        raise ValueError("OpenAI universe response contains out-of-range row numbers")
    return numbers


def response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text).strip()
    choices = getattr(response, "choices", None) or []
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if content:
            return str(content).strip()
    return ""


def ask_openai_for_universe_top10(
    data: Any,
    target: dict[str, Any],
    *,
    max_rows: int | None,
    field_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    key = os.environ.get("AZURE_OPENAI_KEY", "").strip() or os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "").strip() or "2025-03-01-preview"
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "").strip()
    if not endpoint or not key or not deployment:
        return [], {
            "provider": "azure_openai",
            "used": False,
            "fallback": True,
            "error": "Azure OpenAI env vars are missing",
        }

    universe_csv, row_map = enriched_universe_csv(
        data,
        target["cin"],
        max_rows=max_rows,
        field_chars=field_chars,
    )
    prompt = build_universe_openai_prompt(target, universe_csv, len(row_map))
    timeout_seconds = float(os.environ.get("AZURE_OPENAI_UNIVERSE_TIMEOUT", os.environ.get("AZURE_OPENAI_TIMEOUT", "120")))
    max_output_tokens = int(os.environ.get("AZURE_OPENAI_UNIVERSE_MAX_OUTPUT_TOKENS", "4000"))
    reasoning_effort = os.environ.get("AZURE_OPENAI_REASONING_EFFORT", "low").strip().lower()
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=key,
        api_version=api_version,
        timeout=timeout_seconds,
    )

    try:
        request_args: dict[str, Any] = {
            "model": deployment,
            "input": prompt,
            "max_output_tokens": max_output_tokens,
        }
        if reasoning_effort:
            request_args["reasoning"] = {"effort": reasoning_effort}
        response = client.responses.create(**request_args)
        raw_text = response_text(response)
        numbers = parse_universe_row_numbers(raw_text, row_count=len(row_map), expected_count=10)
    except Exception as exc:
        return [], {
            "provider": "azure_openai",
            "model": deployment,
            "used": False,
            "fallback": True,
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            "row_count": len(row_map),
            "prompt_chars": len(prompt),
            "timeout_seconds": timeout_seconds,
            "reasoning_effort": reasoning_effort,
        }

    selected = [row_map[number - 1] for number in numbers]
    return selected, {
        "provider": "azure_openai",
        "model": deployment,
        "used": True,
        "fallback": False,
        "row_count": len(row_map),
        "selected_row_ids": numbers,
        "prompt_chars": len(prompt),
        "timeout_seconds": timeout_seconds,
        "reasoning_effort": reasoning_effort,
    }


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


def print_universe_text(peers: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    print("\nChatGPT full-universe top 10")
    print(
        f"Rows sent: {metadata.get('row_count', 0)} | "
        f"Prompt chars: {metadata.get('prompt_chars', 0)} | "
        f"Fallback: {bool(metadata.get('fallback'))}"
    )
    if metadata.get("error"):
        print(f"OpenAI universe note: {metadata['error']}")
    if not peers:
        return
    print()
    for index, peer in enumerate(peers, start=1):
        revenue = peer.get("revenue_crore")
        revenue_label = f"INR {float(revenue):,.2f} Cr" if revenue not in (None, "") else "Revenue unknown"
        print(f"{index}. {peer['name']}")
        print(f"   row_id: {peer['row_id']} | CIN: {peer['cin']}")
        print(f"   Value chain: {peer.get('value_chain') or 'Unknown'}")
        print(f"   Customer: {peer.get('customer_type') or 'Unknown'}")
        print(f"   Revenue: {revenue_label}")
        print(f"   Core: {peer.get('core_products') or 'Core products not listed'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Return top peers using the current 40+40 -> Cohere -> OpenAI methodology.")
    parser.add_argument("company", nargs="?", help="Company name or CIN. If omitted, you will be prompted.")
    parser.add_argument("--top", type=int, default=5, help="Number of final peers to print. Defaults to 5.")
    parser.add_argument("--use-revenue", action="store_true", help="Apply revenue weighting during candidate generation.")
    parser.add_argument("--use-enum-weighting", action="store_true", help="Apply enum weighting during candidate generation.")
    parser.add_argument("--cohere-retrieval-evidence", action="store_true", help="Include candidate source and similarity scores in Cohere documents.")
    parser.add_argument("--skip-universe-chatgpt", action="store_true", help="Only run the current pipeline; do not ask OpenAI over the full enriched universe.")
    parser.add_argument("--universe-max-rows", type=int, default=None, help="Limit rows sent to OpenAI for testing. Defaults to all enriched companies except the target.")
    parser.add_argument("--universe-field-chars", type=int, default=240, help="Max characters per long CSV field sent to OpenAI. Defaults to 240.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of text.")
    args = parser.parse_args()

    load_local_env()
    data = get_peer_data()
    query = args.company or input("Company name or CIN: ").strip()
    target = find_company(data, query)

    filters = dict(DEFAULT_FILTERS)
    filters["use_revenue"] = args.use_revenue
    filters["use_enum_weighting"] = args.use_enum_weighting

    target, peers, metadata = rank_top_peers(
        data,
        target["cin"],
        filters,
        final_count=max(1, args.top),
        include_retrieval_evidence=args.cohere_retrieval_evidence,
    )
    universe_peers: list[dict[str, Any]] = []
    universe_metadata: dict[str, Any] = {"skipped": True}
    if not args.skip_universe_chatgpt:
        universe_peers, universe_metadata = ask_openai_for_universe_top10(
            data,
            target,
            max_rows=args.universe_max_rows,
            field_chars=max(60, args.universe_field_chars),
        )
    if args.json:
        print(
            json.dumps(
                {
                    "target": target,
                    "current_method_peers": peers,
                    "current_method_metadata": metadata,
                    "chatgpt_universe_peers": universe_peers,
                    "chatgpt_universe_metadata": universe_metadata,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print_text(target, peers, metadata)
        if args.skip_universe_chatgpt:
            print("\nChatGPT full-universe top 10: skipped")
        else:
            print_universe_text(universe_peers, universe_metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
