from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from openai import AzureOpenAI


DEFAULT_OPENAI_FINAL_TOP_N = 10
DEFAULT_OPENAI_CANDIDATE_COUNT = 25


def _load_local_env() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip("'\"")


def parse_openai_rank_numbers(raw: str, candidate_count: int, expected_count: int = DEFAULT_OPENAI_FINAL_TOP_N) -> list[int]:
    text = (raw or "").strip()
    if not re.fullmatch(r"\d{1,2}(?:\s*,\s*\d{1,2})*", text):
        raise ValueError("OpenAI response must contain only comma-separated numbers")

    numbers = [int(part.strip()) for part in text.split(",")]
    if len(numbers) != expected_count:
        raise ValueError(f"OpenAI response must contain exactly {expected_count} numbers")
    if len(set(numbers)) != len(numbers):
        raise ValueError("OpenAI response contains duplicate numbers")
    if any(number < 1 or number > candidate_count for number in numbers):
        raise ValueError("OpenAI response contains out-of-range numbers")
    return numbers


def _join_items(items: list[str]) -> str:
    return "; ".join(str(item).strip() for item in items if str(item).strip()) or "Not listed"


def _value_chain(company: dict[str, Any]) -> str:
    parts = [
        str(company.get("value_chain_primary") or "").strip(),
        str(company.get("value_chain_secondary") or "").strip(),
    ]
    return " / ".join(part for part in parts if part) or "Not listed"


def _revenue_label(company: dict[str, Any]) -> str:
    revenue = company.get("revenue_crore")
    if revenue is None:
        return "Not listed"
    try:
        return f"INR {float(revenue):,.2f} Cr"
    except (TypeError, ValueError):
        return str(revenue) or "Not listed"


def _company_block(company: dict[str, Any], *, number: int | None = None) -> str:
    prefix = f"{number}. " if number is not None else ""
    return "\n".join(
        [
            f"{prefix}{company.get('name', '')}",
            f"   Core products/services: {_join_items(company.get('core_products') or [])}",
            f"   Secondary products/services: {_join_items(company.get('secondary_products') or [])}",
            f"   Value chain: {_value_chain(company)}",
            f"   Customer type: {company.get('customer_type') or 'Not listed'}",
            f"   Revenue: {_revenue_label(company)}",
        ]
    )


def build_openai_final_prompt(target: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    candidate_blocks = "\n\n".join(
        _company_block(candidate, number=index)
        for index, candidate in enumerate(candidates, start=1)
    )
    return "\n\n".join(
        [
            "You are selecting investment banking comparable-company peers.",
            (
                "Choose the 10 companies from the numbered candidate list that would make "
                "the strongest peer set for valuation or benchmarking of the target company. "
                "Prioritize direct business-model and core-product/service similarity over "
                "broad thematic similarity, adjacent suppliers, customers, distributors, or "
                "diversified side businesses. Use revenue as an investment-banking scale "
                "check and tie-breaker, but do not rank a poor business-model match above a "
                "strong direct peer just because revenue is closer."
            ),
            (
                "Response must be 10 numbers between 1 and 25, separated by commas as follows "
                "(nothing more, nothing less):\n\n"
                "3, 7, 1, 12, 4, 9, 2, 18, 6, 10"
            ),
            "Target company:",
            _company_block(target),
            "Candidate companies:",
            candidate_blocks,
        ]
    )


def _response_text(response: Any) -> str:
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


def select_final_peers_with_openai(
    target: dict[str, Any],
    candidates: list[dict[str, Any]],
    final_count: int = DEFAULT_OPENAI_FINAL_TOP_N,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(candidates) < final_count:
        return candidates, {
            "provider": "azure_openai",
            "used": False,
            "fallback": True,
            "error": f"Need at least {final_count} candidates for OpenAI final selection",
            "candidate_count": len(candidates),
            "returned_count": len(candidates),
        }

    _load_local_env()
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    key = os.environ.get("AZURE_OPENAI_KEY", "").strip() or os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "").strip() or "2025-03-01-preview"
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "").strip()
    if not endpoint or not key or not deployment:
        return candidates[:final_count], {
            "provider": "azure_openai",
            "used": False,
            "fallback": True,
            "error": "Azure OpenAI env vars are missing",
            "candidate_count": len(candidates),
            "returned_count": final_count,
        }

    prompt = build_openai_final_prompt(target, candidates)
    timeout_seconds = float(os.environ.get("AZURE_OPENAI_TIMEOUT", "30"))
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=key,
        api_version=api_version,
        timeout=timeout_seconds,
    )
    try:
        max_output_tokens = int(os.environ.get("AZURE_OPENAI_FINAL_MAX_OUTPUT_TOKENS", "4000"))
        reasoning_effort = os.environ.get("AZURE_OPENAI_REASONING_EFFORT", "low").strip().lower()
        request_args: dict[str, Any] = {
            "model": deployment,
            "input": prompt,
            "max_output_tokens": max_output_tokens,
        }
        if reasoning_effort:
            request_args["reasoning"] = {"effort": reasoning_effort}
        response = client.responses.create(
            **request_args,
        )
        raw_text = _response_text(response)
        numbers = parse_openai_rank_numbers(raw_text, candidate_count=len(candidates), expected_count=final_count)
    except Exception as exc:
        return candidates[:final_count], {
            "provider": "azure_openai",
            "model": deployment,
            "reasoning_effort": reasoning_effort,
            "timeout_seconds": timeout_seconds,
            "used": False,
            "fallback": True,
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            "candidate_count": len(candidates),
            "returned_count": final_count,
        }

    selected: list[dict[str, Any]] = []
    for final_rank, number in enumerate(numbers, start=1):
        item = dict(candidates[number - 1])
        item["openai_final_rank"] = final_rank
        item["openai_selected_number"] = number
        item["pre_openai_rank"] = number
        selected.append(item)

    return selected, {
        "provider": "azure_openai",
        "model": deployment,
        "reasoning_effort": reasoning_effort,
        "timeout_seconds": timeout_seconds,
        "used": True,
        "fallback": False,
        "candidate_count": len(candidates),
        "returned_count": len(selected),
        "selected_numbers": numbers,
    }


def summarize_openai_candidate(company: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "rank": index,
        "name": company.get("name", ""),
        "cin": company.get("cin", ""),
        "core_products": company.get("core_products") or [],
        "secondary_products": company.get("secondary_products") or [],
        "value_chain": _value_chain(company),
        "customer_type": company.get("customer_type") or "Not listed",
        "revenue": _revenue_label(company),
        "cohere_rerank_score": company.get("cohere_rerank_score"),
        "product_candidate_score": company.get("product_candidate_score"),
        "company_candidate_score": company.get("company_candidate_score"),
    }
