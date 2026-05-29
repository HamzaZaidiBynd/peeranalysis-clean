from __future__ import annotations

import json
import os
import re
import socket
import urllib.error
import urllib.request
from typing import Any


COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"
DEFAULT_RERANK_MODEL = "rerank-v4.0-fast"
STOPWORDS = {
    "a",
    "and",
    "as",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "plus",
    "services",
    "the",
    "to",
    "with",
}


def _yaml_list(items: list[str]) -> str:
    if not items:
        return "[]"
    return "\n".join(f"  - {item}" for item in items)


def _norm_tokens(value: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", (value or "").lower())
    return {token for token in tokens if len(token) > 2 and token not in STOPWORDS}


def _norm_phrase(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (value or "").lower())).strip()


def _has_any(haystack: str, phrases: tuple[str, ...]) -> bool:
    normalized = _norm_phrase(haystack)
    return any(_norm_phrase(phrase) in normalized for phrase in phrases)


def _all_text(company: dict[str, Any]) -> str:
    fields = [
        company.get("name", ""),
        company.get("business_description", ""),
        company.get("value_chain_primary", ""),
        company.get("value_chain_secondary", ""),
        company.get("customer_type", ""),
        " ".join(company.get("core_products") or []),
        " ".join(company.get("secondary_products") or []),
        " ".join(company.get("minor_products") or []),
        " ".join(company.get("end_markets") or []),
    ]
    return " ".join(str(field) for field in fields if field)


def _primary_value_chain(company: dict[str, Any]) -> str:
    return _norm_phrase(str(company.get("value_chain_primary", "")))


def _primary_is_vehicle_retail_or_distribution(company: dict[str, Any]) -> bool:
    primary_vc = _primary_value_chain(company)
    return any(token in primary_vc for token in ("retailer", "dealer", "distributor", "trader", "wholesaler"))


def _primary_is_manufacturer(company: dict[str, Any]) -> bool:
    return "manufacturer" in _primary_value_chain(company)


def _is_clear_vehicle_oem(company: dict[str, Any]) -> bool:
    text = _all_text(company)
    return _primary_is_manufacturer(company) or _has_any(
        text,
        (
            "own-brand passenger vehicles",
            "own-brand vehicles",
            "own brand passenger vehicles",
            "own brand vehicles",
            "automobile manufacturer",
            "vehicle oem",
        ),
    )


def infer_business_role(company: dict[str, Any]) -> str:
    text = _all_text(company)
    if _has_any(text, ("tyre", "tire", "radial")):
        return "tyre_manufacturer"
    if _primary_is_vehicle_retail_or_distribution(company) and _has_any(
        text,
        ("passenger vehicle", "cars", "suv", "hatchback", "authorized dealer", "dealership", "vehicle dealer"),
    ):
        return "vehicle_dealer"
    if _has_any(text, ("passenger vehicle", "automobile manufacturer", "vehicle oem", "cars", "suv", "hatchback")):
        if _is_clear_vehicle_oem(company):
            return "vehicle_oem"
    if _has_any(text, ("authorized dealer", "dealership", "vehicle dealer")):
        return "vehicle_dealer"
    if _has_any(text, ("passenger vehicle", "automobile manufacturer", "vehicle oem", "cars", "suv", "hatchback")):
        return "vehicle_oem" if _is_clear_vehicle_oem(company) else "vehicle_dealer"
    has_direct_paint = _has_any(text, ("paints", "coating", "coatings", "emulsion", "enamel"))
    has_adjacent_chemical = _has_any(text, ("adhesive", "sealant", "construction chemical", "paint chemicals", "resin", "pigment"))
    if has_direct_paint:
        return "paint_or_coatings_company"
    if has_adjacent_chemical:
        return "adjacent_chemicals_or_materials"
    if _has_any(text, ("smartphone", "laptop", "tablet", "wearables", "consumer electronics")):
        if _has_any(text, ("retail", "retailer", "distribution", "distributor")):
            return "consumer_electronics_retail_or_distribution"
        return "electronics_manufacturer_or_ems"
    if _has_any(
        text,
        (
            "application development",
            "business process",
            "cloud",
            "digital transformation",
            "enterprise it",
            "information technology",
            "it consulting",
            "managed services",
            "software services",
            "system integration",
        ),
    ):
        return "it_services_provider"
    if _has_any(text, ("eyewear", "eyeglasses", "contact lenses", "optical")):
        return "eyewear_or_optical"
    if _has_any(text, ("beauty", "cosmetics", "haircare", "personal care", "skincare")):
        return "beauty_or_personal_care"
    return str(company.get("value_chain_primary", "") or "unknown")


def _product_similarity(left: str, right: str) -> float:
    left_tokens = _norm_tokens(left)
    right_tokens = _norm_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    overlap_score = overlap / min(len(left_tokens), len(right_tokens))
    jaccard_score = overlap / len(left_tokens | right_tokens)
    return round((0.7 * overlap_score) + (0.3 * jaccard_score), 3)


def _best_product_matches(
    target_products: list[str],
    candidate_products_by_primacy: list[tuple[str, str]],
    limit: int = 4,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for target_product in target_products:
        best: tuple[float, str, str] | None = None
        for candidate_product, primacy in candidate_products_by_primacy:
            score = _product_similarity(target_product, candidate_product)
            if best is None or score > best[0]:
                best = (score, candidate_product, primacy)
        if best is not None:
            score, candidate_product, primacy = best
            if score >= 0.5:
                strength = "high"
            elif score >= 0.25:
                strength = "medium"
            elif score > 0:
                strength = "low"
            else:
                strength = "none"
            matches.append(
                {
                    "target_product": target_product,
                    "candidate_product": candidate_product,
                    "candidate_product_primacy": primacy,
                    "text_similarity": score,
                    "match_strength": strength,
                }
            )
    matches.sort(key=lambda item: item["text_similarity"], reverse=True)
    return matches[:limit]


def _overlap(left: list[str], right: list[str]) -> list[str]:
    right_norm = {_norm_phrase(item): item for item in right}
    return [right_norm[_norm_phrase(item)] for item in left if _norm_phrase(item) in right_norm]


def _role_match(target_role: str, candidate_role: str) -> str:
    if target_role == candidate_role:
        return "strong"
    adjacent_pairs = {
        ("paint_or_coatings_company", "adjacent_chemicals_or_materials"),
        ("vehicle_oem", "vehicle_dealer"),
        ("electronics_manufacturer_or_ems", "consumer_electronics_retail_or_distribution"),
    }
    if (target_role, candidate_role) in adjacent_pairs or (candidate_role, target_role) in adjacent_pairs:
        return "partial_adjacent"
    return "weak"


def _unique_ordered(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _category_for_product(product: str, company: dict[str, Any]) -> str | None:
    product_text = _norm_phrase(product)
    if not product_text:
        return None

    if _has_any(product_text, ("passenger vehicle", "passenger vehicles", "suv", "hatchback", "sedan", "mpv", "cars")):
        if _primary_is_vehicle_retail_or_distribution(company) and not _primary_is_manufacturer(company):
            return "Vehicle Retail & Dealership"
        return "Passenger Vehicle OEM"
    if _has_any(product_text, ("motorcycle", "motorcycles", "scooter", "scooters", "two wheeler", "two wheelers", "three wheeler", "three wheelers")):
        return "Two-Wheeler OEM" if _primary_is_manufacturer(company) else "Two-Wheeler Retail & Distribution"
    if _has_any(product_text, ("tyre", "tire", "radial")):
        return "Tyres"
    if _has_any(product_text, ("auto component", "automotive component", "spare parts", "parts and accessories")):
        return "Auto Components"

    if _has_any(product_text, ("adhesive", "adhesives", "sealant", "sealants", "construction chemical", "construction chemicals", "waterproofing", "paint chemicals")):
        return "Adhesives & Construction Chemicals"
    if _has_any(product_text, ("resin", "resins", "pigment", "pigments", "specialty chemical", "speciality chemical")):
        return "Specialty Chemicals"
    if _has_any(
        product_text,
        (
            "industrial paints",
            "decorative paints",
            "emulsion paints",
            "architectural emulsions",
            "wall coatings",
            "paint",
            "paints",
            "coating",
            "coatings",
            "enamel",
        ),
    ):
        return "Paints & Coatings"

    if _has_any(
        product_text,
        (
            "application services",
            "application development",
            "consulting services",
            "business consulting",
            "software services",
            "it consulting",
            "digital transformation",
            "cloud services",
            "cloud platforms",
            "cloud platform",
            "managed services",
            "system integration",
            "cybersecurity",
            "digital infrastructure",
        ),
    ):
        return "IT Services & Consulting"
    if _has_any(product_text, ("business process", "bpo", "contact center", "contact centre", "cx platform", "ccaas", "customer support")):
        return "BPO & Customer Experience Services"
    if _has_any(product_text, ("software platform", "saas", "cloud platform", "analytics platform", "automation platform")):
        return "Software / SaaS"

    if _has_any(product_text, ("smartphone", "smartphones", "mobile phones", "mobiles", "iphone", "laptop", "laptops", "tablet", "tablets", "consumer electronics", "electronics")):
        return "Consumer Electronics"
    if _has_any(product_text, ("wristwatch", "wristwatches", "watches", "wearables")):
        return "Watches & Wearables"
    if _has_any(product_text, ("eyewear", "eyeglasses", "contact lenses", "optical")):
        return "Eyewear & Optical"

    if _has_any(product_text, ("gold jewellery", "diamond jewellery", "silver jewellery", "jewellery", "jewelry", "gemstone")):
        return "Jewellery Retail & Manufacturing"
    if _has_any(product_text, ("bullion", "gold bars", "silver bars", "coins")):
        return "Bullion & Precious Metals Trading"
    if _has_any(product_text, ("apparel", "footwear", "fashion", "garments")):
        return "Fashion & Apparel"

    if _has_any(product_text, ("grocery", "groceries", "fresh produce", "foodgrains", "staples", "dairy", "packaged foods", "supermarket", "hypermarket")):
        return "Grocery & Supermarket Retail"
    if _has_any(product_text, ("quick commerce", "instant delivery", "food delivery marketplace", "restaurant partner", "internet restaurant", "cloud kitchen")):
        return "Food Delivery & Quick Commerce"
    if _has_any(product_text, ("online marketplace", "e commerce marketplace", "digital commerce marketplace", "marketplace platform")):
        return "E-commerce Marketplace"

    if _has_any(product_text, ("personal care", "beauty", "cosmetics", "hair care", "haircare", "skin care", "skincare", "detergent", "dishwashing", "oral care")):
        return "FMCG & Personal Care"
    if _has_any(product_text, ("packaged food", "snacks", "beverages", "food products")):
        return "Packaged Foods & Beverages"
    return None


def derive_normalized_categories(company: dict[str, Any]) -> dict[str, list[str]]:
    primary = _unique_ordered(
        [
            category
            for product in company.get("core_products") or []
            if (category := _category_for_product(product, company))
        ]
    )
    secondary = _unique_ordered(
        [
            category
            for product in [*(company.get("secondary_products") or []), *(company.get("minor_products") or [])]
            if (category := _category_for_product(product, company)) and category not in primary
        ]
    )
    return {
        "primary_categories": primary,
        "secondary_categories": secondary,
    }


def category_match_quality(target_categories: dict[str, list[str]], candidate_categories: dict[str, list[str]]) -> str:
    target_primary = set(target_categories["primary_categories"])
    target_secondary = set(target_categories["secondary_categories"])
    candidate_primary = set(candidate_categories["primary_categories"])
    candidate_secondary = set(candidate_categories["secondary_categories"])
    if not target_primary or not candidate_primary:
        return "unknown_category_match"
    if target_primary & candidate_primary:
        return "primary_category_match"
    if target_primary & candidate_secondary:
        return "candidate_secondary_category_match"
    if target_secondary & candidate_primary:
        return "target_secondary_category_match"
    if target_secondary & candidate_secondary:
        return "secondary_category_match"
    return "no_category_match"


def infer_business_archetype(company: dict[str, Any]) -> str:
    text = _all_text(company)
    core_text = " ".join(company.get("core_products") or [])
    primary_vc = _norm_phrase(str(company.get("value_chain_primary", "")))
    secondary_vc = _norm_phrase(str(company.get("value_chain_secondary", "")))
    value_chain = f"{primary_vc} {secondary_vc}"
    is_retailer = "retailer" in value_chain or _has_any(text, ("retail", "showroom", "store", "stores", "e commerce", "omnichannel"))
    is_primary_vehicle_retailer = _primary_is_vehicle_retail_or_distribution(company)
    is_primary_manufacturer = _primary_is_manufacturer(company)
    is_manufacturer = "manufacturer" in value_chain or _has_any(text, ("manufactures", "manufacturing", "manufacturer"))
    is_trader = "trader" in value_chain or "distributor" in value_chain or "wholesaler" in value_chain

    if _has_any(core_text, ("passenger vehicles", "passenger vehicle", "suv", "hatchback", "sedan", "mpv")):
        if is_primary_vehicle_retailer and not is_primary_manufacturer:
            return "vehicle_dealer_retailer"
        if _is_clear_vehicle_oem(company):
            return "vehicle_oem"
        return "vehicle_dealer_retailer"
    if _has_any(text, ("authorized dealer", "dealership", "new car retail", "vehicle service")):
        return "vehicle_dealer_retailer"
    if _has_any(core_text, ("motorcycles", "motorcycle", "scooters", "scooter", "two-wheelers", "two wheelers", "three-wheelers")):
        return "two_wheeler_oem" if is_manufacturer else "two_wheeler_retail_or_component"
    if _has_any(text, ("tyre", "tire", "automotive component", "auto component", "spare parts", "parts and accessories")):
        return "auto_component_supplier"

    if _has_any(core_text, ("supermarket", "hypermarket", "grocery", "fresh produce", "foodgrains", "staples")) or _has_any(
        text,
        ("supermarket chain", "hypermarket", "discount supermarket", "grocery retail"),
    ):
        return "grocery_supermarket_retailer"

    has_fashion = _has_any(core_text, ("apparel", "footwear", "garments", "fashion")) or _has_any(text, ("fashion retail", "apparel retail"))
    has_jewellery = _has_any(text, ("jewellery", "jewelry", "gold jewellery", "diamond jewellery", "gemstone", "precious metal"))
    if has_fashion and not _has_any(core_text, ("gold", "diamond", "silver", "precious", "gemstone", "bullion")):
        return "fashion_apparel_retailer"
    if has_jewellery:
        if _has_any(core_text, ("bullion", "gold bars", "silver bars", "coins")) or (is_trader and not is_retailer):
            return "bullion_or_jewellery_trader"
        if is_retailer and is_manufacturer:
            return "branded_jewellery_retailer"
        if is_retailer:
            return "jewellery_retailer"
        return "jewellery_manufacturer_or_wholesaler"
    if _has_any(core_text, ("wristwatches", "wristwatch", "watches", "wearables")) and not _has_any(core_text, ("smartphones", "smartphone")):
        return "watch_wearables_brand_retailer" if is_retailer or is_manufacturer else "watch_wearables_supplier"

    if _has_any(core_text, ("smartphones", "smartphone", "iphone", "mobiles")):
        if is_manufacturer:
            return "consumer_electronics_brand_manufacturer"
        return "consumer_electronics_retail_distribution"
    if _has_any(core_text, ("laptops", "desktops", "tablets", "printers", "consumer electronics")):
        return "consumer_electronics_brand_manufacturer" if is_manufacturer else "consumer_electronics_retail_distribution"

    if _has_any(text, ("detergent", "dishwashing", "personal care", "beauty", "hair care", "skin care", "packaged food", "oral care")):
        if _has_any(text, ("ingredient", "labsa", "chemical", "contract manufacturing", "formulation")) or "wholesaler" in value_chain:
            return "fmcg_supplier_or_contract_manufacturer"
        return "fmcg_brand_manufacturer"

    if _has_any(text, ("paint", "paints", "coating", "coatings", "emulsion", "enamel")):
        return "paint_coatings_manufacturer"
    if _has_any(text, ("adhesive", "sealant", "construction chemical", "resin")):
        return "adhesives_chemicals_manufacturer"
    if _has_any(text, ("application development", "cloud", "it consulting", "digital transformation", "managed services", "enterprise it", "software engineering")):
        return "enterprise_it_services"
    return "other"


def business_model_peer_quality(target_archetype: str, candidate_archetype: str) -> str:
    if target_archetype == candidate_archetype:
        return "strong_model_match"
    strong_pairs = {
        ("branded_jewellery_retailer", "jewellery_retailer"),
        ("branded_jewellery_retailer", "watch_wearables_brand_retailer"),
        ("consumer_electronics_retail_distribution", "consumer_electronics_brand_manufacturer"),
        ("consumer_electronics_retail_distribution", "consumer_electronics_retail_distribution"),
    }
    adjacent_pairs = {
        ("vehicle_oem", "vehicle_dealer_retailer"),
        ("vehicle_oem", "auto_component_supplier"),
        ("two_wheeler_oem", "vehicle_oem"),
        ("two_wheeler_oem", "auto_component_supplier"),
        ("fmcg_brand_manufacturer", "fmcg_supplier_or_contract_manufacturer"),
        ("grocery_supermarket_retailer", "fashion_apparel_retailer"),
        ("grocery_supermarket_retailer", "consumer_electronics_retail_distribution"),
        ("branded_jewellery_retailer", "bullion_or_jewellery_trader"),
        ("branded_jewellery_retailer", "jewellery_manufacturer_or_wholesaler"),
        ("branded_jewellery_retailer", "fashion_apparel_retailer"),
        ("consumer_electronics_retail_distribution", "watch_wearables_brand_retailer"),
        ("paint_coatings_manufacturer", "adhesives_chemicals_manufacturer"),
    }
    pair = (target_archetype, candidate_archetype)
    reverse_pair = (candidate_archetype, target_archetype)
    if pair in strong_pairs or reverse_pair in strong_pairs:
        return "strong_model_match"
    if pair in adjacent_pairs or reverse_pair in adjacent_pairs:
        return "adjacent_model_match"
    if target_archetype.split("_", 1)[0] == candidate_archetype.split("_", 1)[0]:
        return "acceptable_model_match"
    return "weak_model_match"


def business_model_priority(evidence: dict[str, Any]) -> str:
    quality = evidence["business_model_peer_quality"]
    if quality != "strong_model_match":
        return quality
    return "strong_model_peer"


def _candidate_sort_score(peer: dict[str, Any], evidence: dict[str, Any]) -> tuple[float, float]:
    company_score = float(peer.get("company_candidate_score") or 0.0)
    product_score = float(peer.get("product_candidate_score") or 0.0)
    return (company_score, product_score)


def _revenue_scale_evidence(target: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    target_revenue = target.get("revenue_crore")
    candidate_revenue = candidate.get("revenue_crore")
    try:
        target_value = float(target_revenue)
        candidate_value = float(candidate_revenue)
    except (TypeError, ValueError):
        return {
            "revenue_ratio_to_target": None,
            "revenue_scale_match": "unknown",
        }
    if target_value <= 0 or candidate_value <= 0:
        return {
            "revenue_ratio_to_target": None,
            "revenue_scale_match": "unknown",
        }
    ratio = max(target_value, candidate_value) / min(target_value, candidate_value)
    if ratio <= 2:
        scale_match = "very_close"
    elif ratio <= 5:
        scale_match = "close"
    elif ratio <= 15:
        scale_match = "moderate"
    elif ratio <= 50:
        scale_match = "far"
    else:
        scale_match = "very_far"
    return {
        "revenue_ratio_to_target": round(ratio, 2),
        "revenue_scale_match": scale_match,
    }


def build_match_evidence(
    target: dict[str, Any],
    candidate: dict[str, Any],
    use_derived_categories: bool = True,
) -> dict[str, Any]:
    target_role = infer_business_role(target)
    candidate_role = infer_business_role(candidate)
    target_archetype = infer_business_archetype(target)
    candidate_archetype = infer_business_archetype(candidate)
    target_categories = derive_normalized_categories(target) if use_derived_categories else {"primary_categories": [], "secondary_categories": []}
    candidate_categories = derive_normalized_categories(candidate) if use_derived_categories else {"primary_categories": [], "secondary_categories": []}
    category_quality = (
        category_match_quality(target_categories, candidate_categories)
        if use_derived_categories
        else "disabled"
    )
    target_vc = {
        _norm_phrase(str(target.get("value_chain_primary", ""))),
        _norm_phrase(str(target.get("value_chain_secondary", ""))),
    }
    candidate_vc_primary = _norm_phrase(str(candidate.get("value_chain_primary", "")))
    candidate_vc_secondary = _norm_phrase(str(candidate.get("value_chain_secondary", "")))
    if candidate_vc_primary and candidate_vc_primary in target_vc:
        value_chain_overlap = "primary"
    elif candidate_vc_secondary and candidate_vc_secondary in target_vc:
        value_chain_overlap = "secondary"
    else:
        value_chain_overlap = "none"

    candidate_products = [
        *((product, "core") for product in candidate.get("core_products") or []),
        *((product, "secondary") for product in candidate.get("secondary_products") or []),
        *((product, "minor") for product in candidate.get("minor_products") or []),
    ]
    core_matches = _best_product_matches(target.get("core_products") or [], candidate_products)
    secondary_matches = _best_product_matches(target.get("secondary_products") or [], candidate_products, limit=2)
    best_core = core_matches[0] if core_matches else {}
    direct_core_match = (
        best_core.get("candidate_product_primacy") == "core"
        and float(best_core.get("text_similarity") or 0.0) >= 0.25
    ) or (target_role == candidate_role and target_role != "unknown")
    adjacent_only = not direct_core_match and (
        _role_match(target_role, candidate_role) == "partial_adjacent"
        or any(match.get("match_strength") in {"high", "medium"} for match in secondary_matches)
    )
    end_market_overlap = _overlap(target.get("end_markets") or [], candidate.get("end_markets") or [])
    product_score = candidate.get("product_candidate_score")
    company_score = candidate.get("company_candidate_score")
    source_count = len(candidate.get("rerank_candidate_sources") or [])
    scale_evidence = _revenue_scale_evidence(target, candidate)
    if use_derived_categories and category_quality == "primary_category_match" and _role_match(target_role, candidate_role) == "strong":
        peer_fit_verdict = "excellent_direct_core_peer"
        ranking_priority = "high"
    elif use_derived_categories and category_quality == "primary_category_match":
        peer_fit_verdict = "good_direct_core_peer"
        ranking_priority = "medium_high"
    elif use_derived_categories and category_quality in {"target_secondary_category_match", "candidate_secondary_category_match", "secondary_category_match"}:
        peer_fit_verdict = "adjacent_or_secondary_category_peer"
        ranking_priority = "medium_low"
    elif direct_core_match and _role_match(target_role, candidate_role) == "strong":
        peer_fit_verdict = "excellent_direct_core_peer"
        ranking_priority = "high"
    elif direct_core_match:
        peer_fit_verdict = "good_direct_core_peer"
        ranking_priority = "medium_high"
    elif adjacent_only:
        peer_fit_verdict = "adjacent_only_not_direct_core_peer"
        ranking_priority = "low"
    else:
        peer_fit_verdict = "weak_or_unclear_peer"
        ranking_priority = "very_low"

    evidence = {
        "candidate_sources": candidate.get("rerank_candidate_sources") or [],
        "product_candidate_score": candidate.get("product_candidate_score"),
        "company_candidate_score": candidate.get("company_candidate_score"),
        "source_consensus": "both_scorers" if source_count > 1 else "single_scorer",
        "product_signal_strength": _score_strength(product_score),
        "company_signal_strength": _score_strength(company_score),
        "revenue_ratio_to_target": scale_evidence["revenue_ratio_to_target"],
        "revenue_scale_match": scale_evidence["revenue_scale_match"],
        "peer_fit_verdict": peer_fit_verdict,
        "ranking_priority": ranking_priority,
        "target_role": target_role,
        "candidate_role": candidate_role,
        "target_archetype": target_archetype,
        "candidate_archetype": candidate_archetype,
        "target_primary_categories": target_categories["primary_categories"],
        "target_secondary_categories": target_categories["secondary_categories"],
        "candidate_primary_categories": candidate_categories["primary_categories"],
        "candidate_secondary_categories": candidate_categories["secondary_categories"],
        "normalized_category_match": category_quality,
        "business_model_peer_quality": business_model_peer_quality(target_archetype, candidate_archetype),
        "role_match": _role_match(target_role, candidate_role),
        "value_chain_overlap": value_chain_overlap,
        "customer_type_match": str(target.get("customer_type", "")) == str(candidate.get("customer_type", "")),
        "end_market_overlap": end_market_overlap,
        "direct_target_core_match": direct_core_match,
        "adjacent_only_match": adjacent_only,
        "target_core_best_matches": core_matches,
        "target_secondary_best_matches": secondary_matches,
    }
    evidence["business_model_priority"] = business_model_priority(evidence)
    return evidence


def _score_strength(score: Any) -> str:
    if score is None:
        return "missing"
    try:
        value = float(score)
    except (TypeError, ValueError):
        return "missing"
    if value >= 0.9:
        return "very_high"
    if value >= 0.75:
        return "high"
    if value >= 0.6:
        return "medium"
    return "low"


def _format_match(match: dict[str, Any]) -> str:
    return (
        f"{match.get('target_product', '')} -> {match.get('candidate_product', '')} "
        f"({match.get('candidate_product_primacy', '')}, "
        f"{match.get('match_strength', '')}, {match.get('text_similarity', '')})"
    )


def format_company_for_rerank(
    company: dict[str, Any],
    target: dict[str, Any] | None = None,
    use_derived_categories: bool = True,
    include_match_evidence: bool = False,
    include_retrieval_evidence: bool = False,
) -> str:
    lines: list[str] = []
    if include_retrieval_evidence:
        sources = company.get("rerank_candidate_sources") or []
        product_score = company.get("product_candidate_score")
        company_score = company.get("company_candidate_score")
        source_consensus = "both_scorers" if len(sources) > 1 else "single_scorer"
        lines.extend(
            [
                "retrieval_evidence:",
                f"  candidate_sources: {', '.join(sources) or 'unknown'}",
                f"  source_consensus: {source_consensus}",
                f"  product_candidate_score: {product_score}",
                f"  product_signal_strength: {_score_strength(product_score)}",
                f"  company_candidate_score: {company_score}",
                f"  company_signal_strength: {_score_strength(company_score)}",
                "",
            ]
        )
    if target is not None and include_match_evidence:
        evidence = build_match_evidence(target, company, use_derived_categories=use_derived_categories)
        lines.extend(
            [
                "match_evidence:",
                f"  peer_fit_verdict: {evidence['peer_fit_verdict']}",
                f"  ranking_priority: {evidence['ranking_priority']}",
                f"  candidate_sources: {', '.join(evidence['candidate_sources']) or 'unknown'}",
                f"  source_consensus: {evidence['source_consensus']}",
                f"  product_candidate_score: {evidence['product_candidate_score']}",
                f"  product_signal_strength: {evidence['product_signal_strength']}",
                f"  company_candidate_score: {evidence['company_candidate_score']}",
                f"  company_signal_strength: {evidence['company_signal_strength']}",
                f"  revenue_ratio_to_target: {evidence['revenue_ratio_to_target']}",
                f"  revenue_scale_match: {evidence['revenue_scale_match']}",
                f"  target_role: {evidence['target_role']}",
                f"  candidate_role: {evidence['candidate_role']}",
                f"  target_archetype: {evidence['target_archetype']}",
                f"  candidate_archetype: {evidence['candidate_archetype']}",
                f"  target_primary_categories: {', '.join(evidence['target_primary_categories']) or 'none'}",
                f"  target_secondary_categories: {', '.join(evidence['target_secondary_categories']) or 'none'}",
                f"  candidate_primary_categories: {', '.join(evidence['candidate_primary_categories']) or 'none'}",
                f"  candidate_secondary_categories: {', '.join(evidence['candidate_secondary_categories']) or 'none'}",
                f"  normalized_category_match: {evidence['normalized_category_match']}",
                f"  business_model_peer_quality: {evidence['business_model_peer_quality']}",
                f"  business_model_priority: {evidence['business_model_priority']}",
                f"  role_match: {evidence['role_match']}",
                f"  value_chain_overlap: {evidence['value_chain_overlap']}",
                f"  customer_type_match: {evidence['customer_type_match']}",
                f"  end_market_overlap: {', '.join(evidence['end_market_overlap']) or 'none'}",
                f"  direct_target_core_match: {evidence['direct_target_core_match']}",
                f"  adjacent_only_match: {evidence['adjacent_only_match']}",
                "  target_core_best_matches:",
                _yaml_list([_format_match(match) for match in evidence["target_core_best_matches"]]),
                "  target_secondary_best_matches:",
                _yaml_list([_format_match(match) for match in evidence["target_secondary_best_matches"]]),
                "",
            ]
        )
    lines.extend([
        f"name: {company.get('name', '')}",
        f"cin: {company.get('cin', '')}",
        f"business_description: {company.get('business_description', '')}",
        f"value_chain_primary: {company.get('value_chain_primary', '')}",
        f"value_chain_secondary: {company.get('value_chain_secondary', '')}",
        f"customer_type: {company.get('customer_type', '')}",
        f"geographic_signals: {company.get('geographic_signals', '')}",
        f"revenue_crore: {company.get('revenue_crore', '')}",
        "core_products:",
        _yaml_list(company.get("core_products") or []),
        "secondary_products:",
        _yaml_list(company.get("secondary_products") or []),
        "minor_products:",
        _yaml_list(company.get("minor_products") or []),
        "end_markets:",
        _yaml_list(company.get("end_markets") or []),
    ])
    return "\n".join(lines)


def filter_enterprise_rerank_candidates(
    target: dict[str, Any],
    peers: list[dict[str, Any]],
    min_count: int,
    use_derived_categories: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept = list(peers)
    category_dropped: list[dict[str, Any]] = []
    business_model_dropped: list[dict[str, Any]] = []
    if use_derived_categories:
        evidence_by_cin = {
            peer.get("cin", ""): build_match_evidence(target, peer, use_derived_categories=use_derived_categories)
            for peer in kept
        }
        primary_category_matches = [
            peer
            for peer in kept
            if evidence_by_cin.get(peer.get("cin", ""), {}).get("normalized_category_match") == "primary_category_match"
        ]
        if len(primary_category_matches) >= min_count:
            kept = primary_category_matches
            kept_cins = {peer.get("cin", "") for peer in kept}
            category_dropped = [peer for peer in peers if peer.get("cin", "") not in kept_cins]

    target_archetype = infer_business_archetype(target)
    business_model_targets = {
        "vehicle_oem",
        "branded_jewellery_retailer",
        "fmcg_brand_manufacturer",
        "grocery_supermarket_retailer",
        "consumer_electronics_retail_distribution",
    }
    if target_archetype in business_model_targets:
        evidence_by_cin = {
            peer.get("cin", ""): build_match_evidence(target, peer, use_derived_categories=use_derived_categories)
            for peer in kept
        }
        preferred = [
            peer
            for peer in kept
            if evidence_by_cin.get(peer.get("cin", ""), {}).get("business_model_priority") == "strong_model_peer"
        ]
        acceptable = [
            peer
            for peer in kept
            if evidence_by_cin.get(peer.get("cin", ""), {}).get("business_model_peer_quality") == "acceptable_model_match"
        ]
        adjacent = [
            peer
            for peer in kept
            if evidence_by_cin.get(peer.get("cin", ""), {}).get("business_model_peer_quality") == "adjacent_model_match"
        ]
        weak = [
            peer
            for peer in kept
            if evidence_by_cin.get(peer.get("cin", ""), {}).get("business_model_peer_quality") == "weak_model_match"
        ]
        selected = list(preferred)
        for bucket in (acceptable, adjacent, weak):
            if len(selected) >= min_count:
                break
            bucket_sorted = sorted(
                bucket,
                key=lambda peer: _candidate_sort_score(peer, evidence_by_cin.get(peer.get("cin", ""), {})),
                reverse=True,
            )
            selected.extend(bucket_sorted[: max(0, min_count - len(selected))])
        if len(selected) >= min_count and len(selected) < len(kept):
            selected_cins = {peer.get("cin", "") for peer in selected}
            business_model_dropped.extend([peer for peer in kept if peer.get("cin", "") not in selected_cins])
            kept = selected

    return kept, {
        "enterprise_scale_filter": False,
        "derived_category_filter": bool(category_dropped),
        "business_model_filter": bool(business_model_dropped),
        "candidate_count_before_filter": len(peers),
        "candidate_count_after_filter": len(kept),
        "filtered_candidate_count": len(category_dropped) + len(business_model_dropped),
        "enterprise_filtered_candidate_names": [],
        "category_filtered_candidate_names": [peer.get("name", "") for peer in category_dropped[:20]],
        "business_model_filtered_candidate_names": [peer.get("name", "") for peer in business_model_dropped[:20]],
        "filtered_candidate_names": [peer.get("name", "") for peer in [*category_dropped, *business_model_dropped][:20]],
    }


def rerank_query(target: dict[str, Any], use_derived_categories: bool = True) -> str:
    return "\n\n".join(
        [
            (
                "Rank candidate companies as investment banking comparable-company peers "
                "for the target company. Choose companies an investment banker would use "
                "in a valuation or benchmarking peer set. A strong peer should share the "
                "target's primary business line, core products or services, value-chain "
                "role, customer type, and end markets. Core product or service overlap is "
                "the most important signal. Secondary or minor product overlap should not "
                "make a company a strong peer if its core business is different. "
                "Rank direct operating peers above adjacent-only matches and above matches "
                "to the target's secondary products. When two candidates both look like "
                "direct operating peers, use the product and company similarity scores as "
                "supporting signals, not as substitutes for business judgment. "
                "If retrieval_evidence is present, treat candidates with both product_max_sim "
                "and company_embedding sources plus medium/high scores as high-confidence "
                "candidates unless the business profile clearly contradicts peer fit. "
                "For branded retail targets, prefer own-brand retail chains over bullion "
                "traders, fashion marketplaces, and wholesalers. For vehicle targets, prefer "
                "OEMs above dealers and component suppliers. For FMCG brand targets, prefer "
                "brand manufacturers above ingredient suppliers or contract manufacturers. "
                "Prefer companies whose main business appears to come from the same category "
                "as the target. Penalize adjacent suppliers, customers, distributors, "
                "diversified side-business matches, and loose semantic matches when direct "
                "operating peers are available. Use geography and end markets as supporting "
                "signals and tie-breakers, not as substitutes for core business similarity. "
                "Do not let broad business-description overlap outweigh direct core-product "
                "or core-category fit from an investment banking peer analysis perspective."
            ),
            "Target company:",
            format_company_for_rerank(target),
        ]
    )


def rerank_peers(
    target: dict[str, Any],
    peers: list[dict[str, Any]],
    top_n: int,
    use_derived_categories: bool = True,
    include_retrieval_evidence: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    api_key = os.environ.get("COHERE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("COHERE_API_KEY is not configured")

    if not peers:
        return [], {
            "provider": "cohere",
            "model": os.environ.get("COHERE_RERANK_MODEL", DEFAULT_RERANK_MODEL),
            "candidate_count": 0,
        }

    model = os.environ.get("COHERE_RERANK_MODEL", DEFAULT_RERANK_MODEL).strip() or DEFAULT_RERANK_MODEL
    top_n = max(1, min(top_n, len(peers)))
    payload = {
        "model": model,
        "query": rerank_query(target, use_derived_categories=use_derived_categories),
        "documents": [
            format_company_for_rerank(
                peer,
                target=target,
                use_derived_categories=use_derived_categories,
                include_match_evidence=False,
                include_retrieval_evidence=include_retrieval_evidence,
            )
            for peer in peers
        ],
        "top_n": top_n,
    }
    request = urllib.request.Request(
        os.environ.get("COHERE_RERANK_URL", COHERE_RERANK_URL),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    timeout = float(os.environ.get("COHERE_RERANK_TIMEOUT", "20"))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"Cohere rerank failed: HTTP {exc.code}: {detail}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise RuntimeError(f"Cohere rerank timed out after {timeout:g}s") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cohere rerank failed: {exc.reason}") from exc

    reranked: list[dict[str, Any]] = []
    for result in body.get("results", []):
        index = result.get("index")
        if not isinstance(index, int) or index < 0 or index >= len(peers):
            continue
        item = dict(peers[index])
        item["pre_rerank_rank"] = index + 1
        item["pre_rerank_score"] = item.get("final_score")
        item["cohere_rerank_score"] = round(float(result.get("relevance_score", 0.0)), 6)
        item["final_score"] = item["cohere_rerank_score"]
        reranked.append(item)

    return reranked, {
        "provider": "cohere",
        "model": model,
        "used": True,
        "fallback": False,
        "candidate_count": len(peers),
        "returned_count": len(reranked),
        "derived_categories": use_derived_categories,
        "retrieval_evidence": include_retrieval_evidence,
    }
