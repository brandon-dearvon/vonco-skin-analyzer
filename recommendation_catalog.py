"""Versioned, server-owned Von & Co recommendation catalog.

The vision model never sees or selects from this catalog. It returns only
validated visible-surface priorities. This module deterministically maps those
priorities to current Von & Co services and products, with conservative area
gates and stable ordering.

Sources reviewed July 11, 2026:
* Von & Co Services Quick Reference, provider guide.
* Von & Co Product Quick Reference, July 2, 2026 provider guide.
* Von & Co Naples compliance service catalog.
* Current Von & Co service pages for public learn-more URLs.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


CATALOG_VERSION = "naples-appearance-recommendations-v2.0.0"

ALL_AREAS = frozenset({"face", "neck_chest", "hands", "back", "legs"})
FACE_ONLY = frozenset({"face"})

# Immutable evidence identifiers make each catalog entry independently auditable.
# The hashes are for the exact provider PDFs reviewed when this catalog version
# was frozen; public pages were rechecked on the recorded review date.
SOURCE_EVIDENCE: dict[str, dict[str, str]] = {
    "services_quick_reference_107c045f": {
        "title": "Von & Co Services Quick Reference",
        "sha256": "107c045f4a4638006aea21d9becae501af690323701c3210a4a27760534bc04b",
        "reviewed": "2026-07-11",
    },
    "products_quick_reference_273ea983": {
        "title": "Von & Co Product Quick Reference",
        "sha256": "273ea9836c0f1e96d5ec8b5f1c962a0f86867f09c8a44b0eb6bd7bcebbd48854",
        "reviewed": "2026-07-11",
    },
    "von_bbl_service_page": {
        "title": "Von & Co BBL treatment page",
        "url": "https://www.vonandcoaesthetics.com/services/bbl-treatment-in-naples/",
        "reviewed": "2026-07-11",
    },
    "von_moxi_service_page": {
        "title": "Von & Co Moxi treatment page",
        "url": "https://www.vonandcoaesthetics.com/services/moxi-laser-in-naples/",
        "reviewed": "2026-07-11",
    },
    "von_microneedling_service_page": {
        "title": "Von & Co Microneedling treatment page",
        "url": "https://www.vonandcoaesthetics.com/services/microneedling-in-naples/",
        "reviewed": "2026-07-11",
    },
    "von_rf_microneedling_service_page": {
        "title": "Von & Co RF Microneedling treatment page",
        "url": "https://www.vonandcoaesthetics.com/services/rf-microneedling-in-naples/",
        "reviewed": "2026-07-11",
    },
    "von_halo_service_page": {
        "title": "Von & Co Halo treatment page",
        "url": "https://www.vonandcoaesthetics.com/services/halo-laser-treatment-in-naples/",
        "reviewed": "2026-07-11",
    },
    "von_peels_service_page": {
        "title": "Von & Co Chemical Peels treatment page",
        "url": "https://www.vonandcoaesthetics.com/services/chemical-peels-in-naples/",
        "reviewed": "2026-07-11",
    },
}

SERVICES_GUIDE_SOURCE = "services_quick_reference_107c045f"
PRODUCTS_GUIDE_SOURCE = "products_quick_reference_273ea983"


SERVICE_CATALOG: dict[str, dict[str, Any]] = {
    "sciton_bbl_photofacial": {
        "name": "Sciton BBL Photofacial",
        "category": "Light treatment",
        "summary": (
            "Von & Co uses BBL in face-and-body plans focused on the appearance "
            "of redness and uneven pigment."
        ),
        "learnMoreUrl": "https://www.vonandcoaesthetics.com/services/bbl-treatment-in-naples/",
        "areas": ALL_AREAS,
        "sourceRefs": (SERVICES_GUIDE_SOURCE, "von_bbl_service_page"),
    },
    "sciton_moxi_laser": {
        "name": "Sciton Moxi Laser",
        "category": "Fractional laser",
        "summary": (
            "Von & Co uses Moxi as a gentle fractional face-and-body option when "
            "visible surface texture is a priority."
        ),
        "learnMoreUrl": "https://www.vonandcoaesthetics.com/services/moxi-laser-in-naples/",
        "areas": ALL_AREAS,
        "sourceRefs": (SERVICES_GUIDE_SOURCE, "von_moxi_service_page"),
    },
    "microneedling": {
        "name": "Microneedling",
        "category": "Collagen renewal",
        "summary": (
            "Von & Co uses microneedling in facial plans focused on the appearance "
            "of fine lines and collagen renewal."
        ),
        "learnMoreUrl": "https://www.vonandcoaesthetics.com/services/microneedling-in-naples/",
        "areas": FACE_ONLY,
        "sourceRefs": (SERVICES_GUIDE_SOURCE, "von_microneedling_service_page"),
    },
    "microneedling_prf": {
        "name": "Microneedling + PRF",
        "category": "Collagen renewal",
        "summary": (
            "Von & Co's provider guide places Microneedling + PRF among the options "
            "for visible pore and scar-like texture concerns."
        ),
        "learnMoreUrl": "https://www.vonandcoaesthetics.com/services/microneedling-in-naples/",
        "areas": FACE_ONLY,
        "sourceRefs": (SERVICES_GUIDE_SOURCE, "von_microneedling_service_page"),
    },
    "rf_microneedling": {
        "name": "RF Microneedling",
        "category": "Collagen renewal",
        "summary": (
            "Von & Co uses RF microneedling when visible laxity, lines, pores, or "
            "scar-like texture is a priority."
        ),
        "learnMoreUrl": "https://www.vonandcoaesthetics.com/services/rf-microneedling-in-naples/",
        "areas": FACE_ONLY,
        "sourceRefs": (
            SERVICES_GUIDE_SOURCE,
            "von_rf_microneedling_service_page",
        ),
    },
    "sciton_halo_laser": {
        "name": "Sciton Halo Laser",
        "category": "Hybrid fractional laser",
        "summary": (
            "Von & Co uses Halo for facial plans focused on visible pigment, lines, "
            "pores, and scar-like texture."
        ),
        "learnMoreUrl": "https://www.vonandcoaesthetics.com/services/halo-laser-treatment-in-naples/",
        "areas": FACE_ONLY,
        "sourceRefs": (SERVICES_GUIDE_SOURCE, "von_halo_service_page"),
    },
    "chemical_peels": {
        "name": "Chemical Peels",
        "category": "Resurfacing treatment",
        "summary": (
            "Von & Co customizes peel options for plans focused on visible "
            "discoloration and scar-like surface texture."
        ),
        "learnMoreUrl": "https://www.vonandcoaesthetics.com/services/chemical-peels-in-naples/",
        "areas": FACE_ONLY,
        "sourceRefs": (SERVICES_GUIDE_SOURCE, "von_peels_service_page"),
    },
}


# Ordered by the provider service guide. A later candidate is used only when an
# earlier candidate is unavailable for the selected area or room remains in the
# recommendation set.
APPEARANCE_SERVICE_MAP: dict[str, tuple[str, ...]] = {
    "visible_lines": ("microneedling", "rf_microneedling"),
    "visible_redness": ("sciton_bbl_photofacial",),
    "pigment_variation": (
        "sciton_bbl_photofacial",
        "sciton_halo_laser",
        "chemical_peels",
    ),
    "surface_texture": (
        "sciton_moxi_laser",
        "rf_microneedling",
        "sciton_halo_laser",
    ),
    "pore_visibility": ("microneedling_prf", "sciton_halo_laser"),
    "laxity_appearance": ("rf_microneedling", "sciton_halo_laser"),
    "blemish_like_spots": (),
    "scar_like_texture": (
        "microneedling_prf",
        "rf_microneedling",
        "chemical_peels",
    ),
    "superficial_vessels": (),
    "visible_flaking": (),
}

PRODUCT_CATALOG: dict[str, dict[str, Any]] = {
    "avene_thermal_water": {
        "name": "Thermal Spring Water",
        "brand": "Avène",
        "category": "Calm + clear",
        "availability": "Carried by Von & Co; confirm current availability.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "skinbetter_alto_defense": {
        "name": "Alto Defense Serum",
        "brand": "SkinBetter Science",
        "category": "Calm + clear",
        "availability": "Carried by Von & Co; confirm current availability.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "skinbetter_even_tone": {
        "name": "Even Tone",
        "brand": "SkinBetter Science",
        "category": "Tone + brightness",
        "availability": "Carried by Von & Co; confirm current availability.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "isdin_melaclear_advanced": {
        "name": "Melaclear Advanced",
        "brand": "ISDIN",
        "category": "Tone + brightness",
        "availability": "Carried by Von & Co; confirm current availability.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "skinbetter_alpharet": {
        "name": "AlphaRet",
        "brand": "SkinBetter Science",
        "category": "Visible aging",
        "availability": "Ask a Von & Co provider before adding this active to your routine.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "zo_wrinkle_texture_repair": {
        "name": "Wrinkle + Texture Repair",
        "brand": "ZO Skin Health",
        "category": "Visible aging",
        "availability": "Ask a Von & Co provider before adding this active to your routine.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "alastin_restorative_skin_complex": {
        "name": "Restorative Skin Complex",
        "brand": "Alastin",
        "category": "Firmness support",
        "availability": "Carried by Von & Co; confirm current availability.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "zo_growth_factor_serum": {
        "name": "Growth Factor Serum",
        "brand": "ZO Skin Health",
        "category": "Firmness support",
        "availability": "Carried by Von & Co; confirm current availability.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "skinbetter_peel_pads": {
        "name": "Peel Pads",
        "brand": "SkinBetter Science",
        "category": "Texture support",
        "availability": "Ask a Von & Co provider before adding this active to your routine.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "zo_complexion_renewal_pads": {
        "name": "Complexion Renewal Pads",
        "brand": "ZO Skin Health",
        "category": "Texture support",
        "availability": "Ask a Von & Co provider before adding this active to your routine.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
}


# Relationship labels mirror the current Product Quick Reference. They describe
# catalog position, not medical suitability.
FACE_PRODUCT_MAP: dict[str, tuple[tuple[str, str], ...]] = {
    "visible_lines": (
        ("skinbetter_alpharet", "Guide match"),
        ("zo_wrinkle_texture_repair", "Alternative match"),
    ),
    "visible_redness": (
        ("avene_thermal_water", "Guide match"),
        ("skinbetter_alto_defense", "Layering option"),
    ),
    "pigment_variation": (
        ("skinbetter_even_tone", "Guide match"),
        ("isdin_melaclear_advanced", "Alternative match"),
    ),
    "surface_texture": (
        ("zo_complexion_renewal_pads", "Guide match"),
        ("skinbetter_peel_pads", "Alternative match"),
    ),
    "pore_visibility": (),
    "laxity_appearance": (
        ("alastin_restorative_skin_complex", "Guide match"),
        ("zo_growth_factor_serum", "Alternative match"),
    ),
    "blemish_like_spots": (),
    "scar_like_texture": (),
    "superficial_vessels": (),
    "visible_flaking": (),
}


def _joined_labels(observation_ids: Sequence[str], labels: Mapping[str, str]) -> str:
    values = [labels[item].lower() for item in observation_ids if item in labels]
    if not values:
        return "the visible priorities"
    if len(values) == 1:
        return values[0]
    return f"{', '.join(values[:-1])} and {values[-1]}"


def _add_match(
    selected: dict[str, dict[str, Any]],
    item_id: str,
    observation_id: str,
    relationship: str | None = None,
) -> None:
    existing = selected.get(item_id)
    if existing is None:
        selected[item_id] = {
            "matchedObservationIds": [observation_id] if observation_id else [],
            "relationship": relationship,
        }
        return
    if observation_id and observation_id not in existing["matchedObservationIds"]:
        existing["matchedObservationIds"].append(observation_id)


def _service_candidates(priorities: Sequence[str], body_area: str) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    max_depth = max((len(APPEARANCE_SERVICE_MAP.get(item, ())) for item in priorities), default=0)
    for depth in range(max_depth):
        for observation_id in priorities:
            candidates = APPEARANCE_SERVICE_MAP.get(observation_id, ())
            if depth >= len(candidates):
                continue
            service_id = candidates[depth]
            service = SERVICE_CATALOG.get(service_id)
            approved_areas = service["areas"] if service else frozenset()
            if not service or body_area not in approved_areas:
                continue
            _add_match(selected, service_id, observation_id)
            if len(selected) >= 3:
                return selected
    return selected


def _product_map_for_area(body_area: str) -> Mapping[str, tuple[tuple[str, str], ...]]:
    if body_area == "face":
        return FACE_PRODUCT_MAP
    return {}


def _product_candidates(
    priorities: Sequence[str],
    body_area: str,
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    product_map = _product_map_for_area(body_area)
    max_depth = max((len(product_map.get(item, ())) for item in priorities), default=0)
    for depth in range(max_depth):
        for observation_id in priorities:
            candidates = product_map.get(observation_id, ())
            if depth >= len(candidates):
                continue
            product_id, relationship = candidates[depth]
            product = PRODUCT_CATALOG.get(product_id)
            if not product or body_area not in product["areas"]:
                continue
            _add_match(selected, product_id, observation_id, relationship)
            if len(selected) >= 2:
                return selected
    return selected


def build_appearance_recommendations(
    priorities: Sequence[str],
    body_area: str,
    labels: Mapping[str, str],
) -> dict[str, list[dict[str, Any]]]:
    """Return deterministic, explainable service and product recommendations."""

    ordered_priorities = [item for item in priorities if item in labels][:2]
    service_matches = _service_candidates(ordered_priorities, body_area)
    product_matches = _product_candidates(ordered_priorities, body_area)

    services: list[dict[str, Any]] = []
    for service_id, match in service_matches.items():
        service = SERVICE_CATALOG[service_id]
        matched_ids = list(match["matchedObservationIds"])
        services.append(
            {
                "id": service_id,
                "name": service["name"],
                "category": service["category"],
                "matchedObservationIds": matched_ids,
                "why": (
                    f"Matched to {_joined_labels(matched_ids, labels)} in your photos. "
                    f"{service['summary']}"
                ),
                "learnMoreUrl": service["learnMoreUrl"],
            }
        )

    products: list[dict[str, Any]] = []
    for product_id, match in product_matches.items():
        product = PRODUCT_CATALOG[product_id]
        matched_ids = list(match["matchedObservationIds"])
        relationship = str(match["relationship"] or "Matched option")
        products.append(
            {
                "id": product_id,
                "name": product["name"],
                "brand": product["brand"],
                "category": product["category"],
                "relationship": relationship,
                "matchedObservationIds": matched_ids,
                "why": (
                    f"Matched to {_joined_labels(matched_ids, labels)} in your photos. "
                    "Von & Co's current product guide includes this option for that "
                    "appearance concern."
                ),
                "availability": product["availability"],
            }
        )

    return {"services": services, "products": products}
