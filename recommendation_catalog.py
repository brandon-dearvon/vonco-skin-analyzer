"""Versioned, server-owned Von & Co recommendation catalog.

The vision model never sees or selects from this catalog. It returns only
validated visible-surface priorities. This module deterministically maps those
priorities to current Von & Co services and products, with conservative area
gates and stable ordering.

Sources reviewed July 11-12, 2026:
* Von & Co Services Quick Reference, provider guide.
* Von & Co Product Quick Reference, July 2, 2026 provider guide.
* Von & Co Naples compliance service catalog.
* Current Von & Co service pages for public learn-more URLs.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


CATALOG_VERSION = "naples-appearance-recommendations-v3.1.0"

ALL_AREAS = frozenset({"face", "neck", "chest", "hands", "back", "legs"})
FACE_ONLY = frozenset({"face"})
FACE_AND_NECK = frozenset({"face", "neck"})
FACE_CHEST_HANDS = frozenset({"face", "chest", "hands"})
FACE_CHEST_LEGS = frozenset({"face", "chest", "legs"})
CHEST_AND_HANDS = frozenset({"chest", "hands"})
NOT_IMAGE_MAPPED = frozenset()

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
    "von_hydrafacial_service_page": {
        "title": "Von & Co HydraFacial treatment page",
        "url": "https://www.vonandcoaesthetics.com/services/hydrafacial-in-naples/",
        "reviewed": "2026-07-12",
    },
    "von_saltfacial_service_page": {
        "title": "Von & Co SaltFacial treatment page",
        "url": "https://www.vonandcoaesthetics.com/services/salt-facial-in-naples/",
        "reviewed": "2026-07-12",
    },
    "von_deep_pore_service_page": {
        "title": "Von & Co Deep Pore Cleansing Facial treatment page",
        "url": "https://www.vonandcoaesthetics.com/services/deep-cleansing-facial-in-naples/",
        "reviewed": "2026-07-12",
    },
    "von_sculptra_service_page": {
        "title": "Von & Co Sculptra treatment page",
        "url": "https://www.vonandcoaesthetics.com/services/sculptra-in-naples/",
        "reviewed": "2026-07-12",
    },
    "von_botox_service_page": {
        "title": "Von & Co Botox treatment page",
        "url": "https://www.vonandcoaesthetics.com/services/botox-in-naples/",
        "reviewed": "2026-07-12",
    },
    "von_dysport_service_page": {
        "title": "Von & Co Dysport treatment page",
        "url": "https://www.vonandcoaesthetics.com/services/dysport-in-naples/",
        "reviewed": "2026-07-12",
    },
    "von_xeomin_service_page": {
        "title": "Von & Co Xeomin treatment page",
        "url": "https://www.vonandcoaesthetics.com/services/xeomin-in-naples/",
        "reviewed": "2026-07-12",
    },
    "von_skinvive_service_page": {
        "title": "Von & Co SkinVive treatment page",
        "url": "https://www.vonandcoaesthetics.com/services/skinvive-in-naples/",
        "reviewed": "2026-07-12",
    },
    "von_facials_service_page": {
        "title": "Von & Co facials treatment page",
        "url": "https://www.vonandcoaesthetics.com/services/facials-in-naples/",
        "reviewed": "2026-07-12",
    },
    "von_all_services_page": {
        "title": "Von & Co all-services page",
        "url": "https://www.vonandcoaesthetics.com/services/all-services/",
        "reviewed": "2026-07-12",
    },
}

SERVICES_GUIDE_SOURCE = "services_quick_reference_107c045f"
PRODUCTS_GUIDE_SOURCE = "products_quick_reference_273ea983"


SERVICE_CATALOG: dict[str, dict[str, Any]] = {
    "botox": {
        "name": "Botox",
        "category": "Neurotoxin",
        "summary": (
            "A provider-selected neurotoxin option for softening the look of "
            "visible face or neck lines."
        ),
        "learnMoreUrl": "https://www.vonandcoaesthetics.com/services/botox-in-naples/",
        "areas": FACE_AND_NECK,
        "sourceRefs": (SERVICES_GUIDE_SOURCE, "von_botox_service_page"),
    },
    "dysport": {
        "name": "Dysport",
        "category": "Neurotoxin",
        "summary": (
            "A provider-selected neurotoxin option for softening the look of "
            "visible face or neck lines."
        ),
        "learnMoreUrl": "https://www.vonandcoaesthetics.com/services/dysport-in-naples/",
        "areas": FACE_AND_NECK,
        "sourceRefs": (SERVICES_GUIDE_SOURCE, "von_dysport_service_page"),
    },
    "xeomin": {
        "name": "Xeomin",
        "category": "Neurotoxin",
        "summary": (
            "A provider-selected neurotoxin option for softening the look of "
            "visible face or neck lines."
        ),
        "learnMoreUrl": "https://www.vonandcoaesthetics.com/services/xeomin-in-naples/",
        "areas": FACE_AND_NECK,
        "sourceRefs": (SERVICES_GUIDE_SOURCE, "von_xeomin_service_page"),
    },
    "dermal_fillers": {
        "name": "Dermal Fillers",
        "category": "HA injectable",
        "summary": (
            "Von & Co offers provider-selected HA fillers for guest goals involving "
            "facial volume and contour."
        ),
        "learnMoreUrl": "https://www.vonandcoaesthetics.com/services/all-services/",
        "areas": FACE_ONLY,
        "sourceRefs": (SERVICES_GUIDE_SOURCE, "von_all_services_page"),
    },
    "lip_filler": {
        "name": "Lip Filler",
        "category": "HA injectable",
        "summary": (
            "Von & Co offers provider-selected lip filler for guests who identify lip "
            "enhancement as a goal."
        ),
        "learnMoreUrl": "https://www.vonandcoaesthetics.com/services/all-services/",
        "areas": FACE_ONLY,
        "sourceRefs": (SERVICES_GUIDE_SOURCE, "von_all_services_page"),
    },
    "hydrafacial_elite": {
        "name": "Elite HydraFacial",
        "category": "Facial",
        "summary": (
            "An expanded cleansing, exfoliation, and hydration facial for fresher, "
            "smoother-looking skin."
        ),
        "learnMoreUrl": "https://www.vonandcoaesthetics.com/services/hydrafacial-in-naples/",
        "areas": FACE_AND_NECK,
        "sourceRefs": (SERVICES_GUIDE_SOURCE, "von_hydrafacial_service_page"),
    },
    "skinvive": {
        "name": "SkinVive",
        "category": "Skin-quality injectable",
        "summary": (
            "A provider-led skin-quality injectable for smoother-looking texture "
            "and softer facial fine lines."
        ),
        "learnMoreUrl": "https://www.vonandcoaesthetics.com/services/skinvive-in-naples/",
        "areas": FACE_ONLY,
        "sourceRefs": (SERVICES_GUIDE_SOURCE, "von_skinvive_service_page"),
    },
    "signature_facial": {
        "name": "Signature Cleansing Facial",
        "category": "Facial",
        "summary": (
            "A customizable cleansing, exfoliation, and hydration facial for fresh, "
            "smooth-looking skin."
        ),
        "learnMoreUrl": "https://www.vonandcoaesthetics.com/services/facials-in-naples/",
        "areas": FACE_ONLY,
        "sourceRefs": (SERVICES_GUIDE_SOURCE, "von_facials_service_page"),
    },
    "anti_aging_facial": {
        "name": "Anti-Aging Facial",
        "category": "Facial",
        "summary": (
            "A resurfacing and hydration facial for smoother-looking texture and "
            "softer visible lines."
        ),
        "learnMoreUrl": "https://www.vonandcoaesthetics.com/services/facials-in-naples/",
        "areas": FACE_ONLY,
        "sourceRefs": (SERVICES_GUIDE_SOURCE, "von_facials_service_page"),
    },
    "kybella": {
        "name": "Kybella",
        "category": "Contour injectable",
        "summary": (
            "Von & Co offers Kybella for provider-evaluated submental contour goals."
        ),
        "learnMoreUrl": "https://www.vonandcoaesthetics.com/services/all-services/",
        "areas": FACE_ONLY,
        "sourceRefs": (SERVICES_GUIDE_SOURCE, "von_all_services_page"),
    },
    "laser_hair_removal": {
        "name": "Laser Hair Removal",
        "category": "Hair reduction",
        "summary": (
            "Von & Co offers laser hair reduction when unwanted hair is an explicit "
            "guest goal."
        ),
        "learnMoreUrl": "https://www.vonandcoaesthetics.com/services/all-services/",
        "areas": ALL_AREAS,
        "sourceRefs": (SERVICES_GUIDE_SOURCE, "von_all_services_page"),
    },
    "hair_restoration_prf": {
        "name": "Hair Restoration (PRF)",
        "category": "Hair restoration",
        "summary": (
            "Von & Co offers provider-led PRF hair restoration for an explicit scalp "
            "hair goal."
        ),
        "learnMoreUrl": "https://www.vonandcoaesthetics.com/services/all-services/",
        "areas": NOT_IMAGE_MAPPED,
        "sourceRefs": (SERVICES_GUIDE_SOURCE, "von_all_services_page"),
    },
    "brow_lamination": {
        "name": "Brow Lamination",
        "category": "Brow styling",
        "summary": (
            "Von & Co offers brow lamination when brow shaping is an explicit guest goal."
        ),
        "learnMoreUrl": "https://www.vonandcoaesthetics.com/services/all-services/",
        "areas": FACE_ONLY,
        "sourceRefs": (SERVICES_GUIDE_SOURCE, "von_all_services_page"),
    },
    "semaglutide_melts": {
        "name": "Semaglutide Melts",
        "category": "Metabolism and wellness",
        "summary": (
            "A consultation-led wellness service that is never inferred from a skin photo."
        ),
        "learnMoreUrl": "https://www.vonandcoaesthetics.com/services/all-services/",
        "areas": NOT_IMAGE_MAPPED,
        "sourceRefs": (SERVICES_GUIDE_SOURCE, "von_all_services_page"),
    },
    "tirzepatide_injections": {
        "name": "Tirzepatide Injections",
        "category": "Metabolism and wellness",
        "summary": (
            "A consultation-led wellness service that is never inferred from a skin photo."
        ),
        "learnMoreUrl": "https://www.vonandcoaesthetics.com/services/all-services/",
        "areas": NOT_IMAGE_MAPPED,
        "sourceRefs": (SERVICES_GUIDE_SOURCE, "von_all_services_page"),
    },
    "sciton_bbl_photofacial": {
        "name": "Sciton BBL Photofacial",
        "category": "Light treatment",
        "summary": (
            "A light treatment for a clearer, more even-looking tone across redness, "
            "visible vessels, uneven pigment, and texture."
        ),
        "learnMoreUrl": "https://www.vonandcoaesthetics.com/services/bbl-treatment-in-naples/",
        "areas": ALL_AREAS,
        "sourceRefs": (SERVICES_GUIDE_SOURCE, "von_bbl_service_page"),
    },
    "sciton_moxi_laser": {
        "name": "Sciton Moxi Laser",
        "category": "Fractional laser",
        "summary": (
            "A gentle fractional laser for smoother-looking texture, refined-looking "
            "pores, and a more even-looking tone."
        ),
        "learnMoreUrl": "https://www.vonandcoaesthetics.com/services/moxi-laser-in-naples/",
        "areas": ALL_AREAS,
        "sourceRefs": (SERVICES_GUIDE_SOURCE, "von_moxi_service_page"),
    },
    "microneedling": {
        "name": "Microneedling",
        "category": "Collagen renewal",
        "summary": (
            "A collagen-renewal treatment for smoother-looking texture, refined-looking "
            "pores, and softer visible lines."
        ),
        "learnMoreUrl": "https://www.vonandcoaesthetics.com/services/microneedling-in-naples/",
        "areas": FACE_ONLY,
        "sourceRefs": (SERVICES_GUIDE_SOURCE, "von_microneedling_service_page"),
    },
    "microneedling_prf": {
        "name": "Microneedling + PRF",
        "category": "Collagen renewal",
        "summary": (
            "Microneedling enhanced with PRF for smoother-looking texture, "
            "refined-looking pores, and softer visible lines."
        ),
        "learnMoreUrl": "https://www.vonandcoaesthetics.com/services/microneedling-in-naples/",
        "areas": FACE_ONLY,
        "sourceRefs": (SERVICES_GUIDE_SOURCE, "von_microneedling_service_page"),
    },
    "rf_microneedling": {
        "name": "RF Microneedling",
        "category": "Collagen renewal",
        "summary": (
            "A collagen-renewal treatment for firmer, smoother-looking skin and "
            "refined-looking pores."
        ),
        "learnMoreUrl": "https://www.vonandcoaesthetics.com/services/rf-microneedling-in-naples/",
        "areas": FACE_AND_NECK,
        "sourceRefs": (
            SERVICES_GUIDE_SOURCE,
            "von_rf_microneedling_service_page",
        ),
    },
    "sciton_halo_laser": {
        "name": "Sciton Halo Laser",
        "category": "Hybrid fractional laser",
        "summary": (
            "A hybrid fractional laser for smoother-looking texture, softer visible "
            "lines, and a more even-looking tone."
        ),
        "learnMoreUrl": "https://www.vonandcoaesthetics.com/services/halo-laser-treatment-in-naples/",
        "areas": FACE_ONLY,
        "sourceRefs": (SERVICES_GUIDE_SOURCE, "von_halo_service_page"),
    },
    "chemical_peels": {
        "name": "Chemical Peels",
        "category": "Resurfacing treatment",
        "summary": (
            "Customizable resurfacing to refine discoloration, surface texture, "
            "blemish-like spots, and visible lines."
        ),
        "learnMoreUrl": "https://www.vonandcoaesthetics.com/services/chemical-peels-in-naples/",
        "areas": FACE_ONLY,
        "sourceRefs": (SERVICES_GUIDE_SOURCE, "von_peels_service_page"),
    },
    "hydrafacial_clarifying": {
        "name": "Clarifying HydraFacial",
        "category": "Facial",
        "summary": (
            "A cleansing and exfoliating facial for clearer-looking skin and "
            "refined-looking pores."
        ),
        "learnMoreUrl": "https://www.vonandcoaesthetics.com/services/hydrafacial-in-naples/",
        "areas": FACE_AND_NECK,
        "sourceRefs": (SERVICES_GUIDE_SOURCE, "von_hydrafacial_service_page"),
    },
    "hydrafacial_customized": {
        "name": "Customized HydraFacial",
        "category": "Facial",
        "summary": (
            "A customizable cleansing, exfoliation, and hydration facial for "
            "smoother, brighter-looking skin."
        ),
        "learnMoreUrl": "https://www.vonandcoaesthetics.com/services/hydrafacial-in-naples/",
        "areas": FACE_AND_NECK,
        "sourceRefs": (SERVICES_GUIDE_SOURCE, "von_hydrafacial_service_page"),
    },
    "saltfacial": {
        "name": "SaltFacial",
        "category": "Facial",
        "summary": (
            "A three-step facial designed to refresh tone, texture, pores, and "
            "visible surface details."
        ),
        "learnMoreUrl": "https://www.vonandcoaesthetics.com/services/salt-facial-in-naples/",
        "areas": FACE_ONLY,
        "sourceRefs": (SERVICES_GUIDE_SOURCE, "von_saltfacial_service_page"),
    },
    "deep_pore_facial": {
        "name": "Deep Pore Cleansing Facial",
        "category": "Facial",
        "summary": (
            "A focused cleansing facial for clearer-looking skin and refined-looking "
            "pores."
        ),
        "learnMoreUrl": "https://www.vonandcoaesthetics.com/services/deep-cleansing-facial-in-naples/",
        "areas": FACE_ONLY,
        "sourceRefs": (SERVICES_GUIDE_SOURCE, "von_deep_pore_service_page"),
    },
    "sculptra": {
        "name": "Sculptra",
        "category": "Biostimulatory injectable",
        "summary": (
            "A provider-led biostimulatory option for firmer-looking skin and "
            "softer visible lines."
        ),
        "learnMoreUrl": "https://www.vonandcoaesthetics.com/services/sculptra-in-naples/",
        "areas": FACE_CHEST_LEGS,
        "sourceRefs": (SERVICES_GUIDE_SOURCE, "von_sculptra_service_page"),
    },
}


# Source-derived from the locked provider guide plus each entry's cited current
# Von page. A service appears under an observation only when one of those sources
# directly supports the appearance relationship. Order follows guide sequence,
# then the cited public-page additions; area gates are applied separately.
APPEARANCE_SERVICE_MAP: dict[str, tuple[str, ...]] = {
    "visible_lines": (
        "botox",
        "dysport",
        "xeomin",
        "microneedling",
        "microneedling_prf",
        "rf_microneedling",
        "sciton_moxi_laser",
        "sciton_halo_laser",
        "chemical_peels",
        "saltfacial",
        "skinvive",
        "anti_aging_facial",
        "sculptra",
    ),
    "visible_redness": ("sciton_bbl_photofacial",),
    "pigment_variation": (
        "sciton_bbl_photofacial",
        "sciton_moxi_laser",
        "sciton_halo_laser",
        "microneedling",
        "microneedling_prf",
        "rf_microneedling",
        "chemical_peels",
        "hydrafacial_customized",
        "hydrafacial_elite",
        "saltfacial",
    ),
    "surface_texture": (
        "sciton_moxi_laser",
        "sciton_bbl_photofacial",
        "hydrafacial_customized",
        "hydrafacial_elite",
        "saltfacial",
        "skinvive",
        "signature_facial",
        "anti_aging_facial",
        "sciton_halo_laser",
        "microneedling",
        "microneedling_prf",
        "rf_microneedling",
        "chemical_peels",
    ),
    "pore_visibility": (
        "hydrafacial_clarifying",
        "hydrafacial_customized",
        "hydrafacial_elite",
        "deep_pore_facial",
        "saltfacial",
        "microneedling",
        "microneedling_prf",
        "sciton_moxi_laser",
        "rf_microneedling",
        "sciton_halo_laser",
        "signature_facial",
    ),
    "laxity_appearance": ("rf_microneedling", "sculptra", "sciton_halo_laser"),
    "blemish_like_spots": (
        "hydrafacial_clarifying",
        "deep_pore_facial",
        "saltfacial",
        "chemical_peels",
        "signature_facial",
    ),
    "scar_like_texture": (
        "microneedling",
        "microneedling_prf",
        "sciton_moxi_laser",
        "rf_microneedling",
        "chemical_peels",
        "sciton_halo_laser",
        "saltfacial",
    ),
    "superficial_vessels": ("sciton_bbl_photofacial",),
    "visible_flaking": (
        "hydrafacial_customized",
        "hydrafacial_elite",
        "signature_facial",
        "anti_aging_facial",
    ),
}

PRODUCT_CATALOG: dict[str, dict[str, Any]] = {
    "avene_thermal_water": {
        "name": "Thermal Spring Water",
        "brand": "Avène",
        "category": "Calm + clear",
        "benefit": (
            "Mineral-rich spring water that calms on contact and can be misted "
            "on whenever skin feels warm or reactive."
        ),
        "availability": "Carried by Von & Co; confirm current availability.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "skinbetter_alto_defense": {
        "name": "Alto Defense Serum",
        "brand": "SkinBetter Science",
        "category": "Calm + clear",
        "benefit": (
            "A fragrance-free antioxidant serum with 19 antioxidants in one pump "
            "to support a calmer-looking routine."
        ),
        "availability": "Carried by Von & Co; confirm current availability.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "skinbetter_even_tone": {
        "name": "Even Tone",
        "brand": "SkinBetter Science",
        "category": "Tone + brightness",
        "benefit": (
            "A retinol-free, hydroquinone-free serum for a more even-looking tone "
            "and visible dark spots."
        ),
        "availability": "Carried by Von & Co; confirm current availability.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "isdin_melaclear_advanced": {
        "name": "Melaclear Advanced",
        "brand": "ISDIN",
        "category": "Tone + brightness",
        "benefit": (
            "Tranexamic acid and niacinamide support a more even-looking tone by "
            "addressing visible pigment at multiple steps."
        ),
        "availability": "Carried by Von & Co; confirm current availability.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "skinbetter_alpharet": {
        "name": "AlphaRet",
        "brand": "SkinBetter Science",
        "category": "Visible aging",
        "benefit": (
            "A starter retinoid that combines a retinoid and AHA to refine the look "
            "of lines and texture with less peeling."
        ),
        "availability": "Ask a Von & Co provider before adding this active to your routine.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "zo_wrinkle_texture_repair": {
        "name": "Wrinkle + Texture Repair",
        "brand": "ZO Skin Health",
        "category": "Visible aging",
        "benefit": (
            "A 0.5% retinol treatment with ZCORE peptides for resilient skin that "
            "already tolerates stronger retinoids."
        ),
        "availability": "Ask a Von & Co provider before adding this active to your routine.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "alastin_restorative_skin_complex": {
        "name": "Restorative Skin Complex",
        "brand": "Alastin",
        "category": "Firmness support",
        "benefit": (
            "TriHex peptide technology supports collagen and elastin renewal for "
            "firmer, smoother-looking skin on the face or body."
        ),
        "availability": "Carried by Von & Co; confirm current availability.",
        "areas": FACE_CHEST_HANDS,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "zo_growth_factor_serum": {
        "name": "Growth Factor Serum",
        "brand": "ZO Skin Health",
        "category": "Firmness support",
        "benefit": (
            "A retinol-free growth factor serum that supports a collagen-focused "
            "routine morning and night."
        ),
        "availability": "Carried by Von & Co; confirm current availability.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "skinbetter_peel_pads": {
        "name": "Peel Pads",
        "brand": "SkinBetter Science",
        "category": "Texture support",
        "benefit": (
            "A gentler exfoliating-pad option for smoother-looking texture when a "
            "stronger acid pad feels like too much."
        ),
        "availability": "Ask a Von & Co provider before adding this active to your routine.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "zo_complexion_renewal_pads": {
        "name": "Complexion Renewal Pads",
        "brand": "ZO Skin Health",
        "category": "Texture support",
        "benefit": (
            "Glycolic and salicylic acids exfoliate to refine the look of pores, "
            "blemish-like spots, and surface texture."
        ),
        "availability": "Ask a Von & Co provider before adding this active to your routine.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "hydrinity_renewing_ha_serum": {
        "name": "Renewing HA Serum",
        "brand": "Hydrinity",
        "category": "Moisture support",
        "benefit": (
            "Multi-weight hyaluronic acid draws in moisture for a smoother, more "
            "hydrated-looking surface."
        ),
        "availability": "Carried by Von & Co; confirm current availability.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "skinbetter_trio_moisture": {
        "name": "Trio Moisture",
        "brand": "SkinBetter Science",
        "category": "Moisture support",
        "benefit": (
            "Ceramides and lipids support the moisture barrier when skin looks dry "
            "or feels depleted."
        ),
        "availability": "Carried by Von & Co; confirm current availability.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "colorscience_face_shield": {
        "name": "Face Shield SPF 50",
        "brand": "Colorescience",
        "category": "Daily protection",
        "benefit": (
            "A 100% mineral SPF 50 that protects from UV and visible light, with "
            "multiple finishes to suit different skin tones and preferences."
        ),
        "availability": "Carried by Von & Co; confirm current availability.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "colorscience_brush_on_shield": {
        "name": "Brush-On Shield",
        "brand": "Colorescience",
        "category": "Daily protection",
        "availability": "Carried by Von & Co; confirm current availability.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "isdin_eryfotona_actinica": {
        "name": "Eryfotona Actinica",
        "brand": "ISDIN",
        "category": "Daily protection",
        "availability": "Carried by Von & Co; confirm current availability.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "isdin_eryfotona_tinted": {
        "name": "Eryfotona Tinted",
        "brand": "ISDIN",
        "category": "Daily protection",
        "availability": "Carried by Von & Co; confirm current availability.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "alastin_hydratint": {
        "name": "HydraTint SPF 36",
        "brand": "Alastin",
        "category": "Daily protection",
        "availability": "Carried by Von & Co; confirm current availability.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "pavise_dynamic_age_defense": {
        "name": "Dynamic Age Defense",
        "brand": "Pavise",
        "category": "Daily protection",
        "availability": "Carried by Von & Co; confirm current availability.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "skinbetter_sunbetter_spf_68": {
        "name": "Sunbetter SPF 68",
        "brand": "SkinBetter Science",
        "category": "Daily protection",
        "availability": "Carried by Von & Co; confirm current availability.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "colorscience_total_eye_spf_35": {
        "name": "Total Eye SPF 35",
        "brand": "Colorescience",
        "category": "Eye protection",
        "availability": "Carried by Von & Co; confirm current availability.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "colorscience_lip_shine_spf_35": {
        "name": "Lip Shine SPF 35",
        "brand": "Colorescience",
        "category": "Lip protection",
        "availability": "Carried by Von & Co; confirm current availability.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "zo_vitamin_c_10": {
        "name": "10% Vitamin C",
        "brand": "ZO Skin Health",
        "category": "Tone + brightness",
        "benefit": (
            "L-ascorbic acid brightens and helps defend against free radicals as a "
            "morning antioxidant step."
        ),
        "availability": "Ask a Von & Co provider before adding this active to your routine.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "hydrinity_vivid_serum": {
        "name": "Vivid Serum",
        "brand": "Hydrinity",
        "category": "Tone + brightness",
        "benefit": (
            "Hexylresorcinol and peptides brighten without retinol, making this a "
            "gentler option for a more even-looking tone."
        ),
        "availability": "Carried by Von & Co; confirm current availability.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "isdin_retinal_advanced": {
        "name": "Retinal Advanced",
        "brand": "ISDIN",
        "category": "Visible aging",
        "availability": "Ask a Von & Co provider before adding this active to your routine.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "hydrinity_retaxome": {
        "name": "RetaXome",
        "brand": "Hydrinity",
        "category": "Visible aging",
        "availability": "Ask a Von & Co provider before adding this active to your routine.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "alastin_skin_nectar": {
        "name": "Skin Nectar",
        "brand": "Alastin",
        "category": "Procedure support",
        "availability": "Use with the pre- or post-procedure plan provided by Von & Co.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "alastin_restorative_eye_cream": {
        "name": "Restorative Eye Cream",
        "brand": "Alastin",
        "category": "Eye care",
        "availability": "Carried by Von & Co; confirm current availability.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "avene_cicalfate": {
        "name": "Cicalfate+",
        "brand": "Avène",
        "category": "Barrier support",
        "availability": "Use with the routine or procedure plan provided by Von & Co.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "hydrinity_hyacyn_mist": {
        "name": "Hyacyn Mist",
        "brand": "Hydrinity",
        "category": "Procedure support",
        "availability": "Use with the routine or procedure plan provided by Von & Co.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "revitalash_conditioner": {
        "name": "RevitaLash Conditioner",
        "brand": "RevitaLash",
        "category": "Lash care",
        "availability": "Carried by Von & Co; confirm current availability.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "revitalash_brow_conditioner": {
        "name": "RevitaLash Brow Conditioner",
        "brand": "RevitaLash",
        "category": "Brow care",
        "availability": "Carried by Von & Co; confirm current availability.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "skinbetter_alpharet_body": {
        "name": "AlphaRet Body",
        "brand": "SkinBetter Science",
        "category": "Body texture + firmness",
        "benefit": (
            "A retinoid body treatment for crepey-looking texture and visible "
            "firmness concerns on the arms, chest, or hands."
        ),
        "availability": "Ask a Von & Co provider before adding this active to your routine.",
        "areas": CHEST_AND_HANDS,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "zo_growth_factor_eye": {
        "name": "Growth Factor Eye",
        "brand": "ZO Skin Health",
        "category": "Eye care",
        "availability": "Carried by Von & Co; confirm current availability.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "hydrinity_eye_renew": {
        "name": "Eye Renew",
        "brand": "Hydrinity",
        "category": "Eye care",
        "availability": "Carried by Von & Co; confirm current availability.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
    "colorscience_lash_mascara": {
        "name": "Lash Mascara",
        "brand": "Colorescience",
        "category": "Lash care",
        "availability": "Carried by Von & Co; confirm current availability.",
        "areas": FACE_ONLY,
        "sourceRefs": (PRODUCTS_GUIDE_SOURCE,),
    },
}

MAX_SERVICE_RECOMMENDATIONS = len(SERVICE_CATALOG)
MAX_PRODUCT_RECOMMENDATIONS = len(PRODUCT_CATALOG)


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
        ("zo_vitamin_c_10", "Additional guide match"),
        ("hydrinity_vivid_serum", "Additional guide match"),
    ),
    "surface_texture": (
        ("zo_complexion_renewal_pads", "Guide match"),
        ("skinbetter_peel_pads", "Alternative match"),
    ),
    "pore_visibility": (
        ("zo_complexion_renewal_pads", "Targeted match"),
    ),
    "laxity_appearance": (
        ("alastin_restorative_skin_complex", "Guide match"),
        ("zo_growth_factor_serum", "Alternative match"),
    ),
    "blemish_like_spots": (
        ("zo_complexion_renewal_pads", "Targeted match"),
    ),
    "scar_like_texture": (),
    "superficial_vessels": (),
    "visible_flaking": (
        ("hydrinity_renewing_ha_serum", "Targeted match"),
        ("skinbetter_trio_moisture", "Alternative match"),
    ),
}

BODY_PRODUCT_MAP: dict[str, tuple[tuple[str, str], ...]] = {
    "visible_lines": (
        ("alastin_restorative_skin_complex", "Guide match"),
        ("skinbetter_alpharet_body", "Alternative match"),
    ),
    "laxity_appearance": (
        ("alastin_restorative_skin_complex", "Guide match"),
        ("skinbetter_alpharet_body", "Alternative match"),
    ),
}

FACE_PROTECTION_PRODUCT_ID = "colorscience_face_shield"


def _joined_labels(observation_ids: Sequence[str], labels: Mapping[str, str]) -> str:
    values = [labels[item].lower() for item in observation_ids if item in labels]
    if not values:
        return "the visible priorities"
    if len(values) == 1:
        return values[0]
    return f"{', '.join(values[:-1])} and {values[-1]}"


def _guest_availability(value: str) -> str:
    if value.startswith("Ask a Von & Co provider"):
        return "A Von & Co provider can help decide whether this active belongs in your routine."
    if value.startswith("Use only") or value.startswith("Use with"):
        return "Use only within the pre- or post-procedure plan provided by Von & Co."
    return "Available at Von & Co. Please confirm current studio availability."


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
    return selected


def _product_map_for_area(body_area: str) -> Mapping[str, tuple[tuple[str, str], ...]]:
    if body_area == "face":
        return FACE_PRODUCT_MAP
    if body_area in {"chest", "hands"}:
        return BODY_PRODUCT_MAP
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

    if body_area == "face" and priorities:
        protection = PRODUCT_CATALOG[FACE_PROTECTION_PRODUCT_ID]
        if body_area in protection["areas"]:
            _add_match(
                selected,
                FACE_PROTECTION_PRODUCT_ID,
                priorities[0],
                "Daily protection",
            )
    return selected


def build_appearance_recommendations(
    priorities: Sequence[str],
    body_area: str,
    labels: Mapping[str, str],
) -> dict[str, list[dict[str, Any]]]:
    """Return deterministic, explainable service and product recommendations."""

    ordered_observations: list[str] = []
    for item in priorities:
        if item in labels and item not in ordered_observations:
            ordered_observations.append(item)
    service_matches = _service_candidates(ordered_observations, body_area)
    product_matches = _product_candidates(ordered_observations, body_area)

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
                "why": service["summary"],
                "learnMoreUrl": service["learnMoreUrl"],
            }
        )

    products: list[dict[str, Any]] = []
    for product_id, match in product_matches.items():
        product = PRODUCT_CATALOG[product_id]
        matched_ids = list(match["matchedObservationIds"])
        relationship = str(match["relationship"] or "Matched option")
        benefit = str(product.get("benefit") or "").strip()
        why = (
            f"Von & Co includes daily SPF with every treatment plan. {benefit}"
            if relationship == "Daily protection"
            else benefit
        )
        products.append(
            {
                "id": product_id,
                "name": product["name"],
                "brand": product["brand"],
                "category": product["category"],
                "relationship": relationship,
                "matchedObservationIds": matched_ids,
                "why": why,
                "availability": _guest_availability(product["availability"]),
            }
        )

    return {"services": services, "products": products}
