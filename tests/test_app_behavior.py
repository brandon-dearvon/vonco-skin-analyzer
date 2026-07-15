"""Behavioral regressions for the restored original app plus approved changes."""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import re
import threading
import time
import unittest
import sqlite3
import stat
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import httpx
from google.genai import errors as genai_errors
from PIL import Image

TEST_REPEAT_CACHE_PATH = (
    Path(__file__).resolve().parents[1]
    / "work/qa/test-analysis-repeat-cache.sqlite3"
)
os.environ.setdefault(
    "ANALYSIS_REPEAT_CACHE_PATH",
    str(TEST_REPEAT_CACHE_PATH),
)
os.environ.setdefault("EXPOSE_ANALYSIS_REPEAT_HEADER", "true")

import server


ROOT = Path(__file__).resolve().parents[1]
BRITTANY_PHOTO = Path(
    os.getenv(
        "BRITTANY_TEST_PHOTO",
        "/Users/macbookair2/Desktop/Von To Do/Brittany Test Photo.jpeg",
    )
)

AREA_CONCERNS = {
    "face": {
        "wrinkles",
        "redness",
        "darkSpots",
        "texture",
        "pores",
        "laxity",
        "sunDamage",
        "unevenTone",
    },
    "neck_chest": {"sunDamage", "laxity", "redness", "texture", "wrinkles"},
    "hands": {"sunDamage", "laxity", "texture", "veins", "dryness"},
    "back": {"acne", "scarring", "texture", "unevenTone", "hairRemoval"},
    "legs": {"veins", "texture", "sunDamage", "hairRemoval", "dryness"},
}

CONCERN_EVIDENCE_LABELS = {
    "wrinkles": "lines and creases",
    "redness": "redness",
    "darkSpots": "pigment variation",
    "texture": "surface texture",
    "pores": "pores",
    "laxity": "crepiness",
    "sunDamage": "pigment variation and sun-exposure signs",
    "unevenTone": "tone variation",
    "acne": "surface congestion",
    "scarring": "textural marks",
    "hairRemoval": "body-hair growth",
    "veins": "veins",
    "dryness": "surface dryness",
}
CONCERN_FUZZ_TERMS = {
    "wrinkles": "lines",
    "redness": "redness",
    "darkSpots": "pigment variation",
    "texture": "texture",
    "pores": "pores",
    "laxity": "crepiness",
    "sunDamage": "sun-exposure signs",
    "unevenTone": "tone variation",
    "acne": "surface congestion",
    "scarring": "textural marks",
    "hairRemoval": "stubble",
    "veins": "veins",
    "dryness": "dryness",
}
PLURAL_EVIDENCE_LABELS = {"wrinkles", "sunDamage", "veins"}


def concern_evidence_description(
    concern_key: str,
    *,
    score: int,
    area: str,
) -> str:
    """Return concern-specific synthetic evidence for parser contract tests."""
    if concern_key == "hairRemoval" and score >= 41:
        return (
            "Distinct visible hair follicles and clearly visible follicular "
            "contrast are present in the photographed area."
        )
    if concern_key == "veins" and score >= 41:
        return (
            "Clearly visible blue surface veins form a branching network "
            "across the photographed area."
        )
    label = CONCERN_EVIDENCE_LABELS[concern_key]
    if (
        score >= 41
        and area == "neck_chest"
        and concern_key in {"laxity", "wrinkles"}
    ):
        verb = "remain" if concern_key == "wrinkles" else "remains"
        return (
            f"Clearly visible persistent {label} {verb} visible at rest in a "
            "neutral resting view, independent of pose."
        )
    qualifier = "Clearly visible" if score >= 41 else "Mild visible"
    verb = "are" if concern_key in PLURAL_EVIDENCE_LABELS else "is"
    return f"{qualifier} {label} {verb} present in the photographed area."


def jpeg_bytes(
    width: int = 48,
    height: int = 64,
    color: tuple[int, int, int] = (178, 146, 126),
) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="JPEG")
    return buffer.getvalue()


def accepted_analysis(area: str = "face") -> dict:
    score_maps = {
        "face": {
            "wrinkles": 20,
            "redness": 5,
            "darkSpots": 55,
            "texture": 35,
            "pores": 15,
            "laxity": 8,
            "sunDamage": 50,
            "unevenTone": 40,
        },
        "neck_chest": {
            "sunDamage": 55,
            "laxity": 45,
            "redness": 20,
            "texture": 30,
            "wrinkles": 15,
        },
        "hands": {
            "sunDamage": 55,
            "laxity": 45,
            "texture": 30,
            "veins": 15,
            "dryness": 25,
        },
        "back": {
            "acne": 55,
            "scarring": 45,
            "texture": 35,
            "unevenTone": 25,
            "hairRemoval": 15,
        },
        "legs": {
            "veins": 55,
            "texture": 45,
            "sunDamage": 35,
            "hairRemoval": 25,
            "dryness": 15,
        },
    }
    concerns = {
        key: {
            "score": score,
            "severity": (
                "none" if score <= 10 else
                "mild" if score <= 40 else
                "moderate" if score <= 60 else
                "severe"
            ),
            "description": concern_evidence_description(
                key,
                score=score,
                area=area,
            ),
        }
        for key, score in score_maps[area].items()
    }
    recommendation_maps = {
        "face": [
            ("Sciton BBL", ["darkSpots", "sunDamage", "unevenTone"]),
            ("Sciton Moxi", ["darkSpots", "sunDamage", "texture"]),
            ("HydraFacial Customized", ["texture", "pores", "unevenTone"]),
        ],
        "neck_chest": [
            ("Sciton BBL", ["sunDamage", "redness"]),
            ("Sciton Moxi", ["sunDamage", "texture", "wrinkles"]),
            ("Microneedling", ["laxity", "texture"]),
        ],
        "hands": [
            ("Sciton BBL", ["sunDamage", "veins"]),
            ("Sciton Moxi", ["sunDamage", "texture"]),
            ("RF Microneedling", ["laxity", "texture"]),
        ],
        "back": [
            ("Chemical Peels", ["acne", "scarring", "texture", "unevenTone"]),
            ("Microneedling + PRF", ["scarring", "texture"]),
            ("Sciton BBL", ["acne", "unevenTone"]),
        ],
        "legs": [
            ("Sciton BBL", ["veins", "sunDamage"]),
            ("Microneedling", ["texture"]),
            ("Laser Hair Removal", ["hairRemoval"]),
        ],
    }
    area_copy = {
        "face": "Your complexion",
        "neck_chest": "Your neck and chest",
        "hands": "Your hands",
        "back": "The skin across your back",
        "legs": "The skin across your legs",
    }[area]
    return {
        "overallScore": 76,
        "observedArea": area,
        "positiveHighlights": [
            {
                "title": "Luminous quality",
                "detail": f"{area_copy} has a bright, luminous quality.",
            },
            {
                "title": "Refined texture",
                "detail": f"{area_copy} appears smooth and polished.",
            },
        ],
        "concerns": concerns,
        "recommendations": [
            {
                "treatment": treatment,
                "reason": f"A mapped option for {', '.join(targets)}.",
                "targets": targets,
                "priority": priority,
            }
            for priority, (treatment, targets) in enumerate(
                recommendation_maps[area],
                start=1,
            )
        ],
        "productRecommendations": [
            {
                "product": "SkinBetter Even Tone",
                "reason": "A skincare option tied to the visible goals.",
            },
            {
                "product": "Colorescience Face Shield SPF 50",
                "reason": "Daily sun protection.",
            },
        ],
        "suggestedCombo": "Hero Combo",
        "summary": f"{area_copy} has a bright, luminous quality. Options follow.",
    }


class FakeModels:
    def __init__(self, payload: dict, fail_first: bool = False):
        self.payload = payload
        self.fail_first = fail_first
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_first and len(self.calls) == 1:
            raise RuntimeError("transient test failure")
        return SimpleNamespace(text=json.dumps(self.payload))


class TimeoutModels:
    def __init__(self):
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        raise httpx.ReadTimeout(
            "synthetic upstream timeout",
            request=httpx.Request("POST", "https://example.test/generate"),
        )


class SequencedModels:
    def __init__(self, steps):
        self.steps = list(steps)
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        step = self.steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        if isinstance(step, dict):
            step = json.dumps(step)
        return SimpleNamespace(text=step)


class BlockingPrimaryModels:
    def __init__(self, first_payload: dict, second_step):
        self.steps = [first_payload, second_step]
        self.calls: list[dict] = []
        self.lock = threading.Lock()
        self.first_started = threading.Event()
        self.release_first = threading.Event()

    def generate_content(self, **kwargs):
        with self.lock:
            call_index = len(self.calls)
            self.calls.append(kwargs)
        if call_index == 0:
            self.first_started.set()
            self.release_first.wait(timeout=1)
        step = self.steps[call_index]
        if isinstance(step, BaseException):
            raise step
        return SimpleNamespace(text=json.dumps(step))


def google_api_error(code: int, status: str) -> genai_errors.APIError:
    error_type = genai_errors.ServerError if code >= 500 else genai_errors.ClientError
    return error_type(
        code,
        {
            "error": {
                "code": code,
                "message": "synthetic provider failure",
                "status": status,
            }
        },
    )


class FakePart:
    @staticmethod
    def from_bytes(*, data: bytes, mime_type: str):
        return {"data": data, "mime_type": mime_type}


def fake_genai_types():
    return SimpleNamespace(
        Part=FakePart,
        GenerateContentConfig=lambda **kwargs: kwargs,
        HttpOptions=lambda **kwargs: kwargs,
        HttpRetryOptions=lambda **kwargs: kwargs,
        ThinkingConfig=lambda **kwargs: kwargs,
        ThinkingLevel=SimpleNamespace(HIGH="HIGH"),
    )


class RestoredAnalyzerBehaviorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.saved = {
            "LIVE_MODE": server.LIVE_MODE,
            "MODE": server.MODE,
            "gemini_client": server.gemini_client,
            "genai_types": server.genai_types,
            "GOOGLE_TOTAL_BUDGET_MS": server.GOOGLE_TOTAL_BUDGET_MS,
            "GOOGLE_HEDGE_DELAY_MS": server.GOOGLE_HEDGE_DELAY_MS,
        }
        server.rate_tracker.clear()
        server._clear_analysis_repeat_cache()

    def tearDown(self) -> None:
        for name, value in self.saved.items():
            setattr(server, name, value)
        server.rate_tracker.clear()
        server._clear_analysis_repeat_cache()

    def test_demo_contract_for_every_original_body_area(self) -> None:
        for index, (area, expected_keys) in enumerate(AREA_CONCERNS.items()):
            with self.subTest(area=area):
                random.seed(1000 + index)
                result = server.generate_demo_analysis(area)
                self.assertEqual(set(result["concerns"]), expected_keys)
                self.assertNotIn("skinAge", result)
                self.assertEqual(len(result["positiveHighlights"]), 2)
                for highlight in result["positiveHighlights"]:
                    self.assertTrue(highlight["title"].strip())
                    self.assertTrue(highlight["detail"].strip())
                self.assertTrue(
                    result["summary"].startswith(
                        result["positiveHighlights"][0]["detail"]
                    )
                )
                self.assertIsInstance(result["recommendations"], list)
                self.assertIsInstance(result["productRecommendations"], list)

    def test_demo_copy_and_catalog_are_bounded_across_2500_seeded_samples(self) -> None:
        audit_risk = re.compile(
            r"\b(?:due to|likely from|from chronic|from cumulative|"
            r"chronic sun exposure|cumulative sun exposure|volume loss|"
            r"skin elasticity|natural elasticity|skin thickness|"
            r"hydration level|hydration status|well-hydrated|dehydrated|"
            r"gold standard|permanent(?:ly)? reduction|"
            r"safe for all skin tones|perfect|makes? skin act younger)\b",
            re.IGNORECASE,
        )
        max_service_count = {area: 0 for area in AREA_CONCERNS}

        for area_index, (area, expected_keys) in enumerate(AREA_CONCERNS.items()):
            random.seed(9100 + area_index)
            for sample_index in range(500):
                with self.subTest(area=area, sample=sample_index):
                    result = server.generate_demo_analysis(area)
                    self.assertEqual(set(result["concerns"]), expected_keys)
                    self.assertIsNone(result["suggestedCombo"])

                    for text in server._guest_facing_strings(result):
                        self.assertIsNone(
                            server._PROHIBITED_MEDICAL_TERM_PATTERN.search(text)
                        )
                        self.assertIsNone(
                            server._UNSUPPORTED_APPEARANCE_TERM_PATTERN.search(text)
                        )
                        self.assertIsNone(
                            server._DIAGNOSTIC_CLAIM_PATTERN.search(text)
                        )
                        self.assertIsNone(audit_risk.search(text))
                        self.assertEqual(
                            server._sanitize_response({"copy": text})["copy"],
                            text,
                        )

                    recommendations = result["recommendations"]
                    max_service_count[area] = max(
                        max_service_count[area],
                        len(recommendations),
                    )
                    self.assertEqual(
                        [item["priority"] for item in recommendations],
                        list(range(1, len(recommendations) + 1)),
                    )
                    treatment_names = [
                        item["treatment"] for item in recommendations
                    ]
                    self.assertEqual(len(treatment_names), len(set(treatment_names)))
                    for item in recommendations:
                        self.assertIn(item["treatment"], server.AREA_TREATMENTS[area])
                        self.assertTrue(item["targets"])
                        self.assertTrue(set(item["targets"]).issubset(expected_keys))
                        self.assertTrue(
                            set(item["targets"]).issubset(
                                server.TREATMENT_TARGETS[item["treatment"]]
                            )
                        )
                    if area == "hands":
                        self.assertNotIn("Sculptra", treatment_names)

                    product_names = [
                        item["product"]
                        for item in result["productRecommendations"]
                    ]
                    self.assertEqual(len(product_names), len(set(product_names)))
                    self.assertTrue(
                        set(product_names).issubset(
                            server._product_names_for_area(area)
                        )
                    )

        self.assertGreater(
            max_service_count["face"],
            3,
            "The demo recommendation list must not regress to a three-service cap.",
        )

    def test_frontend_demo_duplicate_uses_safe_copy_and_area_catalog_names(self) -> None:
        html = (ROOT / "public" / "index.html").read_text(encoding="utf-8")
        demo = html[
            html.index("function generateDemoResults()"):
            html.index("function toggleAllFindings()")
        ]
        lowered_demo = demo.lower()
        for phrase in (
            "rosacea",
            "keratosis pilaris",
            "hyperpigmentation",
            "photoaging",
            "dehydration",
            "dehydrated",
            "due to",
            "likely from",
            "chronic sun exposure",
            "cumulative sun exposure",
            "volume loss",
            "elasticity",
            "thickness",
            "well-hydrated",
            "gold standard",
            "permanent reduction",
            "safe for all skin tones",
            "makes skin act younger",
        ):
            self.assertNotIn(phrase, lowered_demo)
        self.assertIsNone(re.search(r"\bperfect\b", demo, re.IGNORECASE))

        self.assertNotIn("recs.length >= 3", demo)
        self.assertNotIn("treatment:'HydraFacial'", demo)
        self.assertNotIn("treatment:'VI Peel'", demo)
        self.assertIn("treatment:'HydraFacial Clarifying'", demo)
        self.assertIn(
            "opt.treatment === 'Laser Hair Removal' && data.score < 41",
            demo,
        )

        treatment_maps = demo[
            demo.index("const treatmentMaps"):
            demo.index("const txMap")
        ]
        area_order = list(AREA_CONCERNS)
        for area_index, area in enumerate(area_order):
            start = treatment_maps.index(f"{area}: {{")
            if area_index + 1 < len(area_order):
                end = treatment_maps.index(
                    f"{area_order[area_index + 1]}: {{",
                    start,
                )
            else:
                end = len(treatment_maps)
            area_block = treatment_maps[start:end]
            names = set(re.findall(r"treatment:'([^']+)'", area_block))
            self.assertTrue(
                names.issubset(server.AREA_TREATMENTS[area]),
                (area, names - server.AREA_TREATMENTS[area]),
            )

        string_literals = [
            first or second
            for first, second in re.findall(
                r"'([^'\\]*(?:\\.[^'\\]*)*)'|`([^`\\]*(?:\\.[^`\\]*)*)`",
                demo,
                re.DOTALL,
            )
        ]
        internal_labels = set().union(*AREA_CONCERNS.values())
        for literal in string_literals:
            if literal.strip() in internal_labels:
                continue
            self.assertIsNone(
                server._PROHIBITED_MEDICAL_TERM_PATTERN.search(literal)
            )
            self.assertIsNone(
                server._UNSUPPORTED_APPEARANCE_TERM_PATTERN.search(literal)
            )

    def test_structured_schema_requires_two_to_three_positives_and_no_age(self) -> None:
        rejected = server.ANALYSIS_RESPONSE_SCHEMA["anyOf"][0]
        accepted = server.ANALYSIS_RESPONSE_SCHEMA["anyOf"][1]
        self.assertIn("observedArea", rejected["properties"])
        self.assertNotIn("observedArea", rejected["required"])
        positives = accepted["properties"]["positiveHighlights"]
        self.assertEqual(positives["minItems"], 2)
        self.assertEqual(positives["maxItems"], 3)
        self.assertIn("positiveHighlights", accepted["required"])
        self.assertIn("observedArea", accepted["required"])
        self.assertEqual(
            set(accepted["properties"]["observedArea"]["enum"]),
            {"face", "neck_chest", "hands", "back", "legs", "other"},
        )
        self.assertNotIn("skinAge", accepted["properties"])
        self.assertNotIn("skinAge", accepted["required"])

    def test_live_schema_is_constrained_to_each_selected_body_area(self) -> None:
        for area, expected_keys in AREA_CONCERNS.items():
            with self.subTest(area=area):
                accepted = server._analysis_schema_for_area(area)["anyOf"][1]
                concerns = accepted["properties"]["concerns"]
                self.assertEqual(set(concerns["properties"]), expected_keys)
                self.assertEqual(set(concerns["required"]), expected_keys)
                self.assertIs(concerns["additionalProperties"], False)

                recommendations = accepted["properties"]["recommendations"]
                self.assertEqual(recommendations["minItems"], 0)
                self.assertNotIn("maxItems", recommendations)
                priority_schema = recommendations["items"]["properties"]["priority"]
                self.assertEqual(priority_schema["minimum"], 1)
                self.assertNotIn("maximum", priority_schema)
                self.assertEqual(
                    set(recommendations["items"]["properties"]["treatment"]["enum"]),
                    server.AREA_TREATMENTS[area],
                )
                self.assertEqual(
                    set(recommendations["items"]["properties"]["targets"]["items"]["enum"]),
                    expected_keys,
                )

                products = accepted["properties"]["productRecommendations"]
                self.assertEqual(products["minItems"], 1)
                self.assertNotIn("maxItems", products)
                self.assertNotIn(
                    "RevitaLash Conditioner",
                    products["items"]["properties"]["product"]["enum"],
                )
                if area != "face":
                    self.assertNotIn(
                        "ALASTIN Restorative Eye Treatment",
                        products["items"]["properties"]["product"]["enum"],
                    )
                    self.assertNotIn(
                        "ZO Growth Factor Eye",
                        products["items"]["properties"]["product"]["enum"],
                    )

        fallback = server._analysis_schema_for_area("unknown")["anyOf"][1]
        self.assertEqual(
            set(fallback["properties"]["concerns"]["properties"]),
            AREA_CONCERNS["face"],
        )
        self.assertNotIn("SaltFacial", server.AREA_TREATMENTS["back"])
        self.assertNotIn("SaltFacial", server.AREA_TREATMENTS["legs"])

    def test_catalog_normalizer_filters_unsupported_targets_and_combo(self) -> None:
        analysis = accepted_analysis("legs")
        analysis["concerns"]["hairRemoval"].update({
            "score": 55,
            "severity": "moderate",
            "description": (
                "Distinct dark hair follicles and visible follicular contrast "
                "are clearly present across the selected skin."
            ),
        })
        analysis["recommendations"] = [
            {
                "treatment": "Laser Hair Removal",
                "reason": "For visible hair and a smoother routine.",
                "targets": ["hairRemoval", "texture"],
                "priority": 3,
            },
            {
                "treatment": "Microneedling",
                "reason": "For visible surface texture.",
                "targets": ["texture"],
                "priority": 2,
            },
            {
                "treatment": "Sciton BBL",
                "reason": "For visible vascularity and sun exposure.",
                "targets": ["veins", "sunDamage"],
                "priority": 1,
            },
        ]
        analysis["productRecommendations"] = [
            {
                "product": "ZO Complexion Renewal Pads",
                "reason": "For visible texture.",
            },
            {
                "product": "ISDIN Eryfotona Actinica",
                "reason": "For daily sun protection.",
            },
        ]
        analysis["suggestedCombo"] = "Hero Combo: BBL plus Halo"

        normalized = server._normalize_catalog_recommendations(analysis, "legs")

        laser = next(
            item
            for item in normalized["recommendations"]
            if item["treatment"] == "Laser Hair Removal"
        )
        self.assertEqual(laser["targets"], ["hairRemoval"])
        self.assertEqual(
            [item["priority"] for item in normalized["recommendations"]],
            [1, 2, 3],
        )
        self.assertIsNone(normalized["suggestedCombo"])

    def test_catalog_normalizer_has_no_five_service_cap(self) -> None:
        analysis = accepted_analysis("face")
        for concern_key, concern in analysis["concerns"].items():
            concern["score"] = 55
            concern["severity"] = "moderate"
            concern["description"] = concern_evidence_description(
                concern_key,
                score=55,
                area="face",
            )
        analysis["recommendations"] = [
            {"treatment": "Sciton Halo", "reason": "A mapped option.", "targets": ["texture"], "priority": 1},
            {"treatment": "Sciton BBL", "reason": "A mapped option.", "targets": ["redness"], "priority": 2},
            {"treatment": "Botox", "reason": "A mapped option.", "targets": ["wrinkles"], "priority": 3},
            {"treatment": "Sculptra", "reason": "A mapped option.", "targets": ["laxity"], "priority": 4},
            {"treatment": "HydraFacial Customized", "reason": "A mapped option.", "targets": ["pores"], "priority": 5},
            {"treatment": "Chemical Peels", "reason": "A mapped option.", "targets": ["unevenTone"], "priority": 6},
        ]

        normalized = server._normalize_catalog_recommendations(analysis, "face")

        self.assertEqual(len(normalized["recommendations"]), 6)
        self.assertEqual(
            [item["priority"] for item in normalized["recommendations"]],
            [1, 2, 3, 4, 5, 6],
        )

    def test_treatment_reason_names_every_displayed_target(self) -> None:
        analysis = accepted_analysis("face")
        for concern in analysis["concerns"].values():
            concern.update({
                "score": 10,
                "severity": "minimal",
                "description": "Only subtle variation is visible in the photograph.",
            })
        for key, score in (
            ("darkSpots", 65),
            ("redness", 60),
            ("sunDamage", 55),
            ("unevenTone", 50),
        ):
            analysis["concerns"][key].update({
                "score": score,
                "severity": "moderate",
                "description": concern_evidence_description(
                    key,
                    score=score,
                    area="face",
                ),
            })
        analysis["recommendations"] = [{
            "treatment": "Sciton BBL",
            "reason": "Model-authored copy is replaced.",
            "targets": ["darkSpots", "redness", "sunDamage", "unevenTone"],
            "priority": 1,
        }]

        normalized = server._normalize_catalog_recommendations(analysis, "face")
        bbl = next(
            item
            for item in normalized["recommendations"]
            if item["treatment"] == "Sciton BBL"
        )

        self.assertEqual(
            bbl["targets"],
            ["darkSpots", "redness", "sunDamage", "unevenTone"],
        )
        for label in (
            "visible pigmentation",
            "visible redness",
            "visible sun-exposure signs",
            "visible tone variation",
        ):
            self.assertIn(label, bbl["reason"].lower())

    def test_chemical_peels_is_one_canonical_service_card(self) -> None:
        analysis = accepted_analysis("face")
        analysis["recommendations"] = [
            {
                "treatment": "Chemical Peels (VI Peel)",
                "reason": "Legacy duplicate label.",
                "targets": ["darkSpots"],
                "priority": 1,
            },
            {
                "treatment": "Chemical Peels",
                "reason": "Canonical peel service.",
                "targets": ["darkSpots"],
                "priority": 2,
            },
        ]

        normalized = server._normalize_catalog_recommendations(analysis, "face")
        peel_cards = [
            item
            for item in normalized["recommendations"]
            if "Chemical Peel" in item["treatment"]
        ]

        self.assertEqual(len(peel_cards), 1)
        self.assertEqual(peel_cards[0]["treatment"], "Chemical Peels")
        self.assertNotIn(
            "Chemical Peels (VI Peel)",
            server.AREA_TREATMENTS["face"],
        )

    def test_treatment_reason_cannot_claim_an_unsupported_target(self) -> None:
        analysis = accepted_analysis("face")
        analysis["concerns"]["redness"].update({
            "score": 35,
            "severity": "mild",
            "description": "Mild visible redness appears around the nose.",
        })
        analysis["concerns"]["unevenTone"].update({
            "score": 35,
            "severity": "mild",
            "description": "Mild visible tone variation appears centrally.",
        })
        analysis["recommendations"] = [{
            "treatment": "Sciton Moxi",
            "reason": "This gentle laser addresses visible redness and uneven tone.",
            "targets": ["unevenTone"],
            "priority": 1,
        }]

        normalized = server._normalize_catalog_recommendations(analysis, "face")

        self.assertEqual(len(normalized["recommendations"]), 1)
        reason = normalized["recommendations"][0]["reason"]
        self.assertNotIn("redness", reason.lower())
        self.assertIn("visible tone variation", reason.lower())
        self.assertIn("approach to discuss", reason.lower())

    def test_treatment_reason_is_limited_to_this_guests_actual_targets(self) -> None:
        analysis = accepted_analysis("face")
        for concern in analysis["concerns"].values():
            concern.update({
                "score": 10,
                "severity": "minimal",
                "description": "No prominent variation is visible in the photograph.",
            })
        analysis["concerns"]["unevenTone"].update({
            "score": 35,
            "severity": "mild",
            "description": "Mild visible tone variation appears centrally.",
        })
        analysis["recommendations"] = [{
            "treatment": "Sciton BBL",
            "reason": "For visible vascularity.",
            "targets": ["unevenTone"],
            "priority": 1,
        }]

        normalized = server._normalize_catalog_recommendations(analysis, "face")

        bbl = next(
            item
            for item in normalized["recommendations"]
            if item["treatment"] == "Sciton BBL"
        )
        self.assertEqual(bbl["targets"], ["unevenTone"])
        self.assertNotIn("vascularity", bbl["reason"].lower())
        self.assertIn("visible tone variation", bbl["reason"].lower())
        self.assertIn("option to discuss", bbl["reason"].lower())

    def test_model_authored_recommendation_reasons_are_replaced_with_grounded_copy(self) -> None:
        analysis = accepted_analysis("face")
        for item in analysis["recommendations"]:
            item["reason"] = "A guaranteed miracle with incredible results."
        for item in analysis["productRecommendations"]:
            item["reason"] = "A guaranteed miracle with incredible results."

        normalized = server._normalize_catalog_recommendations(analysis, "face")

        for item in normalized["recommendations"]:
            self.assertNotIn("guaranteed miracle", item["reason"].lower())
            self.assertRegex(item["reason"].lower(), r"option|approach")
        for item in normalized["productRecommendations"]:
            self.assertNotIn("guaranteed miracle", item["reason"].lower())
            self.assertRegex(
                item["reason"].lower(),
                r"provider-guided home routine|broad-spectrum sunscreen",
            )

    def test_completed_summary_is_deterministic_positive_first_and_photo_bounded(self) -> None:
        analysis = accepted_analysis("face")
        analysis["summary"] = "Variable model marketing copy should not survive."
        normalized = server._normalize_catalog_recommendations(analysis, "face")

        finalized = server._finalize_completed_analysis(normalized, "face")

        self.assertTrue(
            finalized["summary"].startswith(
                finalized["positiveHighlights"][0]["detail"]
            )
        )
        self.assertIn("photo-based preview", finalized["summary"].lower())
        self.assertIn("what a standard photo cannot", finalized["summary"].lower())
        self.assertIn("VISIA consultation", finalized["summary"])
        self.assertNotIn("Variable model marketing", finalized["summary"])

    def test_laser_hair_removal_requires_moderate_treatment_relevant_evidence(self) -> None:
        unsupported_descriptions = (
            "Some very fine natural hair is visible on the upper back.",
            "Moderate fine natural hair is visible across the area.",
            "Noticeable peach-fuzz hair appears on the selected skin.",
            "Dense light vellus hairs are present.",
            "No visible hair follicles or stubble are present.",
            "No dark or coarse body hair is present, only fine natural hair.",
            "Stubble is not visible.",
            "Visible follicles are absent.",
            "Distinct follicular contrast cannot be confirmed.",
            "Long dark scalp hair is draped across the shoulders and back.",
            "Long dark hair lies across the shoulders and back.",
        )
        for description in unsupported_descriptions:
            with self.subTest(description=description):
                unsupported = accepted_analysis("back")
                unsupported["concerns"]["hairRemoval"].update({
                    "score": 55,
                    "severity": "moderate",
                    "description": description,
                })
                unsupported["recommendations"].append({
                    "treatment": "Laser Hair Removal",
                    "reason": "For the visible hair.",
                    "targets": ["hairRemoval"],
                    "priority": 4,
                })

                normalized = server._normalize_catalog_recommendations(
                    unsupported,
                    "back",
                )

                self.assertFalse(server._supports_laser_hair_removal(unsupported))
                self.assertNotIn(
                    "Laser Hair Removal",
                    [item["treatment"] for item in normalized["recommendations"]],
                )

        follicular_evidence = accepted_analysis("legs")
        follicular_evidence["concerns"]["hairRemoval"].update({
            "score": 55,
            "severity": "moderate",
            "description": (
                "Visible dark follicles and distinct follicular contrast are "
                "present across the selected skin."
            ),
        })
        self.assertTrue(server._supports_laser_hair_removal(follicular_evidence))

        stubble_evidence = accepted_analysis("legs")
        stubble_evidence["concerns"]["hairRemoval"].update({
            "score": 55,
            "severity": "moderate",
            "description": "Short stubble is visible across the selected skin.",
        })
        self.assertTrue(server._supports_laser_hair_removal(stubble_evidence))

    def test_hair_evidence_parser_rejects_post_phrase_negation(self) -> None:
        for description in (
            "Stubble is not visible.",
            "Stubble: not visible.",
            "Stubble? Not visible.",
            "Stubble, if any, is not visible.",
            "I cannot confirm visible hair follicles.",
            "Visible hair follicles cannot reliably be confirmed.",
            "Stubble is questionable and not clearly confirmed.",
            "There may be stubble, but the image does not establish it.",
            "Possible visible follicles are not enough to confirm body hair.",
            "Stubble is invisible.",
            "Stubble is imperceptible.",
            "Stubble is undetectable.",
            "Stubble is indiscernible.",
            "Stubble is doubtful.",
            "Stubble is equivocal.",
            "Stubble is ambiguous.",
            "Stubble is unlikely.",
            "Potential stubble is present.",
            "Perhaps stubble is present.",
            "Maybe stubble is present.",
            "Stubble could be present.",
            "Stubble is neither visible nor distinct.",
            "Stubble is anything but visible.",
            "The image rules out stubble.",
            "The image excludes stubble.",
            "Zero stubble is present.",
            "Stubble: none.",
            "Follicular contrast is negligible.",
            "Follicular contrast is weak.",
            "Follicular contrast is poor.",
            "Visible follicles could be present.",
            "Dark body hair is unlikely.",
            "Dark body hair could be present.",
            "The dark line could be hair.",
            "Visible follicles are absent.",
            "Distinct follicular contrast cannot be confirmed.",
            "Dark body hair could not be clearly seen.",
            "Follicular visibility is difficult to confirm.",
        ):
            with self.subTest(description=description):
                self.assertFalse(
                    server._has_nonnegated_hair_evidence(description)
                )

        for description in (
            "Short stubble is visible across the photographed leg.",
            "Distinct follicular contrast is clearly visible.",
            "Clearly visible dark body hair is present.",
            (
                "No stubble is visible, but coarse body hair is clearly visible "
                "across the photographed area."
            ),
            "Stubble is absent but coarse body hair is clearly visible.",
            "Stubble is absent although coarse body hair is clearly visible.",
            "Stubble is absent yet coarse body hair is clearly visible.",
            "Stubble is absent however coarse body hair is clearly visible.",
            "Visible follicles are present, though contrast may be subtle.",
            (
                "Scalp hair drapes over the shoulders, but visible stubble is "
                "present on the lower back."
            ),
        ):
            with self.subTest(description=description):
                self.assertTrue(
                    server._has_nonnegated_hair_evidence(description)
                )

    def test_uncertain_hair_copy_cannot_trigger_laser_hair_removal(self) -> None:
        uncertain_descriptions = (
            "Stubble: not visible",
            "Stubble? Not visible",
            "Stubble, if any, is not visible",
            "I cannot confirm visible hair follicles",
            "Visible hair follicles cannot reliably be confirmed",
            "Stubble is questionable and not clearly confirmed",
            "There may be stubble, but the image does not establish it",
            "Possible visible follicles are not enough to confirm body hair",
            "Stubble is invisible",
            "Stubble is imperceptible",
            "Stubble is undetectable",
            "Stubble is indiscernible",
            "Stubble is doubtful",
            "Stubble is equivocal",
            "Stubble is ambiguous",
            "Stubble is unlikely",
            "Potential stubble is present",
            "Perhaps stubble is present",
            "Maybe stubble is present",
            "Stubble could be present",
            "Stubble is neither visible nor distinct",
            "Stubble is anything but visible",
            "The image rules out stubble",
            "The image excludes stubble",
            "Zero stubble is present",
            "Stubble: none",
            "Follicular contrast is negligible",
            "Follicular contrast is weak",
            "Follicular contrast is poor",
            "Visible follicles could be present",
            "Dark body hair is unlikely",
            "Dark body hair could be present",
            "The dark line could be hair",
            "Visible stubble is speculative.",
            "Stubble is indeterminate and visible.",
            "Visible stubble remains unverified.",
            "Nothing resembling visible stubble is present.",
            "Visible stubble was not observed.",
            "Visible stubble is not evident.",
            "Visible stubble is not demonstrated.",
            "Visible stubble is unsupported.",
            "Visible stubble is hypothetical.",
            "I couldn't confirm visible stubble.",
            "The image fails to show visible stubble.",
            "Stubble is uncertain.",
            "Fine vellus hair is visible.",
            "Scalp hair is visible over the shoulders.",
            "Stubble is allegedly visible.",
            "Stubble is reportedly visible.",
            "Stubble is hard to discern while still labeled visible.",
            "Stubble seems borderline and visible.",
            "Stubble is inconclusive and visible.",
            "Stubble is debatable and visible.",
            "Stubble is only theoretically visible.",
        )
        for area in ("back", "legs"):
            for description in uncertain_descriptions:
                with self.subTest(area=area, description=description):
                    analysis = accepted_analysis(area)
                    analysis["concerns"]["hairRemoval"].update({
                        "score": 60,
                        "severity": "moderate",
                        "description": description,
                    })
                    analysis["recommendations"].append({
                        "treatment": "Laser Hair Removal",
                        "reason": "Raw model reason.",
                        "targets": ["hairRemoval"],
                        "priority": 99,
                    })

                    normalized = server._normalize_catalog_recommendations(
                        analysis,
                        area,
                    )
                    finalized = server._finalize_completed_analysis(
                        normalized,
                        area,
                    )

                    self.assertNotIn(
                        "Laser Hair Removal",
                        [
                            item["treatment"]
                            for item in finalized["recommendations"]
                        ],
                    )
                    self.assertEqual(
                        finalized["concerns"]["hairRemoval"]["description"],
                        "No prominent body-hair growth is visible in the photographed area.",
                    )
                    self.assertEqual(
                        finalized["concerns"]["hairRemoval"]["score"],
                        10,
                    )

    def test_grooming_inference_is_removed_without_losing_visible_hair_evidence(self) -> None:
        analysis = accepted_analysis("legs")
        for concern in analysis["concerns"].values():
            concern.update({
                "score": 10,
                "severity": "none",
                "description": "No prominent variation is visible.",
            })
        analysis["concerns"]["hairRemoval"].update({
            "score": 55,
            "severity": "moderate",
            "description": (
                "Distinct visible follicles and stubble contrast are noticeable "
                "on the upper thighs, indicating regular surface hair removal."
            ),
        })
        analysis["recommendations"] = [{
            "treatment": "Laser Hair Removal",
            "reason": "Model-authored copy.",
            "targets": ["hairRemoval"],
            "priority": 1,
        }]
        analysis["productRecommendations"] = [{
            "product": "Colorescience Face Shield SPF 50",
            "reason": "Daily broad-spectrum sunscreen.",
        }]
        analysis["suggestedCombo"] = None

        repaired = server._repair_photo_observation_inferences(
            analysis,
            "legs",
        )

        description = repaired["concerns"]["hairRemoval"]["description"]
        self.assertNotIn("hair removal", description.lower())
        self.assertRegex(
            description.lower(),
            r"visible (?:stubble|hair follicles)",
        )
        self.assertTrue(server._supports_laser_hair_removal(repaired))

        normalized = server._normalize_catalog_recommendations(
            repaired,
            "legs",
        )
        self.assertIn(
            "Laser Hair Removal",
            [item["treatment"] for item in normalized["recommendations"]],
        )

    def test_catalog_normalizer_retains_only_a_fully_represented_combo(self) -> None:
        analysis = accepted_analysis()
        analysis["recommendations"] = [
            {
                "treatment": "Sciton BBL",
                "reason": "For visible pigment.",
                "targets": ["darkSpots"],
                "priority": 1,
            },
            {
                "treatment": "Sciton Halo",
                "reason": "For visible texture.",
                "targets": ["sunDamage"],
                "priority": 2,
            },
            {
                "treatment": "HydraFacial Customized",
                "reason": "For visible pore texture.",
                "targets": ["pores"],
                "priority": 3,
            },
        ]
        analysis["suggestedCombo"] = "Hero Combo: BBL plus Halo"

        normalized = server._normalize_catalog_recommendations(analysis, "face")

        self.assertEqual(normalized["suggestedCombo"], "Hero Combo")

    def test_catalog_normalizer_expands_selected_services_to_supported_visible_targets(self) -> None:
        analysis = accepted_analysis()
        score_map = {
            "darkSpots": 50,
            "texture": 35,
            "pores": 20,
            "wrinkles": 10,
            "redness": 5,
            "laxity": 5,
            "sunDamage": 12,
            "unevenTone": 11,
        }
        for key, score in score_map.items():
            analysis["concerns"][key]["score"] = score
        analysis["recommendations"] = [
            {
                "treatment": "Sciton BBL",
                "reason": "For the visible pigment.",
                "targets": ["darkSpots"],
                "priority": 1,
            },
            {
                "treatment": "Sciton Moxi",
                "reason": "Another pigment-focused option.",
                "targets": ["darkSpots"],
                "priority": 2,
            },
            {
                "treatment": "Chemical Peels",
                "reason": "A third pigment-focused option.",
                "targets": ["darkSpots"],
                "priority": 3,
            },
        ]

        normalized = server._normalize_catalog_recommendations(analysis, "face")

        self.assertTrue(
            all(
                "darkSpots" in item["targets"]
                for item in normalized["recommendations"]
            )
        )
        self.assertTrue(
            any(
                "texture" in item["targets"]
                for item in normalized["recommendations"]
            )
        )
        for item in normalized["recommendations"]:
            self.assertNotIn("sunDamage", item["targets"])
            self.assertNotIn("unevenTone", item["targets"])

    def test_catalog_normalizer_adds_bounded_fallback_for_missing_moderate_service(self) -> None:
        analysis = accepted_analysis("face")
        for concern in analysis["concerns"].values():
            concern["score"] = 10
        analysis["concerns"]["wrinkles"]["score"] = 55
        analysis["concerns"]["wrinkles"]["description"] = (
            concern_evidence_description(
                "wrinkles",
                score=55,
                area="face",
            )
        )
        analysis["recommendations"] = [
            {
                "treatment": "Sciton BBL",
                "reason": "For visible tone.",
                "targets": ["unevenTone"],
                "priority": 1,
            },
        ]

        normalized = server._normalize_catalog_recommendations(analysis, "face")

        self.assertEqual(
            [item["treatment"] for item in normalized["recommendations"]],
            ["Sciton Moxi"],
        )
        self.assertEqual(
            normalized["recommendations"][0]["targets"],
            ["wrinkles"],
        )
        self.assertIn(
            "approach to discuss",
            normalized["recommendations"][0]["reason"].lower(),
        )

    def test_every_moderate_area_concern_has_an_exact_catalog_fallback(self) -> None:
        for area, mappings in server.FALLBACK_TREATMENTS_BY_AREA_CONCERN.items():
            for target, candidates in mappings.items():
                with self.subTest(area=area, target=target):
                    analysis = accepted_analysis(area)
                    for concern in analysis["concerns"].values():
                        concern["score"] = 10
                        concern["severity"] = "none"
                        concern["description"] = "No prominent concern is visible."
                    analysis["concerns"][target]["score"] = 55
                    analysis["concerns"][target]["severity"] = "moderate"
                    analysis["concerns"][target]["description"] = (
                        concern_evidence_description(
                            target,
                            score=55,
                            area=area,
                        )
                    )
                    analysis["recommendations"] = []
                    analysis["productRecommendations"] = [{
                        "product": "Colorescience Face Shield SPF 50",
                        "reason": "A daily broad-spectrum baseline.",
                    }]
                    analysis["suggestedCombo"] = None

                    normalized = server._normalize_catalog_recommendations(
                        analysis,
                        area,
                    )

                    fallback = normalized["recommendations"][0]
                    self.assertEqual(fallback["treatment"], candidates[0])
                    self.assertIn(target, fallback["targets"])
                    self.assertIn(
                        fallback["treatment"],
                        server.AREA_TREATMENTS[area],
                    )
                    self.assertIn(
                        target,
                        server.TREATMENT_TARGETS[fallback["treatment"]],
                    )

    def test_uncovered_moderate_dryness_gets_exact_skincare_fallback(self) -> None:
        for area in ("hands", "legs"):
            with self.subTest(area=area):
                analysis = accepted_analysis(area)
                for concern in analysis["concerns"].values():
                    concern["score"] = 10
                    concern["severity"] = "none"
                    concern["description"] = "No prominent concern is visible."
                analysis["concerns"]["dryness"] = {
                    "score": 55,
                    "severity": "moderate",
                    "description": "Clearly visible surface dryness is present.",
                }
                analysis["recommendations"] = []
                analysis["productRecommendations"] = [{
                    "product": "Colorescience Face Shield SPF 50",
                    "reason": "A daily broad-spectrum baseline.",
                }]
                analysis["suggestedCombo"] = None

                normalized = server._normalize_catalog_recommendations(
                    analysis,
                    area,
                )

                products = {
                    item["product"]: item
                    for item in normalized["productRecommendations"]
                }
                self.assertIn("SkinBetter Trio Moisture", products)
                self.assertIn(
                    "provider-guided home routine",
                    products["SkinBetter Trio Moisture"]["reason"],
                )

    def test_product_reason_keeps_visible_goal_when_name_contains_face_word(self) -> None:
        analysis = accepted_analysis("neck_chest")
        for concern in analysis["concerns"].values():
            concern.update({
                "score": 10,
                "severity": "none",
                "description": "No prominent variation is visible.",
            })
        analysis["concerns"]["texture"].update({
            "score": 55,
            "severity": "moderate",
            "description": "Clearly visible surface texture variation appears.",
        })
        analysis["recommendations"] = [{
            "treatment": "Sciton Moxi",
            "reason": "Model-authored copy.",
            "targets": ["texture"],
            "priority": 1,
        }]
        analysis["productRecommendations"] = [{
            "product": "ZO Complexion Renewal Pads",
            "reason": "Model-authored copy.",
        }]
        analysis["suggestedCombo"] = None

        normalized = server._normalize_catalog_recommendations(
            analysis,
            "neck_chest",
        )
        finalized = server._finalize_completed_analysis(
            normalized,
            "neck_chest",
        )
        product = next(
            item
            for item in finalized["productRecommendations"]
            if item["product"] == "ZO Complexion Renewal Pads"
        )

        self.assertIn("visible texture", product["reason"].lower())
        self.assertIn("provider-guided home routine", product["reason"])
        self.assertNotIn("photographed neck and chest", product["reason"])
        self.assertNotIn("ZO Complexion Renewal Pads", product["reason"])

    def test_product_only_moderate_does_not_create_a_mild_service(self) -> None:
        analysis = accepted_analysis("legs")
        for concern in analysis["concerns"].values():
            concern["score"] = 10
            concern["severity"] = "none"
            concern["description"] = "No prominent concern is visible."
        analysis["concerns"]["dryness"] = {
            "score": 55,
            "severity": "moderate",
            "description": "Clearly visible surface dryness is present.",
        }
        analysis["concerns"]["texture"] = {
            "score": 20,
            "severity": "mild",
            "description": "A little visible texture is present.",
        }
        analysis["recommendations"] = []
        analysis["productRecommendations"] = [{
            "product": "Colorescience Face Shield SPF 50",
            "reason": "A daily broad-spectrum baseline.",
        }]

        normalized = server._normalize_catalog_recommendations(analysis, "legs")

        self.assertEqual(normalized["recommendations"], [])
        self.assertIn(
            "SkinBetter Trio Moisture",
            {
                item["product"]
                for item in normalized["productRecommendations"]
            },
        )

    def test_all_mild_result_keeps_top_concern_coverage(self) -> None:
        analysis = accepted_analysis("legs")
        for concern in analysis["concerns"].values():
            concern["score"] = 10
            concern["severity"] = "none"
            concern["description"] = "No prominent concern is visible."
        analysis["concerns"]["texture"] = {
            "score": 30,
            "severity": "mild",
            "description": "Some visible texture is present.",
        }
        analysis["recommendations"] = []
        analysis["productRecommendations"] = [{
            "product": "Colorescience Face Shield SPF 50",
            "reason": "A daily broad-spectrum baseline.",
        }]

        normalized = server._normalize_catalog_recommendations(analysis, "legs")

        self.assertEqual(normalized["recommendations"], [])
        self.assertIn(
            "ZO Complexion Renewal Pads",
            {
                item["product"]
                for item in normalized["productRecommendations"]
            },
        )

    def test_mild_hand_veins_do_not_force_bbl_or_make_the_result_fail(self) -> None:
        analysis = accepted_analysis("hands")
        for concern in analysis["concerns"].values():
            concern["score"] = 10
            concern["severity"] = "none"
            concern["description"] = "No prominent concern is visible."
        analysis["concerns"]["veins"] = {
            "score": 40,
            "severity": "mild",
            "description": "Mild visible vascularity appears on the hands.",
        }
        analysis["concerns"]["dryness"] = {
            "score": 30,
            "severity": "mild",
            "description": "Some visible surface dryness is present.",
        }
        analysis["recommendations"] = [{
            "treatment": "Sciton BBL",
            "reason": "For visible vascularity.",
            "targets": ["veins"],
            "priority": 1,
        }]
        analysis["productRecommendations"] = [{
            "product": "Colorescience Face Shield SPF 50",
            "reason": "A daily broad-spectrum baseline.",
        }]

        normalized = server._normalize_catalog_recommendations(analysis, "hands")

        self.assertNotIn(
            "Sciton BBL",
            [item["treatment"] for item in normalized["recommendations"]],
        )
        self.assertIn(
            "SkinBetter Trio Moisture",
            {
                item["product"]
                for item in normalized["productRecommendations"]
            },
        )

        only_mild_veins = accepted_analysis("hands")
        for concern in only_mild_veins["concerns"].values():
            concern["score"] = 10
            concern["severity"] = "none"
            concern["description"] = "No prominent concern is visible."
        only_mild_veins["concerns"]["veins"] = {
            "score": 40,
            "severity": "mild",
            "description": "Mild visible vascularity appears on the hands.",
        }
        only_mild_veins["recommendations"] = []
        only_mild_veins["productRecommendations"] = []

        normalized = server._normalize_catalog_recommendations(
            only_mild_veins,
            "hands",
        )

        self.assertEqual(normalized["recommendations"], [])
        self.assertEqual(
            [
                item["product"]
                for item in normalized["productRecommendations"]
            ],
            ["Colorescience Face Shield SPF 50"],
        )

    def test_every_single_mild_area_concern_can_be_normalized_without_503(self) -> None:
        intentionally_unmapped = {("hands", "veins")}
        for area, concern_keys in server.AREA_CONCERN_KEYS.items():
            for target in concern_keys:
                with self.subTest(area=area, target=target):
                    analysis = accepted_analysis(area)
                    for concern in analysis["concerns"].values():
                        concern["score"] = 10
                        concern["severity"] = "none"
                        concern["description"] = "No prominent concern is visible."
                    analysis["concerns"][target] = {
                        "score": 40,
                        "severity": "mild",
                        "description": concern_evidence_description(
                            target,
                            score=40,
                            area=area,
                        ),
                    }
                    analysis["recommendations"] = []
                    analysis["productRecommendations"] = []
                    analysis["suggestedCombo"] = None

                    normalized = server._normalize_catalog_recommendations(
                        analysis,
                        area,
                    )

                    treatments = {
                        item["treatment"]
                        for item in normalized["recommendations"]
                    }
                    self.assertFalse(
                        treatments.intersection({
                            "Sciton Halo",
                            "Sculptra",
                            "Laser Hair Removal",
                        })
                    )
                    if area == "hands":
                        self.assertNotIn("Sciton BBL", treatments)
                    covered = {
                        covered_target
                        for item in normalized["recommendations"]
                        for covered_target in item["targets"]
                    }
                    for item in normalized["productRecommendations"]:
                        covered.update(
                            server.PRODUCT_TARGETS[item["product"]].intersection(
                                {target}
                            )
                        )
                    if target == "hairRemoval" or (area, target) in intentionally_unmapped:
                        self.assertNotIn(target, covered)
                    else:
                        self.assertIn(target, covered)

    def test_fallback_map_exactly_covers_treatment_eligible_area_concerns(self) -> None:
        for area, area_concerns in server.AREA_CONCERN_KEYS.items():
            eligible = {
                target
                for target in area_concerns
                if any(
                    target in server.TREATMENT_TARGETS[treatment]
                    for treatment in server.AREA_TREATMENTS[area]
                )
            }
            self.assertEqual(
                set(server.FALLBACK_TREATMENTS_BY_AREA_CONCERN[area]),
                eligible,
            )
            for target, candidates in (
                server.FALLBACK_TREATMENTS_BY_AREA_CONCERN[area].items()
            ):
                self.assertTrue(candidates)
                for treatment in candidates:
                    self.assertIn(treatment, server.AREA_TREATMENTS[area])
                    self.assertIn(target, server.TREATMENT_TARGETS[treatment])
                    self.assertIn(
                        treatment,
                        server._FALLBACK_TREATMENT_REASON_TEMPLATES,
                    )

    def test_redness_compatible_pore_fallback_avoids_needling(self) -> None:
        analysis = accepted_analysis("face")
        for concern in analysis["concerns"].values():
            concern["score"] = 10
            concern["severity"] = "none"
            concern["description"] = "No prominent concern is visible."
        analysis["concerns"]["redness"] = {
            "score": 55,
            "severity": "moderate",
            "description": "Clearly visible moderate redness is present.",
        }
        analysis["concerns"]["pores"] = {
            "score": 50,
            "severity": "moderate",
            "description": "Clearly visible moderate pore visibility is present.",
        }
        analysis["recommendations"] = []
        analysis["suggestedCombo"] = None

        normalized = server._normalize_catalog_recommendations(analysis, "face")

        treatments = {
            item["treatment"]: item
            for item in normalized["recommendations"]
        }
        self.assertIn("Sciton BBL", treatments)
        self.assertIn("HydraFacial Clarifying", treatments)
        self.assertIn("pores", treatments["HydraFacial Clarifying"]["targets"])
        self.assertNotIn("Microneedling", treatments)
        self.assertNotIn("RF Microneedling", treatments)

    def test_shared_fallback_service_copy_names_every_added_goal(self) -> None:
        analysis = accepted_analysis("face")
        for concern in analysis["concerns"].values():
            concern["score"] = 10
            concern["severity"] = "none"
            concern["description"] = "No prominent concern is visible."
        for target, score in (
            ("darkSpots", 55),
            ("sunDamage", 50),
            ("unevenTone", 45),
        ):
            analysis["concerns"][target] = {
                "score": score,
                "severity": "moderate",
                "description": concern_evidence_description(
                    target,
                    score=score,
                    area="face",
                ),
            }
        analysis["recommendations"] = []
        analysis["suggestedCombo"] = "Hero Combo"

        normalized = server._normalize_catalog_recommendations(analysis, "face")

        bbl = next(
            item
            for item in normalized["recommendations"]
            if item["treatment"] == "Sciton BBL"
        )
        self.assertEqual(
            set(bbl["targets"]),
            {"darkSpots", "sunDamage", "unevenTone"},
        )
        for phrase in (
            "visible pigmentation",
            "visible sun-exposure signs",
            "visible tone variation",
        ):
            self.assertIn(phrase, bbl["reason"])
        self.assertIsNone(normalized["suggestedCombo"])

    def test_randomized_moderate_coverage_invariants(self) -> None:
        rng = random.Random(20260713)
        areas = tuple(AREA_CONCERNS)
        for iteration in range(2000):
            area = rng.choice(areas)
            analysis = accepted_analysis(area)
            scores = {
                key: rng.randint(0, 100)
                for key in AREA_CONCERNS[area]
            }
            if max(scores.values()) < 41:
                scores[rng.choice(tuple(scores))] = rng.randint(41, 100)
            for key, score in scores.items():
                analysis["concerns"][key] = {
                    "score": score,
                    "severity": "moderate" if score >= 41 else "mild",
                    "description": (
                        "Clearly visible persistent evidence remains visible at "
                        "rest in a neutral resting view, independent of pose."
                        if area == "neck_chest"
                        and key in {"laxity", "wrinkles"}
                        and score >= 41
                        else (
                            "Distinct dark hair follicles and clearly visible "
                            "follicular contrast are present."
                        )
                        if key == "hairRemoval" and score >= 41
                        else f"Clearly visible evidence for {key}."
                    ),
                }

            allowed_treatments = tuple(server.AREA_TREATMENTS[area])
            analysis["recommendations"] = []
            for priority, treatment in enumerate(
                rng.sample(
                    allowed_treatments,
                    k=rng.randint(0, min(4, len(allowed_treatments))),
                ),
                start=1,
            ):
                candidate_targets = tuple(AREA_CONCERNS[area])
                analysis["recommendations"].append({
                    "treatment": treatment,
                    "reason": "A model-selected exact-catalog option.",
                    "targets": rng.sample(
                        candidate_targets,
                        k=rng.randint(1, min(3, len(candidate_targets))),
                    ),
                    "priority": priority,
                })

            allowed_products = tuple(server._product_names_for_area(area))
            analysis["productRecommendations"] = [
                {
                    "product": product,
                    "reason": "A model-selected exact-catalog product.",
                }
                for product in rng.sample(
                    allowed_products,
                    k=rng.randint(0, min(3, len(allowed_products))),
                )
            ]
            analysis["suggestedCombo"] = rng.choice((None, "Hero Combo"))

            try:
                normalized = server._normalize_catalog_recommendations(
                    analysis,
                    area,
                )
            except server._GoogleResponseError as error:
                self.fail(
                    f"iteration={iteration} area={area} scores={scores} "
                    f"recommendations={analysis['recommendations']} error={error}"
                )

            concern_scores = {
                key: value["score"]
                for key, value in normalized["concerns"].items()
            }
            moderate = {
                key for key, score in concern_scores.items() if score >= 41
            }
            service_covered = {
                target
                for item in normalized["recommendations"]
                for target in item["targets"]
            }
            product_covered = {
                target
                for item in normalized["productRecommendations"]
                for target in server.PRODUCT_TARGETS[item["product"]]
                if concern_scores.get(target, 0) > 10
            }
            treatment_eligible = {
                target
                for target in moderate
                if any(
                    target in server.TREATMENT_TARGETS[treatment]
                    for treatment in server.AREA_TREATMENTS[area]
                )
            }
            self.assertTrue(
                moderate.issubset(service_covered.union(product_covered)),
                (iteration, area, concern_scores, normalized),
            )
            self.assertTrue(
                treatment_eligible.issubset(service_covered),
                (iteration, area, concern_scores, normalized),
            )
            self.assertEqual(
                [item["priority"] for item in normalized["recommendations"]],
                list(range(1, len(normalized["recommendations"]) + 1)),
            )
            self.assertEqual(
                len({item["treatment"] for item in normalized["recommendations"]}),
                len(normalized["recommendations"]),
            )
            for item in normalized["recommendations"]:
                self.assertIn(item["treatment"], server.AREA_TREATMENTS[area])
                self.assertTrue(item["targets"])
                self.assertEqual(
                    item["reason"],
                    server._fallback_treatment_reason(
                        item["treatment"],
                        item["targets"],
                    ),
                    (iteration, area, item),
                )
                self.assertTrue(
                    set(item["targets"]).issubset(
                        server.TREATMENT_TARGETS[item["treatment"]]
                    )
                )
                self.assertTrue(
                    all(concern_scores[target] > 10 for target in item["targets"])
                )

    def test_sculptra_is_not_a_hand_or_vein_recommendation(self) -> None:
        self.assertNotIn("Sculptra", server.AREA_TREATMENTS["hands"])
        self.assertNotIn("veins", server.TREATMENT_TARGETS["Sculptra"])
        hands_schema = server._analysis_schema_for_area("hands")
        treatment_enum = (
            hands_schema["anyOf"][1]["properties"]["recommendations"]
            ["items"]["properties"]["treatment"]["enum"]
        )
        self.assertNotIn("Sculptra", treatment_enum)

    def test_sculptra_requires_a_moderate_supported_target(self) -> None:
        analysis = accepted_analysis("face")
        for concern in analysis["concerns"].values():
            concern["score"] = 10
        analysis["concerns"]["laxity"]["score"] = 30
        analysis["recommendations"] = [
            {
                "treatment": "Sculptra",
                "reason": "A biostimulatory option for visible contour concerns.",
                "targets": ["laxity"],
                "priority": 1,
            },
            {
                "treatment": "RF Microneedling",
                "reason": "A resurfacing option for visible contour concerns.",
                "targets": ["laxity"],
                "priority": 2,
            },
        ]

        normalized = server._normalize_catalog_recommendations(analysis, "face")

        self.assertNotIn(
            "Sculptra",
            [item["treatment"] for item in normalized["recommendations"]],
        )

    def test_hand_bbl_requires_a_moderate_supported_target(self) -> None:
        analysis = accepted_analysis("hands")
        for concern in analysis["concerns"].values():
            concern["score"] = 10
        analysis["concerns"]["dryness"]["score"] = 32
        analysis["concerns"]["veins"]["score"] = 25
        analysis["concerns"]["texture"]["score"] = 15
        analysis["recommendations"] = [
            {
                "treatment": "Sciton BBL",
                "reason": "For visible vascularity.",
                "targets": ["veins"],
                "priority": 1,
            },
            {
                "treatment": "Microneedling",
                "reason": "For visible surface texture.",
                "targets": ["texture"],
                "priority": 2,
            },
        ]
        analysis["productRecommendations"] = [
            {
                "product": "Hydrinity Renewing HA Serum",
                "reason": "For visible surface dryness.",
            },
            {
                "product": "ISDIN Eryfotona Actinica",
                "reason": "A daily broad-spectrum baseline.",
            },
        ]

        normalized = server._normalize_catalog_recommendations(analysis, "hands")

        self.assertNotIn(
            "Sciton BBL",
            [item["treatment"] for item in normalized["recommendations"]],
        )

    def test_catalog_normalizer_adds_fixed_spf_without_inventing_other_products(self) -> None:
        analysis = accepted_analysis("face")
        analysis["productRecommendations"] = []

        normalized = server._normalize_catalog_recommendations(analysis, "face")

        self.assertEqual(
            normalized["productRecommendations"],
            [{
                "product": "Colorescience Face Shield SPF 50",
                "reason": server._spf_product_reason(
                    "Colorescience Face Shield SPF 50"
                ),
            }],
        )

    def test_mild_result_does_not_require_a_marketing_entry_service(self) -> None:
        analysis = accepted_analysis("face")
        for concern in analysis["concerns"].values():
            concern["score"] = 10
        analysis["concerns"]["texture"]["score"] = 30
        analysis["recommendations"] = [
            {
                "treatment": "Sciton Moxi",
                "reason": "A gentle fractional option for visible texture.",
                "targets": ["texture"],
                "priority": 1,
            },
        ]
        analysis["productRecommendations"] = [
            {
                "product": "Colorescience Face Shield SPF 50",
                "reason": "A daily broad-spectrum baseline.",
            },
        ]

        normalized = server._normalize_catalog_recommendations(analysis, "face")

        self.assertEqual(
            [item["treatment"] for item in normalized["recommendations"]],
            ["Sciton Moxi"],
        )

    def test_catalog_normalizer_repairs_priority_order_from_visible_scores(self) -> None:
        analysis = accepted_analysis()
        analysis["recommendations"][0], analysis["recommendations"][2] = (
            analysis["recommendations"][2],
            analysis["recommendations"][0],
        )
        analysis["recommendations"][0]["priority"] = 1
        analysis["recommendations"][1]["priority"] = 2
        analysis["recommendations"][2]["priority"] = 3

        normalized = server._normalize_catalog_recommendations(analysis, "face")

        self.assertIn(
            "darkSpots",
            normalized["recommendations"][0]["targets"],
        )
        self.assertEqual(
            [item["priority"] for item in normalized["recommendations"]],
            [1, 2, 3],
        )

    def test_catalog_normalizer_derives_severity_and_rejects_invalid_scores(self) -> None:
        analysis = accepted_analysis()
        score_map = {
            "wrinkles": 0,
            "redness": 10,
            "darkSpots": 11,
            "texture": 40,
            "pores": 41,
            "laxity": 60,
            "sunDamage": 61,
            "unevenTone": 100,
        }
        for key, score in score_map.items():
            analysis["concerns"][key]["score"] = score
        server._normalize_concern_severity(analysis)
        self.assertEqual(
            {key: value["severity"] for key, value in analysis["concerns"].items()},
            {
                "wrinkles": "none",
                "redness": "none",
                "darkSpots": "mild",
                "texture": "mild",
                "pores": "moderate",
                "laxity": "moderate",
                "sunDamage": "severe",
                "unevenTone": "severe",
            },
        )

        analysis["concerns"]["wrinkles"]["score"] = 101
        with self.assertRaises(server._GoogleResponseError):
            server._normalize_catalog_recommendations(analysis, "face")

    def test_catalog_normalizer_drops_halo_without_forcing_replacement(self) -> None:
        analysis = accepted_analysis()
        analysis["recommendations"] = [
            {
                "treatment": "Sciton BBL",
                "reason": "For visible pigment.",
                "targets": ["darkSpots", "sunDamage"],
                "priority": 1,
            },
            {
                "treatment": "Sciton Moxi",
                "reason": "For visible pigment and texture.",
                "targets": ["darkSpots", "sunDamage", "texture"],
                "priority": 2,
            },
            {
                "treatment": "Sciton Halo",
                "reason": "For mild texture.",
                "targets": ["texture"],
                "priority": 3,
            },
        ]

        normalized = server._normalize_catalog_recommendations(analysis, "face")

        self.assertEqual(
            [item["treatment"] for item in normalized["recommendations"]],
            ["Sciton BBL", "Sciton Moxi"],
        )

    def test_catalog_normalizer_allows_clear_photo_without_forced_treatment(self) -> None:
        analysis = accepted_analysis("face")
        for concern in analysis["concerns"].values():
            concern["score"] = 10
        analysis["recommendations"] = []
        analysis["productRecommendations"] = [
            {
                "product": "Colorescience Face Shield SPF 50",
                "reason": "Daily broad-spectrum sun protection.",
            }
        ]
        analysis["suggestedCombo"] = None

        normalized = server._normalize_catalog_recommendations(analysis, "face")

        self.assertEqual(normalized["recommendations"], [])
        self.assertEqual(len(normalized["productRecommendations"]), 1)
        self.assertEqual(
            normalized["productRecommendations"][0]["product"],
            "Colorescience Face Shield SPF 50",
        )

    def test_installed_google_sdk_accepts_the_exact_high_thinking_config(self) -> None:
        self.assertIsNotNone(server.genai_types)
        config = server.genai_types.GenerateContentConfig(
            seed=12345,
            max_output_tokens=server.GOOGLE_MAX_OUTPUT_TOKENS,
            response_mime_type="application/json",
            response_json_schema=server.ANALYSIS_RESPONSE_SCHEMA,
            thinking_config=server.genai_types.ThinkingConfig(
                thinking_level=server.genai_types.ThinkingLevel.HIGH,
            ),
        )
        self.assertEqual(config.thinking_config.thinking_level.value, "HIGH")
        self.assertEqual(config.seed, 12345)
        self.assertEqual(config.max_output_tokens, 32_768)
        self.assertEqual(config.response_mime_type, "application/json")
        self.assertIsNotNone(config.response_json_schema)

    def test_user_prompt_never_requests_an_adult_skin_age_estimate(self) -> None:
        prompt = server.build_user_prompt("35", "face")
        self.assertIn("guest entered age 35", prompt)
        self.assertIn("never return, infer, or compare an adult skin-age estimate", prompt.lower())
        self.assertNotIn("estimated skin age", prompt.lower())

    def test_positive_first_summary_is_enforced_and_idempotent(self) -> None:
        analysis = accepted_analysis()
        analysis["summary"] = "Areas to discuss follow."
        expected_opening = analysis["positiveHighlights"][0]["detail"]

        first = server._ensure_positive_first_summary(analysis)
        second = server._ensure_positive_first_summary(first)

        self.assertTrue(first["summary"].startswith(expected_opening))
        self.assertEqual(second["summary"].count(expected_opening), 1)

    def test_absence_based_positive_is_replaced_with_direct_area_copy(self) -> None:
        analysis = accepted_analysis()
        old_opening = "No visible veins stand out in this image."
        analysis["positiveHighlights"][0] = {
            "title": "No Visible Veins",
            "detail": old_opening,
        }
        analysis["concerns"] = {
            "veins": {"score": 8, "severity": "none", "description": "Low."},
            "texture": {"score": 30, "severity": "mild", "description": "Visible."},
            "sunDamage": {"score": 40, "severity": "mild", "description": "Visible."},
            "hairRemoval": {"score": 50, "severity": "moderate", "description": "Visible."},
            "dryness": {"score": 60, "severity": "moderate", "description": "Visible."},
        }
        analysis["summary"] = f"{old_opening} Areas to discuss follow."

        repaired = server._repair_positive_highlights(analysis, "legs")
        repaired = server._ensure_positive_first_summary(repaired)

        first = repaired["positiveHighlights"][0]
        self.assertEqual(first["title"], "Balanced Appearance")
        self.assertNotIn("no visible", first["detail"].lower())
        self.assertTrue(repaired["summary"].startswith(first["detail"]))
        self.assertNotIn(old_opening, repaired["summary"])

    def test_positive_copy_is_always_derived_from_the_lowest_scores(self) -> None:
        analysis = accepted_analysis()
        original = deepcopy(analysis["positiveHighlights"])

        repaired = server._repair_positive_highlights(analysis, "face")

        self.assertNotEqual(repaired["positiveHighlights"], original)
        self.assertEqual(
            [item["groundedIn"] for item in repaired["positiveHighlights"]],
            ["redness", "laxity"],
        )
        for highlight in repaired["positiveHighlights"]:
            self.assertLessEqual(
                repaired["concerns"][highlight["groundedIn"]]["score"],
                40,
            )

    def test_no_signs_and_elasticity_are_not_used_as_positive_highlights(self) -> None:
        analysis = accepted_analysis("back")
        analysis["positiveHighlights"] = [
            {
                "title": "Clear Complexion",
                "detail": "Your back has absolutely no signs of congestion.",
            },
            {
                "title": "Excellent Elasticity",
                "detail": "The area shows excellent natural elasticity.",
            },
        ]

        repaired = server._repair_positive_highlights(analysis, "back")

        combined = json.dumps(repaired["positiveHighlights"]).lower()
        self.assertNotIn(" no ", f" {combined} ")
        self.assertNotIn("elasticity", combined)

    def test_positive_highlight_cannot_contradict_a_severe_score(self) -> None:
        contradictions = {
            "texture": ("Refined texture", "Your skin has a smooth, refined surface."),
            "darkSpots": ("Clear, Luminous Finish", "Your complexion has a clear, luminous finish."),
            "unevenTone": ("Luminous Tone", "Your complexion has an even, luminous quality."),
            "laxity": ("Natural Definition", "Your contours have a softly defined, balanced appearance."),
        }
        for concern_key, (title, detail) in contradictions.items():
            with self.subTest(concern=concern_key):
                analysis = accepted_analysis("face")
                for concern in analysis["concerns"].values():
                    concern["score"] = 30
                analysis["concerns"][concern_key]["score"] = 95
                analysis["positiveHighlights"] = [
                    {"title": title, "detail": detail},
                    {"title": title, "detail": detail},
                ]
                analysis["summary"] = f"{detail} Options follow."

                finalized = server._finalize_completed_analysis(analysis, "face")
                self.assertNotIn(
                    detail,
                    [item["detail"] for item in finalized["positiveHighlights"]],
                )
                for highlight in finalized["positiveHighlights"]:
                    grounding = highlight["groundedIn"]
                    if grounding in finalized["concerns"]:
                        self.assertNotEqual(grounding, concern_key)
                        self.assertLessEqual(
                            finalized["concerns"][grounding]["score"],
                            40,
                        )
                    else:
                        self.assertIn(
                            grounding,
                            {"guestIdentity", "photoClarity"},
                        )

    def test_all_moderate_scores_get_two_honest_human_first_positives(self) -> None:
        analysis = accepted_analysis("face")
        for concern_key, concern in analysis["concerns"].items():
            concern["score"] = 50
            concern["description"] = concern_evidence_description(
                concern_key,
                score=50,
                area="face",
            )

        finalized = server._finalize_completed_analysis(analysis, "face")

        self.assertEqual(
            [item["groundedIn"] for item in finalized["positiveHighlights"]],
            ["guestIdentity", "photoClarity"],
        )
        self.assertEqual(
            [item["title"] for item in finalized["positiveHighlights"]],
            ["Distinctly Yours", "A Clear Starting Point"],
        )
        combined = json.dumps(finalized["positiveHighlights"]).lower()
        self.assertNotIn("a stronger quality", combined)
        self.assertNotIn("one of the strongest", combined)
        self.assertTrue(
            finalized["summary"].startswith(
                finalized["positiveHighlights"][0]["detail"]
            )
        )

    def test_one_low_score_gets_one_direct_and_one_human_first_positive(self) -> None:
        analysis = accepted_analysis("face")
        for concern_key, concern in analysis["concerns"].items():
            concern["score"] = 50
            concern["description"] = concern_evidence_description(
                concern_key,
                score=50,
                area="face",
            )
        analysis["concerns"]["redness"]["score"] = 20
        analysis["concerns"]["redness"]["description"] = (
            concern_evidence_description(
                "redness",
                score=20,
                area="face",
            )
        )

        finalized = server._finalize_completed_analysis(analysis, "face")

        self.assertEqual(
            [item["groundedIn"] for item in finalized["positiveHighlights"]],
            ["redness", "guestIdentity"],
        )
        self.assertEqual(
            finalized["positiveHighlights"][0]["title"],
            server._DIRECT_POSITIVE_COPY["face"]["redness"][0],
        )
        self.assertEqual(
            finalized["positiveHighlights"][1]["title"],
            "Distinctly Yours",
        )

    def test_all_severe_scores_never_receive_absolute_praise(self) -> None:
        analysis = accepted_analysis("legs")
        for concern_key, concern in analysis["concerns"].items():
            concern["score"] = 80
            concern["description"] = concern_evidence_description(
                concern_key,
                score=80,
                area="legs",
            )

        finalized = server._finalize_completed_analysis(analysis, "legs")

        self.assertEqual(len(finalized["positiveHighlights"]), 2)
        self.assertEqual(
            [item["groundedIn"] for item in finalized["positiveHighlights"]],
            ["guestIdentity", "photoClarity"],
        )
        combined = json.dumps(finalized["positiveHighlights"]).lower()
        self.assertNotIn("a stronger quality", combined)
        self.assertNotIn("one of the strongest", combined)

    def test_low_score_direct_copy_downgrades_when_another_score_contradicts_it(self) -> None:
        analysis = accepted_analysis("neck_chest")
        for concern_key, concern in analysis["concerns"].items():
            concern["score"] = 70
            concern["description"] = concern_evidence_description(
                concern_key,
                score=70,
                area="neck_chest",
            )
        analysis["concerns"]["wrinkles"]["score"] = 10
        analysis["concerns"]["wrinkles"]["description"] = (
            "No prominent lines and creases are visible in the photographed area."
        )
        analysis["concerns"]["texture"]["score"] = 80

        finalized = server._finalize_completed_analysis(analysis, "neck_chest")

        self.assertEqual(
            finalized["positiveHighlights"][0]["groundedIn"],
            "guestIdentity",
        )
        self.assertEqual(
            finalized["positiveHighlights"][0]["title"],
            "Distinctly Yours",
        )
        self.assertNotIn(
            "smooth, elegant surface",
            finalized["positiveHighlights"][0]["detail"].lower(),
        )

    def test_positive_derivation_has_deterministic_tie_order(self) -> None:
        analysis = accepted_analysis("hands")
        for concern in analysis["concerns"].values():
            concern["score"] = 65

        first = server._derive_positive_highlights(analysis["concerns"], "hands")
        second = server._derive_positive_highlights(analysis["concerns"], "hands")

        self.assertEqual(first, second)
        self.assertEqual(
            [item["groundedIn"] for item in first],
            ["guestIdentity", "photoClarity"],
        )

    def test_final_validator_rejects_tampered_positive_grounding(self) -> None:
        analysis = accepted_analysis("face")
        analysis = server._repair_positive_highlights(analysis, "face")
        analysis = server._ensure_positive_first_summary(analysis)
        analysis["positiveHighlights"][0]["groundedIn"] = "texture"

        with self.assertRaises(server._GoogleResponseError):
            server._validate_final_completed_analysis(analysis, "face")

    def test_raw_provider_positive_shape_never_gates_a_completed_result(self) -> None:
        variants = [
            [],
            [{"title": "No redness", "detail": "No redness is visible."}],
            [
                {"title": f"Discarded {index}", "detail": "Redness does not stand out."}
                for index in range(4)
            ],
        ]
        for index, raw_highlights in enumerate(variants):
            with self.subTest(raw_count=len(raw_highlights)):
                server._clear_analysis_repeat_cache()
                analysis = accepted_analysis("face")
                analysis["positiveHighlights"] = raw_highlights
                analysis["summary"] = "A focused plan follows."
                models = FakeModels(analysis)
                server.LIVE_MODE = True
                server.gemini_client = SimpleNamespace(models=models)
                server.genai_types = fake_genai_types()

                response = server.app.test_client().post(
                    "/api/analyze",
                    data={
                        "image": (io.BytesIO(jpeg_bytes()), "photo.jpg"),
                        "body_area": "face",
                        "age": "35",
                    },
                    content_type="multipart/form-data",
                    headers={"X-Forwarded-For": f"unit-raw-positive-{index}"},
                )

                self.assertEqual(response.status_code, 200)
                self.assertEqual(len(models.calls), 1)
                payload = response.get_json()
                self.assertEqual(len(payload["positiveHighlights"]), 2)
                self.assertNotIn("does not stand out", json.dumps(payload).lower())
                self.assertNotIn("no redness", json.dumps(payload).lower())

    def test_discarded_raw_medical_positive_cannot_leak_or_fail(self) -> None:
        analysis = accepted_analysis("face")
        analysis["positiveHighlights"] = [
            {"title": "Rosacea", "detail": "Your hands show rosacea."},
        ]
        analysis["summary"] = "A focused plan follows."

        finalized = server._finalize_completed_analysis(analysis, "face")

        self.assertNotIn("rosacea", json.dumps(finalized).lower())
        self.assertNotIn("hands", json.dumps(finalized).lower())

    def test_area_copy_repair_removes_wrong_anatomy(self) -> None:
        analysis = accepted_analysis("back")
        analysis["concerns"]["acne"]["description"] = (
            "Congestion is visible along the jawline."
        )
        analysis["summary"] = "The jawline would benefit from a focused plan."

        repaired = server._repair_anatomical_mismatches(analysis, "back")

        self.assertNotIn("jawline", json.dumps(repaired).lower())
        self.assertIn("photographed back", repaired["concerns"]["acne"]["description"])

    def test_anatomy_guard_handles_compounds_without_false_positives(self) -> None:
        allowed = {
            "hands": "The backs of your hands have a smooth surface.",
            "neck_chest": "The back of your neck has a refined texture.",
            "legs": "The backs of the knees and calves look even.",
        }
        for area, copy in allowed.items():
            with self.subTest(area=area):
                self.assertFalse(server._has_anatomical_mismatch(copy, area))

        for copy, area in (
            ("The back of each hand looks smooth.", "hands"),
            ("The back of her hand has visible texture.", "hands"),
            ("The back of her neck looks smooth.", "neck_chest"),
            ("The back of his neck has an even tone.", "neck_chest"),
            ("The back of her calf has visible texture.", "legs"),
        ):
            with self.subTest(copy=copy, area=area):
                self.assertFalse(server._has_anatomical_mismatch(copy, area))

        mismatches = {
            "face": "The backs of the hands show visible tone variation.",
            "hands": "The jawline has visible definition.",
            "back": "The calf has visible texture.",
            "legs": "The upper back has visible tone variation.",
            "neck_chest": "The forehead has a smooth surface.",
        }
        for area, copy in mismatches.items():
            with self.subTest(area=area):
                self.assertTrue(server._has_anatomical_mismatch(copy, area))

    def test_body_anatomy_repair_does_not_send_guests_to_visia(self) -> None:
        analysis = accepted_analysis("back")
        analysis["summary"] = "The forehead has a smooth surface."

        repaired = server._repair_anatomical_mismatches(analysis, "back")

        self.assertIn("An in-person consultation", repaired["summary"])
        self.assertNotIn("VISIA", repaired["summary"])

    def test_overclaim_sanitizer_uses_bounded_language(self) -> None:
        analysis = accepted_analysis()
        analysis["summary"] = (
            "This gold standard option is safe for all skin tones, reverses years "
            "of sun damage, and permanently reduces hair to eliminate shaving "
            "irritation entirely for a flawless finish. It shows great foundational "
            "support, excellent natural elasticity, and natural bounce. This go-to "
            "product will instantly deliver gorgeous results with a potent formula."
        )

        cleaned = server._sanitize_response(analysis)
        summary = cleaned["summary"].lower()

        for phrase in (
            "gold standard",
            "safe for all skin tones",
            "reverses years",
            "permanently reduces",
            "entirely",
            "flawless",
            "foundational support",
            "elasticity",
            "bounce",
            "go-to",
            "instantly",
            "gorgeous",
            "potent",
        ):
            self.assertNotIn(phrase, summary)

    def test_overclaim_sanitizer_preserves_grammar_for_known_failure_phrases(self) -> None:
        analysis = accepted_analysis()
        analysis["summary"] = (
            "It eliminates redness entirely. It is permanently reducing hair. "
            "It guarantees clear skin. Your skin is beautiful. This can boost "
            "collagen by 40-50%. It is highly effective and helps prevent future "
            "fine lines and prevent future UV damage. Your complexion has a "
            "natural, healthy reflection of light."
        )

        summary = server._sanitize_response(analysis)["summary"]

        self.assertIn("It helps reduce redness.", summary)
        self.assertIn("It supports long-term reduction of hair.", summary)
        self.assertIn("It is designed to support clear skin.", summary)
        self.assertIn("Your skin is refined.", summary)
        self.assertIn("This supports collagen renewal.", summary)
        self.assertIn("helps soften the look of fine lines", summary)
        self.assertIn("support daily protection from UV exposure", summary)
        self.assertIn("natural reflection of light", summary)
        for fragment in (
            "It help reduce",
            "is support long-term",
            "It designed",
            "skin is.",
            "highly effective",
            "prevent future",
            "healthy reflection",
        ):
            self.assertNotIn(fragment, summary)

    def test_overclaim_sanitizer_removes_live_audit_hype(self) -> None:
        analysis = accepted_analysis()
        analysis["summary"] = (
            "This is incredibly common and is highly responsive to treatment. "
            "It can look exceptionally smooth. This is an excellent way to begin, "
            "a highly appropriate option, and highly beneficial for great natural contours."
        )

        summary = server._sanitize_response(analysis)["summary"]

        self.assertIn("common and can often be addressed with treatment", summary)
        self.assertIn("smoother-looking", summary)
        self.assertIn("an option to begin", summary)
        self.assertIn("potentially appropriate option", summary)
        self.assertIn("worth discussing", summary)
        self.assertIn("visible natural contours", summary)
        for phrase in (
            "incredibly common",
            "highly responsive",
            "exceptionally smooth",
            "excellent way",
            "highly appropriate",
            "highly beneficial",
            "great natural contours",
        ):
            self.assertNotIn(phrase, summary.lower())

    def test_final_copy_repairs_known_live_ledger_polish_defects(self) -> None:
        source = (
            "This is a appealing option and a great option. The skin looks refinedeven and "
            "refinedsmooth, refinedcalm, and refinedclear. A premium plan is giving you "
            "long-term smooth results. A personalized combination of targeted treatments "
            "can be focused. It can deeply exfoliate the skin, address dynamic lines, "
            "and rebuild structural support from within over time. The refinedstrong finish "
            "looks incredibly smooth. Add a complementary VISIA scan."
        )

        cleaned = server._sanitize_response({"copy": source})["copy"]

        self.assertEqual(
            cleaned,
            (
                "This is an appealing option and a focused option. The skin looks refined, "
                "even and refined, smooth, refined, calm, and refined, clear. A "
                "provider-selected plan is supporting longer-term hair reduction after a "
                "provider confirms candidacy. A personalized treatment plan can focus on "
                "these visible goals. It can exfoliate the skin's surface, address visible "
                "expression lines, and can gradually support visible contour goals "
                "after a provider confirms candidacy. The refined, strong finish looks "
                "visibly smooth. Add a complimentary VISIA scan."
            ),
        )
        for concern_key in server._PLURAL_OBSERVABLE_CONCERNS:
            mild = server._bounded_visible_concern_description(
                concern_key,
                {"score": 25},
            )
            moderate = server._bounded_visible_concern_description(
                concern_key,
                {"score": 55},
            )
            self.assertIn(" are visible ", mild)
            self.assertIn(" appear in ", moderate)
            self.assertNotIn(" is visible ", mild)
            self.assertNotIn(" appears in ", moderate)

    def test_overclaim_sanitizer_handles_combined_instant_elimination_claim(self) -> None:
        analysis = accepted_analysis()
        analysis["summary"] = "It will instantly eliminate redness entirely."

        summary = server._sanitize_response(analysis)["summary"]

        self.assertEqual(summary, "It can help reduce redness.")
        self.assertNotIn("help help", summary.lower())

    def test_sanitizer_repairs_preserved_live_copy_polish_defects(self) -> None:
        source = (
            "The variations are very responsive to treatment. This is a great way "
            "to begin. A healthy sheen looks completely natural. Sun damage is "
            "visible. This light based option addresses texture often referred to "
            "as strawberry legs. Fine lines do not stand out prominently. "
            "One spot does not stand out."
        )

        cleaned = server._sanitize_response({"copy": source})["copy"]

        for defect in (
            "very responsive",
            "great way",
            "healthy sheen",
            "completely natural",
            "sun damage",
            "light based",
            "strawberry legs",
            "do not stand out",
            "does not stand out",
        ):
            self.assertNotIn(defect, cleaned.lower())
        self.assertIn("can often be addressed with treatment", cleaned)
        self.assertIn("visible sun-exposure signs", cleaned.lower())
        self.assertIn("light-based option", cleaned.lower())
        self.assertIn("fine lines appear subtle", cleaned.lower())
        self.assertIn("one spot appears subtle", cleaned.lower())

    def test_overclaim_sanitizer_deduplicates_repaired_coordination_and_is_idempotent(self) -> None:
        analysis = accepted_analysis()
        analysis["summary"] = "Your skin is absolutely beautiful and truly lovely."

        once = server._sanitize_response(analysis)
        twice = server._sanitize_response(deepcopy(once))

        self.assertEqual(once["summary"], "Your skin is refined.")
        self.assertEqual(twice, once)

    def test_diagnostic_claim_guard_covers_plain_english_diagnoses(self) -> None:
        claims = (
            "Rosacea is visible in this photo.",
            "This looks like rosacea.",
            "This is likely rosacea.",
            "The appearance is suggestive of rosacea.",
            "There appears to be rosacea.",
            "We see eczema.",
            "Acne is present.",
            "Your complexion suggests melasma.",
            "This may be dermatitis.",
            "The photo reveals psoriasis.",
            "This looks like keratosis pilaris.",
            "The bumps are keratosis pilaris.",
            "Those bumps are acne.",
            "That is eczema.",
            "These spots look like melasma.",
            "It appears to be rosacea.",
            "I see psoriasis.",
            "The rash is dermatitis.",
            "Likely rosacea across the cheeks.",
        )
        for claim in claims:
            with self.subTest(claim=claim):
                self.assertIsNotNone(
                    server._DIAGNOSTIC_CLAIM_PATTERN.search(claim)
                )

        self.assertIsNone(
            server._DIAGNOSTIC_CLAIM_PATTERN.search(
                "An acne-focused blue-light protocol may be discussed in person."
            )
        )

    def test_every_prohibited_diagnostic_variant_is_hard_rejected_before_repair(self) -> None:
        diagnostic_terms = (
            "rosacea",
            "melasma",
            "acne",
            "dermatitis",
            "eczema",
            "psoriasis",
            "keratosis pilaris",
            "skin cancer",
            "melanoma",
            "malignant",
            "malignancy",
            "basal cell carcinoma",
            "basal-cell carcinoma",
            "BCC",
            "squamous cell carcinoma",
            "squamous-cell carcinoma",
            "SCC",
            "actinic keratosis",
            "seborrheic keratosis",
            "lentigo maligna",
            "dysplastic nevus",
            "concerning mole",
            "telangiectasia",
            "xerosis",
            "venous insufficiency",
            "folliculitis",
        )
        for term in diagnostic_terms:
            for field in ("summary", "concern"):
                with self.subTest(term=term, field=field):
                    analysis = accepted_analysis("face")
                    diagnostic_copy = f"Visible {term} is present in the photo."
                    if field == "summary":
                        analysis["summary"] = diagnostic_copy
                    else:
                        analysis["concerns"]["redness"]["description"] = (
                            diagnostic_copy
                        )
                    with self.assertRaises(server._GoogleResponseError):
                        server._repair_photo_observation_inferences(
                            analysis,
                            "face",
                        )

    def test_medical_condition_terms_are_removed_from_guest_copy_structurally(self) -> None:
        claims = (
            "This is rosacea.",
            "The redness is rosacea.",
            "Redness indicates rosacea.",
            "The findings indicate melasma.",
            "The features are consistent with psoriasis.",
            "Consistent with dermatitis.",
            "Visible signs point to eczema.",
            "The rash represents dermatitis.",
            "This confirms melasma.",
            "The appearance fits rosacea.",
            "The bumps fit keratosis pilaris.",
            "An acne-focused option for acne scars.",
        )
        for claim in claims:
            with self.subTest(claim=claim):
                cleaned = server._sanitize_response({"copy": claim})["copy"]
                self.assertIsNone(
                    server._PROHIBITED_MEDICAL_TERM_PATTERN.search(cleaned)
                )

        exact_copy_repairs = {
            "An acne-focused blue-light protocol may be discussed in person.": (
                "A congestion-focused blue-light protocol may be discussed in person."
            ),
            "This kills acne bacteria.": (
                "This is used as part of a congestion-focused blue-light protocol."
            ),
            "The redness is rosacea.": "Visible redness is present.",
            "Redness indicates rosacea.": "Visible redness is present.",
            "Acne scars may be discussed in person.": (
                "Textural marks may be discussed in person."
            ),
        }
        for source, expected in exact_copy_repairs.items():
            with self.subTest(source=source):
                self.assertEqual(
                    server._sanitize_response({"copy": source})["copy"],
                    expected,
                )

        unsupported_appearance_copy = (
            "Hyper-pigmentation, photo aging, scarring, scars, a scarred surface, "
            "dehydration, and dehydrated skin."
        )
        cleaned_appearance_copy = server._sanitize_response(
            {"copy": unsupported_appearance_copy}
        )["copy"]
        self.assertIsNone(
            server._UNSUPPORTED_APPEARANCE_TERM_PATTERN.search(cleaned_appearance_copy)
        )
        self.assertNotRegex(cleaned_appearance_copy.lower(), r"visible\s+visible")

        grammar = server._sanitize_response({
            "copy": (
                "Visible hyper-pigmentation is present. Photo aging is visible. "
                "A scarred surface is present."
            )
        })["copy"]
        self.assertEqual(
            grammar,
            (
                "Visible pigment variation is present. Visible sun-exposure signs "
                "are visible. Visible textural marks are present."
            ),
        )
        plural_grammar_repairs = {
            "Visible photoaging appears across the cheeks.": (
                "Visible sun-exposure signs appear across the cheeks."
            ),
            "A scarred surface appears in the photo.": (
                "Visible textural marks appear in the photo."
            ),
            "Scarring looks noticeable in the photo.": (
                "Visible textural marks look noticeable in the photo."
            ),
            "Photo aging has become visible.": (
                "Visible sun-exposure signs have become visible."
            ),
            "Scarring was visible.": "Visible textural marks were visible.",
        }
        for source, expected in plural_grammar_repairs.items():
            with self.subTest(source=source):
                self.assertEqual(
                    server._sanitize_response({"copy": source})["copy"],
                    expected,
                )

    def test_raw_photo_observation_cannot_label_a_medical_condition(self) -> None:
        analysis = accepted_analysis("back")
        analysis["concerns"]["acne"]["description"] = (
            "The bumps are acne."
        )

        with self.assertRaises(server._GoogleResponseError):
            server._finalize_completed_analysis(analysis, "back")

        for unsupported_label in (
            "Visible hyperpigmentation is present.",
            "Visible hyper-pigmentation is present.",
            "Photoaging is visible.",
            "Photo aging is visible.",
            "There is mild scarring.",
            "The surface looks scarred.",
            "The surface appears dehydrated.",
        ):
            with self.subTest(unsupported_label=unsupported_label):
                unsupported = accepted_analysis("back")
                unsupported["concerns"]["texture"]["description"] = unsupported_label
                finalized = server._finalize_completed_analysis(unsupported, "back")
                for text in server._guest_facing_strings(finalized):
                    self.assertIsNone(
                        server._UNSUPPORTED_APPEARANCE_TERM_PATTERN.search(text)
                    )

    def test_raw_photo_observation_cannot_infer_history_cause_or_physical_state(self) -> None:
        unsupported_observations = (
            "The pigment variation likely relates to past sun exposure.",
            "These spots indicate signs of cumulative sun exposure.",
            "The freckling can result from routine sun exposure.",
            "Years spent in the sun explain the visible variation.",
            "The pigment variation likely reflects time spent in the sun.",
            "These textural marks are from past breakouts.",
            "The photo reveals a history of shaving.",
            "Visible follicles are present, indicating regular surface hair removal.",
            "The pattern suggests recent hair removal.",
            "This is evidence of a hair removal routine.",
            "The area appears recently shaved.",
            "A razor bump is present.",
            "Visible dark follicles suggest regular shaving.",
            "The skin appears very firm.",
            "The surface shows excellent firmness and elasticity.",
            "There is a lack of surface moisture and suppleness.",
            "The image shows strong hydration and greater skin thickness.",
            "The skin appears well-hydrated and moisture balanced.",
            "The skin barrier appears intact.",
            "The area has elastic-looking skin and high collagen levels.",
            "The contour reflects volume loss and collagen loss.",
            "The pigment variation reflects some sun exposure.",
            "The darker areas are reflecting general sun exposure.",
            "These variations often accompany normal sun exposure.",
            "The pigment variation suggests sun exposure.",
            "This is evidence of sun exposure.",
            "The spots point to sun exposure.",
            "The variation is associated with sun exposure.",
            "The pattern indicates UV exposure.",
            "This may be related to incidental sun exposure.",
            "The photo shows visible signs of environmental exposure.",
            "The contours appear well supported.",
            "The folds reflect shifts in volume.",
            "The surface softens over time.",
            "The textural variation may indicate minor past marks.",
            "The pigment variation is from incidental sun exposure.",
            "The follicular pattern makes this an ideal target for laser hair reduction.",
        )
        for copy in unsupported_observations:
            with self.subTest(copy=copy):
                analysis = accepted_analysis("face")
                analysis["concerns"]["texture"]["description"] = copy
                with self.assertRaises(server._GoogleResponseError):
                    server._validate_raw_model_observation_copy(analysis)

        supported = accepted_analysis("face")
        supported["concerns"]["sunDamage"]["description"] = (
            "Visible pigment variation and several small freckles appear across "
            "the photographed cheeks."
        )
        self.assertIs(
            server._validate_raw_model_observation_copy(supported),
            supported,
        )

        visible_contour_copy = accepted_analysis("face")
        visible_contour_copy["concerns"]["laxity"]["description"] = (
            "Visible contour softness appears along the photographed jawline."
        )
        self.assertIs(
            server._validate_raw_model_observation_copy(visible_contour_copy),
            visible_contour_copy,
        )

    def test_photo_inference_repair_preserves_completion_with_visible_only_copy(self) -> None:
        inference_phrases = (
            "Years spent in the sun explain the visible variation.",
            "The pigment variation likely reflects time spent in the sun.",
            "These textural marks are from past breakouts.",
            "The photo reveals a history of shaving.",
            "Visible follicles are present, indicating regular surface hair removal.",
            "The pattern suggests recent hair removal.",
            "This is evidence of a hair removal routine.",
            "The area appears recently shaved.",
            "A razor bump is present.",
            "The skin barrier appears intact.",
            "Prior prolonged sun exposure is visible.",
            "The surface appears well-hydrated and moisture balanced.",
            "The area has elastic-looking skin with high collagen levels.",
            "The skin naturally thins over time.",
            "Visible photo aging and a scarred surface are present.",
            "The pigment variation reflects some sun exposure.",
            "The darker areas are reflecting general sun exposure.",
            "These variations often accompany normal sun exposure.",
            "The pigment variation suggests sun exposure.",
            "This is evidence of sun exposure.",
            "The spots point to sun exposure.",
            "The variation is associated with sun exposure.",
            "The pattern indicates UV exposure.",
            "This may be related to incidental sun exposure.",
            "The photo shows visible signs of environmental exposure.",
            "The contours appear well supported.",
            "The folds reflect shifts in volume.",
            "The surface softens over time.",
            "The textural variation may indicate minor past marks.",
            "The pigment variation is from incidental sun exposure.",
            "The follicular pattern makes this an ideal target for laser hair reduction.",
        )
        for copy in inference_phrases:
            with self.subTest(copy=copy):
                analysis = accepted_analysis("face")
                analysis["concerns"]["texture"]["description"] = copy
                analysis["summary"] = (
                    f"{copy} An in-person plan can address the visible goals."
                )

                repaired = server._repair_photo_observation_inferences(
                    analysis,
                    "face",
                )

                self.assertIs(
                    server._validate_raw_model_observation_copy(repaired),
                    repaired,
                )
                observation_copy = " ".join(
                    [repaired["summary"]]
                    + [
                        concern["description"]
                        for concern in repaired["concerns"].values()
                    ]
                )
                self.assertIsNone(
                    server._PHOTO_HISTORY_OR_CAUSE_PATTERN.search(observation_copy)
                )
                self.assertIsNone(
                    server._UNMEASURED_PHYSICAL_STATE_PATTERN.search(
                        observation_copy
                    )
                )
                self.assertIsNone(
                    server._UNSUPPORTED_APPEARANCE_TERM_PATTERN.search(
                        observation_copy
                    )
                )

    def test_photo_inference_repair_covers_preserved_real_photo_failure_copy(self) -> None:
        analysis = accepted_analysis("hands")
        analysis["concerns"]["laxity"]["description"] = (
            "Loose folding is visible, indicating thinning surface skin."
        )
        analysis["concerns"]["sunDamage"]["description"] = (
            "Pigment variation is a common sign of cumulative environmental exposure."
        )
        analysis["concerns"]["veins"]["description"] = (
            "Surface veins are prominent, which happens as the skin naturally thins."
        )
        analysis["summary"] = (
            "Since the skin on our hands naturally thins and shows sun exposure "
            "over time, a restorative plan may be discussed in person."
        )

        repaired = server._repair_photo_observation_inferences(analysis, "hands")

        observation_copy = " ".join(
            [repaired["summary"]]
            + [
                concern["description"]
                for concern in repaired["concerns"].values()
            ]
        )
        self.assertIsNone(
            server._PHOTO_HISTORY_OR_CAUSE_PATTERN.search(observation_copy)
        )
        self.assertIsNone(
            server._UNMEASURED_PHYSICAL_STATE_PATTERN.search(observation_copy)
        )
        self.assertIs(
            server._validate_raw_model_observation_copy(repaired),
            repaired,
        )

    def test_every_model_authored_observation_sentence_is_canonicalized(self) -> None:
        for area, concern_keys in server.AREA_CONCERN_KEYS.items():
            with self.subTest(area=area):
                analysis = accepted_analysis(area)
                original_summary = f"Unique raw summary sentinel for {area}."
                analysis["summary"] = original_summary
                original_descriptions = {}
                for index, concern_key in enumerate(concern_keys):
                    sentinel = (
                        f"Unique raw concern sentinel number {index} for {area}."
                    )
                    original_descriptions[concern_key] = sentinel
                    analysis["concerns"][concern_key].update({
                        "score": 25,
                        "severity": "mild",
                        "description": sentinel,
                    })

                repaired = server._repair_photo_observation_inferences(
                    analysis,
                    area,
                )

                self.assertEqual(
                    repaired["summary"],
                    "A photo-based preview of visible surface features.",
                )
                self.assertNotIn(original_summary, repaired["summary"])
                for concern_key, sentinel in original_descriptions.items():
                    description = repaired["concerns"][concern_key]["description"]
                    self.assertNotIn(sentinel, description)
                    self.assertEqual(
                        description,
                        server._bounded_visible_concern_description(
                            concern_key,
                            repaired["concerns"][concern_key],
                        ),
                    )

    def test_preserved_audit_inferences_cannot_survive_canonicalization(self) -> None:
        analysis = accepted_analysis("hands")
        analysis["concerns"]["veins"]["description"] = (
            "Visible veins often accompany a loss of natural tissue volume."
        )
        analysis["concerns"]["laxity"]["description"] = (
            "Noticeable thinning appears around the visible contours."
        )
        analysis["concerns"]["sunDamage"]["description"] = (
            "Pigment variation is visible, reflecting everyday environmental exposure."
        )

        repaired = server._repair_photo_observation_inferences(analysis, "hands")
        observation_copy = " ".join(
            [repaired["summary"]]
            + [
                concern["description"]
                for concern in repaired["concerns"].values()
            ]
        ).lower()

        for unsupported_phrase in (
            "loss of natural tissue volume",
            "noticeable thinning",
            "everyday environmental exposure",
        ):
            self.assertNotIn(unsupported_phrase, observation_copy)

    def test_neck_canonical_copy_preserves_only_validated_neutral_rest_evidence(self) -> None:
        supported = accepted_analysis("neck_chest")
        supported["concerns"]["wrinkles"].update({
            "score": 55,
            "severity": "moderate",
            "description": (
                "Multiple persistent lines remain clearly visible at rest in a "
                "neutral resting view, independent of pose."
            ),
        })
        repaired_supported = server._repair_photo_observation_inferences(
            supported,
            "neck_chest",
        )
        self.assertEqual(
            repaired_supported["concerns"]["wrinkles"]["score"],
            55,
        )
        self.assertEqual(
            repaired_supported["concerns"]["wrinkles"]["description"],
            (
                "Clearly visible lines and creases appear at rest in a neutral "
                "resting view, independent of pose."
            ),
        )

        unsupported = accepted_analysis("neck_chest")
        unsupported["concerns"]["wrinkles"].update({
            "score": 55,
            "severity": "moderate",
            "description": "Several visible horizontal lines appear in this photo.",
        })
        repaired_unsupported = server._repair_photo_observation_inferences(
            unsupported,
            "neck_chest",
        )
        self.assertEqual(
            repaired_unsupported["concerns"]["wrinkles"]["score"],
            40,
        )
        self.assertEqual(
            repaired_unsupported["concerns"]["wrinkles"]["description"],
            "Mild lines and creases are visible in the photographed area.",
        )

    def test_mild_written_evidence_cannot_support_a_moderate_score(self) -> None:
        analysis = accepted_analysis("neck_chest")
        analysis["concerns"]["laxity"]["score"] = 45
        analysis["concerns"]["laxity"]["description"] = (
            "Some slight crepiness and soft natural folds are visible."
        )

        server._validate_score_description_coherence(analysis, "neck_chest")

        self.assertEqual(analysis["concerns"]["laxity"]["score"], 40)
        self.assertEqual(analysis["concerns"]["laxity"]["severity"], "mild")

    def test_clear_moderate_evidence_can_support_a_moderate_score(self) -> None:
        analysis = accepted_analysis("neck_chest")
        analysis["concerns"]["laxity"]["score"] = 45
        analysis["concerns"]["laxity"]["description"] = (
            "Clearly visible persistent crepiness remains visible at rest in a "
            "neutral resting view, independent of pose."
        )

        self.assertIs(
            server._validate_score_description_coherence(
                analysis,
                "neck_chest",
            ),
            analysis,
        )

    def test_generic_clear_wording_cannot_rescue_a_positional_neck_fold(self) -> None:
        analysis = accepted_analysis("neck_chest")
        analysis["concerns"]["laxity"]["score"] = 45
        analysis["concerns"]["laxity"]["description"] = (
            "A clearly visible single fold appears because the neck is turned."
        )

        server._validate_score_description_coherence(analysis, "neck_chest")

        self.assertEqual(analysis["concerns"]["laxity"]["score"], 40)

    def test_neck_lines_need_neutral_rest_evidence_for_a_moderate_score(self) -> None:
        analysis = accepted_analysis("neck_chest")
        analysis["concerns"]["wrinkles"]["score"] = 50
        analysis["concerns"]["wrinkles"]["description"] = (
            "A few noticeable horizontal bands are visible."
        )

        server._validate_score_description_coherence(analysis, "neck_chest")

        self.assertEqual(analysis["concerns"]["wrinkles"]["score"], 40)

    def test_negated_neutral_view_cannot_support_a_moderate_neck_score(self) -> None:
        descriptions = (
            "Moderate horizontal lines are visible, but this is not a neutral resting view.",
            "Clearly visible crepiness is present, but it cannot be confirmed as independent of pose.",
            "Horizontal lines appear in a neutral resting view that cannot be confirmed.",
            "Lines are not visible at rest, but appear with rotation.",
            "There is no neutral resting view for this photo.",
            "There is no evidence that the lines are visible at rest.",
            "Without a neutral resting view, the lines cannot be separated from pose.",
            "The photo lacks a neutral resting view.",
            "A neutral resting view cannot be established.",
            "This is a non-neutral view, although the lines are visible at rest.",
            "The photograph is oblique, although the lines appear independent of pose.",
            "The neck is turned, although lines are visible at rest in a neutral resting view.",
            "The neck is flexed, although the lines appear independent of pose.",
            "An oblique view shows lines that appear independent of pose.",
        )
        for description in descriptions:
            with self.subTest(description=description):
                analysis = accepted_analysis("neck_chest")
                analysis["concerns"]["wrinkles"]["score"] = 50
                analysis["concerns"]["wrinkles"]["description"] = description
                server._validate_score_description_coherence(
                    analysis,
                    "neck_chest",
                )
                self.assertLessEqual(
                    analysis["concerns"]["wrinkles"]["score"],
                    40,
                )

        neutral = accepted_analysis("neck_chest")
        neutral["concerns"]["wrinkles"]["score"] = 50
        neutral["concerns"]["wrinkles"]["description"] = (
            "The neck is not turned, and multiple persistent lines remain visible "
            "at rest in a neutral resting view."
        )
        self.assertIs(
            server._validate_score_description_coherence(
                neutral,
                "neck_chest",
            ),
            neutral,
        )

    def test_secondary_mild_feature_does_not_invalidate_the_scored_concern(self) -> None:
        analysis = accepted_analysis("legs")
        analysis["concerns"]["dryness"]["score"] = 45
        analysis["concerns"]["dryness"]["description"] = (
            "The surface shows some visible dryness and minor irritation."
        )
        analysis["concerns"]["texture"]["score"] = 55
        analysis["concerns"]["texture"]["description"] = (
            "Clearly visible bumpiness appears with slight redness."
        )

        self.assertIs(
            server._validate_score_description_coherence(analysis, "legs"),
            analysis,
        )

    def test_negated_or_mixed_mild_wording_does_not_corrupt_score(self) -> None:
        descriptions = (
            "No subtle texture is present; clearly visible roughness is prominent.",
            "The visible appearance is anything but mild texture.",
            "Mild-to-moderate texture is visible across the area.",
            "Mild texture is absent; clearly visible roughness is prominent.",
            "There is no evidence of mild texture; prominent roughness is clearly visible.",
            "No sign of subtle texture is present; prominent roughness is clearly visible.",
            "Mild or moderate texture is visible.",
            "Mild/moderate texture is visible.",
            "Mild but clearly moderate texture is visible.",
            "The texture isn't mild; roughness is clearly visible.",
        )
        for description in descriptions:
            with self.subTest(description=description):
                analysis = accepted_analysis("legs")
                analysis["concerns"]["texture"]["score"] = 55
                analysis["concerns"]["texture"]["description"] = description

                server._validate_score_description_coherence(analysis, "legs")

                self.assertEqual(
                    analysis["concerns"]["texture"]["score"],
                    55,
                )

    def test_later_affirmative_mild_clause_can_conservatively_cap_score(self) -> None:
        analysis = accepted_analysis("legs")
        analysis["concerns"]["texture"]["score"] = 55
        analysis["concerns"]["texture"]["description"] = (
            "No subtle texture is present in one area, but texture is mild overall."
        )

        server._validate_score_description_coherence(analysis, "legs")

        self.assertEqual(analysis["concerns"]["texture"]["score"], 40)

    def test_moderate_vein_score_requires_two_visible_vascular_details(self) -> None:
        for area in ("hands", "legs"):
            with self.subTest(area=area, evidence="generic"):
                analysis = accepted_analysis(area)
                analysis["concerns"]["veins"].update({
                    "score": 55,
                    "severity": "moderate",
                    "description": (
                        "Clearly visible surface veins appear in the "
                        "photographed area."
                    ),
                })
                server._validate_score_description_coherence(analysis, area)
                self.assertEqual(analysis["concerns"]["veins"]["score"], 40)
                self.assertEqual(analysis["concerns"]["veins"]["severity"], "mild")

            with self.subTest(area=area, evidence="corroborated"):
                analysis = accepted_analysis(area)
                analysis["concerns"]["veins"].update({
                    "score": 55,
                    "severity": "moderate",
                    "description": (
                        "Clearly visible blue surface veins form a branching "
                        "network across the photographed area."
                    ),
                })
                server._validate_score_description_coherence(analysis, area)
                self.assertEqual(analysis["concerns"]["veins"]["score"], 55)

    def test_explicitly_absent_or_uncertain_evidence_never_becomes_clearly_visible(self) -> None:
        templates = (
            "No {term} is visible.",
            "{term} is not visible.",
            "I cannot confirm visible {term}.",
            "Visible {term} cannot be confirmed.",
            "Possible visible {term} is present.",
            "{term} is uncertain.",
            "There is not enough evidence to establish {term}.",
            "The image rules out {term}.",
            "The image excludes {term}.",
            "{term} is absent.",
            "Zero {term} is present.",
            "Visible {term} was not observed.",
            "Visible {term} is not evident.",
            "Visible {term} is not demonstrated.",
            "Visible {term} is unsupported.",
            "Visible {term} is indeterminate.",
            "Visible {term} is speculative.",
            "Visible {term} is hypothetical.",
        )
        for area, concern_keys in server.AREA_CONCERN_KEYS.items():
            for concern_key in concern_keys:
                term = CONCERN_FUZZ_TERMS[concern_key]
                for template in templates:
                    description = template.format(term=term)
                    with self.subTest(
                        area=area,
                        concern=concern_key,
                        description=description,
                    ):
                        analysis = accepted_analysis(area)
                        analysis["concerns"][concern_key].update({
                            "score": 60,
                            "severity": "moderate",
                            "description": description,
                        })

                        repaired = server._repair_photo_observation_inferences(
                            analysis,
                            area,
                        )

                        concern = repaired["concerns"][concern_key]
                        self.assertLessEqual(concern["score"], 10)
                        self.assertFalse(
                            concern["description"].startswith("Clearly visible")
                        )

    def test_clear_concern_specific_evidence_preserves_each_moderate_score(self) -> None:
        for area, concern_keys in server.AREA_CONCERN_KEYS.items():
            for concern_key in concern_keys:
                with self.subTest(area=area, concern=concern_key):
                    analysis = accepted_analysis(area)
                    for other_key, other in analysis["concerns"].items():
                        other.update({
                            "score": 10,
                            "severity": "none",
                            "description": (
                                "No prominent "
                                f"{CONCERN_EVIDENCE_LABELS[other_key]} is visible."
                            ),
                        })
                    analysis["concerns"][concern_key].update({
                        "score": 60,
                        "severity": "moderate",
                        "description": concern_evidence_description(
                            concern_key,
                            score=60,
                            area=area,
                        ),
                    })

                    repaired = server._repair_photo_observation_inferences(
                        analysis,
                        area,
                    )

                    concern = repaired["concerns"][concern_key]
                    self.assertEqual(concern["score"], 60)
                    self.assertTrue(
                        concern["description"].startswith("Clearly visible")
                        or concern_key == "hairRemoval"
                    )

    def test_mild_synonyms_for_scored_concern_are_not_missed(self) -> None:
        examples = {
            "redness": "Mild warmth is visible across the area.",
            "darkSpots": "Subtle freckling is visible across the area.",
            "laxity": "A slight crepey appearance is visible.",
            "sunDamage": (
                "We notice some subtle freckling and mild pigment variation "
                "typical of UV exposure in this area."
            ),
            "unevenTone": "Mild pigment variation is visible.",
            "dryness": "The surface appears slightly matte.",
        }
        area_for_concern = {
            "redness": "face",
            "darkSpots": "face",
            "laxity": "hands",
            "sunDamage": "neck_chest",
            "unevenTone": "face",
            "dryness": "legs",
        }
        for concern_key, description in examples.items():
            with self.subTest(concern=concern_key):
                area = area_for_concern[concern_key]
                analysis = accepted_analysis(area)
                analysis["concerns"][concern_key]["score"] = 45
                analysis["concerns"][concern_key]["description"] = description

                server._validate_score_description_coherence(analysis, area)

                self.assertEqual(
                    analysis["concerns"][concern_key]["score"],
                    40,
                )

    def test_mild_qualifier_for_the_scored_concern_rejects_moderate_score(self) -> None:
        examples = (
            ("dryness", "Minor visible dryness appears across the surface."),
            ("dryness", "The skin appears slightly dry."),
            (
                "sunDamage",
                "There are slight pigment variations and sun-exposure signs.",
            ),
            ("texture", "The visible texture is subtle in this photo."),
            ("texture", "Mild texture is visible in this photo."),
        )
        for concern_key, description in examples:
            with self.subTest(concern=concern_key):
                analysis = accepted_analysis("legs")
                analysis["concerns"][concern_key]["score"] = 45
                analysis["concerns"][concern_key]["description"] = description
                server._validate_score_description_coherence(
                    analysis,
                    "legs",
                )
                self.assertEqual(
                    analysis["concerns"][concern_key]["score"],
                    40,
                )

    def test_moderate_score_without_written_evidence_is_reset(self) -> None:
        analysis = accepted_analysis("legs")
        analysis["concerns"]["texture"]["score"] = 45
        analysis["concerns"]["texture"]["description"] = ""

        server._validate_score_description_coherence(analysis, "legs")

        self.assertEqual(analysis["concerns"]["texture"]["score"], 10)
        self.assertEqual(analysis["concerns"]["texture"]["severity"], "none")

    def test_invalid_score_cannot_be_rescued_by_mild_wording(self) -> None:
        for invalid_score in (101, -1, True):
            with self.subTest(score=invalid_score):
                analysis = accepted_analysis("legs")
                analysis["concerns"]["texture"]["score"] = invalid_score
                analysis["concerns"]["texture"]["description"] = (
                    "Mild texture is visible."
                )

                with self.assertRaises(server._GoogleResponseError):
                    server._validate_score_description_coherence(
                        analysis,
                        "legs",
                    )

    def test_mild_written_evidence_remains_valid_with_a_mild_score(self) -> None:
        analysis = accepted_analysis("neck_chest")
        analysis["concerns"]["laxity"]["score"] = 35
        analysis["concerns"]["laxity"]["description"] = (
            "A slight natural fold is visible with the neck turned."
        )

        self.assertIs(
            server._validate_score_description_coherence(
                analysis,
                "neck_chest",
            ),
            analysis,
        )

    def test_internal_positive_grounding_key_is_not_rewritten_as_guest_copy(self) -> None:
        analysis = accepted_analysis("back")
        analysis["concerns"]["acne"]["score"] = 5

        finalized = server._finalize_completed_analysis(analysis, "back")

        self.assertEqual(finalized["positiveHighlights"][0]["groundedIn"], "acne")
        self.assertTrue(
            any(
                "acne" in recommendation["targets"]
                for recommendation in finalized["recommendations"]
            )
        )
        for text in server._guest_facing_strings(finalized):
            self.assertIsNone(server._PROHIBITED_MEDICAL_TERM_PATTERN.search(text))
            self.assertIsNone(server._UNSUPPORTED_APPEARANCE_TERM_PATTERN.search(text))

    def test_final_copy_pass_sanitizes_repaired_positive_templates(self) -> None:
        analysis = accepted_analysis("face")
        analysis["positiveHighlights"] = [
            {"title": "No redness", "detail": "Redness does not stand out."},
            {"title": "No lines", "detail": "No wrinkles are visible."},
        ]
        analysis["summary"] = "Redness does not stand out. A simple plan follows."

        analysis = server._sanitize_response(analysis)
        analysis = server._repair_positive_highlights(analysis, "face")
        analysis = server._sanitize_response(analysis)
        analysis = server._ensure_positive_first_summary(analysis)

        all_copy = json.dumps(analysis).lower()
        for phrase in ("beautiful", "gorgeous", "lovely", "fantastic", "perfect"):
            self.assertNotIn(phrase, all_copy)
        self.assertTrue(
            analysis["summary"].lower().startswith(
                analysis["positiveHighlights"][0]["detail"].lower()
            )
        )

    def test_overall_score_is_transparent_and_does_not_mutate_concerns(self) -> None:
        analysis = accepted_analysis()
        for index, concern in enumerate(analysis["concerns"].values()):
            concern["score"] = index * 10
        original_concerns = deepcopy(analysis["concerns"])

        server._apply_score_correction(analysis)

        expected = round(
            100
            - sum(item["score"] for item in original_concerns.values())
            / len(original_concerns)
        )
        self.assertEqual(analysis["overallScore"], expected)
        self.assertEqual(analysis["concerns"], original_concerns)

    def test_model_punctuation_sanitizer_preserves_structure(self) -> None:
        analysis = accepted_analysis()
        analysis["summary"] = "Bright — balanced – refined."
        cleaned = server._sanitize_response(analysis)
        self.assertNotIn("—", cleaned["summary"])
        self.assertNotIn("–", cleaned["summary"])
        self.assertEqual(len(cleaned["positiveHighlights"]), 2)

    def test_server_take_home_report_leads_with_positives(self) -> None:
        response = server.app.test_client().post(
            "/api/report",
            json={"name": "Guest", "analysis": accepted_analysis()},
        )
        self.assertEqual(response.status_code, 200)
        report = response.get_data(as_text=True)

        positive = report.index("Begin With the Positive")
        score = report.index("Overall Score:")
        summary = report.index(">Summary<")
        concerns = report.index("Skin Analysis Results")
        recommendations = report.index("Recommended Treatments")
        self.assertLess(positive, score)
        self.assertLess(score, summary)
        self.assertLess(summary, concerns)
        self.assertLess(concerns, recommendations)
        self.assertIn('/logo.png"', report)
        self.assertIn("Personalized Skin Analysis Report", report)
        self.assertIn("font-family:'Arsenica'", report)
        self.assertIn("/arsenica-regular.otf", report)
        self.assertIn('name="viewport"', report)
        self.assertIn('class="report-results-table"', report)
        self.assertIn('data-label="Assessment"', report)
        self.assertIn("Book Your Complimentary Consultation", report)
        self.assertIn(
            "https://booking.vonandcoaesthetics.com/webstoreNew/services?utm_source=skin-analyzer",
            report,
        )
        self.assertIn(
            "Any concerning lesion needs an in-person medical evaluation.",
            report,
        )
        self.assertNotIn("Skin Age", report)
        self.assertNotIn("radar", report.lower())

    def test_server_take_home_report_escapes_guest_and_model_copy(self) -> None:
        analysis = accepted_analysis()
        analysis["summary"] = '<img src=x onerror="window.__unsafe=1">'
        analysis["positiveHighlights"][0]["title"] = "<script>unsafe()</script>"
        analysis["recommendations"][0]["treatment"] = "<b>Unsafe treatment</b>"
        analysis["productRecommendations"] = [
            {"product": "<i>Unsafe product</i>", "reason": "<svg onload=unsafe()>"},
        ]
        response = server.app.test_client().post(
            "/api/report",
            json={"name": "<img src=x onerror=unsafe()>", "analysis": analysis},
        )
        self.assertEqual(response.status_code, 200)
        report = response.get_data(as_text=True)

        self.assertNotIn("<script>unsafe()</script>", report)
        self.assertNotIn('<img src=x onerror="window.__unsafe=1">', report)
        self.assertNotIn("<b>Unsafe treatment</b>", report)
        self.assertNotIn("<i>Unsafe product</i>", report)
        self.assertNotIn("<svg onload=unsafe()>", report)
        self.assertIn("&lt;script&gt;unsafe()&lt;/script&gt;", report)
        self.assertIn("&lt;b&gt;Unsafe treatment&lt;/b&gt;", report)
        self.assertIn("&lt;i&gt;Unsafe product&lt;/i&gt;", report)

    def test_minor_age_gate_remains_before_demo_results(self) -> None:
        server.LIVE_MODE = False
        for age in ("17", "17.0", "17.9"):
            with self.subTest(age=age):
                response = server.app.test_client().post(
                    "/api/analyze",
                    data={
                        "image": (io.BytesIO(jpeg_bytes()), "photo.jpg"),
                        "body_area": "face",
                        "age": age,
                    },
                    content_type="multipart/form-data",
                    headers={"X-Forwarded-For": "unit-minor"},
                )
                self.assertEqual(response.status_code, 422)
                self.assertTrue(response.get_json()["rejected"])

    def test_malformed_age_is_rejected_instead_of_bypassing_gate(self) -> None:
        server.LIVE_MODE = False
        for age in ("seventeen", "NaN", "-1"):
            with self.subTest(age=age):
                response = server.app.test_client().post(
                    "/api/analyze",
                    data={
                        "image": (io.BytesIO(jpeg_bytes()), "photo.jpg"),
                        "body_area": "face",
                        "age": age,
                    },
                    content_type="multipart/form-data",
                    headers={"X-Forwarded-For": "unit-invalid-age"},
                )
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.get_json()["code"], "invalid_age")

    def test_original_single_image_upload_contract_remains(self) -> None:
        server.LIVE_MODE = False
        response = server.app.test_client().post(
            "/api/analyze",
            data={
                "image": (io.BytesIO(jpeg_bytes()), "photo.jpg"),
                "body_area": "hands",
                "age": "35",
            },
            content_type="multipart/form-data",
            headers={"X-Forwarded-For": "unit-single-image"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(set(payload["concerns"]), AREA_CONCERNS["hands"])
        self.assertIs(payload["_isDemo"], True)

    def test_original_file_extension_rules_remain(self) -> None:
        for filename in ("photo.jpg", "photo.jpeg", "photo.png", "photo.webp"):
            with self.subTest(filename=filename):
                self.assertTrue(server.allowed_file(filename))
        for filename in ("photo.heic", "photo.gif", "photo", "photo.txt"):
            with self.subTest(filename=filename):
                self.assertFalse(server.allowed_file(filename))

    def test_severely_underexposed_photo_is_rejected_before_google(self) -> None:
        models = FakeModels(accepted_analysis("back"))
        server.LIVE_MODE = True
        server.gemini_client = SimpleNamespace(models=models)
        server.genai_types = fake_genai_types()

        response = server.app.test_client().post(
            "/api/analyze",
            data={
                "image": (
                    io.BytesIO(jpeg_bytes(color=(28, 28, 28))),
                    "underexposed.jpg",
                ),
                "body_area": "back",
                "age": "35",
            },
            content_type="multipart/form-data",
            headers={"X-Forwarded-For": "unit-underexposed-preflight"},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(models.calls, [])
        payload = response.get_json()
        self.assertIs(payload["rejected"], True)
        self.assertEqual(payload["reasonCode"], "quality")
        self.assertEqual(payload["rejectionSource"], "local_underexposure")
        self.assertIn("well-lit", payload["reason"])

    def test_brightness_preflight_does_not_reject_a_lit_dark_surface(self) -> None:
        image = Image.new("RGB", (640, 480), (95, 95, 95))
        self.assertIsNone(server._local_image_quality_rejection(image))

    def test_brightness_preflight_preserves_dark_scene_with_clear_highlights(self) -> None:
        image = Image.new("RGB", (640, 480), (45, 45, 45))
        image.paste((185, 185, 185), (0, 0, 40, 480))
        self.assertIsNone(server._local_image_quality_rejection(image))

    def test_brightness_preflight_separates_pinned_dark_control_from_accepted_photos(self) -> None:
        fixture_dir = ROOT / "work/test-images/real-world-10"
        accepted_filenames = (
            "01-face-black-woman.jpg",
            "02-face-older-man.jpg",
            "03-neck-closeup.jpg",
            "04d-neck-clavicle-7067815.jpg",
            "05-hands-older.jpg",
            "06b-hands-dark-skin-man-8276212.jpg",
            "07b-back-mature-welllit.jpg",
            "08-back-clothed-shoulders.jpg",
            "09-legs-male.jpg",
            "10-legs-knees.jpg",
        )
        for filename in accepted_filenames:
            with self.subTest(filename=filename):
                with Image.open(fixture_dir / filename) as image:
                    self.assertIsNone(server._local_image_quality_rejection(image))
        with Image.open(fixture_dir / "07-back-bare-shoulders.jpg") as image:
            rejection = server._local_image_quality_rejection(image)
        self.assertIsNotNone(rejection)
        self.assertEqual(rejection["reasonCode"], "quality")
        if BRITTANY_PHOTO.exists():
            with Image.open(BRITTANY_PHOTO) as image:
                self.assertIsNone(server._local_image_quality_rejection(image))

    @unittest.skipUnless(BRITTANY_PHOTO.exists(), "Brittany test photo is unavailable")
    def test_exact_brittany_photo_is_accepted_in_restored_demo_intake(self) -> None:
        source = BRITTANY_PHOTO.read_bytes()
        self.assertEqual(
            hashlib.sha256(source).hexdigest(),
            "df1d305937419e60fadacaf120365162a80a828cda379aea90507a1de430132a",
        )
        server.LIVE_MODE = False
        response = server.app.test_client().post(
            "/api/analyze",
            data={
                "image": (io.BytesIO(source), BRITTANY_PHOTO.name),
                "body_area": "face",
                "age": "35",
            },
            content_type="multipart/form-data",
            headers={"X-Forwarded-For": "unit-brittany-demo"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(len(payload["positiveHighlights"]), 2)
        self.assertIs(payload["_isDemo"], True)

    @unittest.skipUnless(BRITTANY_PHOTO.exists(), "Brittany test photo is unavailable")
    def test_exact_brittany_photo_reaches_mocked_gemini_as_clean_rgb_jpeg(self) -> None:
        models = FakeModels(accepted_analysis())
        server.LIVE_MODE = True
        server.gemini_client = SimpleNamespace(models=models)
        server.genai_types = fake_genai_types()

        response = server.app.test_client().post(
            "/api/analyze",
            data={
                "image": (io.BytesIO(BRITTANY_PHOTO.read_bytes()), BRITTANY_PHOTO.name),
                "body_area": "face",
                "age": "35",
            },
            content_type="multipart/form-data",
            headers={"X-Forwarded-For": "unit-brittany-live"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(models.calls), 1)
        call = models.calls[0]
        self.assertEqual(call["model"], "gemini-3.1-pro-preview")
        self.assertEqual(call["config"]["thinking_config"], {"thinking_level": "HIGH"})
        self.assertIsInstance(call["config"]["seed"], int)
        self.assertNotIn("temperature", call["config"])
        self.assertEqual(call["config"]["max_output_tokens"], 32_768)
        self.assertEqual(
            call["config"]["http_options"]["retry_options"],
            {"attempts": 1},
        )
        self.assertGreaterEqual(call["config"]["http_options"]["timeout"], 69_000)
        self.assertLessEqual(call["config"]["http_options"]["timeout"], 70_000)
        self.assertEqual(
            call["config"]["response_json_schema"],
            server._analysis_schema_for_area("face"),
        )
        image_part = call["contents"][0]
        self.assertEqual(image_part["mime_type"], "image/jpeg")
        with Image.open(io.BytesIO(image_part["data"])) as image:
            self.assertEqual(image.format, "JPEG")
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.size, (480, 640))
            self.assertNotEqual(image.getexif().get(274), 6)

    def test_identical_live_input_reuses_byte_identical_canonical_result(self) -> None:
        first_payload = accepted_analysis("face")
        different_second_payload = accepted_analysis("face")
        different_second_payload["overallScore"] = 64
        different_second_payload["summary"] = (
            "This deliberately different provider response must never be used."
        )
        models = SequencedModels([first_payload, different_second_payload])
        server.LIVE_MODE = True
        server.gemini_client = SimpleNamespace(models=models)
        server.genai_types = fake_genai_types()
        photo = jpeg_bytes(color=(181, 147, 124))

        def submit(client_ip: str):
            return server.app.test_client().post(
                "/api/analyze",
                data={
                    "image": (io.BytesIO(photo), "same-photo.jpg"),
                    "body_area": "face",
                    "age": "35",
                },
                content_type="multipart/form-data",
                headers={"X-Forwarded-For": client_ip},
            )

        first = submit("unit-repeat-first")
        second = submit("unit-repeat-second")

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.data, second.data)
        self.assertEqual(first.get_json(), second.get_json())
        self.assertEqual(len(models.calls), 1)
        self.assertEqual(first.headers["X-Von-Analysis-Repeat"], "generated")
        self.assertEqual(second.headers["X-Von-Analysis-Repeat"], "reused")

    def test_identical_live_input_survives_repeat_cache_reopen(self) -> None:
        models = FakeModels(accepted_analysis("face"))
        server.LIVE_MODE = True
        server.gemini_client = SimpleNamespace(models=models)
        server.genai_types = fake_genai_types()
        photo = jpeg_bytes(color=(176, 143, 121))

        first = server.app.test_client().post(
            "/api/analyze",
            data={
                "image": (io.BytesIO(photo), "restart-photo.jpg"),
                "body_area": "face",
                "age": "35",
            },
            content_type="multipart/form-data",
            headers={"X-Forwarded-For": "unit-repeat-before-reopen"},
        )
        self.assertEqual(first.status_code, 200)
        self.assertEqual(len(models.calls), 1)

        server._reopen_analysis_repeat_cache()
        fail_if_called = SequencedModels([
            AssertionError("Gemini must not run for a persisted repeat"),
        ])
        server.gemini_client = SimpleNamespace(models=fail_if_called)
        second = server.app.test_client().post(
            "/api/analyze",
            data={
                "image": (io.BytesIO(photo), "restart-photo.jpg"),
                "body_area": "face",
                "age": "35.0",
            },
            content_type="multipart/form-data",
            headers={"X-Forwarded-For": "unit-repeat-after-reopen"},
        )

        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.data, second.data)
        self.assertEqual(fail_if_called.calls, [])
        self.assertEqual(second.headers["X-Von-Analysis-Repeat"], "reused")

    def test_repeat_key_separates_photo_area_age_and_build_contract(self) -> None:
        first_bytes = jpeg_bytes(color=(180, 145, 122))
        second_bytes = jpeg_bytes(color=(120, 145, 180))

        baseline = server._analysis_repeat_key(
            first_bytes,
            "face",
            "35",
        )
        self.assertEqual(
            baseline,
            server._analysis_repeat_key(
                first_bytes,
                "face",
                "35.0",
            ),
        )
        self.assertNotEqual(
            baseline,
            server._analysis_repeat_key(
                second_bytes,
                "face",
                "35",
            ),
        )
        self.assertNotEqual(
            baseline,
            server._analysis_repeat_key(
                first_bytes,
                "hands",
                "35",
            ),
        )
        self.assertNotEqual(
            baseline,
            server._analysis_repeat_key(
                first_bytes,
                "face",
                "36",
            ),
        )
        original_fingerprint = server.BUILD_FINGERPRINT
        try:
            server.BUILD_FINGERPRINT = "different-analysis-contract"
            self.assertNotEqual(
                baseline,
                server._analysis_repeat_key(first_bytes, "face", "35"),
            )
        finally:
            server.BUILD_FINGERPRINT = original_fingerprint

    def test_primary_model_seed_is_stable_across_build_fingerprints(self) -> None:
        photo = jpeg_bytes(color=(178, 144, 121))
        baseline = server._analysis_model_seed(photo, "face", "35")
        original_fingerprint = server.BUILD_FINGERPRINT
        try:
            server.BUILD_FINGERPRINT = "a-future-release-fingerprint"
            self.assertEqual(
                baseline,
                server._analysis_model_seed(photo, "face", "35.0"),
            )
        finally:
            server.BUILD_FINGERPRINT = original_fingerprint

        self.assertNotEqual(
            baseline,
            server._analysis_model_seed(photo, "hands", "35"),
        )
        self.assertNotEqual(
            baseline,
            server._analysis_model_seed(photo, "face", "36"),
        )

    def test_client_claimed_hash_cannot_alias_two_different_uploads(self) -> None:
        models = FakeModels(accepted_analysis("face"))
        server.LIVE_MODE = True
        server.gemini_client = SimpleNamespace(models=models)
        server.genai_types = fake_genai_types()
        claimed_hash = "a" * 64

        responses = []
        for index, photo in enumerate((
            jpeg_bytes(color=(181, 147, 124)),
            jpeg_bytes(color=(124, 147, 181)),
        )):
            responses.append(server.app.test_client().post(
                "/api/analyze",
                data={
                    "image": (io.BytesIO(photo), "photo.jpg"),
                    "body_area": "face",
                    "age": "35",
                    "source_sha256": claimed_hash,
                },
                content_type="multipart/form-data",
                headers={"X-Forwarded-For": f"unit-forged-hash-{index}"},
            ))

        self.assertEqual([response.status_code for response in responses], [200, 200])
        self.assertEqual(len(models.calls), 2)
        self.assertEqual(
            [response.headers["X-Von-Analysis-Repeat"] for response in responses],
            ["generated", "generated"],
        )

    def test_identical_concurrent_requests_share_one_provider_call(self) -> None:
        class SlowModels:
            def __init__(inner_self):
                inner_self.calls = []
                inner_self.lock = threading.Lock()

            def generate_content(inner_self, **kwargs):
                with inner_self.lock:
                    inner_self.calls.append(kwargs)
                time.sleep(0.05)
                return SimpleNamespace(text=json.dumps(accepted_analysis("face")))

        models = SlowModels()
        server.LIVE_MODE = True
        server.gemini_client = SimpleNamespace(models=models)
        server.genai_types = fake_genai_types()
        photo = jpeg_bytes(color=(183, 149, 126))

        def submit(index: int):
            return server.app.test_client().post(
                "/api/analyze",
                data={
                    "image": (io.BytesIO(photo), "concurrent-photo.jpg"),
                    "body_area": "face",
                    "age": "35",
                },
                content_type="multipart/form-data",
                headers={"X-Forwarded-For": f"unit-concurrent-{index}"},
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            first, second = list(executor.map(submit, (1, 2)))

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.data, second.data)
        self.assertEqual(len(models.calls), 1)
        self.assertEqual(
            sorted((
                first.headers["X-Von-Analysis-Repeat"],
                second.headers["X-Von-Analysis-Repeat"],
            )),
            ["generated", "reused"],
        )

    def test_persistent_cache_failure_falls_back_without_breaking_analysis(self) -> None:
        class FailingCacheConnection:
            def execute(inner_self, *_args, **_kwargs):
                raise sqlite3.OperationalError("synthetic cache failure")

            def rollback(inner_self):
                return None

            def close(inner_self):
                return None

        models = FakeModels(accepted_analysis("face"))
        server.LIVE_MODE = True
        server.gemini_client = SimpleNamespace(models=models)
        server.genai_types = fake_genai_types()
        photo = jpeg_bytes(color=(179, 146, 123))
        original_db = server._analysis_repeat_db
        server._analysis_repeat_db = FailingCacheConnection()

        def submit(client_ip: str):
            return server.app.test_client().post(
                "/api/analyze",
                data={
                    "image": (io.BytesIO(photo), "cache-failure.jpg"),
                    "body_area": "face",
                    "age": "35",
                },
                content_type="multipart/form-data",
                headers={"X-Forwarded-For": client_ip},
            )

        try:
            first = submit("unit-cache-failure-first")
            second = submit("unit-cache-failure-second")
        finally:
            server._analysis_repeat_db = original_db

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.data, second.data)
        self.assertEqual(len(models.calls), 1)
        self.assertEqual(second.headers["X-Von-Analysis-Repeat"], "reused")

    def test_repeat_cache_files_are_private_to_the_current_os_account(self) -> None:
        server._harden_analysis_repeat_cache_permissions()
        cache_path = Path(server.ANALYSIS_REPEAT_CACHE_PATH)
        protected_paths = (
            cache_path,
            Path(f"{cache_path}-wal"),
            Path(f"{cache_path}-shm"),
        )
        self.assertTrue(cache_path.exists())
        for protected_path in protected_paths:
            if protected_path.exists():
                self.assertEqual(
                    stat.S_IMODE(protected_path.stat().st_mode),
                    0o600,
                )

    def test_model_observed_area_mismatch_returns_fixed_actionable_rejection(self) -> None:
        mismatch = {
            "rejected": True,
            "reason": "area_mismatch",
            "observedArea": "back",
        }
        models = SequencedModels([mismatch, mismatch])
        server.LIVE_MODE = True
        server.gemini_client = SimpleNamespace(models=models)
        server.genai_types = fake_genai_types()

        response = server.app.test_client().post(
            "/api/analyze",
            data={
                "image": (io.BytesIO(jpeg_bytes()), "photo.jpg"),
                "body_area": "face",
                "age": "35",
            },
            content_type="multipart/form-data",
            headers={"X-Forwarded-For": "unit-area-mismatch"},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(len(models.calls), 2)
        payload = response.get_json()
        self.assertIs(payload["rejected"], True)
        self.assertEqual(payload["reasonCode"], "area_mismatch")
        self.assertEqual(payload["observedArea"], "back")
        self.assertIn("back", payload["reason"].lower())
        self.assertIn("face", payload["reason"].lower())

    def test_area_mismatch_survives_transient_backup_failure(self) -> None:
        mismatch = {
            "rejected": True,
            "reason": "area_mismatch",
            "observedArea": "face",
        }
        models = SequencedModels([
            mismatch,
            google_api_error(503, "UNAVAILABLE"),
        ])
        server.LIVE_MODE = True
        server.gemini_client = SimpleNamespace(models=models)
        server.genai_types = fake_genai_types()

        response = server.app.test_client().post(
            "/api/analyze",
            data={
                "image": (io.BytesIO(jpeg_bytes()), "photo.jpg"),
                "body_area": "legs",
                "age": "35",
            },
            content_type="multipart/form-data",
            headers={"X-Forwarded-For": "unit-area-mismatch-provider-fallback"},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(len(models.calls), 2)
        payload = response.get_json()
        self.assertEqual(payload["reasonCode"], "area_mismatch")
        self.assertEqual(payload["observedArea"], "face")
        self.assertIn("face", payload["reason"].lower())
        self.assertIn("legs", payload["reason"].lower())

    def test_area_mismatch_survives_backup_that_never_finishes(self) -> None:
        mismatch = {
            "rejected": True,
            "reason": "area_mismatch",
            "observedArea": "back",
        }

        class HangingBackupModels:
            def __init__(self):
                self.calls = []
                self.lock = threading.Lock()
                self.release = threading.Event()

            def generate_content(inner_self, **kwargs):
                with inner_self.lock:
                    call_index = len(inner_self.calls)
                    inner_self.calls.append(kwargs)
                if call_index == 0:
                    return SimpleNamespace(text=json.dumps(mismatch))
                inner_self.release.wait(timeout=1)
                return SimpleNamespace(text=json.dumps(mismatch))

        models = HangingBackupModels()
        server.LIVE_MODE = True
        server.gemini_client = SimpleNamespace(models=models)
        server.genai_types = fake_genai_types()
        server.GOOGLE_TOTAL_BUDGET_MS = 120
        server.GOOGLE_HEDGE_DELAY_MS = 10

        started = time.monotonic()
        try:
            response = server.app.test_client().post(
                "/api/analyze",
                data={
                    "image": (io.BytesIO(jpeg_bytes()), "photo.jpg"),
                    "body_area": "hands",
                    "age": "35",
                },
                content_type="multipart/form-data",
                headers={"X-Forwarded-For": "unit-area-mismatch-hung-backup"},
            )
        finally:
            models.release.set()
        elapsed = time.monotonic() - started

        self.assertEqual(response.status_code, 422)
        self.assertEqual(len(models.calls), 2)
        self.assertLess(elapsed, 0.5)
        payload = response.get_json()
        self.assertEqual(payload["reasonCode"], "area_mismatch")
        self.assertEqual(payload["observedArea"], "back")

    def test_completed_analysis_still_beats_single_area_mismatch(self) -> None:
        mismatch = {
            "rejected": True,
            "reason": "area_mismatch",
            "observedArea": "back",
        }
        models = SequencedModels([mismatch, accepted_analysis("face")])
        server.LIVE_MODE = True
        server.gemini_client = SimpleNamespace(models=models)
        server.genai_types = fake_genai_types()

        response = server.app.test_client().post(
            "/api/analyze",
            data={
                "image": (io.BytesIO(jpeg_bytes()), "photo.jpg"),
                "body_area": "face",
                "age": "35",
            },
            content_type="multipart/form-data",
            headers={"X-Forwarded-For": "unit-false-area-mismatch-recovers"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(models.calls), 2)
        self.assertNotIn("rejected", response.get_json())

    def test_single_false_quality_rejection_cannot_beat_clean_retry(self) -> None:
        false_rejection = {
            "rejected": True,
            "reason": "We couldn't get a clear enough read on this photo.",
            "observedArea": "face",
        }
        models = SequencedModels([false_rejection, accepted_analysis("face")])
        server.LIVE_MODE = True
        server.gemini_client = SimpleNamespace(models=models)
        server.genai_types = fake_genai_types()

        response = server.app.test_client().post(
            "/api/analyze",
            data={
                "image": (io.BytesIO(jpeg_bytes()), "photo.jpg"),
                "body_area": "face",
                "age": "35",
            },
            content_type="multipart/form-data",
            headers={"X-Forwarded-For": "unit-false-rejection-recovers"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(models.calls), 2)
        self.assertNotIn("rejected", response.get_json())
        self.assertNotEqual(
            models.calls[0]["config"]["seed"],
            models.calls[1]["config"]["seed"],
        )
        self.assertIn(
            "A clear close-up or tight crop",
            models.calls[1]["contents"][1],
        )

    def test_two_model_quality_rejections_are_retryable_and_not_cached(self) -> None:
        rejection = {
            "rejected": True,
            "reason": "We couldn't get a clear enough read on this photo.",
            "observedArea": "face",
        }
        models = SequencedModels([
            rejection,
            rejection,
            accepted_analysis("face"),
        ])
        server.LIVE_MODE = True
        server.gemini_client = SimpleNamespace(models=models)
        server.genai_types = fake_genai_types()

        def submit(client_ip):
            return server.app.test_client().post(
                "/api/analyze",
                data={
                    "image": (io.BytesIO(jpeg_bytes()), "photo.jpg"),
                    "body_area": "face",
                    "age": "35",
                },
                content_type="multipart/form-data",
                headers={"X-Forwarded-For": client_ip},
            )

        first = submit("unit-model-quality-uncertain-first")
        self.assertEqual(first.status_code, 503)
        self.assertEqual(first.get_json()["code"], "analysis_unavailable")
        self.assertNotIn("X-Von-Analysis-Repeat", first.headers)
        self.assertEqual(len(models.calls), 2)
        self.assertNotEqual(
            models.calls[0]["config"]["seed"],
            models.calls[1]["config"]["seed"],
        )

        second = submit("unit-model-quality-uncertain-second")
        self.assertEqual(second.status_code, 200)
        self.assertEqual(len(models.calls), 3)
        self.assertEqual(second.headers["X-Von-Analysis-Repeat"], "generated")

    def test_model_quality_vote_plus_provider_failure_is_not_cached(self) -> None:
        quality_rejection = {
            "rejected": True,
            "reason": "We couldn't get a clear enough read on this photo.",
            "observedArea": "face",
        }
        models = SequencedModels([
            quality_rejection,
            google_api_error(503, "UNAVAILABLE"),
            accepted_analysis("face"),
        ])
        server.LIVE_MODE = True
        server.gemini_client = SimpleNamespace(models=models)
        server.genai_types = fake_genai_types()

        def submit(client_ip):
            return server.app.test_client().post(
                "/api/analyze",
                data={
                    "image": (io.BytesIO(jpeg_bytes()), "photo.jpg"),
                    "body_area": "face",
                    "age": "35",
                },
                content_type="multipart/form-data",
                headers={"X-Forwarded-For": client_ip},
            )

        first = submit("unit-quality-provider-failure-first")
        self.assertEqual(first.status_code, 503)
        self.assertIs(first.get_json()["retryable"], True)
        self.assertNotIn("X-Von-Analysis-Repeat", first.headers)
        self.assertEqual(len(models.calls), 2)

        second = submit("unit-quality-provider-failure-second")
        self.assertEqual(second.status_code, 200)
        self.assertEqual(len(models.calls), 3)
        self.assertEqual(second.headers["X-Von-Analysis-Repeat"], "generated")

    def test_quality_prompt_accepts_clear_closeups_and_tight_crops(self) -> None:
        self.assertIn(
            "A clear, adequately lit close-up",
            server.SYSTEM_PROMPT,
        )
        self.assertIn(
            "Do not require the full head, full limb, or full body",
            server.SYSTEM_PROMPT,
        )
        self.assertIn(
            "prefer a conservative completed analysis over a quality rejection",
            server.SYSTEM_PROMPT,
        )

    def test_unclassified_model_rejection_copy_cannot_reach_the_guest(self) -> None:
        raw_reason = "Arbitrary provider prose that should never be displayed."
        models = FakeModels({
            "rejected": True,
            "reason": raw_reason,
            "observedArea": "face",
        })
        server.LIVE_MODE = True
        server.gemini_client = SimpleNamespace(models=models)
        server.genai_types = fake_genai_types()

        response = server.app.test_client().post(
            "/api/analyze",
            data={
                "image": (io.BytesIO(jpeg_bytes()), "photo.jpg"),
                "body_area": "face",
                "age": "35",
            },
            content_type="multipart/form-data",
            headers={"X-Forwarded-For": "unit-unclassified-rejection"},
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(len(models.calls), 2)
        payload = response.get_json()
        self.assertEqual(payload["code"], "analysis_unavailable")
        self.assertNotIn(raw_reason, json.dumps(payload))

    def test_model_minor_rejection_requires_two_matching_attempts(self) -> None:
        models = FakeModels({
            "rejected": True,
            "reason": "This appears to be a minor under 18.",
            "observedArea": "face",
        })
        server.LIVE_MODE = True
        server.gemini_client = SimpleNamespace(models=models)
        server.genai_types = fake_genai_types()

        response = server.app.test_client().post(
            "/api/analyze",
            data={
                "image": (io.BytesIO(jpeg_bytes()), "photo.jpg"),
                "body_area": "face",
                "age": "35",
            },
            content_type="multipart/form-data",
            headers={"X-Forwarded-For": "unit-minor-model-rejection"},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(len(models.calls), 2)
        self.assertEqual(response.get_json()["reasonCode"], "minor")

    def test_single_false_minor_rejection_cannot_beat_clean_retry(self) -> None:
        minor_rejection = {
            "rejected": True,
            "reason": "This appears to be a minor under 18.",
            "observedArea": "face",
        }
        models = SequencedModels([minor_rejection, accepted_analysis("face")])
        server.LIVE_MODE = True
        server.gemini_client = SimpleNamespace(models=models)
        server.genai_types = fake_genai_types()

        response = server.app.test_client().post(
            "/api/analyze",
            data={
                "image": (io.BytesIO(jpeg_bytes()), "photo.jpg"),
                "body_area": "face",
                "age": "35",
            },
            content_type="multipart/form-data",
            headers={"X-Forwarded-For": "unit-false-minor-recovers"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(models.calls), 2)

    def test_model_positive_anatomy_cannot_leak_into_completed_result(self) -> None:
        invalid = accepted_analysis("face")
        invalid["positiveHighlights"] = [
            {"title": "Graceful hands", "detail": "Your hands look polished."},
            {"title": "Smooth legs", "detail": "Your legs look smooth."},
        ]
        invalid["summary"] = "Your hands look polished. Options follow."
        for concern in invalid["concerns"].values():
            if concern["score"] <= 25:
                concern["score"] = 26
        models = FakeModels(invalid)
        server.LIVE_MODE = True
        server.gemini_client = SimpleNamespace(models=models)
        server.genai_types = fake_genai_types()

        response = server.app.test_client().post(
            "/api/analyze",
            data={
                "image": (io.BytesIO(jpeg_bytes()), "photo.jpg"),
                "body_area": "face",
                "age": "35",
            },
            content_type="multipart/form-data",
            headers={"X-Forwarded-For": "unit-final-anatomy-retry"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(models.calls), 1)
        payload = response.get_json()
        self.assertNotIn("hands", json.dumps(payload).lower())
        self.assertNotIn("legs", json.dumps(payload).lower())
        self.assertTrue(
            all(item.get("groundedIn") for item in payload["positiveHighlights"])
        )

    def test_diagnostic_claim_cannot_win_and_clean_retry_can(self) -> None:
        invalid = accepted_analysis("back")
        invalid["summary"] = (
            f"{invalid['positiveHighlights'][0]['detail']} You have acne."
        )
        models = SequencedModels([invalid, accepted_analysis("back")])
        server.LIVE_MODE = True
        server.gemini_client = SimpleNamespace(models=models)
        server.genai_types = fake_genai_types()

        response = server.app.test_client().post(
            "/api/analyze",
            data={
                "image": (io.BytesIO(jpeg_bytes()), "photo.jpg"),
                "body_area": "back",
                "age": "35",
            },
            content_type="multipart/form-data",
            headers={"X-Forwarded-For": "unit-diagnostic-retry"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(models.calls), 2)
        self.assertNotIn("you have acne", json.dumps(response.get_json()).lower())

    def test_undeclared_model_fields_cannot_survive_into_the_api_payload(self) -> None:
        invalid_top_level = accepted_analysis("face")
        invalid_top_level["providerNarrative"] = "Raw top-level model prose."
        invalid_concern = accepted_analysis("face")
        invalid_concern["concerns"]["redness"]["providerNarrative"] = (
            "Raw nested model prose."
        )

        for index, invalid in enumerate((invalid_top_level, invalid_concern)):
            with self.subTest(index=index):
                server._clear_analysis_repeat_cache()
                models = SequencedModels([invalid, accepted_analysis("face")])
                server.LIVE_MODE = True
                server.gemini_client = SimpleNamespace(models=models)
                server.genai_types = fake_genai_types()

                response = server.app.test_client().post(
                    "/api/analyze",
                    data={
                        "image": (io.BytesIO(jpeg_bytes()), "photo.jpg"),
                        "body_area": "face",
                        "age": "35",
                    },
                    content_type="multipart/form-data",
                    headers={
                        "X-Forwarded-For": f"unit-extra-model-field-{index}"
                    },
                )

                self.assertEqual(response.status_code, 200)
                self.assertEqual(len(models.calls), 2)
                payload_text = json.dumps(response.get_json()).lower()
                self.assertNotIn("providernarrative", payload_text)
                self.assertNotIn("raw top-level model prose", payload_text)
                self.assertNotIn("raw nested model prose", payload_text)

    def test_score_description_contradiction_is_conservatively_capped(self) -> None:
        invalid = accepted_analysis("neck_chest")
        invalid["concerns"]["laxity"]["score"] = 45
        invalid["concerns"]["laxity"]["description"] = (
            "Some slight crepiness and a natural fold are visible."
        )
        models = SequencedModels([invalid, accepted_analysis("neck_chest")])
        server.LIVE_MODE = True
        server.gemini_client = SimpleNamespace(models=models)
        server.genai_types = fake_genai_types()

        response = server.app.test_client().post(
            "/api/analyze",
            data={
                "image": (io.BytesIO(jpeg_bytes()), "photo.jpg"),
                "body_area": "neck_chest",
                "age": "35",
            },
            content_type="multipart/form-data",
            headers={"X-Forwarded-For": "unit-score-copy-retry"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(models.calls), 1)
        self.assertEqual(
            response.get_json()["concerns"]["laxity"]["score"],
            40,
        )
        self.assertEqual(
            response.get_json()["concerns"]["laxity"]["severity"],
            "mild",
        )

    def test_matching_observed_area_is_preserved_in_completed_response(self) -> None:
        models = FakeModels(accepted_analysis("hands"))
        server.LIVE_MODE = True
        server.gemini_client = SimpleNamespace(models=models)
        server.genai_types = fake_genai_types()

        response = server.app.test_client().post(
            "/api/analyze",
            data={
                "image": (io.BytesIO(jpeg_bytes()), "photo.jpg"),
                "body_area": "hands",
                "age": "35",
            },
            content_type="multipart/form-data",
            headers={"X-Forwarded-For": "unit-area-match"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["observedArea"], "hands")

    def test_mild_hand_veins_endpoint_returns_200_without_forcing_bbl(self) -> None:
        analysis = accepted_analysis("hands")
        for concern in analysis["concerns"].values():
            concern["score"] = 10
            concern["severity"] = "none"
            concern["description"] = "No prominent concern is visible."
        analysis["concerns"]["veins"] = {
            "score": 40,
            "severity": "mild",
            "description": "Mild visible vascularity appears on the hands.",
        }
        analysis["concerns"]["dryness"] = {
            "score": 30,
            "severity": "mild",
            "description": "Some visible surface dryness is present.",
        }
        analysis["recommendations"] = [{
            "treatment": "Sciton BBL",
            "reason": "For visible vascularity.",
            "targets": ["veins"],
            "priority": 1,
        }]
        analysis["productRecommendations"] = []
        analysis["suggestedCombo"] = None
        models = FakeModels(analysis)
        server.LIVE_MODE = True
        server.gemini_client = SimpleNamespace(models=models)
        server.genai_types = fake_genai_types()

        response = server.app.test_client().post(
            "/api/analyze",
            data={
                "image": (io.BytesIO(jpeg_bytes()), "photo.jpg"),
                "body_area": "hands",
                "age": "35",
            },
            content_type="multipart/form-data",
            headers={"X-Forwarded-For": "unit-mild-hand-veins-200"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(models.calls), 1)
        payload = response.get_json()
        self.assertEqual(payload["concerns"]["veins"]["score"], 40)
        treatments = {
            item["treatment"]
            for item in payload["recommendations"]
        }
        self.assertFalse(treatments.intersection({
            "Sciton BBL",
            "Sciton Halo",
            "Sculptra",
            "Laser Hair Removal",
        }))
        self.assertIn(
            "SkinBetter Trio Moisture",
            {
                item["product"]
                for item in payload["productRecommendations"]
            },
        )

    def test_uncorroborated_moderate_hand_veins_cap_before_catalog_mapping(self) -> None:
        analysis = accepted_analysis("hands")
        for concern in analysis["concerns"].values():
            concern.update({
                "score": 10,
                "severity": "none",
                "description": "No prominent concern is visible.",
            })
        analysis["concerns"]["veins"] = {
            "score": 55,
            "severity": "moderate",
            "description": (
                "Clearly visible surface veins appear in the photographed area."
            ),
        }
        analysis["concerns"]["dryness"] = {
            "score": 30,
            "severity": "mild",
            "description": "Some visible surface dryness is present.",
        }
        analysis["recommendations"] = [{
            "treatment": "Sciton BBL",
            "reason": "For visible vascularity.",
            "targets": ["veins"],
            "priority": 1,
        }]
        analysis["productRecommendations"] = []
        analysis["suggestedCombo"] = None
        models = FakeModels(analysis)
        server.LIVE_MODE = True
        server.gemini_client = SimpleNamespace(models=models)
        server.genai_types = fake_genai_types()

        response = server.app.test_client().post(
            "/api/analyze",
            data={
                "image": (io.BytesIO(jpeg_bytes()), "photo.jpg"),
                "body_area": "hands",
                "age": "35",
            },
            content_type="multipart/form-data",
            headers={"X-Forwarded-For": "unit-moderate-hand-veins-cap"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["concerns"]["veins"]["score"], 40)
        self.assertNotIn(
            "Sciton BBL",
            {item["treatment"] for item in payload["recommendations"]},
        )

    def test_gemini_transport_timeout_retries_once_within_total_budget(self) -> None:
        models = TimeoutModels()
        server.LIVE_MODE = True
        server.gemini_client = SimpleNamespace(models=models)
        server.genai_types = fake_genai_types()

        response = server.app.test_client().post(
            "/api/analyze",
            data={
                "image": (io.BytesIO(jpeg_bytes()), "photo.jpg"),
                "body_area": "face",
                "age": "35",
            },
            content_type="multipart/form-data",
            headers={"X-Forwarded-For": "unit-timeout"},
        )

        self.assertEqual(len(models.calls), 2)
        for call in models.calls:
            options = call["config"]["http_options"]
            self.assertEqual(options["retry_options"], {"attempts": 1})
            self.assertGreater(options["timeout"], 0)
            self.assertLessEqual(options["timeout"], 70_000)
        self.assertEqual(response.status_code, 504)
        payload = response.get_json()
        self.assertEqual(payload["code"], "analysis_timeout")
        self.assertIs(payload["retryable"], True)
        self.assertIsInstance(payload["error"], str)
        self.assertNotIn("synthetic upstream timeout", payload["error"].lower())

    def test_google_deadline_error_recovers_on_second_high_thinking_attempt(self) -> None:
        models = SequencedModels([
            google_api_error(504, "DEADLINE_EXCEEDED"),
            accepted_analysis(),
        ])
        server.LIVE_MODE = True
        server.gemini_client = SimpleNamespace(models=models)
        server.genai_types = fake_genai_types()

        response = server.app.test_client().post(
            "/api/analyze",
            data={
                "image": (io.BytesIO(jpeg_bytes()), "photo.jpg"),
                "body_area": "face",
                "age": "35",
            },
            content_type="multipart/form-data",
            headers={"X-Forwarded-For": "unit-google-deadline-recovers"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(models.calls), 2)
        self.assertEqual(
            [call["config"]["thinking_config"] for call in models.calls],
            [{"thinking_level": "HIGH"}, {"thinking_level": "HIGH"}],
        )

    def test_repeated_google_deadline_error_returns_retryable_504(self) -> None:
        models = SequencedModels([
            google_api_error(504, "DEADLINE_EXCEEDED"),
            google_api_error(504, "DEADLINE_EXCEEDED"),
        ])
        server.LIVE_MODE = True
        server.gemini_client = SimpleNamespace(models=models)
        server.genai_types = fake_genai_types()

        response = server.app.test_client().post(
            "/api/analyze",
            data={
                "image": (io.BytesIO(jpeg_bytes()), "photo.jpg"),
                "body_area": "face",
                "age": "35",
            },
            content_type="multipart/form-data",
            headers={"X-Forwarded-For": "unit-google-deadline-exhausted"},
        )

        self.assertEqual(len(models.calls), 2)
        self.assertEqual(response.status_code, 504)
        payload = response.get_json()
        self.assertEqual(payload["code"], "analysis_timeout")
        self.assertIs(payload["retryable"], True)
        self.assertNotIn("synthetic provider failure", payload["error"].lower())

    def test_slow_primary_starts_hedge_and_returns_without_waiting_for_loser(self) -> None:
        second_payload = accepted_analysis()
        second_payload["concerns"]["redness"].update({
            "score": 9,
            "severity": "none",
        })
        models = BlockingPrimaryModels(accepted_analysis(), second_payload)
        server.LIVE_MODE = True
        server.gemini_client = SimpleNamespace(models=models)
        server.genai_types = fake_genai_types()
        server.GOOGLE_TOTAL_BUDGET_MS = 500
        server.GOOGLE_HEDGE_DELAY_MS = 10

        started = time.monotonic()
        try:
            response = server.app.test_client().post(
                "/api/analyze",
                data={
                    "image": (io.BytesIO(jpeg_bytes()), "photo.jpg"),
                    "body_area": "face",
                },
                content_type="multipart/form-data",
                headers={"X-Forwarded-For": "unit-total-budget"},
            )
        finally:
            models.release_first.set()
        elapsed = time.monotonic() - started

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(models.calls), 2)
        self.assertLess(elapsed, 0.2)
        self.assertNotEqual(
            models.calls[0]["config"]["seed"],
            models.calls[1]["config"]["seed"],
        )
        self.assertEqual(
            response.get_json()["concerns"]["redness"]["score"],
            9,
        )
        timeouts = [
            call["config"]["http_options"]["timeout"] for call in models.calls
        ]
        self.assertGreater(timeouts[0], timeouts[1])
        self.assertGreater(timeouts[1], 0)

    def test_slow_primary_can_finish_after_hedge_fails(self) -> None:
        models = BlockingPrimaryModels(
            accepted_analysis(),
            google_api_error(503, "UNAVAILABLE"),
        )
        server.LIVE_MODE = True
        server.gemini_client = SimpleNamespace(models=models)
        server.genai_types = fake_genai_types()
        server.GOOGLE_TOTAL_BUDGET_MS = 500
        server.GOOGLE_HEDGE_DELAY_MS = 10

        release_timer = threading.Timer(0.06, models.release_first.set)
        release_timer.start()
        started = time.monotonic()
        try:
            response = server.app.test_client().post(
                "/api/analyze",
                data={
                    "image": (io.BytesIO(jpeg_bytes()), "photo.jpg"),
                    "body_area": "face",
                },
                content_type="multipart/form-data",
                headers={"X-Forwarded-For": "unit-primary-survives-hedge"},
            )
        finally:
            models.release_first.set()
            release_timer.join()
        elapsed = time.monotonic() - started

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(models.calls), 2)
        self.assertGreaterEqual(elapsed, 0.05)
        self.assertLess(elapsed, 0.2)

    def test_google_503_retries_once_then_returns_sanitized_503(self) -> None:
        models = SequencedModels([
            google_api_error(503, "UNAVAILABLE"),
            google_api_error(503, "UNAVAILABLE"),
        ])
        server.LIVE_MODE = True
        server.gemini_client = SimpleNamespace(models=models)
        server.genai_types = fake_genai_types()

        response = server.app.test_client().post(
            "/api/analyze",
            data={
                "image": (io.BytesIO(jpeg_bytes()), "photo.jpg"),
                "body_area": "face",
            },
            content_type="multipart/form-data",
            headers={"X-Forwarded-For": "unit-google-unavailable"},
        )

        self.assertEqual(len(models.calls), 2)
        self.assertEqual(response.status_code, 503)
        payload = response.get_json()
        self.assertEqual(payload["code"], "analysis_unavailable")
        self.assertIs(payload["retryable"], True)
        self.assertNotIn("synthetic provider failure", payload["error"].lower())

    def test_remote_protocol_disconnect_retries_then_returns_sanitized_503(self) -> None:
        disconnect = httpx.RemoteProtocolError(
            "synthetic upstream disconnect",
            request=httpx.Request("POST", "https://example.test/generate"),
        )
        models = SequencedModels([disconnect, disconnect])
        server.LIVE_MODE = True
        server.gemini_client = SimpleNamespace(models=models)
        server.genai_types = fake_genai_types()

        response = server.app.test_client().post(
            "/api/analyze",
            data={
                "image": (io.BytesIO(jpeg_bytes()), "photo.jpg"),
                "body_area": "face",
            },
            content_type="multipart/form-data",
            headers={"X-Forwarded-For": "unit-google-disconnect"},
        )

        self.assertEqual(len(models.calls), 2)
        self.assertEqual(response.status_code, 503)
        payload = response.get_json()
        self.assertEqual(payload["code"], "analysis_unavailable")
        self.assertIs(payload["retryable"], True)
        self.assertNotIn("synthetic upstream disconnect", payload["error"].lower())

    def test_google_429_retries_once_then_recovers(self) -> None:
        models = SequencedModels([
            google_api_error(429, "RESOURCE_EXHAUSTED"),
            accepted_analysis(),
        ])
        server.LIVE_MODE = True
        server.gemini_client = SimpleNamespace(models=models)
        server.genai_types = fake_genai_types()

        response = server.app.test_client().post(
            "/api/analyze",
            data={
                "image": (io.BytesIO(jpeg_bytes()), "photo.jpg"),
                "body_area": "face",
            },
            content_type="multipart/form-data",
            headers={"X-Forwarded-For": "unit-google-rate-retry"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(models.calls), 2)

    def test_nonretryable_google_error_does_not_start_a_second_call(self) -> None:
        models = SequencedModels([
            google_api_error(400, "INVALID_ARGUMENT"),
            accepted_analysis(),
        ])
        server.LIVE_MODE = True
        server.gemini_client = SimpleNamespace(models=models)
        server.genai_types = fake_genai_types()

        response = server.app.test_client().post(
            "/api/analyze",
            data={
                "image": (io.BytesIO(jpeg_bytes()), "photo.jpg"),
                "body_area": "face",
            },
            content_type="multipart/form-data",
            headers={"X-Forwarded-For": "unit-google-nonretryable"},
        )

        self.assertEqual(len(models.calls), 1)
        self.assertEqual(response.status_code, 502)
        payload = response.get_json()
        self.assertEqual(payload["code"], "analysis_unavailable")
        self.assertIs(payload["retryable"], False)

    def test_empty_google_response_retries_once_then_recovers(self) -> None:
        models = SequencedModels(["", accepted_analysis()])
        server.LIVE_MODE = True
        server.gemini_client = SimpleNamespace(models=models)
        server.genai_types = fake_genai_types()

        response = server.app.test_client().post(
            "/api/analyze",
            data={
                "image": (io.BytesIO(jpeg_bytes()), "photo.jpg"),
                "body_area": "face",
            },
            content_type="multipart/form-data",
            headers={"X-Forwarded-For": "unit-empty-response-recovers"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(models.calls), 2)

    def test_partial_json_response_cannot_win_the_hedge(self) -> None:
        models = SequencedModels([{}, accepted_analysis()])
        server.LIVE_MODE = True
        server.gemini_client = SimpleNamespace(models=models)
        server.genai_types = fake_genai_types()

        response = server.app.test_client().post(
            "/api/analyze",
            data={
                "image": (io.BytesIO(jpeg_bytes()), "photo.jpg"),
                "body_area": "face",
            },
            content_type="multipart/form-data",
            headers={"X-Forwarded-For": "unit-partial-response-recovers"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(models.calls), 2)

    def test_health_endpoint_reports_current_mode(self) -> None:
        server.MODE = "demo"
        response = server.app.test_client().get("/api/health")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["mode"], "demo")
        self.assertEqual(payload["model"], "gemini-3.1-pro-preview")
        self.assertEqual(payload["thinkingLevel"], "HIGH")
        self.assertEqual(payload["totalBudgetMs"], 70_000)
        self.assertEqual(payload["hedgeDelayMs"], 15_000)
        self.assertEqual(payload["maxOutputTokens"], 32_768)
        self.assertTrue(payload["buildFingerprint"])

    def test_default_build_fingerprint_matches_the_runtime_source_contract(self) -> None:
        expected = hashlib.sha256()
        for relative_path in server.RUNTIME_SOURCE_FILES:
            expected.update(relative_path.encode("utf-8"))
            expected.update(b"\0")
            expected.update((ROOT / relative_path).read_bytes())
            expected.update(b"\0")

        self.assertEqual(server._runtime_source_fingerprint(), expected.hexdigest())
        if "BUILD_FINGERPRINT" not in os.environ:
            self.assertEqual(server.BUILD_FINGERPRINT, expected.hexdigest())

    def test_lead_export_is_disabled_without_an_admin_token(self) -> None:
        original_admin_token = os.environ.pop("ADMIN_TOKEN", None)
        try:
            response = server.app.test_client().get(
                "/api/leads?token=vonco-admin-2026"
            )
        finally:
            if original_admin_token is not None:
                os.environ["ADMIN_TOKEN"] = original_admin_token

        self.assertEqual(response.status_code, 401)

    def test_lead_export_accepts_only_the_configured_admin_token(self) -> None:
        original_admin_token = os.environ.get("ADMIN_TOKEN")
        os.environ["ADMIN_TOKEN"] = "unit-test-admin-token"
        try:
            rejected = server.app.test_client().get("/api/leads?token=wrong")
            accepted = server.app.test_client().get(
                "/api/leads?token=unit-test-admin-token"
            )
        finally:
            if original_admin_token is None:
                os.environ.pop("ADMIN_TOKEN", None)
            else:
                os.environ["ADMIN_TOKEN"] = original_admin_token

        self.assertEqual(rejected.status_code, 401)
        self.assertEqual(accepted.status_code, 200)


if __name__ == "__main__":
    unittest.main()
