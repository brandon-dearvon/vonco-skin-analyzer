"""Behavioral regressions for the restored original app plus approved changes."""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

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


def jpeg_bytes(width: int = 48, height: int = 64) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (178, 146, 126)).save(buffer, format="JPEG")
    return buffer.getvalue()


def accepted_analysis(area: str = "face") -> dict:
    concerns = {
        key: {
            "score": 30 + index,
            "severity": "mild",
            "description": f"Visible description for {key}.",
        }
        for index, key in enumerate(sorted(AREA_CONCERNS[area]))
    }
    first_key = next(iter(concerns))
    return {
        "overallScore": 76,
        "positiveHighlights": [
            {
                "title": "Luminous quality",
                "detail": "The visible skin has a bright, luminous quality.",
            },
            {
                "title": "Refined texture",
                "detail": "The visible surface appears smooth and polished.",
            },
        ],
        "concerns": concerns,
        "recommendations": [
            {
                "treatment": "Sciton BBL",
                "reason": "An option tied to the visible goals.",
                "targets": [first_key],
                "priority": 1,
            },
            {
                "treatment": "Sciton Moxi",
                "reason": "Another option tied to the visible goals.",
                "targets": [first_key],
                "priority": 2,
            },
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
        "summary": "The visible skin has a bright, luminous quality. Options follow.",
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


class FakePart:
    @staticmethod
    def from_bytes(*, data: bytes, mime_type: str):
        return {"data": data, "mime_type": mime_type}


class RestoredAnalyzerBehaviorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.saved = {
            "LIVE_MODE": server.LIVE_MODE,
            "MODE": server.MODE,
            "gemini_client": server.gemini_client,
            "genai_types": server.genai_types,
        }
        server.rate_tracker.clear()

    def tearDown(self) -> None:
        for name, value in self.saved.items():
            setattr(server, name, value)
        server.rate_tracker.clear()

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

    def test_structured_schema_requires_two_to_three_positives_and_no_age(self) -> None:
        accepted = server.ANALYSIS_RESPONSE_SCHEMA["anyOf"][1]
        positives = accepted["properties"]["positiveHighlights"]
        self.assertEqual(positives["minItems"], 2)
        self.assertEqual(positives["maxItems"], 3)
        self.assertIn("positiveHighlights", accepted["required"])
        self.assertNotIn("skinAge", accepted["properties"])
        self.assertNotIn("skinAge", accepted["required"])

    def test_installed_google_sdk_accepts_the_exact_high_thinking_config(self) -> None:
        self.assertIsNotNone(server.genai_types)
        config = server.genai_types.GenerateContentConfig(
            max_output_tokens=65536,
            response_mime_type="application/json",
            response_json_schema=server.ANALYSIS_RESPONSE_SCHEMA,
            thinking_config=server.genai_types.ThinkingConfig(
                thinking_level=server.genai_types.ThinkingLevel.HIGH,
            ),
        )
        self.assertEqual(config.thinking_config.thinking_level.value, "HIGH")
        self.assertEqual(config.max_output_tokens, 65536)
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

        positive = report.index("What Looks Especially Good")
        score = report.index("Overall Score:")
        summary = report.index(">Summary<")
        concerns = report.index("Skin Analysis Results")
        recommendations = report.index("Recommended Treatments")
        self.assertLess(positive, score)
        self.assertLess(score, summary)
        self.assertLess(summary, concerns)
        self.assertLess(concerns, recommendations)
        self.assertIn('src="/logo.png"', report)
        self.assertIn("PERSONALIZED SKIN ANALYSIS REPORT", report)
        self.assertIn(
            "Any concerning lesion needs an in-person medical evaluation.",
            report,
        )
        self.assertNotIn("Skin Age", report)
        self.assertNotIn("radar", report.lower())

    def test_minor_age_gate_remains_before_demo_results(self) -> None:
        server.LIVE_MODE = False
        response = server.app.test_client().post(
            "/api/analyze",
            data={
                "image": (io.BytesIO(jpeg_bytes()), "photo.jpg"),
                "body_area": "face",
                "age": "17",
            },
            content_type="multipart/form-data",
            headers={"X-Forwarded-For": "unit-minor"},
        )
        self.assertEqual(response.status_code, 422)
        self.assertTrue(response.get_json()["rejected"])

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
        models = FakeModels(accepted_analysis(), fail_first=True)
        server.LIVE_MODE = True
        server.gemini_client = SimpleNamespace(models=models)
        server.genai_types = SimpleNamespace(
            Part=FakePart,
            GenerateContentConfig=lambda **kwargs: kwargs,
            ThinkingConfig=lambda **kwargs: kwargs,
            ThinkingLevel=SimpleNamespace(HIGH="HIGH"),
        )

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
        self.assertEqual(len(models.calls), 2)
        call = models.calls[-1]
        self.assertEqual(call["model"], "gemini-3.1-pro-preview")
        self.assertEqual(call["config"]["thinking_config"], {"thinking_level": "HIGH"})
        self.assertIs(
            call["config"]["response_json_schema"],
            server.ANALYSIS_RESPONSE_SCHEMA,
        )
        image_part = call["contents"][0]
        self.assertEqual(image_part["mime_type"], "image/jpeg")
        with Image.open(io.BytesIO(image_part["data"])) as image:
            self.assertEqual(image.format, "JPEG")
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.size, (480, 640))
            self.assertNotEqual(image.getexif().get(274), 6)

    def test_health_endpoint_reports_current_mode(self) -> None:
        server.MODE = "demo"
        response = server.app.test_client().get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "ok", "mode": "demo"})


if __name__ == "__main__":
    unittest.main()
