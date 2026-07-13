"""Behavioral regressions for the restored original app plus approved changes."""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import httpx
from google.genai import errors as genai_errors
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

    def test_live_schema_is_constrained_to_each_selected_body_area(self) -> None:
        for area, expected_keys in AREA_CONCERNS.items():
            with self.subTest(area=area):
                accepted = server._analysis_schema_for_area(area)["anyOf"][1]
                concerns = accepted["properties"]["concerns"]
                self.assertEqual(set(concerns["properties"]), expected_keys)
                self.assertEqual(set(concerns["required"]), expected_keys)
                self.assertIs(concerns["additionalProperties"], False)

        fallback = server._analysis_schema_for_area("unknown")["anyOf"][1]
        self.assertEqual(
            set(fallback["properties"]["concerns"]["properties"]),
            AREA_CONCERNS["face"],
        )

    def test_installed_google_sdk_accepts_the_exact_high_thinking_config(self) -> None:
        self.assertIsNotNone(server.genai_types)
        config = server.genai_types.GenerateContentConfig(
            max_output_tokens=server.GOOGLE_MAX_OUTPUT_TOKENS,
            response_mime_type="application/json",
            response_json_schema=server.ANALYSIS_RESPONSE_SCHEMA,
            thinking_config=server.genai_types.ThinkingConfig(
                thinking_level=server.genai_types.ThinkingLevel.HIGH,
            ),
        )
        self.assertEqual(config.thinking_config.thinking_level.value, "HIGH")
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
        second_payload["positiveHighlights"][0]["title"] = "Hedge winner"
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
        self.assertEqual(
            response.get_json()["positiveHighlights"][0]["title"],
            "Hedge winner",
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
        self.assertEqual(response.get_json(), {"status": "ok", "mode": "demo"})


if __name__ == "__main__":
    unittest.main()
