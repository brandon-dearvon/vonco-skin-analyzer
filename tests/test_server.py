"""Contract tests for the production visible-surface analyzer backend."""

from __future__ import annotations

import io
import json
import os
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image, ImageCms, ImageDraw

import analysis_engine
from analysis_engine import FINAL_RESULT_KEYS
from backend_app import AnalysisRateLimiter, create_app


def image_bytes(size: tuple[int, int] = (640, 640)) -> bytes:
    image = Image.new("RGB", size, (176, 151, 137))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size[0] // 2, size[1]), fill=(116, 89, 82))
    draw.ellipse(
        (size[0] // 4, size[1] // 4, 3 * size[0] // 4, 3 * size[1] // 4),
        fill=(207, 179, 164),
    )
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=90)
    return output.getvalue()


def mpo_image_bytes(size: tuple[int, int] = (640, 480)) -> bytes:
    primary = Image.new("RGB", size, (176, 151, 137))
    draw = ImageDraw.Draw(primary)
    draw.rectangle((0, 0, size[0] // 2, size[1]), fill=(116, 89, 82))
    orientation = primary.getexif()
    orientation[274] = 6
    secondary = Image.new("L", (320, 240), 151)
    output = io.BytesIO()
    primary.save(
        output,
        format="MPO",
        save_all=True,
        append_images=[secondary],
        quality=90,
        exif=orientation,
    )
    return output.getvalue()


def complete_model_result(angles: list[str] | None = None) -> dict:
    angles = angles or ["single"]
    return {
        "status": "complete",
        "quality": {"overall": "suitable", "issues": [], "guidance": []},
        "observations": [
            {
                "id": "visible_redness",
                "label": "Visible redness",
                "level": "visible",
                "description": "Visible redness is present across part of the submitted area.",
                "angles": angles,
            },
            {
                "id": "surface_texture",
                "label": "Visible surface texture",
                "level": "subtle",
                "description": "Surface texture appears subtle in the available light.",
                "angles": angles,
            },
        ],
        "strengths": ["surface_texture"],
        "priorities": ["visible_redness"],
        "medicalReview": {"suggested": False, "reason": "none"},
    }


def medical_model_result() -> dict:
    return {
        "status": "medical_review",
        "quality": {
            "overall": "limited",
            "issues": ["obstruction"],
            "guidance": ["remove_obstructions"],
        },
        "observations": [
            {
                "id": "visible_redness",
                "label": "Visible redness",
                "level": "visible",
                "description": "A visibly red area falls outside this limited cosmetic review.",
                "angles": ["single"],
            }
        ],
        "strengths": [],
        "priorities": ["visible_redness"],
        "medicalReview": {
            "suggested": True,
            "reason": "visible_concern_outside_cosmetic_scope",
        },
    }


def retake_model_result() -> dict:
    return {
        "status": "retake",
        "quality": {
            "overall": "retake",
            "issues": ["blur"],
            "guidance": ["hold_camera_steady"],
        },
        "observations": [],
        "strengths": [],
        "priorities": [],
        "medicalReview": {"suggested": False, "reason": "none"},
    }


class ServerContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = mock.patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "ignored-legacy-openai-key",
                "GOOGLE_API_KEY": "test-google-key",
                "GEMINI_API_KEY": "",
                "ANTHROPIC_API_KEY": "ignored-legacy-anthropic-key",
                "AI_PROVIDER_ORDER": "openai,gemini,anthropic",
                "GEMINI_MODEL": "gemini-3.5-flash",
                "ALLOWED_ORIGINS": "https://allowed.example",
                "RATE_LIMIT": "100",
                "RATE_WINDOW": "3600",
                "TRUST_PROXY_HOPS": "1",
            },
            clear=False,
        )
        self.environment.start()
        self.app = create_app()
        self.app.config.update(TESTING=True)
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        self.environment.stop()

    def _post_single(self, **form_fields):
        data = {
            "image": (io.BytesIO(image_bytes()), "capture.jpg"),
            "body_area": "face",
            "age_confirmed": "true",
        }
        data.update(form_fields)
        return self.client.post("/api/analyze", data=data, content_type="multipart/form-data")

    def test_three_image_intake_uses_all_images_and_canonical_metadata(self) -> None:
        observed = {}

        def fake_provider(images, body_area, api_key, model):
            observed.update(
                image_count=len(images),
                angles=[image.angle for image in images],
                body_area=body_area,
                api_key=api_key,
                model=model,
            )
            return complete_model_result(["front", "left", "right"])

        data = {
            "images": [
                (io.BytesIO(image_bytes()), "front.jpg"),
                (io.BytesIO(image_bytes()), "left.jpg"),
                (io.BytesIO(image_bytes()), "right.jpg"),
            ],
            "angle_labels": json.dumps(["front", "left", "right"]),
            "body_area": "face",
            "age_confirmed": "true",
        }
        with mock.patch.dict(
            analysis_engine.PROVIDER_CALLS, {"gemini": fake_provider}, clear=False
        ):
            response = self.client.post(
                "/api/analyze", data=data, content_type="multipart/form-data"
            )

        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(set(result), FINAL_RESULT_KEYS)
        self.assertEqual(result["imageCount"], 3)
        self.assertEqual(result["bodyArea"], "face")
        self.assertEqual(
            result["model"],
            {
                "provider": "gemini",
                "name": "gemini-3.5-flash",
                "promptVersion": analysis_engine.PROMPT_VERSION,
            },
        )
        self.assertEqual(observed["image_count"], 3)
        self.assertEqual(observed["angles"], ["front", "left", "right"])
        self.assertEqual(observed["body_area"], "face")

    def test_three_image_intake_rejects_duplicate_or_misordered_angles(self) -> None:
        provider_calls = []

        def unexpected_provider(*_):
            provider_calls.append(True)
            return complete_model_result(["front", "left", "right"])

        for labels in (
            ["front", "front", "right"],
            ["left", "front", "right"],
        ):
            with self.subTest(labels=labels), mock.patch.dict(
                analysis_engine.PROVIDER_CALLS,
                {"gemini": unexpected_provider},
                clear=False,
            ):
                response = self.client.post(
                    "/api/analyze",
                    data={
                        "images": [
                            (io.BytesIO(image_bytes()), "first.jpg"),
                            (io.BytesIO(image_bytes()), "second.jpg"),
                            (io.BytesIO(image_bytes()), "third.jpg"),
                        ],
                        "angle_labels": json.dumps(labels),
                        "body_area": "face",
                        "age_confirmed": "true",
                    },
                    content_type="multipart/form-data",
                )
                self.assertEqual(response.status_code, 400)

        self.assertEqual(provider_calls, [])

    def test_internal_analysis_requires_canonical_image_angle_sequence(self) -> None:
        invalid_sequences = (
            ["front", "front", "right"],
            ["left", "front", "right"],
            ["front", "left"],
        )
        for angles in invalid_sequences:
            images = [
                analysis_engine.NormalizedImage(
                    angle, b"jpeg", "image/jpeg", 640, 640
                )
                for angle in angles
            ]
            with self.subTest(angles=angles), self.assertRaises(
                analysis_engine.SchemaValidationError
            ):
                analysis_engine.analyze(images, "face")

    def test_legacy_provider_configuration_cannot_enable_removed_providers(self) -> None:
        statuses = analysis_engine.provider_status()
        self.assertEqual(
            statuses,
            [
                {
                    "provider": "gemini",
                    "available": True,
                    "model": "gemini-3.5-flash",
                    "thinkingLevel": "high",
                }
            ],
        )
        self.assertEqual(analysis_engine._provider_order(), ("gemini",))
        self.assertEqual(set(analysis_engine.PROVIDER_CALLS), {"gemini"})

        health = self.client.get("/api/health").get_json()
        self.assertEqual(health["providers"], statuses)
        self.assertTrue(health["providerAvailable"])
        self.assertNotIn("openaiStore", health["privacy"])

    def test_invalid_gemini_schema_fails_closed_without_fallback(self) -> None:
        invalid = complete_model_result()
        invalid["observations"] = invalid["observations"] * 6
        calls = []

        def invalid_gemini(*_):
            calls.append("gemini")
            return invalid

        normalized = [
            analysis_engine.NormalizedImage("single", b"jpeg", "image/jpeg", 640, 640)
        ]
        with mock.patch.dict(
            os.environ,
            {
                "GOOGLE_API_KEY": "test-google-key",
                "AI_PROVIDER_ORDER": "gemini,anthropic",
            },
            clear=False,
        ), mock.patch.dict(
            analysis_engine.PROVIDER_CALLS,
            {"gemini": invalid_gemini},
            clear=False,
        ):
            with self.assertRaises(analysis_engine.ProviderUnavailable):
                analysis_engine.analyze(normalized, "face")

        self.assertEqual(calls, ["gemini"])

    def test_gemini_uses_system_instruction_and_strict_response_schema(self) -> None:
        captured = {}

        class FakePart:
            @staticmethod
            def from_text(*, text):
                return {"kind": "text", "text": text}

            @staticmethod
            def from_bytes(*, data, mime_type):
                return {"kind": "image", "data": data, "mime_type": mime_type}

        class FakeModels:
            def generate_content(self, **kwargs):
                captured["request"] = kwargs
                return SimpleNamespace(text=json.dumps(complete_model_result()))

        class FakeClient:
            def __init__(self, **kwargs):
                captured["client_kwargs"] = kwargs
                self.models = FakeModels()

        class FakeTypes:
            Part = FakePart

            @staticmethod
            def Content(**kwargs):
                return kwargs

            @staticmethod
            def HttpRetryOptions(**kwargs):
                return SimpleNamespace(**kwargs)

            @staticmethod
            def HttpOptions(**kwargs):
                return SimpleNamespace(**kwargs)

            @staticmethod
            def GenerateContentConfig(**kwargs):
                captured["config"] = kwargs
                return SimpleNamespace(**kwargs)

            @staticmethod
            def ThinkingConfig(**kwargs):
                captured["thinking_config"] = kwargs
                return SimpleNamespace(**kwargs)

        fake_genai = SimpleNamespace(Client=FakeClient, types=FakeTypes)
        normalized = [
            analysis_engine.NormalizedImage("single", b"jpeg", "image/jpeg", 640, 640)
        ]
        with mock.patch.dict(
            sys.modules,
            {
                "google": SimpleNamespace(genai=fake_genai),
                "google.genai": fake_genai,
            },
        ):
            result = analysis_engine._call_gemini(
                normalized, "face", "test-key", "gemini-3.5-flash"
            )

        self.assertEqual(captured["config"]["system_instruction"], analysis_engine.SYSTEM_PROMPT)
        self.assertNotIn("temperature", captured["config"])
        self.assertEqual(captured["config"]["max_output_tokens"], 8192)
        self.assertEqual(captured["thinking_config"], {"thinking_level": "high"})
        self.assertEqual(captured["config"]["response_mime_type"], "application/json")
        self.assertEqual(captured["request"]["model"], "gemini-3.5-flash")
        self.assertEqual(captured["client_kwargs"]["http_options"].retry_options.attempts, 1)
        self.assertEqual(
            captured["config"]["response_json_schema"],
            analysis_engine.gemini_output_schema("face"),
        )
        projected_schema = captured["config"]["response_json_schema"]
        observations_schema = projected_schema["properties"]["observations"]
        self.assertNotIn("maxItems", observations_schema)
        self.assertNotIn("minItems", observations_schema)
        self.assertEqual(projected_schema["properties"]["strengths"]["maxItems"], 2)
        self.assertEqual(projected_schema["properties"]["priorities"]["maxItems"], 2)
        self.assertIn(
            "maxItems",
            analysis_engine.model_output_schema("face")["properties"]["observations"],
        )
        user_parts = captured["request"]["contents"][0]["parts"]
        self.assertNotIn(
            analysis_engine.SYSTEM_PROMPT,
            [part.get("text") for part in user_parts if part["kind"] == "text"],
        )
        self.assertEqual(result["status"], "complete")

    def test_named_angle_fields_are_supported(self) -> None:
        data = {
            "front": (io.BytesIO(image_bytes()), "front.jpg"),
            "left": (io.BytesIO(image_bytes()), "left.jpg"),
            "right": (io.BytesIO(image_bytes()), "right.jpg"),
            "body_area": "face",
            "age_confirmed": "true",
        }
        with mock.patch.dict(
            analysis_engine.PROVIDER_CALLS,
            {
                "gemini": lambda *_: complete_model_result(
                    ["front", "left", "right"]
                )
            },
            clear=False,
        ):
            response = self.client.post(
                "/api/analyze", data=data, content_type="multipart/form-data"
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["imageCount"], 3)

    def test_missing_gemini_key_fails_closed_even_with_legacy_keys(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "ignored-legacy-openai-key",
                "GOOGLE_API_KEY": "",
                "GEMINI_API_KEY": "",
                "ANTHROPIC_API_KEY": "ignored-legacy-anthropic-key",
            },
            clear=False,
        ):
            response = self._post_single()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.get_json(), {"error": "Analysis service is temporarily unavailable."}
        )

    def test_schema_rejection_fails_closed(self) -> None:
        invalid = complete_model_result()
        invalid["overallScore"] = 91
        with mock.patch.dict(
            analysis_engine.PROVIDER_CALLS,
            {"gemini": lambda *_: invalid},
            clear=False,
        ):
            response = self._post_single()
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("overallScore", response.get_data(as_text=True))

    def test_body_area_rejects_out_of_area_observation(self) -> None:
        invalid = complete_model_result()
        invalid["observations"][0] = {
            "id": "pore_visibility",
            "label": "Visible pore appearance",
            "level": "visible",
            "description": "Pore appearance is visible in the submitted area.",
            "angles": ["single"],
        }
        invalid["priorities"] = ["pore_visibility"]
        with mock.patch.dict(
            analysis_engine.PROVIDER_CALLS,
            {"gemini": lambda *_: invalid},
            clear=False,
        ):
            response = self._post_single(body_area="hands")
        self.assertEqual(response.status_code, 503)

    def test_quality_states_require_consistent_actionable_codes(self) -> None:
        limited = complete_model_result()
        limited["quality"] = {"overall": "limited", "issues": [], "guidance": []}
        with self.assertRaises(analysis_engine.SchemaValidationError):
            analysis_engine.validate_model_output(limited, ["single"], "face")

        contradictory = complete_model_result()
        contradictory["quality"] = {
            "overall": "suitable",
            "issues": ["uneven_lighting"],
            "guidance": ["use_natural_even_light"],
        }
        with self.assertRaises(analysis_engine.SchemaValidationError):
            analysis_engine.validate_model_output(contradictory, ["single"], "face")

    def test_discussion_topics_are_omitted_for_unapproved_body_area_pairing(self) -> None:
        hands_result = {
            "status": "complete",
            "quality": {"overall": "suitable", "issues": [], "guidance": []},
            "observations": [
                {
                    "id": "visible_lines",
                    "label": "Visible lines",
                    "level": "visible",
                    "description": "Visible lines can be seen in the submitted area.",
                    "angles": ["single"],
                }
            ],
            "strengths": [],
            "priorities": ["visible_lines"],
            "medicalReview": {"suggested": False, "reason": "none"},
        }
        with mock.patch.dict(
            analysis_engine.PROVIDER_CALLS,
            {"gemini": lambda *_: hands_result},
            clear=False,
        ):
            response = self._post_single(body_area="hands")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["discussionTopics"], [])

    def test_pigment_topics_follow_current_provider_guide(self) -> None:
        topics = analysis_engine._discussion_topics(["pigment_variation"], "face")
        self.assertEqual(
            [topic["id"] for topic in topics],
            ["sciton_bbl_photofacial", "sciton_halo_laser"],
        )

    def test_medical_review_suppresses_cosmetic_topics(self) -> None:
        with mock.patch.dict(
            analysis_engine.PROVIDER_CALLS,
            {"gemini": lambda *_: medical_model_result()},
            clear=False,
        ):
            response = self._post_single()
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result["status"], "medical_review")
        self.assertTrue(result["medicalReview"]["suggested"])
        self.assertEqual(result["observations"], [])
        self.assertEqual(result["strengths"], [])
        self.assertEqual(result["priorities"], [])
        self.assertEqual(result["discussionTopics"], [])
        self.assertIn("cannot diagnose or rule out disease", result["disclaimer"])

    def test_model_requested_retake_returns_canonical_422(self) -> None:
        with mock.patch.dict(
            analysis_engine.PROVIDER_CALLS,
            {"gemini": lambda *_: retake_model_result()},
            clear=False,
        ):
            response = self._post_single()
        self.assertEqual(response.status_code, 422)
        result = response.get_json()
        self.assertEqual(set(result), FINAL_RESULT_KEYS)
        self.assertEqual(result["status"], "retake")
        self.assertEqual(result["observations"], [])
        self.assertEqual(result["discussionTopics"], [])

    def test_model_diagnostic_description_is_rejected(self) -> None:
        invalid = complete_model_result()
        invalid["observations"][0]["description"] = "This appears to be dermatitis."
        with mock.patch.dict(
            analysis_engine.PROVIDER_CALLS,
            {"gemini": lambda *_: invalid},
            clear=False,
        ):
            response = self._post_single()
        self.assertEqual(response.status_code, 503)

    def test_leads_endpoint_is_unavailable(self) -> None:
        for method in (self.client.get, self.client.post):
            response = method("/api/leads")
            self.assertEqual(response.status_code, 404)
            self.assertEqual(response.get_json(), {"error": "API endpoint not found."})

    def test_gunicorn_access_log_is_disabled_to_avoid_ip_logging(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "gunicorn.conf.py"
        namespace = {}
        exec(compile(config_path.read_text(encoding="utf-8"), str(config_path), "exec"), namespace)
        self.assertIsNone(namespace["accesslog"])

    def test_cors_reflects_only_exact_configured_origin(self) -> None:
        allowed = self.client.get(
            "/api/health", headers={"Origin": "https://allowed.example"}
        )
        lookalike = self.client.get(
            "/api/health", headers={"Origin": "https://allowed.example.evil"}
        )
        self.assertEqual(
            allowed.headers.get("Access-Control-Allow-Origin"),
            "https://allowed.example",
        )
        self.assertIsNone(lookalike.headers.get("Access-Control-Allow-Origin"))
        self.assertIn("Origin", allowed.headers.get("Vary", ""))

    def test_cross_origin_post_is_rejected_before_upload_parsing(self) -> None:
        with mock.patch("backend_app._extract_uploads") as extract_uploads:
            response = self.client.post(
                "/api/analyze",
                data=b"unparsed-form-body",
                content_type="multipart/form-data; boundary=not-used",
                headers={"Origin": "https://malicious.example"},
            )
        self.assertEqual(response.status_code, 403)
        extract_uploads.assert_not_called()

    def test_rate_limiter_identifier_storage_is_bounded(self) -> None:
        limiter = AnalysisRateLimiter(25, 3600, b"test-secret", bucket_cap=2)
        limiter.consume("first")
        limiter.consume("second")
        limiter.consume("third")
        self.assertEqual(limiter.bucket_count, 2)

    def test_repeated_result_has_no_random_score_behavior(self) -> None:
        fixed = complete_model_result()
        with mock.patch.dict(
            analysis_engine.PROVIDER_CALLS,
            {"gemini": lambda *_: deepcopy(fixed)},
            clear=False,
        ):
            first = self._post_single().get_json()
            second = self._post_single().get_json()
        self.assertEqual(first, second)
        serialized = json.dumps(first).lower()
        self.assertNotIn("overallscore", serialized)
        self.assertNotIn("skinage", serialized)
        self.assertNotIn(
            "import random", Path(analysis_engine.__file__).read_text(encoding="utf-8")
        )

    def test_low_resolution_image_returns_structured_422_retake(self) -> None:
        data = {
            "image": (io.BytesIO(image_bytes((320, 320))), "small.jpg"),
            "body_area": "hands",
            "age_confirmed": "true",
        }
        response = self.client.post(
            "/api/analyze", data=data, content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 422)
        result = response.get_json()
        self.assertEqual(set(result), FINAL_RESULT_KEYS)
        self.assertEqual(result["status"], "retake")
        self.assertEqual(result["bodyArea"], "hands")
        self.assertEqual(result["quality"]["overall"], "retake")
        self.assertEqual(
            result["quality"]["guidance"], ["use_a_higher_resolution_image"]
        )
        self.assertEqual(result["observations"], [])
        self.assertEqual(result["discussionTopics"], [])

    def test_mpo_jpeg_container_uses_primary_frame(self) -> None:
        encoded = mpo_image_bytes()
        with Image.open(io.BytesIO(encoded)) as source:
            self.assertEqual(source.format, "MPO")
            self.assertEqual(source.n_frames, 2)
            self.assertEqual(source.getexif().get(274), 6)
            source.seek(1)
            self.assertEqual(source.size, (320, 240))
            self.assertEqual(source.mode, "L")

        normalized = analysis_engine.normalize_image(io.BytesIO(encoded), "single")
        self.assertEqual((normalized.width, normalized.height), (480, 640))
        self.assertEqual(normalized.media_type, "image/jpeg")
        with Image.open(io.BytesIO(normalized.data)) as output:
            self.assertEqual(output.format, "JPEG")
            self.assertEqual(output.mode, "RGB")
            self.assertEqual(getattr(output, "n_frames", 1), 1)

    def test_mpo_jpeg_upload_reaches_provider(self) -> None:
        observed = {}

        def fake_provider(images, body_area, api_key, model):
            observed.update(
                image_count=len(images),
                format=Image.open(io.BytesIO(images[0].data)).format,
                body_area=body_area,
            )
            return complete_model_result(["single"])

        with mock.patch.dict(
            analysis_engine.PROVIDER_CALLS, {"gemini": fake_provider}, clear=False
        ):
            response = self.client.post(
                "/api/analyze",
                data={
                    "image": (io.BytesIO(mpo_image_bytes()), "portrait.jpeg"),
                    "body_area": "face",
                    "age_confirmed": "true",
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(observed, {"image_count": 1, "format": "JPEG", "body_area": "face"})
        self.assertEqual(response.get_json()["status"], "complete")

    def test_age_confirmation_is_required(self) -> None:
        data = {
            "image": (io.BytesIO(image_bytes()), "capture.jpg"),
            "body_area": "face",
        }
        response = self.client.post(
            "/api/analyze", data=data, content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 403)

    def test_privacy_route_serves_privacy_notice(self) -> None:
        response = self.client.get("/privacy")
        try:
            self.assertEqual(response.status_code, 200)
            html = response.get_data(as_text=True)
            self.assertIn("privacy", html.lower())
            self.assertIn("Google Gemini", html)
            self.assertNotIn("OpenAI", html)
            self.assertNotIn("Anthropic", html)
        finally:
            response.close()

    def test_guest_disclaimer_hierarchy_is_compact_and_actionable(self) -> None:
        response = self.client.get("/")
        try:
            self.assertEqual(response.status_code, 200)
            html = response.get_data(as_text=True)
            submit_start = html.index('<div class="submit-row">')
            consent = html.index('id="photoConsent"')
            submit_button = html.index('id="analyzeButton"')
            self.assertLess(submit_start, consent)
            self.assertLess(consent, submit_button)
            self.assertIn("Von &amp; Co and Google Gemini", html)
            self.assertIn('class="footer-fine-print"', html)
            self.assertIn("Photo-based cosmetic preview only.", html)
            self.assertIn(
                "An in-person evaluation is required before treatment", html
            )
            self.assertIn(
                "any concerning lesion should be evaluated by a qualified medical professional",
                html,
            )
            self.assertIn(".server-disclaimer:not([hidden])", html)
            retake_block = html[
                html.index("function renderRetake(data)") : html.index(
                    "function renderUnavailable", html.index("function renderRetake(data)")
                )
            ]
            self.assertIn("renderServerDisclaimer(data);", retake_block)
            self.assertNotIn("Important limitation:", html)
            self.assertNotIn("No sample result was shown", html)
        finally:
            response.close()

    def test_hero_preserves_consumer_value_and_brand_cta_typography(self) -> None:
        response = self.client.get("/")
        try:
            self.assertEqual(response.status_code, 200)
            html = response.get_data(as_text=True)
            self.assertIn("Your skin is personal.", html)
            self.assertIn("Your next step should be too.", html)
            self.assertIn("Visible strengths", html)
            self.assertIn("Visible priorities", html)
            self.assertIn("Matched studio services", html)
            self.assertIn("Skincare shortlist", html)
            self.assertIn("Your Von &amp; Co matches", html)
            self.assertIn('aria-label="Your skin analysis may include"', html)
            self.assertNotIn("Start with what", html)
            self.assertIn('font-family: "Arsenica";', html)

            cta_css = html[
                html.index(".main-site-link {") : html.index(
                    ".main-site-link:hover", html.index(".main-site-link {")
                )
            ]
            self.assertIn('font-family: "Fira Sans", "Trebuchet MS", sans-serif;', cta_css)
            self.assertIn("font-size: 15px;", cta_css)
            self.assertIn("letter-spacing: 1.5px;", cta_css)
            self.assertIn("line-height: 16px;", cta_css)
        finally:
            response.close()

        font_response = self.client.get("/arsenica-regular.otf")
        try:
            self.assertEqual(font_response.status_code, 200)
            self.assertEqual(font_response.content_type, "font/otf")
            self.assertGreater(len(font_response.data), 100_000)
        finally:
            font_response.close()

    def test_recommendation_ui_uses_server_catalog_and_safe_dom_rendering(self) -> None:
        response = self.client.get("/")
        try:
            self.assertEqual(response.status_code, 200)
            html = response.get_data(as_text=True)
            self.assertIn('id="servicesSection"', html)
            self.assertIn('id="productsSection"', html)
            self.assertIn('id="recommendationsEmpty"', html)
            self.assertIn("renderRecommendations(data.appearanceRecommendations);", html)
            self.assertIn("function officialServiceUrl(value)", html)
            self.assertIn('url.pathname.indexOf("/services/") === 0', html)
            self.assertIn("services.slice(0, 3)", html)
            self.assertIn("products.slice(0, 2)", html)
            self.assertIn("Book your complimentary VISIA consultation", html)
            self.assertNotIn("Topics to discuss with a provider", html)
            self.assertNotIn("innerHTML", html)
        finally:
            response.close()

    def test_embedded_icc_profile_is_converted_then_stripped(self) -> None:
        image = Image.open(io.BytesIO(image_bytes()))
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        encoded = io.BytesIO()
        image.save(encoded, format="JPEG", quality=90, icc_profile=profile)
        original_transform = analysis_engine.ImageCms.profileToProfile
        with mock.patch.object(
            analysis_engine.ImageCms,
            "profileToProfile",
            wraps=original_transform,
        ) as transform:
            normalized = analysis_engine.normalize_image(
                io.BytesIO(encoded.getvalue()), "single"
            )
        self.assertTrue(transform.called)
        with Image.open(io.BytesIO(normalized.data)) as output:
            self.assertEqual(output.mode, "RGB")
            self.assertNotIn("icc_profile", output.info)

    def test_cmyk_profile_transform_runs_before_rgb_conversion(self) -> None:
        image = Image.new("CMYK", (640, 640), (0, 80, 80, 20))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 319, 639), fill=(80, 0, 20, 0))
        encoded = io.BytesIO()
        image.save(encoded, format="JPEG", quality=90, icc_profile=b"test-cmyk-profile")
        observed = {}

        def transform(source, _source_profile, _target_profile, *, outputMode):
            observed["source_mode"] = source.mode
            observed["output_mode"] = outputMode
            return source.convert(outputMode)

        with mock.patch.object(
            analysis_engine.ImageCms, "ImageCmsProfile", return_value=object()
        ), mock.patch.object(
            analysis_engine.ImageCms, "createProfile", return_value=object()
        ), mock.patch.object(
            analysis_engine.ImageCms, "profileToProfile", side_effect=transform
        ):
            normalized = analysis_engine.normalize_image(
                io.BytesIO(encoded.getvalue()), "single"
            )

        self.assertEqual(observed["source_mode"], "CMYK")
        self.assertEqual(observed["output_mode"], "RGB")
        with Image.open(io.BytesIO(normalized.data)) as output:
            self.assertEqual(output.mode, "RGB")

    def test_non_face_angle_context_is_explicit(self) -> None:
        images = [
            analysis_engine.NormalizedImage(angle, b"jpeg", "image/jpeg", 640, 640)
            for angle in ("front", "left", "right")
        ]
        context = analysis_engine._image_context(images, "hands")
        self.assertIn("front (left hand capture slot)", context)
        self.assertIn("left (right hand capture slot)", context)
        self.assertIn("right (both hands capture slot)", context)

    def test_rate_limit_returns_retry_after(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"RATE_LIMIT": "1", "RATE_WINDOW": "60"},
            clear=False,
        ):
            limited_app = create_app()
        limited_app.config.update(TESTING=True)
        client = limited_app.test_client()
        with mock.patch.dict(
            analysis_engine.PROVIDER_CALLS,
            {"gemini": lambda *_: complete_model_result()},
            clear=False,
        ):
            first = client.post(
                "/api/analyze",
                data={
                    "image": (io.BytesIO(image_bytes()), "one.jpg"),
                    "body_area": "face",
                    "age_confirmed": "true",
                },
                content_type="multipart/form-data",
            )
            second = client.post(
                "/api/analyze",
                data={
                    "image": (io.BytesIO(image_bytes()), "two.jpg"),
                    "body_area": "face",
                    "age_confirmed": "true",
                },
                content_type="multipart/form-data",
            )
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)
        self.assertGreaterEqual(int(second.headers["Retry-After"]), 1)


if __name__ == "__main__":
    unittest.main()
