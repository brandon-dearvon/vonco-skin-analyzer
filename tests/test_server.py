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


def schema_contains_key(value, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(schema_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(schema_contains_key(item, key) for item in value)
    return False


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
                "OPENAI_API_KEY": "test-openai-key",
                "GOOGLE_API_KEY": "",
                "GEMINI_API_KEY": "",
                "ANTHROPIC_API_KEY": "",
                "AI_PROVIDER_ORDER": "openai,gemini,anthropic",
                "OPENAI_MODEL": "gpt-5.6-terra",
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
            analysis_engine.PROVIDER_CALLS, {"openai": fake_provider}, clear=False
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
                "provider": "openai",
                "name": "gpt-5.6-terra",
                "promptVersion": analysis_engine.PROMPT_VERSION,
            },
        )
        self.assertEqual(observed["image_count"], 3)
        self.assertEqual(observed["angles"], ["front", "left", "right"])
        self.assertEqual(observed["body_area"], "face")

    def test_openai_request_uses_three_original_detail_images_and_strict_schema(self) -> None:
        captured = {}

        class FakeResponses:
            def create(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    output_text=json.dumps(
                        complete_model_result(["front", "left", "right"])
                    )
                )

        class FakeOpenAI:
            def __init__(self, **kwargs):
                captured["client_kwargs"] = kwargs
                self.responses = FakeResponses()

        normalized = [
            analysis_engine.NormalizedImage(angle, b"jpeg", "image/jpeg", 640, 640)
            for angle in ("front", "left", "right")
        ]
        with mock.patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=FakeOpenAI)}):
            result = analysis_engine._call_openai(
                normalized, "face", "test-key", "gpt-5.6-terra"
            )
        image_parts = [
            item
            for item in captured["input"][0]["content"]
            if item["type"] == "input_image"
        ]
        self.assertEqual(len(image_parts), 3)
        self.assertTrue(all(item["detail"] == "original" for item in image_parts))
        self.assertIs(captured["store"], False)
        self.assertTrue(captured["text"]["format"]["strict"])
        self.assertEqual(captured["client_kwargs"]["max_retries"], 0)
        self.assertLessEqual(captured["client_kwargs"]["timeout"], 38)
        self.assertNotIn("temperature", captured)
        self.assertEqual(result["status"], "complete")

    def test_anthropic_request_omits_sampling_parameters(self) -> None:
        from anthropic import transform_schema

        captured = {}

        def fake_transform_schema(schema):
            captured["schema_before_transform"] = schema
            captured["schema_after_transform"] = transform_schema(schema)
            return captured["schema_after_transform"]

        class FakeMessages:
            def create(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    content=[SimpleNamespace(text=json.dumps(complete_model_result()))]
                )

        class FakeAnthropic:
            def __init__(self, **kwargs):
                captured["client_kwargs"] = kwargs
                self.messages = FakeMessages()

        normalized = [
            analysis_engine.NormalizedImage("single", b"jpeg", "image/jpeg", 640, 640)
        ]
        with mock.patch.dict(
            sys.modules,
            {
                "anthropic": SimpleNamespace(
                    Anthropic=FakeAnthropic,
                    transform_schema=fake_transform_schema,
                )
            },
        ):
            result = analysis_engine._call_anthropic(
                normalized, "face", "test-key", "claude-sonnet-5"
            )
        self.assertNotIn("temperature", captured)
        self.assertNotIn("top_p", captured)
        self.assertEqual(
            captured["output_config"],
            {
                "format": {
                    "type": "json_schema",
                    "schema": captured["schema_after_transform"],
                }
            },
        )
        self.assertEqual(
            captured["schema_before_transform"],
            analysis_engine.model_output_schema("face"),
        )
        self.assertTrue(schema_contains_key(captured["schema_before_transform"], "maxItems"))
        self.assertFalse(schema_contains_key(captured["schema_after_transform"], "maxItems"))
        self.assertEqual(captured["client_kwargs"]["max_retries"], 0)
        self.assertLessEqual(captured["client_kwargs"]["timeout"], 38)
        self.assertEqual(result["status"], "complete")

    def test_relaxed_provider_schema_still_fails_closed_and_falls_back(self) -> None:
        invalid = complete_model_result()
        invalid["observations"][1]["level"] = "visible"
        invalid["observations"].append(
            {
                "id": "visible_lines",
                "label": "Visible lines",
                "level": "visible",
                "description": "Visible lines can be seen in the submitted area.",
                "angles": ["single"],
            }
        )
        invalid["strengths"] = []
        invalid["priorities"] = [
            "visible_redness",
            "surface_texture",
            "visible_lines",
        ]
        calls = []

        def invalid_gemini(*_):
            calls.append("gemini")
            return invalid

        def valid_anthropic(*_):
            calls.append("anthropic")
            return complete_model_result()

        normalized = [
            analysis_engine.NormalizedImage("single", b"jpeg", "image/jpeg", 640, 640)
        ]
        with mock.patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "",
                "GOOGLE_API_KEY": "test-google-key",
                "ANTHROPIC_API_KEY": "test-anthropic-key",
                "AI_PROVIDER_ORDER": "gemini,anthropic",
            },
            clear=False,
        ), mock.patch.dict(
            analysis_engine.PROVIDER_CALLS,
            {"gemini": invalid_gemini, "anthropic": valid_anthropic},
            clear=False,
        ):
            result = analysis_engine.analyze(normalized, "face")

        self.assertEqual(calls, ["gemini", "anthropic"])
        self.assertEqual(result["model"]["provider"], "anthropic")
        self.assertLessEqual(len(result["priorities"]), 2)

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
                normalized, "face", "test-key", "gemini-2.5-flash"
            )

        self.assertEqual(captured["config"]["system_instruction"], analysis_engine.SYSTEM_PROMPT)
        self.assertEqual(
            captured["config"]["response_json_schema"],
            analysis_engine.gemini_output_schema("face"),
        )
        projected_schema = captured["config"]["response_json_schema"]
        self.assertNotIn("maxItems", json.dumps(projected_schema))
        self.assertNotIn("minItems", json.dumps(projected_schema))
        self.assertIn("maxItems", json.dumps(analysis_engine.model_output_schema("face")))
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
                "openai": lambda *_: complete_model_result(
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

    def test_no_provider_keys_fails_closed_without_demo_result(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "",
                "GOOGLE_API_KEY": "",
                "GEMINI_API_KEY": "",
                "ANTHROPIC_API_KEY": "",
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
            {"openai": lambda *_: invalid},
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
            {"openai": lambda *_: invalid},
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
            {"openai": lambda *_: hands_result},
            clear=False,
        ):
            response = self._post_single(body_area="hands")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["discussionTopics"], [])

    def test_approved_moxi_discussion_topic_is_preserved(self) -> None:
        topics = analysis_engine._discussion_topics(["pigment_variation"], "face")
        self.assertEqual([topic["id"] for topic in topics], ["sciton_moxi_laser"])

    def test_medical_review_suppresses_cosmetic_topics(self) -> None:
        with mock.patch.dict(
            analysis_engine.PROVIDER_CALLS,
            {"openai": lambda *_: medical_model_result()},
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
            {"openai": lambda *_: retake_model_result()},
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
            {"openai": lambda *_: invalid},
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
            {"openai": lambda *_: deepcopy(fixed)},
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
            analysis_engine.PROVIDER_CALLS, {"openai": fake_provider}, clear=False
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
            self.assertIn("privacy", response.get_data(as_text=True).lower())
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
            self.assertIn("Von &amp; Co and one or more AI providers", html)
            self.assertIn('class="footer-fine-print"', html)
            self.assertIn("Photo-based cosmetic preview only.", html)
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
            {"openai": lambda *_: complete_model_result()},
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
