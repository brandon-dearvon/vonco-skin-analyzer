"""Production Flask surface for the visible-surface analyzer."""

from __future__ import annotations

import json
import hashlib
import hmac
import logging
import math
import os
import secrets
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, send_from_directory
from werkzeug.middleware.proxy_fix import ProxyFix

from analysis_engine import (
    ANALYSIS_VERSION,
    BODY_AREA_OBSERVATIONS,
    PROMPT_VERSION,
    SCHEMA_VERSION,
    TOPIC_MAPPING_VERSION,
    ImageIntakeError,
    ProviderUnavailable,
    analyze,
    build_local_retake,
    normalize_image,
    prompt_hash,
    provider_status,
)


BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"

# Local development files never override deployment environment variables.
load_dotenv(BASE_DIR / ".env", override=False)
load_dotenv(BASE_DIR / "env.txt", override=False)

PORT = int(os.getenv("PORT", "5002"))
DEBUG = os.getenv("DEBUG", "false").strip().lower() == "true"

LOGGER = logging.getLogger(__name__)
MAX_REQUEST_BYTES = 38 * 1024 * 1024


class IntakeContractError(ValueError):
    """Multipart fields do not match an accepted public contract."""


class AnalysisRateLimiter:
    """Per-process sliding-window limiter that stores only HMAC identifiers."""

    def __init__(
        self, limit: int, window_seconds: int, secret: bytes, bucket_cap: int = 10_000
    ) -> None:
        self.limit = max(1, limit)
        self.window_seconds = max(1, window_seconds)
        self.bucket_cap = max(1, bucket_cap)
        self._secret = secret
        self._events: defaultdict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()
        self._last_sweep = time.monotonic()

    def consume(self, raw_identifier: str) -> tuple[bool, int]:
        token = hmac.new(
            self._secret, raw_identifier.encode("utf-8", errors="replace"), hashlib.sha256
        ).hexdigest()
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            if now - self._last_sweep >= min(60, self.window_seconds):
                for identifier, timestamps in list(self._events.items()):
                    recent_timestamps = [item for item in timestamps if item > cutoff]
                    if recent_timestamps:
                        self._events[identifier] = recent_timestamps
                    else:
                        del self._events[identifier]
                self._last_sweep = now
            if token not in self._events and len(self._events) >= self.bucket_cap:
                # Bound memory even if upstream identity headers are spoofed.
                self._events.pop(next(iter(self._events)))
            recent = [timestamp for timestamp in self._events[token] if timestamp > cutoff]
            if len(recent) >= self.limit:
                retry_after = max(1, math.ceil(self.window_seconds - (now - recent[0])))
                self._events[token] = recent
                return False, retry_after
            recent.append(now)
            self._events[token] = recent
            return True, 0

    @property
    def bucket_count(self) -> int:
        with self._lock:
            return len(self._events)


def _positive_int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _allowed_origins() -> set[str]:
    origins: set[str] = set()
    for value in os.getenv("ALLOWED_ORIGINS", "").split(","):
        origin = value.strip().rstrip("/")
        if origin and origin != "*" and origin.startswith(("https://", "http://localhost:")):
            origins.add(origin)
    return origins


def _cors_origin() -> str | None:
    origin = request.headers.get("Origin", "").strip().rstrip("/")
    return origin if origin in _allowed_origins() else None


def _request_origin_is_allowed() -> bool:
    """Reject browser cross-origin writes even though HTML forms ignore CORS."""

    supplied = request.headers.get("Origin")
    if supplied is None:
        return True
    origin = supplied.strip().rstrip("/")
    same_origin = f"{request.scheme}://{request.host}".rstrip("/")
    return bool(origin) and (origin == same_origin or origin in _allowed_origins())


def _request_error(message: str, status: int) -> tuple[Response, int]:
    return jsonify({"error": message}), status


def _parse_angle_labels(raw_value: str | None) -> list[str] | None:
    if raw_value is None or not raw_value.strip():
        return None
    try:
        value = json.loads(raw_value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise IntakeContractError("angle_labels must be valid JSON") from exc
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise IntakeContractError("angle_labels must be an array of strings")
    return value


def _extract_uploads() -> list[tuple[str, Any]]:
    """Accept exactly one supported multipart shape and normalize angle labels."""

    file_keys = set(request.files.keys())
    repeated = request.files.getlist("images")
    singular = request.files.getlist("image")
    named_keys = {"front", "left", "right"}
    supplied_named = named_keys & file_keys
    labels = _parse_angle_labels(request.form.get("angle_labels"))

    if file_keys == {"image"}:
        if len(singular) != 1 or labels not in (None, ["single"]):
            raise IntakeContractError("single image contract is invalid")
        return [("single", singular[0])]

    if file_keys == {"images"}:
        if len(repeated) == 1:
            if labels not in (None, ["single"]):
                raise IntakeContractError("single image labels are invalid")
            return [("single", repeated[0])]
        if len(repeated) == 3 and labels == ["front", "left", "right"]:
            return list(zip(labels, repeated))
        raise IntakeContractError("images must contain one file or the three required angles")

    if file_keys == named_keys:
        if labels not in (None, ["front", "left", "right"]):
            raise IntakeContractError("named angle labels are invalid")
        uploads: list[tuple[str, Any]] = []
        for angle in ("front", "left", "right"):
            files = request.files.getlist(angle)
            if len(files) != 1:
                raise IntakeContractError("each named angle must contain exactly one image")
            uploads.append((angle, files[0]))
        return uploads

    if supplied_named or file_keys:
        raise IntakeContractError("mixed, partial, or unsupported image fields are not accepted")
    raise IntakeContractError("one image or three guided images are required")


def _security_headers(response: Response) -> Response:
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = (
        "camera=(self), microphone=(), geolocation=(), payment=(), usb=()"
    )
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; base-uri 'self'; object-src 'none'; frame-ancestors 'none'; "
        "form-action 'self'; img-src 'self' data: blob:; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com data:; "
        "script-src 'self' 'unsafe-inline'; connect-src 'self'"
    )
    if request.path.startswith("/api/") or response.mimetype == "text/html":
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

    allowed_origin = _cors_origin()
    if allowed_origin:
        response.headers["Access-Control-Allow-Origin"] = allowed_origin
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Max-Age"] = "600"
    response.headers.add("Vary", "Origin")
    return response


def create_app() -> Flask:
    app = Flask(__name__, static_folder=str(PUBLIC_DIR), static_url_path="")
    proxy_hops = _positive_int_env("TRUST_PROXY_HOPS", 1)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=proxy_hops, x_proto=proxy_hops)
    app.config.update(
        MAX_CONTENT_LENGTH=MAX_REQUEST_BYTES,
        JSON_SORT_KEYS=False,
    )
    rate_limit = _positive_int_env("RATE_LIMIT", 25)
    rate_window = _positive_int_env("RATE_WINDOW", 3600)
    configured_secret = os.getenv("RATE_LIMIT_SECRET", "").encode("utf-8")
    limiter = AnalysisRateLimiter(
        rate_limit,
        rate_window,
        configured_secret or secrets.token_bytes(32),
        bucket_cap=_positive_int_env("RATE_LIMIT_BUCKET_CAP", 10_000),
    )
    app.extensions["analysis_rate_limiter"] = limiter
    app.after_request(_security_headers)

    @app.route("/api/health", methods=["GET", "OPTIONS"])
    def health() -> Response | tuple[str, int]:
        if request.method == "OPTIONS":
            return "", 204
        providers = provider_status()
        return jsonify(
            {
                "status": "ok",
                "analysisVersion": ANALYSIS_VERSION,
                "schemaVersion": SCHEMA_VERSION,
                "topicMappingVersion": TOPIC_MAPPING_VERSION,
                "promptVersion": PROMPT_VERSION,
                "promptHash": prompt_hash(),
                "acceptedImageCounts": [1, 3],
                "rateLimit": {"requests": rate_limit, "windowSeconds": rate_window},
                "providers": providers,
                "providerAvailable": any(item["available"] for item in providers),
                "leadCaptureEnabled": False,
                "privacy": {
                    "applicationImageStorage": False,
                    "applicationPiiStorage": False,
                    "openaiStore": False,
                    "externalProviderProcessing": True,
                    "providerAccountRetentionMayApply": True,
                    "captureQualityChecks": (
                        "basic technical heuristics, not clinical confidence"
                    ),
                },
            }
        )

    @app.route("/api/analyze", methods=["POST", "OPTIONS"])
    def analyze_route() -> Response | tuple[Response, int] | tuple[str, int]:
        if request.method == "OPTIONS":
            return "", 204
        if not _request_origin_is_allowed():
            return _request_error("Request origin is not allowed.", 403)
        if request.form.get("age_confirmed", "").strip().lower() != "true":
            return _request_error("Adult confirmation is required.", 403)

        body_area = request.form.get("body_area", "face").strip().lower()
        if body_area not in BODY_AREA_OBSERVATIONS:
            return _request_error("Unsupported body area.", 400)

        try:
            uploads = _extract_uploads()
        except IntakeContractError:
            return _request_error("Invalid image submission.", 400)

        remote_identifier = request.remote_addr or "unknown"
        allowed, retry_after = limiter.consume(remote_identifier)
        if not allowed:
            response, status = _request_error("Too many analysis requests. Try again later.", 429)
            response.headers["Retry-After"] = str(retry_after)
            return response, status

        normalized = []
        try:
            for angle, upload in uploads:
                normalized.append(normalize_image(upload.stream, angle))
        except ImageIntakeError as exc:
            result = build_local_retake(
                image_count=len(uploads),
                issue=exc.issue,
                guidance=exc.guidance,
                message=exc.public_message,
                body_area=body_area,
            )
            return jsonify(result), 422

        try:
            result = analyze(normalized, body_area)
        except ProviderUnavailable:
            return _request_error("Analysis service is temporarily unavailable.", 503)
        except Exception as exc:
            # The class name is enough for operations without exposing request data.
            LOGGER.error("Analyzer request failed (%s)", type(exc).__name__)
            return _request_error("Analysis service is temporarily unavailable.", 503)

        return jsonify(result), 422 if result["status"] == "retake" else 200

    @app.route(
        "/api/<path:unused>",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    def api_not_found(unused: str) -> tuple[Response, int]:
        # This intentionally keeps retired endpoints, including /api/leads, unavailable.
        return _request_error("API endpoint not found.", 404)

    @app.route("/")
    def serve_index() -> Response:
        return send_from_directory(str(PUBLIC_DIR), "index.html")

    @app.route("/privacy")
    def serve_privacy() -> Response:
        return send_from_directory(str(PUBLIC_DIR), "privacy.html")

    @app.route("/<path:path>")
    def serve_static(path: str) -> Response | tuple[str, int]:
        file_path = PUBLIC_DIR / path
        try:
            resolved = file_path.resolve()
            resolved.relative_to(PUBLIC_DIR.resolve())
        except (OSError, ValueError):
            return "Not found", 404
        if file_path.is_file():
            return send_from_directory(str(PUBLIC_DIR), path)
        return send_from_directory(str(PUBLIC_DIR), "index.html")

    @app.errorhandler(413)
    def request_too_large(_: Exception) -> tuple[Response, int]:
        return _request_error("Image submission is too large.", 413)

    @app.errorhandler(500)
    def internal_error(_: Exception) -> tuple[Response, int]:
        return _request_error("The server could not complete the request.", 500)

    return app


app = create_app()


def main() -> None:
    configured = [item["provider"] for item in provider_status() if item["available"]]
    print(
        f"Von & Co visible-surface analyzer {ANALYSIS_VERSION}; "
        f"configured providers: {', '.join(configured) if configured else 'none'}"
    )
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG)
