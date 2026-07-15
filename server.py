"""
Von & Co Aesthetics Skin Analyzer Backend
Flask server for AI-powered skin analysis using Google Gemini
"""

import os
import json
import hashlib
import hmac
import random
import re
import sqlite3
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from html import escape as html_escape
from pathlib import Path
from io import BytesIO

from datetime import datetime
from collections import OrderedDict, defaultdict
import time
import httpx

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

# Load environment variables (override=True so env.txt takes precedence over system vars)
load_dotenv(override=True)
# Also load from env.txt if .env doesn't exist (macOS won't let users create dotfiles in Finder)
if not os.path.exists(os.path.join(Path(__file__).parent, '.env')):
    load_dotenv(os.path.join(Path(__file__).parent, 'env.txt'), override=True)

# Configuration
BASE_DIR = Path(__file__).parent
PUBLIC_DIR = BASE_DIR / "public"

# Initialize Flask app
app = Flask(__name__, static_folder=str(PUBLIC_DIR), static_url_path="")
CORS(app)

@app.after_request
def add_no_cache_headers(response):
    """Prevent browser from caching API responses and HTML."""
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_MODEL = "gemini-3.1-pro-preview"
GOOGLE_TOTAL_BUDGET_MS = int(os.getenv("GOOGLE_TOTAL_BUDGET_MS", str(70_000)))
GOOGLE_HEDGE_DELAY_MS = int(os.getenv("GOOGLE_HEDGE_DELAY_MS", str(15_000)))
GOOGLE_MAX_OUTPUT_TOKENS = int(os.getenv("GOOGLE_MAX_OUTPUT_TOKENS", str(32_768)))
GOOGLE_TRANSIENT_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})

RUNTIME_SOURCE_FILES = (
    "server.py",
    "public/index.html",
    "public/arsenica-regular.otf",
    "public/logo.png",
    "public/logo_clean.png",
    "public/logo_white.png",
)


def _runtime_source_fingerprint():
    """Identify the exact runtime files even when the host injects no release ID."""
    fingerprint = hashlib.sha256()
    for relative_path in RUNTIME_SOURCE_FILES:
        fingerprint.update(relative_path.encode("utf-8"))
        fingerprint.update(b"\0")
        fingerprint.update((BASE_DIR / relative_path).read_bytes())
        fingerprint.update(b"\0")
    return fingerprint.hexdigest()


BUILD_FINGERPRINT = os.getenv("BUILD_FINGERPRINT") or _runtime_source_fingerprint()
ANALYSIS_REPEAT_VERSION = "2026-07-15-v2"
ANALYSIS_MODEL_SEED_VERSION = "2026-07-15-v1"
ANALYSIS_REPEAT_TTL_SECONDS = int(
    os.getenv("ANALYSIS_REPEAT_TTL_SECONDS", str(30 * 24 * 60 * 60))
)
ANALYSIS_REPEAT_MAX_ENTRIES = int(
    os.getenv("ANALYSIS_REPEAT_MAX_ENTRIES", "5000")
)
ANALYSIS_REPEAT_CACHE_PATH = os.getenv(
    "ANALYSIS_REPEAT_CACHE_PATH",
    str(
        BASE_DIR
        / "work"
        / "private-analysis-cache"
        / "analysis-repeat-cache.sqlite3"
    ),
)
EXPOSE_ANALYSIS_REPEAT_HEADER = (
    os.getenv("EXPOSE_ANALYSIS_REPEAT_HEADER", "false").lower() == "true"
)
_analysis_repeat_hmac_key = hashlib.sha256(
    b"von-analysis-repeat\0"
    + (os.getenv("ANALYSIS_REPEAT_HMAC_KEY") or GOOGLE_API_KEY or BUILD_FINGERPRINT).encode(
        "utf-8"
    )
).digest()

# Keep only the completed JSON result in memory and SQLite. Images are not
# retained by the repeat-result cache.
# The cache makes an identical photo + area + age reuse one canonical result
# instead of asking a generative model to improvise a second answer.
_analysis_repeat_cache = OrderedDict()
_analysis_repeat_inflight = set()
_analysis_repeat_condition = threading.Condition(threading.Lock())


def _harden_analysis_repeat_cache_permissions():
    """Restrict result-cache files to the current operating-system account."""
    cache_path = Path(ANALYSIS_REPEAT_CACHE_PATH)
    for protected_path in (
        cache_path,
        Path(f"{cache_path}-wal"),
        Path(f"{cache_path}-shm"),
    ):
        try:
            if protected_path.exists():
                os.chmod(protected_path, 0o600)
        except OSError as permission_error:
            print(
                "[Repeat] Could not restrict cache file permissions: "
                f"{permission_error.__class__.__name__}"
            )


def _open_analysis_repeat_db():
    try:
        cache_path = Path(ANALYSIS_REPEAT_CACHE_PATH)
        cache_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if cache_path.parent.name == "private-analysis-cache":
            os.chmod(cache_path.parent, 0o700)
        connection = sqlite3.connect(
            str(cache_path),
            timeout=5,
            check_same_thread=False,
        )
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_repeat_cache (
                cache_key TEXT PRIMARY KEY,
                stored_at REAL NOT NULL,
                status_code INTEGER NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        connection.commit()
        _harden_analysis_repeat_cache_permissions()
        return connection
    except Exception as cache_error:
        print(f"[Repeat] Persistent cache unavailable: {cache_error}")
        return None


_analysis_repeat_db = _open_analysis_repeat_db()


def _normalized_repeat_age(user_age):
    if not user_age:
        return ""
    try:
        age_value = Decimal(str(user_age).strip())
    except (InvalidOperation, ValueError):
        return str(user_age).strip()
    normalized = format(age_value.normalize(), "f")
    return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized


def _analysis_repeat_key(
    upload_bytes,
    body_area,
    user_age,
):
    """HMAC the photo fingerprint plus every input that affects the result."""
    message = hashlib.sha256()
    for value in (
        ANALYSIS_REPEAT_VERSION,
        BUILD_FINGERPRINT,
        GOOGLE_MODEL,
        body_area,
        _normalized_repeat_age(user_age),
    ):
        message.update(str(value).encode("utf-8"))
        message.update(b"\0")
    message.update(b"server-upload-sha256\0")
    message.update(hashlib.sha256(upload_bytes).digest())
    return hmac.new(
        _analysis_repeat_hmac_key,
        message.digest(),
        hashlib.sha256,
    ).hexdigest()


def _analysis_model_seed(upload_bytes, body_area, user_age):
    """Keep the primary model seed stable when an unchanged app is redeployed."""
    message = hashlib.sha256()
    for value in (
        ANALYSIS_MODEL_SEED_VERSION,
        GOOGLE_MODEL,
        body_area,
        _normalized_repeat_age(user_age),
    ):
        message.update(str(value).encode("utf-8"))
        message.update(b"\0")
    message.update(b"server-upload-sha256\0")
    message.update(hashlib.sha256(upload_bytes).digest())
    return int.from_bytes(message.digest()[:4], "big") & 0x7FFFFFFF


def _independent_rejection_review_seed(analysis_seed, reason_code):
    """Give a validation retry a deterministic but independent model sample."""
    digest = hashlib.sha256(
        b"von-independent-rejection-review\0"
        + str(analysis_seed).encode("ascii")
        + b"\0"
        + str(reason_code or "unknown").encode("utf-8")
    ).digest()
    review_seed = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
    return review_seed if review_seed != analysis_seed else analysis_seed ^ 1


def _disable_persistent_analysis_repeat_cache(cache_error):
    """Keep analysis available when the optional durable cache is unhealthy."""
    global _analysis_repeat_db
    failed_connection = _analysis_repeat_db
    _analysis_repeat_db = None
    print(
        "[Repeat] Persistent cache disabled for this process: "
        f"{cache_error.__class__.__name__}"
    )
    if failed_connection is not None:
        try:
            failed_connection.rollback()
        except Exception:
            pass
        try:
            failed_connection.close()
        except Exception:
            pass


def _read_persistent_analysis_repeat(cache_key, now):
    if _analysis_repeat_db is None:
        return None
    try:
        row = _analysis_repeat_db.execute(
            "SELECT stored_at, status_code, payload_json "
            "FROM analysis_repeat_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        stored_at, status_code, payload_json = row
        if now - stored_at >= ANALYSIS_REPEAT_TTL_SECONDS:
            _analysis_repeat_db.execute(
                "DELETE FROM analysis_repeat_cache WHERE cache_key = ?",
                (cache_key,),
            )
            _analysis_repeat_db.commit()
            return None
        payload = json.loads(payload_json)
        if not isinstance(payload, dict):
            raise ValueError("Cached analysis payload is not an object")
        return stored_at, payload, int(status_code)
    except Exception as cache_error:
        _disable_persistent_analysis_repeat_cache(cache_error)
        return None


def _write_persistent_analysis_repeat(cache_key, analysis, status_code):
    canonical_json = json.dumps(
        analysis,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    canonical_analysis = json.loads(canonical_json)
    now = time.time()
    if _analysis_repeat_db is None:
        return now, canonical_analysis, int(status_code)
    try:
        _analysis_repeat_db.execute(
            "INSERT OR IGNORE INTO analysis_repeat_cache "
            "(cache_key, stored_at, status_code, payload_json) VALUES (?, ?, ?, ?)",
            (cache_key, now, int(status_code), canonical_json),
        )
        _analysis_repeat_db.commit()
        _harden_analysis_repeat_cache_permissions()
        row = _analysis_repeat_db.execute(
            "SELECT stored_at, status_code, payload_json "
            "FROM analysis_repeat_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            raise RuntimeError("Canonical cache row was not readable after insert")
        persisted_analysis = json.loads(row[2])
        if not isinstance(persisted_analysis, dict):
            raise ValueError("Cached analysis payload is not an object")
        return row[0], persisted_analysis, int(row[1])
    except Exception as cache_error:
        _disable_persistent_analysis_repeat_cache(cache_error)
        return now, canonical_analysis, int(status_code)


def _prune_analysis_repeat_cache(now):
    expired = [
        key
        for key, entry in _analysis_repeat_cache.items()
        if now - entry[0] >= ANALYSIS_REPEAT_TTL_SECONDS
    ]
    for key in expired:
        _analysis_repeat_cache.pop(key, None)
    while len(_analysis_repeat_cache) > ANALYSIS_REPEAT_MAX_ENTRIES:
        _analysis_repeat_cache.popitem(last=False)
    if _analysis_repeat_db is not None:
        try:
            cutoff = time.time() - ANALYSIS_REPEAT_TTL_SECONDS
            _analysis_repeat_db.execute(
                "DELETE FROM analysis_repeat_cache WHERE stored_at < ?",
                (cutoff,),
            )
            overflow = _analysis_repeat_db.execute(
                "SELECT COUNT(*) FROM analysis_repeat_cache"
            ).fetchone()[0] - ANALYSIS_REPEAT_MAX_ENTRIES
            if overflow > 0:
                _analysis_repeat_db.execute(
                    "DELETE FROM analysis_repeat_cache WHERE cache_key IN ("
                    "SELECT cache_key FROM analysis_repeat_cache "
                    "ORDER BY stored_at ASC LIMIT ?)",
                    (overflow,),
                )
            _analysis_repeat_db.commit()
            _harden_analysis_repeat_cache_permissions()
        except Exception as cache_error:
            _disable_persistent_analysis_repeat_cache(cache_error)


def _claim_analysis_repeat_key(cache_key):
    """Return a cached result or become the one request allowed to compute it."""
    wait_deadline = time.monotonic() + (
        (GOOGLE_TOTAL_BUDGET_MS / 1000) + 5
    )
    with _analysis_repeat_condition:
        while True:
            now = time.monotonic()
            _prune_analysis_repeat_cache(now)
            cached = _analysis_repeat_cache.get(cache_key)
            if cached is not None:
                _analysis_repeat_cache.move_to_end(cache_key)
                return "hit", deepcopy(cached[1]), cached[2]
            persistent = _read_persistent_analysis_repeat(cache_key, time.time())
            if persistent is not None:
                _analysis_repeat_cache[cache_key] = (
                    time.monotonic(),
                    persistent[1],
                    persistent[2],
                )
                _analysis_repeat_cache.move_to_end(cache_key)
                return "hit", deepcopy(persistent[1]), persistent[2]
            if cache_key not in _analysis_repeat_inflight:
                _analysis_repeat_inflight.add(cache_key)
                return "owner", None, None
            remaining = wait_deadline - now
            if remaining <= 0:
                return "timeout", None, None
            _analysis_repeat_condition.wait(timeout=remaining)


def _release_analysis_repeat_key(cache_key, analysis=None, status_code=None):
    with _analysis_repeat_condition:
        if analysis is not None and status_code is not None:
            stored_at, canonical_analysis, canonical_status = (
                _write_persistent_analysis_repeat(
                    cache_key,
                    analysis,
                    status_code,
                )
            )
            _analysis_repeat_cache[cache_key] = (
                time.monotonic(),
                canonical_analysis,
                canonical_status,
            )
            _analysis_repeat_cache.move_to_end(cache_key)
            _prune_analysis_repeat_cache(time.monotonic())
        _analysis_repeat_inflight.discard(cache_key)
        _analysis_repeat_condition.notify_all()
        if analysis is not None and status_code is not None:
            return deepcopy(canonical_analysis), canonical_status
        return None, None


def _clear_analysis_repeat_cache():
    """Test-only reset hook for process-memory and persisted repeat results."""
    with _analysis_repeat_condition:
        _analysis_repeat_cache.clear()
        _analysis_repeat_inflight.clear()
        if _analysis_repeat_db is not None:
            _analysis_repeat_db.execute("DELETE FROM analysis_repeat_cache")
            _analysis_repeat_db.commit()
        _analysis_repeat_condition.notify_all()


def _reopen_analysis_repeat_cache():
    """Test-only hook that simulates a process restart without losing results."""
    global _analysis_repeat_db
    with _analysis_repeat_condition:
        if _analysis_repeat_db is not None:
            _analysis_repeat_db.commit()
            _analysis_repeat_db.close()
        _analysis_repeat_db = _open_analysis_repeat_db()
        _analysis_repeat_cache.clear()
        _analysis_repeat_inflight.clear()
        _analysis_repeat_condition.notify_all()


def _analysis_json_response(payload, status_code, repeat_state):
    response = jsonify(deepcopy(payload))
    response.status_code = status_code
    if EXPOSE_ANALYSIS_REPEAT_HEADER:
        response.headers["X-Von-Analysis-Repeat"] = repeat_state
    return response


def _google_http_options(timeout_ms):
    """Return a bounded, non-retrying HTTP contract for one model attempt."""
    return genai_types.HttpOptions(
        timeout=timeout_ms,
        retry_options=genai_types.HttpRetryOptions(attempts=1),
    )


class _GoogleResponseError(Exception):
    """The provider returned an empty or malformed structured response."""


ANALYSIS_RESPONSE_SCHEMA = {
    "anyOf": [
        {
            "type": "object",
            "properties": {
                "rejected": {"type": "boolean"},
                "reason": {"type": "string"},
                "observedArea": {
                    "type": "string",
                    "enum": ["face", "neck_chest", "hands", "back", "legs", "other"],
                },
            },
            "required": ["rejected", "reason"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "overallScore": {"type": "integer", "minimum": 0, "maximum": 100},
                "observedArea": {
                    "type": "string",
                    "enum": ["face", "neck_chest", "hands", "back", "legs", "other"],
                },
                "positiveHighlights": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "detail": {"type": "string"},
                        },
                        "required": ["title", "detail"],
                        "additionalProperties": False,
                    },
                },
                "concerns": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "score": {"type": "integer", "minimum": 0, "maximum": 100},
                            "severity": {
                                "type": "string",
                                "enum": ["none", "mild", "moderate", "severe"],
                            },
                            "description": {"type": "string"},
                        },
                        "required": ["score", "severity", "description"],
                        "additionalProperties": False,
                    },
                },
                "recommendations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "treatment": {"type": "string"},
                            "reason": {"type": "string"},
                            "targets": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "priority": {"type": "integer", "minimum": 1},
                        },
                        "required": ["treatment", "reason", "targets", "priority"],
                        "additionalProperties": False,
                    },
                },
                "productRecommendations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "product": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["product", "reason"],
                        "additionalProperties": False,
                    },
                },
                "suggestedCombo": {"type": ["string", "null"]},
                "summary": {"type": "string"},
            },
            "required": [
                "overallScore",
                "observedArea",
                "positiveHighlights",
                "concerns",
                "recommendations",
                "productRecommendations",
                "suggestedCombo",
                "summary",
            ],
            "additionalProperties": False,
        },
    ]
}
PORT = int(os.getenv("PORT", "5002"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
FORCE_DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

# Determine mode
LIVE_MODE = bool(GOOGLE_API_KEY) and not FORCE_DEMO_MODE
MODE = "live" if LIVE_MODE else "demo"

gemini_client = genai.Client(api_key=GOOGLE_API_KEY) if LIVE_MODE else None
if gemini_client:
    print(f"[Gemini] Initialized google-genai client for {GOOGLE_MODEL} with high thinking")

# Rate limiting - protects against API cost abuse
# Max 5 analyses per IP per hour
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "25"))
RATE_WINDOW = 3600  # 1 hour in seconds
rate_tracker = defaultdict(list)

def check_rate_limit(ip):
    """Returns True if request is allowed, False if rate limited"""
    now = time.time()
    # Clean old entries
    rate_tracker[ip] = [t for t in rate_tracker[ip] if now - t < RATE_WINDOW]
    if len(rate_tracker[ip]) >= RATE_LIMIT:
        return False
    rate_tracker[ip].append(now)
    return True

SYSTEM_PROMPT = """You are an expert skin analysis AI for Von & Co Medical Aesthetics Studio, a physician-led medical aesthetics studio in Naples, FL. Our team has over 35 years of combined aesthetic experience. We are open 7 days a week.

Your task is to analyze skin images and provide a preliminary, non-diagnostic preview based only on visible characteristics in the submitted photo. A standard photo is not a VISIA scan and cannot measure subsurface features.

IMPORTANT: You are NOT providing medical diagnoses. You are analyzing visible skin characteristics to suggest aesthetic treatments that may be beneficial.

LANGUAGE RULES: Always say "guest" (never "patient" or "client"). Always say "studio" (never "clinic" or "office"). Always say "provider" (never "technician").

=== IMAGE VALIDATION (CHECK FIRST. BEFORE ANY ANALYSIS!) ===

STEP 1: Is this actually human skin?
If the image is NOT a photo of human skin (e.g. a dog, cat, pet, object, landscape, food, hat, shoe, cartoon, meme, text screenshot, car, furniture, or anything else that is clearly not a person's skin/face/body), REJECT with:
{"rejected": true, "reason": "This doesn't appear to be a photo of skin. Please upload a clear, well-lit photo of the area you'd like analyzed, such as your face, neck, hands, back, or legs."}

STEP 2: Is the photo quality sufficient for a credible analysis?
If the image is too blurry, too dark, or too low resolution to meaningfully analyze, REJECT with:
{"rejected": true, "reason": "We couldn't get a clear enough read on this photo. For the best results, please upload a well-lit, in-focus photo taken at arm's length in natural lighting."}
A clear, adequately lit close-up of the selected face or body area is acceptable whenever enough visible skin surface remains to assess conservatively. Do not require the full head, full limb, or full body. Do not reject merely because the photo is close, tightly cropped, professionally photographed, or not literally taken at arm's length. When the selected area is identifiable and surface features are visible, prefer a conservative completed analysis over a quality rejection.

STEP 3: Does the photo visibly include screenshot UI, a watermark, or a heavy beauty filter?
If the image shows obvious browser chrome, UI elements, status bars, a visible stock-photo watermark, or heavy beauty filters (poreless, airbrushed, Snapchat/Instagram filters with visual effects), REJECT with:
{"rejected": true, "reason": "This photo appears to be filtered or not an original photo. Our AI needs an unfiltered, natural-light photo taken with your camera for an accurate analysis. Beauty filters and screenshots can't give you a reliable result."}
Do not reject a photo merely because it looks professionally photographed. Judge only visible image artifacts.

STEP 4: Is this a minor / child?
If the person in the photo appears to be under 18 years old (a child, preteen, or young teenager), you MUST reject. Medical aesthetic treatments like Botox, lasers, and peels are not appropriate for minors. REJECT with:
{"rejected": true, "reason": "Our skin analysis is designed for adults (18+). Medical aesthetic treatments are not recommended for minors. If you're over 18 and we got it wrong, we apologize! Try a different photo and we'll take another look!"}
This applies even if the user enters an adult age. Trust what you SEE in the photo, not the age they typed.

STEP 5: Do not produce or compare an estimated skin age for an adult guest. The entered age is context only and must never appear as an apparent-age comparison in the result.

STEP 6: Compare the dominant anatomy actually visible with the area selected in the user prompt. For a completed result, set observedArea to the dominant visible area using exactly one of: face, neck_chest, hands, back, legs, other. Do not force the photo to match the selection. If the dominant visible area differs from the selected area, do not fabricate a completed analysis for the selection. Return only {"rejected": true, "reason": "area_mismatch", "observedArea": "<dominant visible area>"}. The server will provide the guest-facing correction.

MEDICAL BOUNDARY: Do not identify, assess, flag, rule out, or comment on lesions or medical conditions. A fixed small-print referral disclaimer is added separately after the analysis.

Do NOT attempt to analyze non-skin images. Do NOT make jokes about non-skin uploads. Just return the rejection JSON.
=== END IMAGE VALIDATION ===

VOICE & TONE: Speak like a knowledgeable, warm aesthetics provider. Be specific about what you see. Use clinical language but keep it accessible. Make the guest feel understood and cared for, not judged. Frame everything positively. Focus on what can be improved and how great they will feel.

COPY STANDARD:
- Write concise, polished American English. Favor one clear, specific sentence over strings of adjectives.
- Be complimentary without sounding gushy or promotional. Do not use "gorgeous," "beautiful," "lovely," "wonderful," "fantastic," "perfect," "amazing," "stunning," "healthy," "highly effective," "potent," "instantly," "go-to," "prevent future," or "absolute best."
- Describe only surface appearance that a standard photo can show. Do not infer skin-barrier health, hydration status, collagen levels, elasticity, underlying support, or general health.
- Do not infer why a feature is present or reconstruct the guest's history from one photo. Never attribute pigment to past, cumulative, chronic, routine, or repeated sun exposure; never infer shaving habits; and never claim volume or collagen loss. Describe only the visible pigment, follicular pattern, contour, lines, crepiness, or surface finish.
- Never describe firmness, elasticity, hydration, moisture, suppleness, softness, or skin thickness as if a standard photograph measured it.
- Every concern description must be a visual observation, not an explanation. Do not say skin is thinning, a skin barrier is intact, spots reflect environmental exposure over time, or bumps come from shaving or razor use.
- In positiveHighlights, concern descriptions, and summary, never label the photographed appearance as rosacea, melasma, acne, dermatitis, eczema, psoriasis, keratosis pilaris, or any cancer. Use non-diagnostic appearance language such as visible redness, pigmentation, surface congestion, visible irritation, texture, or bumpiness.
- In all guest-facing copy, also avoid inferred labels such as hyperpigmentation, photoaging, scarring or scars, and dehydration or dehydrated. Describe only what the photo supports: visible pigment variation, sun-exposure signs, textural marks, or surface dryness.
- Do not praise "bounce," "foundational support," or "natural firmness." Those are not reliable observations from one standard photo.
- Treatment reasons should connect a visible target to the realistic role of the service and leave candidacy to the in-person provider.
- Product reasons should connect a visible target to a realistic role in a home routine. Do not promise a timeline or guaranteed result.

POSITIVE-FIRST REQUIREMENT:
- Begin every completed analysis with 2 or 3 specific positiveHighlights grounded in appealing qualities that are actually visible in the submitted photo.
- State the appealing quality directly. Never phrase a positive as the absence of a concern, such as "redness does not stand out," "no wrinkles," or "minimal pores."
- Each highlight needs a concise title and one natural sentence of detail.
- The summary must open with a genuine positive observation before discussing areas the guest may want to refine.

FORMATTING RULES: NEVER use em dashes (the long dash character). Use commas, periods, colons, or semicolons instead. NEVER use en dashes. Use "to" for ranges. This is a hard requirement for every string in your response.

=== COMPLETE VON & CO TREATMENT MENU ===

LINES & WRINKLES:
- Botox: Botulinum toxin type A treatment. 15-30 min. Placement, dose, onset, and duration must be personalized by a provider.
- Dysport: AbobotulinumtoxinA treatment. 15-30 min. Placement, dose, onset, and duration must be personalized by a provider.
- Xeomin: IncobotulinumtoxinA treatment with an accessory-protein-free formulation. 15-30 min. Placement, dose, onset, and duration must be personalized by a provider.
- Microneedling: Collagen induction therapy. 30-60 min. 3-6 sessions. Builds collagen naturally. Add PRF for enhanced results.
- RF Microneedling: Needles + radiofrequency energy. 45-60 min. Deeper collagen remodeling. 2-4 sessions. Tightens + resurfaces simultaneously.

VOLUME LOSS & CONTOURING:
- Dermal Fillers: HA filler (Versa, Lyft, Contour, Kysse, Refyne, Defyne) for cheeks, jaw, temples. 30-60 min. Immediate results + collagen stimulation over time. Lasts 6-24 mo. "Putting the scaffolding back."
- Sculptra: PLLA biostimulatory injectable. 60 min. Not an HA filler. Results develop gradually over a treatment series; candidacy and expected duration require an in-person provider assessment.

LIP ENHANCEMENT:
- Lip Filler (Versa): HA filler for lips. 30-60 min. "We enhance, not exaggerate." 1-2 sessions for ideal shape. ~12 mo duration.

SKIN GLOW & TEXTURE:
- HydraFacial Clarifying: Deep cleanse + exfoliation + extraction. 60 min. Targets congestion/breakouts and includes an acne-focused blue-light step. Ideal for oily/acne-prone skin. Great entry point.
- HydraFacial Customized: Dermaplaning + LED + booster serum. 70 min. Exfoliation and the booster are tailored to the guest's visible goals.
- HydraFacial Elite: VIP: lymphatic drainage, massage, aroma. 80 min. All Customized steps PLUS scalp/face/arm massage, ice globes, aromatherapy. Exclusive to V&C.
- SaltFacial: Sea salt exfoliation + ultrasound + LED. 60 min. An accessible facial option for new guests.
- SkinVive: Micro-droplet HA skin quality injectable. 30-60 min. NOT a filler. Texture + hydration enhancer. Dewy glow from within. Single session. Lasts 6-9 months.
- Sciton Moxi: Gentle fractionated laser. 60 min. "The weekend laser." 2-3 day recovery. Candidacy and settings must be confirmed by a provider for each guest. 3-4 sessions. "Treat Friday, glow Monday."

ACNE, SCARS & PORES:
- Chemical Peels: Controlled superficial exfoliation. 45-60 min. VI Peel Precision, VI Peel Purify, and VI Peel Advanced are provider-selected peel options for the guest's visible goals. Series may be discussed in person.
- Microneedling + PRF: Collagen induction + growth factors. 30-60 min. PRF is added to the microneedling protocol to support collagen renewal. 3-6 sessions.
- Deep Pore Facial: Classic deep cleansing facial. 45-60 min. An accessible first facial with extractions and moisturizing steps.
- Signature Facial: Customized cleansing + mask + massage. 45-60 min. A maintenance facial for returning guests.
- Anti-Aging Facial: Resurfacing + deep hydration facial. 50-60 min. Targets visible aging signs. Ideal for mature skin.
- RF Microneedling: Microneedling + radiofrequency. 45-60 min. Next-level scar reduction + skin tightening. 2-4 sessions for stubborn scarring.

SUN DAMAGE & PIGMENTATION:
- Sciton BBL: Broadband light photofacial. 30-60 min. "Lunchtime laser." Redness, sun spots, rosacea. Face + body. Quick recovery. Pair with Halo = Hero Combo.
- Sciton Halo: Hybrid fractional laser (ablative + non-ablative). 60 min. Used for appropriate candidates with visible texture, pigment, pores, tone, wrinkles, firmness, or scarring concerns. Typical course and recovery vary by plan and settings.
- Chemical Peels: Controlled surface exfoliation, including provider-selected VI Peel options. 45-60 min. An option for visible tone variation, pores, and surface discoloration after provider evaluation.

SKIN TIGHTENING & FIRMING:
- RF Microneedling: 45-60 min. Tightens + resurfaces. 2-4 sessions. Minimal downtime. Great for jowls, neck.
- Sculptra: 60 min. Rebuilds structure from within. Lasts 2+ years. Face + body (arms, chest, buttocks).
- Sciton Halo: 60 min. Deep collagen remodeling + surface renewal. 1-2 treatments.

DOUBLE CHIN & JAWLINE:
- Kybella: Deoxycholic acid injection for appropriate submental-fullness candidates. 60 min. A provider must confirm candidacy, treatment course, risks, and expectations in person.

UNWANTED HAIR:
- Laser Hair Removal: Light-based treatment used for long-term hair reduction. 30-60 min. Treatment count, interval, and candidacy vary by area, hair, and skin type and must be confirmed by a provider.

HAIR THINNING & BROW SHAPING:
- Hair Restoration (PRF): Platelet-rich fibrin into scalp. 30 min. Next-gen of PRP: sustained growth factor release up to 1 week. Every 3-6 months.
- Brow Lamination: Semi-permanent brow styling. 60 min. "The brow perm." Instant fullness. Lasts 6-8 wks.

COMBO PLAYS (recommend these stacks when multiple concerns align):
- New Guest Starter: VISIA scan + HydraFacial + toxin consult
- Anti-Aging Power: Toxin (wait 14 days) + Filler + Moxi laser series
- Glow-Up Package: HydraFacial Elite + SkinVive + daily SPF
- Scar Reduction: Chemical Peel then Microneedling + PRF series
- Hero Combo: BBL + Halo. Surface clearing + deep remodeling
- Full Rejuvenation: Halo + Sculptra + Toxin maintenance plan

=== VON & CO SKINCARE PRODUCTS (recommend all and only clearly mapped options, including SPF) ===

CONCERN-TO-PRODUCT MAP:
Wrinkles/Fine Lines: SkinBetter AlphaRet (primary) or ZO Wrinkle+Texture Repair
Redness/Rosacea: Avene Thermal Water (primary) + Avene Cicalfate+ or Alastin HydraTint
Dark Spots/Pigmentation: SkinBetter Even Tone (primary) or ISDIN Melaclear Advanced
Uneven Tone/Dullness: ZO 10% Vitamin C (primary) or Hydrinity Vivid Serum
Texture/Roughness: ZO Complexion Renewal Pads (primary) or SkinBetter Peel Pads
Firmness/Laxity: Alastin Restorative Skin Complex (primary) or ZO Growth Factor Serum
Dehydration/Dryness: Hydrinity Renewing HA Serum (primary) or SkinBetter Trio Moisture
Crow's Feet/Eye Area: ALASTIN Restorative Eye Treatment or ZO Growth Factor Eye
Sun Protection (ALWAYS): Colorescience Face Shield SPF 50 (primary), or ISDIN Eryfotona Actinica, or SkinBetter Sunbetter SPF 68

GOLDEN RULES FOR PRODUCT RECS:
- ALWAYS include an SPF product in the productRecommendations array
- Recommend only products that clearly map to visible concerns in the selected body area. Do not add unrelated products simply to increase the count.
- Post-procedure sequence: Alastin Skin Nectar + Hydrinity Hyacyn Mist + Avene Cicalfate+ first, then transition to regular routine

=== THE CLUB MEMBERSHIP ===
$149/month or $1,499/year. All dues convert to Club Funds. Spend on any treatment or product.
20% off lasers, hair removal, microneedling, hair restoration.
15% off injectables, facials, peels, brow lamination, skincare products.
Funds can be banked, shared, or gifted. Member-only events + exclusive specials.
First-time guests: 15% off their first visit, treatments + same-day skincare.

=== ANALYSIS INSTRUCTIONS ===

Analyze the skin image and respond with ONLY valid JSON (no markdown, no code blocks, no explanation text). The JSON must follow this exact structure:

{
  "overallScore": <0-100 integer>,
  "observedArea": "<face|neck_chest|hands|back|legs|other based on what is actually visible>",
  "positiveHighlights": [
    {
      "title": "<concise name for a visible strength>",
      "detail": "<specific, complimentary observation grounded in the photo>"
    }
  ],
  "concerns": {
    "wrinkles": {"score": <0-100>, "severity": "<none|mild|moderate|severe>", "description": "<warm, specific description of what you see>"},
    "redness": {"score": <0-100>, "severity": "<none|mild|moderate|severe>", "description": "<warm, specific description>"},
    "darkSpots": {"score": <0-100>, "severity": "<none|mild|moderate|severe>", "description": "<warm, specific description>"},
    "texture": {"score": <0-100>, "severity": "<none|mild|moderate|severe>", "description": "<warm, specific description>"},
    "pores": {"score": <0-100>, "severity": "<none|mild|moderate|severe>", "description": "<warm, specific description>"},
    "laxity": {"score": <0-100>, "severity": "<none|mild|moderate|severe>", "description": "<warm, specific description of visible contour definition or crepiness only>"},
    "sunDamage": {"score": <0-100>, "severity": "<none|mild|moderate|severe>", "description": "<warm, specific description of visible sun-exposure signs>"},
    "unevenTone": {"score": <0-100>, "severity": "<none|mild|moderate|severe>", "description": "<warm, specific description of tone uniformity>"}
  },
  "recommendations": [
    {
      "treatment": "<exact treatment name from menu above>",
      "reason": "<specific, warm explanation of why. Reference what you saw in their skin>",
      "targets": ["<concern1>", "<concern2>"],
      "priority": <positive integer, 1=highest>
    }
  ],
  "productRecommendations": [
    {
      "product": "<exact product name from skincare menu above>",
      "reason": "<one-sentence warm explanation of why this product helps their specific concern>"
    }
  ],
  "suggestedCombo": "<suggest ONE combo play only when every named treatment is also present in recommendations; otherwise null>",
  "summary": "<2-3 warm, encouraging sentences about their visible skin and a suggested treatment journey. For a face result, mention a VISIA consultation. For any other area, mention an in-person consultation.>"
}

CONCERN SCORE GUIDELINES (these are severity scores, NOT health scores. Higher = worse):
- 0-10: No visible concern at all
- 11-25: Minimal, barely visible
- 26-40: Mild, some signs present
- 41-60: Moderate, clearly visible
- 61-80: Significant, very noticeable
- 81-100: Advanced, prominent concern

USE THE FULL 0-100 RANGE FOR EACH CONCERN. Most people will have a mix: some concerns near 10-20, others at 40-60, maybe one at 70+. DO NOT score all 8 concerns in the same narrow band. Differentiate. If someone has great texture but bad sun damage, texture should be 10-15 and sunDamage should be 55-75. The overallScore is recalculated server-side from your concern scores, so focus on making each individual concern score accurate to what you actually see.

EXAMPLES OF REALISTIC CONCERN SPREADS:
- Low visible-concern example: wrinkles 5, redness 12, darkSpots 8, texture 10, pores 22, laxity 3, sunDamage 15, unevenTone 18 (avg ~12, overall ~88)
- Moderate mixed example: wrinkles 25, redness 35, darkSpots 42, texture 20, pores 30, laxity 15, sunDamage 48, unevenTone 28 (avg ~30, overall ~70)
- Higher visible-concern example: wrinkles 62, redness 45, darkSpots 72, texture 55, pores 40, laxity 58, sunDamage 78, unevenTone 50 (avg ~58, overall ~42)
- Varied example: wrinkles 35, redness 18, darkSpots 55, texture 28, pores 45, laxity 22, sunDamage 60, unevenTone 40 (avg ~38, overall ~62)

=== CREDIBILITY GUARDRAILS (CRITICAL FOR TRUST) ===

1. MAKE EACH CONCERN SCORE REFLECT WHAT YOU ACTUALLY SEE. Do not default to safe middle scores. If a concern is barely visible, score it under 15. If it is very noticeable, score it above 55. Each of the 8 concerns should be scored independently. The overallScore is computed automatically from your concern scores, so do not worry about what the overall number will be. Just score each concern honestly.

2. If a concern is not actually visible, keep its score at 10 or below. Never raise a score merely to create a recommendation.

3. Be SPECIFIC about what you actually see. Don't use generic filler descriptions. Reference specific areas of the face/body, visible features, and real observations. Vague analysis = not credible.

4. The positiveHighlights must describe attractive qualities actually visible in the submitted image. Do not invent praise and do not describe only the absence of a concern.

5. Treatment recommendations must match the actual severity. Don't recommend Halo (aggressive laser) for barely visible concerns. Match treatment intensity to severity.

6. If the overall skin looks good, lead with positivity. Frame recommendations as maintenance and support, not correction or prevention promises.

7. Keep the numeric score and written evidence internally consistent. A score of 41 or higher means the feature is clearly visible and moderate. If the most accurate description is slight, subtle, minimal, faint, barely visible, minor, or very mild, keep the score at 40 or below. Do not use a moderate score for a natural fold or line created by pose, rotation, extension, or camera angle.
=== END CREDIBILITY GUARDRAILS ===

RECOMMENDATION RULES:
- Recommend all and only TREATMENTS with a specific visible rationale. There is no fixed minimum or maximum. Do not pad a plan to hit a count, and do not omit a distinct clearly supported option merely to hit a count.
- Recommend all and only SKINCARE PRODUCTS that map to a visible concern, plus an SPF. There is no fixed count. Do not add unrelated products.
- Use ONLY the exact treatment and product names from the menus above
- Every treatment target must be a visible concern scored above 10 and must be one that treatment actually addresses
- Priority 1 must address the highest-scoring treatment-eligible concern; priorities 1 and 2 together must address the two highest-scoring treatment-eligible concerns
- Cover every concern scored 41 or higher. When no concern reaches 41, cover at least the single highest concern scored above 10
- Prioritize treatments that address multiple concerns simultaneously
- When it directly matches a visible concern, favor an accessible entry point such as HydraFacial Clarifying, HydraFacial Customized, Deep Pore Facial, Signature Facial, or SaltFacial
- Recommend Sciton Halo only when at least one of its listed targets scores 41 or higher
- Recommend Laser Hair Removal only when hairRemoval scores 41 or higher AND the photo shows treatment-relevant evidence such as clearly visible dark or coarse body hair, stubble, distinct visible follicles, or follicular contrast. Fine, very fine, natural, minimal, vellus, or peach-fuzz hair is not enough. Scalp hair draped over the back or shoulders is not body-hair evidence.
- When redness scores 41 or higher, do not recommend RF Microneedling or standard Microneedling
- Return suggestedCombo as one exact combo name from the menu only when every component is present in recommendations; otherwise return null

TREATMENT PRIORITY BY CONCERN:

Visible Redness:
  1. Sciton BBL (focused option for visible vascular redness and broken capillaries, with provider-confirmed candidacy)
  2. HydraFacial Clarifying (gentle cleansing and exfoliation option)
  3. Signature Facial (provider-personalized soothing facial option)
  Products: Avene Thermal Water + Avene Cicalfate+ or Alastin HydraTint

Dark Spots & Sun Damage:
  1. Sciton BBL (targets melanin and pigment directly)
  2. Sciton Halo (deeper combined resurfacing for stubborn pigmentation)
  3. Chemical Peels / VI Peel (surface pigment)
  4. Sciton Moxi (gentle option for mild discoloration)
  Products: SkinBetter Even Tone, ISDIN Melaclear Advanced, Colorescience Face Shield SPF 50

Wrinkles & Fine Lines:
  1. Botox / Dysport / Xeomin (for visible expression lines on the forehead, around the eyes, or between the brows)
  2. Sciton Halo or Sciton Moxi (for textural fine lines and skin renewal)
  3. RF Microneedling (for deeper skin tightening and collagen)
  4. Dermal Fillers (for static lines and volume-related folds)
  Products: SkinBetter AlphaRet, ZO Wrinkle+Texture Repair

Skin Texture & Smoothness:
  1. Sciton Halo (focused full-resurfacing option for appropriate candidates)
  2. Sciton Moxi (gentler resurfacing, great for maintenance)
  3. HydraFacial Customized (hydration + exfoliation + booster)
  4. RF Microneedling (tightening + collagen stimulation)
  Products: ZO Complexion Renewal Pads, SkinBetter Peel Pads

Pore Size:
  1. Microneedling (collagen induction can soften the appearance of pores)
  2. RF Microneedling (deeper collagen remodeling)
  3. SaltFacial (exfoliation + LED)
  4. HydraFacial Clarifying (deep cleansing)
  Products: ZO Complexion Renewal Pads

Volume Loss & Sagging:
  1. Sculptra (gradual collagen building, lasts 2+ years)
  2. Dermal Fillers (immediate volume (cheeks, jaw, temples)

Skin Laxity & Firmness:
  1. RF Microneedling (focused tightening option for appropriate candidates)
  2. Sculptra (rebuilds structure from within, face + body)
  3. Sciton Halo (deep collagen remodeling + surface renewal)
  Products: Alastin Restorative Skin Complex, ZO Growth Factor Serum

Uneven Skin Tone:
  1. Sciton BBL (evens tone by targeting pigment)
  2. Chemical Peels / VI Peel (resurfacing for tone correction)
  3. Sciton Moxi (gentle laser for mild unevenness)
  Products: SkinBetter Even Tone, ZO 10% Vitamin C

Overall Skin Quality / Prejuvenation:
  1. HydraFacial Customized (accessible option, with the protocol personalized in person)
  2. SkinVive (injectable hydration. Dewy glow)
  3. Sciton Moxi (gentle laser prejuvenation)
  4. SaltFacial (great gateway treatment for new guests)
  Products: Hydrinity Renewing HA Serum, SkinBetter Alto Defense
"""


def generate_demo_analysis(body_area="face"):
    """Generate clearly labeled sample results with bounded appearance copy."""
    # Pick a random skin profile to ensure score variety
    profile = random.choice(["excellent", "good", "average", "moderate", "significant"])
    profile_offsets = {"excellent": -20, "good": -10, "average": 0, "moderate": 15, "significant": 25}
    offset = profile_offsets[profile]

    # Area-specific concern templates with wide base ranges
    area_concerns = {
        "face": {
            "wrinkles": {"range": (5, 75), "descriptions": [
                "Fine lines are visible around the eyes and forehead",
                "Minimal fine lines visible with certain expressions",
                "Deep wrinkles visible across the forehead and around the eyes",
                "Minimal visible lines appear in the photographed area"]},
            "redness": {"range": (5, 65), "descriptions": [
                "Some visible redness appears through the central face",
                "Mild redness in the cheek and nose area",
                "Noticeable redness and flushing across the cheeks and nose",
                "Skin tone is generally even with minimal redness"]},
            "darkSpots": {"range": (5, 75), "descriptions": [
                "Visible brown spots and pigment variation appear across the complexion",
                "A few dark spots are visible across the photographed area",
                "Multiple brown spots and areas of pigment variation are visible",
                "Minimal pigment variation is visible"]},
            "texture": {"range": (5, 65), "descriptions": [
                "Texture is generally smooth with minor imperfections",
                "Some roughness and surface dryness are visible",
                "Rough, uneven texture with visible textural marks",
                "Skin texture appears smooth and refined"]},
            "pores": {"range": (5, 70), "descriptions": [
                "Pores are subtly visible across the photographed area",
                "Some enlarged pores visible in the T-zone",
                "Enlarged pores very visible across the nose and cheeks",
                "Pores appear softly diffused"]},
            "laxity": {"range": (5, 65), "descriptions": [
                "The facial contours look softly defined",
                "Mild contour softness is visible along the jawline",
                "Noticeable contour softness is visible along the jawline and cheeks",
                "The facial contours look smooth and well defined"]},
            "sunDamage": {"range": (5, 75), "descriptions": [
                "Some visible freckling and pigment variation appear across the complexion",
                "Mild sun-exposure signs are visible, particularly across the cheeks",
                "Pronounced freckling and multiple brown spots are visible",
                "Minimal sun-exposure signs are visible"]},
            "unevenTone": {"range": (5, 65), "descriptions": [
                "Some tonal variation visible across different facial zones",
                "Mild unevenness in skin tone, particularly around the chin and forehead",
                "Pronounced tonal unevenness across the face",
                "Skin tone is largely uniform with minimal variation"]}
        },
        "neck_chest": {
            "sunDamage": {"range": (25, 65), "descriptions": [
                "Freckling and brown spots are visible across the décolletage",
                "Moderate visible sun-exposure signs with brown spots on the chest",
                "Mild sun-exposure signs are visible on the neck and upper chest"]},
            "laxity": {"range": (20, 55), "descriptions": [
                "A pronounced crepey appearance is visible",
                "Mild contour softness is visible, especially on the neck",
                "The visible contours look smooth and softly defined"]},
            "redness": {"range": (10, 45), "descriptions": [
                "Redness and flushing visible on the chest area",
                "Mild diffuse redness across the décolletage",
                "Minimal redness observed"]},
            "texture": {"range": (15, 50), "descriptions": [
                "Rough surface texture is visible across the photographed area",
                "Mild textural irregularities on the chest",
                "Texture is relatively smooth"]},
            "wrinkles": {"range": (15, 50), "descriptions": [
                "Horizontal neck lines (necklace lines) are visible",
                "Fine lines are visible on the chest",
                "Minimal wrinkling in this area"]}
        },
        "hands": {
            "sunDamage": {"range": (25, 65), "descriptions": [
                "Brown spots and freckling are visible on the backs of the hands",
                "Several visible brown spots appear across the backs of the hands",
                "Mild freckling and a few brown spots are visible"]},
            "laxity": {"range": (20, 60), "descriptions": [
                "A crepey-looking surface appears with visible tendons and veins",
                "Visible contour softness accompanies prominent veins and tendons",
                "The contours look smooth and naturally defined"]},
            "texture": {"range": (15, 50), "descriptions": [
                "Rough, dry texture on the backs of the hands",
                "Mild textural variation is visible on the backs of the hands",
                "Texture is relatively smooth"]},
            "veins": {"range": (15, 50), "descriptions": [
                "Veins and tendons are prominently visible",
                "Some veins are visible across the backs of the hands",
                "Veins are minimally visible"]},
            "dryness": {"range": (15, 45), "descriptions": [
                "Significant surface dryness is visible",
                "Moderate dryness especially around the knuckles",
                "The surface looks smooth with little visible flaking"]}
        },
        "back": {
            "acne": {"range": (20, 60), "descriptions": [
                "Visible breakout-like bumps and surface congestion appear on the back",
                "Mild breakout-like bumps are visible",
                "Minimal congestion observed"]},
            "scarring": {"range": (15, 50), "descriptions": [
                "Visible textural marks appear across the photographed area",
                "Mild textural marks and dark spots are visible",
                "Minimal textural marking is visible"]},
            "texture": {"range": (15, 50), "descriptions": [
                "Rough, uneven texture across the upper back",
                "Mild textural irregularities",
                "Texture is generally smooth"]},
            "unevenTone": {"range": (15, 45), "descriptions": [
                "Noticeable pigment variation and dark marks are visible",
                "Mild tone variation and several dark marks are visible",
                "Skin tone is relatively even"]},
            "hairRemoval": {"range": (10, 40), "descriptions": [
                "Unwanted hair growth on the upper back",
                "Moderate hair growth is visible across the upper back",
                "Minimal visible hair growth"]}
        },
        "legs": {
            "veins": {"range": (20, 55), "descriptions": [
                "Spider veins visible on the thighs and calves",
                "Mild spider veins beginning to appear",
                "Minimal visible veins"]},
            "texture": {"range": (15, 45), "descriptions": [
                "Rough, bumpy surface texture is visible",
                "Mild textural irregularities",
                "Texture is generally smooth"]},
            "sunDamage": {"range": (15, 45), "descriptions": [
                "Visible dark spots and freckling appear on the shins",
                "Mild freckling is visible across the photographed area",
                "Minimal sun-exposure signs are visible"]},
            "hairRemoval": {"range": (15, 50), "descriptions": [
                "Visible hair growth appears across the photographed area",
                "Moderate hair growth is visible",
                "Minimal visible hair growth"]},
            "dryness": {"range": (15, 45), "descriptions": [
                "Dry, flaky skin especially on the shins",
                "Moderate dryness visible",
                "The visible surface has an even, smooth-looking finish"]}
        }
    }

    # Generate concerns for the selected body area, shifted by skin profile
    area_data = area_concerns.get(body_area, area_concerns["face"])
    concerns = {}
    for key, template in area_data.items():
        lo, hi = template["range"]
        shifted_lo = max(2, lo + offset)
        shifted_hi = min(90, hi + offset)
        if shifted_lo >= shifted_hi:
            shifted_lo = max(2, shifted_hi - 15)
        score = random.randint(shifted_lo, shifted_hi)
        # Pick description that matches score level
        descs = template["descriptions"]
        if score <= 25:
            desc = descs[-1] if len(descs) > 1 else descs[0]  # lowest severity desc
        elif score >= 55:
            desc = descs[-2] if len(descs) > 2 else descs[0]  # high severity desc
        else:
            desc = random.choice(descs[:2]) if len(descs) > 1 else descs[0]
        concerns[key] = {
            "score": score,
            "description": desc
        }

    # Use the same score-to-severity bands as completed live results.
    for key in concerns:
        s = concerns[key]["score"]
        concerns[key]["severity"] = "none" if s <= 10 else "mild" if s <= 40 else "moderate" if s <= 60 else "severe"

    # Calculate overall score with the same transparent live formula.
    avg_concern = sum(c["score"] for c in concerns.values()) / len(concerns)
    base_score = max(0, min(100, int(round(100 - avg_concern))))

    # Sort concerns by score (highest = most visible = treat first)
    ranked = sorted(concerns.items(), key=lambda x: x[1]["score"], reverse=True)

    positive_copy = {
        "wrinkles": ("Graceful expression", "Your natural expression gives the area warmth and character."),
        "redness": ("Balanced-looking tone", "The visible tone has a fresh, naturally balanced quality."),
        "darkSpots": ("Clear-looking complexion", "The complexion has an appealing clarity worth preserving."),
        "texture": ("Refined texture", "The visible surface appears smooth and polished."),
        "pores": ("Smooth-looking finish", "The skin has a refined, even-looking finish."),
        "laxity": ("Contour definition", "The visible contours have a softly defined, balanced appearance."),
        "sunDamage": ("Luminous quality", "The visible skin has a bright, luminous quality."),
        "unevenTone": ("Harmonious tone", "The overall tone looks harmonious and composed."),
        "veins": ("Elegant definition", "The area has a naturally elegant definition."),
        "dryness": ("Smooth-looking surface", "The visible surface has a smooth-looking, even finish."),
        "acne": ("Fresh foundation", "The surrounding skin has a fresh-looking foundation."),
        "scarring": ("Even-looking surface", "The visible surface has areas of smooth, even texture."),
        "hairRemoval": ("Clean visual finish", "The photographed area has a clean, cohesive visual finish."),
    }
    positive_highlights = [
        {"title": positive_copy[key][0], "detail": positive_copy[key][1]}
        for key, _ in sorted(concerns.items(), key=lambda item: item[1]["score"])[:2]
    ]

    # Use the same conservative exact-catalog fallback choices as live results.
    # One supported service may cover multiple visible concerns; there is no
    # arbitrary recommendation cap.
    area = body_area if body_area in AREA_CONCERN_KEYS else "face"
    concern_scores = {
        key: value["score"]
        for key, value in concerns.items()
    }
    recommendations_by_treatment = {}
    for concern_key, concern_data in ranked:
        if concern_data["score"] < 20:
            continue
        treatment = _fallback_treatment_for_target(
            area,
            concern_key,
            concern_scores,
        )
        if treatment is None:
            continue
        existing = recommendations_by_treatment.get(treatment)
        if existing is None:
            recommendations_by_treatment[treatment] = [concern_key]
        elif concern_key not in existing:
            existing.append(concern_key)

    recs = []
    for treatment, targets in recommendations_by_treatment.items():
        recs.append({
            "treatment": treatment,
            "reason": _fallback_treatment_reason(treatment, targets),
            "targets": targets,
            "priority": len(recs) + 1,
        })

    # Use exact current product names and bounded discussion language. There is
    # no arbitrary product cap; duplicate products remain de-duplicated.
    product_recs = []
    used_products = set()
    for concern_key, concern_data in ranked:
        if concern_data["score"] < 15:
            continue
        products = FALLBACK_PRODUCTS_BY_CONCERN.get(concern_key, ())
        if not products:
            continue
        product = products[0]
        if product in used_products or product not in _product_names_for_area(area):
            continue
        product_recs.append({
            "product": product,
            "reason": _fallback_product_reason(product, concern_key),
        })
        used_products.add(product)

    spf_product = "Colorescience Face Shield SPF 50"
    if spf_product not in used_products:
        product_recs.append({
            "product": spf_product,
            "reason": (
                "A daily broad-spectrum SPF is a simple baseline for exposed "
                "skin, and a provider can help choose the formula that best "
                "fits your routine."
            ),
        })

    # A generated sample should not imply that a named package is appropriate.
    suggested_combo = None

    # Build a natural summary
    area_labels = {"face": "face", "neck_chest": "neck and chest", "hands": "hands", "back": "back", "legs": "legs"}
    area_label = area_labels.get(body_area, "skin")
    concern_names_map = {
        "wrinkles": "visible lines", "redness": "visible redness",
        "darkSpots": "visible pigment variation", "texture": "visible texture",
        "pores": "pore visibility", "laxity": "visible contour softness",
        "sunDamage": "visible sun-exposure signs", "veins": "visible vascularity",
        "scarring": "visible textural marks", "hairRemoval": "visible hair growth",
        "acne": "visible surface congestion", "dryness": "surface dryness",
        "unevenTone": "visible tone variation"
    }
    top_two = [concern_names_map.get(r[0], r[0]) for r in ranked[:2] if r[1]["score"] > 20]
    top_treatments = [r["treatment"] for r in recs[:2]]
    positive_opening = positive_highlights[0]["detail"]

    if top_two and top_treatments:
        summary = (
            f"{positive_opening} This sample for the {area_label} highlights "
            f"{' and '.join(top_two)}. The service ideas below are starting "
            "points only. An in-person consultation is needed to confirm fit."
        )
    else:
        summary = (
            f"{positive_opening} This sample shows how a simple maintenance "
            "plan may be presented. An in-person consultation is needed to "
            "confirm fit."
        )

    return {
        "overallScore": base_score,
        "positiveHighlights": positive_highlights,
        "concerns": concerns,
        "recommendations": recs,
        "productRecommendations": product_recs,
        "suggestedCombo": suggested_combo,
        "summary": summary
    }


def allowed_file(filename):
    """Check if file type is allowed"""
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Lead storage (in-memory for now, can upgrade to DB/CRM later)
leads = []

@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "buildFingerprint": BUILD_FINGERPRINT,
        "hedgeDelayMs": GOOGLE_HEDGE_DELAY_MS,
        "maxOutputTokens": GOOGLE_MAX_OUTPUT_TOKENS,
        "status": "ok",
        "mode": MODE,
        "model": GOOGLE_MODEL,
        "thinkingLevel": "HIGH",
        "totalBudgetMs": GOOGLE_TOTAL_BUDGET_MS,
    })


@app.route("/api/lead", methods=["POST"])
def capture_lead():
    """Capture lead information from the lead gate form"""
    try:
        data = request.get_json()
        name = data.get("name", "").strip()
        email = data.get("email", "").strip()
        phone = data.get("phone", "").strip()

        if not name or not email:
            return jsonify({"error": "Name and email are required"}), 400

        lead = {
            "name": name,
            "email": email,
            "phone": phone,
            "timestamp": datetime.now().isoformat(),
            "ip": request.headers.get('X-Forwarded-For', request.remote_addr)
        }
        leads.append(lead)
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/leads", methods=["GET"])
def get_leads():
    """View captured leads (protected by simple token)"""
    token = request.args.get("token", "")
    expected = os.getenv("ADMIN_TOKEN", "")
    if not expected or not hmac.compare_digest(token, expected):
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify({"leads": leads, "total": len(leads)})


@app.route("/api/report", methods=["POST"])
def generate_report():
    """Generate a branded Von & Co treatment plan one-pager as HTML"""
    try:
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            data = {}
        analysis = data.get("analysis", {})
        if not isinstance(analysis, dict):
            analysis = {}
        name = html_escape(str(data.get("name", "Guest")))

        overall = html_escape(str(analysis.get("overallScore", "...")))
        positive_highlights = analysis.get("positiveHighlights", [])
        if not isinstance(positive_highlights, list):
            positive_highlights = []
        summary = html_escape(str(analysis.get("summary", "")))
        concerns = analysis.get("concerns", {})
        if not isinstance(concerns, dict):
            concerns = {}
        recs = analysis.get("recommendations", [])
        if not isinstance(recs, list):
            recs = []
        products = analysis.get("productRecommendations", [])
        if not isinstance(products, list):
            products = []
        asset_root = request.host_url.rstrip("/")

        concern_labels = {
            "wrinkles": "Wrinkles & Fine Lines",
            "redness": "Visible Redness",
            "darkSpots": "Dark Spots & Pigment Variation",
            "texture": "Skin Texture & Smoothness",
            "pores": "Pore Size & Visibility",
            "laxity": "Skin Laxity & Crepiness",
            "sunDamage": "Visible Sun-Exposure Signs",
            "veins": "Visible Veins",
            "scarring": "Visible Textural Marks",
            "hairRemoval": "Unwanted Hair",
            "acne": "Visible Breakouts & Congestion",
            "dryness": "Visible Dryness",
            "unevenTone": "Uneven Skin Tone",
        }

        # Build concerns HTML rows
        concern_rows = ""
        for key, c in concerns.items():
            if not isinstance(c, dict):
                continue
            label = html_escape(str(concern_labels.get(key, key)))
            try:
                score = float(c.get("score", 0))
            except (TypeError, ValueError):
                score = 0
            severity = html_escape(str(c.get("severity", "none")).upper())
            desc = html_escape(str(c.get("description", "")))
            color = "#C0C8A9" if score <= 30 else "#C68D2F" if score <= 60 else "#C58B74"
            concern_rows += f"""
            <tr class="report-result-row">
                <td data-label="Concern" style="padding:10px 12px; border-bottom:1px solid #F2F3F5; font-weight:500; color:#4C4C4E;">{label}</td>
                <td data-label="Level" style="padding:10px 12px; border-bottom:1px solid #F2F3F5; text-align:center;">
                    <span style="display:inline-block; padding:3px 10px; border-radius:12px; background:{color}20; color:{color}; font-weight:600; font-size:0.85em;">{severity}</span>
                </td>
                <td data-label="Assessment" style="padding:10px 12px; border-bottom:1px solid #F2F3F5; color:#4C4C4E; font-size:0.9em;">{desc}</td>
            </tr>"""

        # Build recommendations HTML
        rec_items = ""
        for i, rec in enumerate(recs, 1):
            if not isinstance(rec, dict):
                continue
            treatment = html_escape(str(rec.get("treatment", "")))
            reason = html_escape(str(rec.get("reason", rec.get("description", ""))))
            rec_items += f"""
            <div class="report-item" style="display:flex; gap:12px; margin-bottom:14px; align-items:flex-start;">
                <div style="min-width:28px; height:28px; background:#516862; color:#fff; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:600; font-size:0.85em;">{i}</div>
                <div>
                    <div style="font-weight:600; color:#4C4C4E; margin-bottom:2px;">{treatment}</div>
                    <div style="color:#4C4C4E; font-size:0.9em;">{reason}</div>
                </div>
            </div>"""

        product_items = ""
        for product in products:
            if not isinstance(product, dict):
                continue
            product_name = html_escape(str(product.get("product", "")))
            product_reason = html_escape(str(product.get("reason", "")))
            product_items += f"""
            <div class="report-item" style="display:flex; gap:12px; margin-bottom:12px; align-items:flex-start; padding:10px 14px; background:#F5F5F5; border-radius:8px; border-left:3px solid #C1A890;">
                <div>
                    <div style="font-weight:600; color:#4C4C4E; margin-bottom:2px;">{product_name}</div>
                    <div style="color:#4C4C4E; font-size:0.9em;">{product_reason}</div>
                </div>
            </div>"""

        positive_items = ""
        for highlight in positive_highlights[:3]:
            if not isinstance(highlight, dict):
                continue
            title = html_escape(str(highlight.get("title", "")))
            detail = html_escape(str(highlight.get("detail", "")))
            positive_items += f"""
            <div style="flex:1; min-width:190px; padding:14px 16px; background:#F5F5F5; border-left:3px solid #C0C8A9; border-radius:0 8px 8px 0;">
                <div class="report-display" style="color:#516862; margin-bottom:3px;">{title}</div>
                <div style="color:#4C4C4E; font-size:0.9em; line-height:1.5;">{detail}</div>
            </div>"""

        html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Von &amp; Co Aesthetics - Treatment Plan</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fira+Sans:wght@300;400;500;600&display=swap');
  @font-face {{ font-family:'Arsenica'; src:url('{asset_root}/arsenica-regular.otf') format('opentype'); font-style:normal; font-weight:400; font-display:swap; }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Fira Sans','Trebuchet MS',sans-serif; color:#4C4C4E; background:#fff; }}
  .page {{ max-width:800px; margin:0 auto; padding:40px; }}
  .report-display {{ font-family:'Arsenica',Georgia,serif; font-weight:400; }}
  .report-title {{ color:#516862; font-family:'Arsenica',Georgia,serif; font-size:1.55rem; font-weight:400; line-height:1.15; }}
  table {{ table-layout:fixed; }}
  th, td {{ overflow-wrap:anywhere; }}
  @media screen and (max-width:600px) {{
    .page {{ padding:18px; }}
    .report-header {{ padding:22px 18px 18px !important; }}
    .report-meta {{ padding:14px 18px !important; }}
    .report-summary {{ padding:22px 20px !important; }}
    .report-offers {{ flex-direction:column; }}
    .report-offer-card {{ min-width:0 !important; }}
    .report-footer {{ padding:22px 18px !important; }}
    .report-results-table, .report-results-table tbody {{ display:block; width:100%; }}
    .report-results-head {{ display:none; }}
    .report-result-row {{ display:block; width:100%; margin-bottom:10px; padding:8px 12px; background:#F5F5F5; border-left:3px solid #C0C8A9; border-radius:0 8px 8px 0; }}
    .report-result-row td {{ display:grid; grid-template-columns:88px minmax(0, 1fr); align-items:start; gap:10px; width:100%; padding:8px 0 !important; border:0 !important; text-align:left !important; font-size:0.95rem !important; line-height:1.4; }}
    .report-result-row td + td {{ border-top:1px solid #F2F3F5 !important; }}
    .report-result-row td::before {{ content:attr(data-label); color:#516862; font-size:0.72rem; font-weight:600; letter-spacing:0.7px; line-height:1.4; text-transform:uppercase; }}
  }}
  @media print {{
    .page {{ max-width:100%; padding:12px; }}
    table {{ font-size:0.8rem; }}
    th, td {{ padding-left:6px !important; padding-right:6px !important; }}
    .report-section-title {{ break-after:avoid-page; page-break-after:avoid; }}
    .report-item, tr {{ break-inside:avoid; page-break-inside:avoid; }}
  }}
</style>
</head><body>
<div class="page">
  <!-- Header -->
  <div class="report-header" style="background:#fff; padding:24px 35px 20px; border:1px solid #EBEAE5; border-radius:12px 12px 0 0; text-align:center;">
    <img src="{asset_root}/logo.png" alt="Von & Co Medical Aesthetics Studio" width="1549" height="848" style="display:block; width:190px; height:auto; margin:0 auto 10px;">
    <div class="report-title">Personalized Skin Analysis Report</div>
  </div>

  <!-- Patient Info Bar -->
  <div class="report-meta" style="background:#F2F3F5; padding:16px 35px;">
    <div><span style="color:rgba(76,76,78,0.72); font-size:0.85em;">Prepared for </span><strong>{name}</strong></div>
  </div>

  <!-- Positives first -->
  <div style="margin:25px 0 20px;">
    <div class="report-display" style="font-size:1.4em; color:#516862; margin-bottom:12px;">Begin With the Positive</div>
    <div style="display:flex; gap:10px; flex-wrap:wrap;">{positive_items}</div>
  </div>

  <div style="padding:14px 18px; border:1px solid #F2F3F5; border-radius:8px; margin-bottom:20px;">
    <span style="color:rgba(76,76,78,0.72); font-size:0.85em;">Overall Score: </span><strong style="color:#516862; font-size:1.1em;">{overall}/100</strong>
  </div>

  <!-- Summary -->
  <div class="report-summary" style="padding:25px 35px; border-left:3px solid #C1A890; margin:25px 0; background:#F5F5F5; border-radius:0 8px 8px 0;">
    <div style="font-size:0.8em; text-transform:uppercase; letter-spacing:1.5px; color:#C1A890; margin-bottom:6px;">Summary</div>
    <p style="color:#4C4C4E; line-height:1.6; font-size:0.95em;">{summary}</p>
  </div>

  <!-- Concerns Table -->
  <div style="margin:0 0 30px;">
    <div class="report-section-title report-display" style="font-size:1.4em; color:#516862; margin-bottom:14px; padding:0 0 8px; border-bottom:2px solid #F2F3F5;">Skin Analysis Results</div>
    <table class="report-results-table" style="width:100%; border-collapse:collapse;">
      <tr class="report-results-head" style="background:#516862;">
        <th style="padding:10px 12px; text-align:left; font-size:0.85em; text-transform:uppercase; letter-spacing:0.5px; color:#FFFFFF;">Concern</th>
        <th style="padding:10px 12px; text-align:center; font-size:0.85em; text-transform:uppercase; letter-spacing:0.5px; color:#FFFFFF;">Level</th>
        <th style="padding:10px 12px; text-align:left; font-size:0.85em; text-transform:uppercase; letter-spacing:0.5px; color:#FFFFFF;">Assessment</th>
      </tr>
      {concern_rows}
    </table>
  </div>

  <!-- Recommendations -->
  <div style="margin:0 0 30px;">
    <div class="report-section-title report-display" style="font-size:1.4em; color:#516862; margin-bottom:14px; padding:0 0 8px; border-bottom:2px solid #F2F3F5;">Recommended Treatments</div>
    {rec_items}
  </div>

  {f'''<div style="margin:0 0 30px;">
    <div class="report-section-title report-display" style="font-size:1.4em; color:#516862; margin-bottom:14px; padding:0 0 8px; border-bottom:2px solid #F2F3F5;">Your Skincare Essentials</div>
    {product_items}
  </div>''' if product_items else ''}

  <!-- Promo + Club -->
  <div class="report-offers" style="display:flex; gap:16px; margin:0 0 25px; flex-wrap:wrap;">
    <div class="report-offer-card" style="flex:1; min-width:200px; background:#FFFFFF; border:1px solid #516862; border-radius:10px; padding:18px 20px; text-align:center;">
      <div class="report-display" style="font-size:1.2rem; color:#516862; margin-bottom:4px;">New Guest Offer</div>
      <div style="font-size:0.88em; color:#4C4C4E;">15% off your first visit, including treatments and same-day skincare.</div>
    </div>
    <div class="report-offer-card" style="flex:1; min-width:200px; background:#516862; border-radius:10px; padding:18px 20px; text-align:center; color:#fff;">
      <div class="report-display" style="font-size:1.2rem; margin-bottom:4px;">The Club. $149/month</div>
      <div style="font-size:0.88em; color:#EBEAE5;">20% off lasers and microneedling. 15% off injectables, facials, and skincare. Funds bank, share, or gift.</div>
    </div>
  </div>

  <!-- CTA Footer -->
  <div class="report-footer" style="background:#516862; padding:24px 35px; border-radius:0 0 12px 12px; text-align:center; color:#fff;">
    <div class="report-display" style="font-size:1.3em; margin-bottom:6px;">Ready to Start Your Journey?</div>
    <div style="font-size:0.9em; color:#EBEAE5; margin-bottom:8px;">Book a complimentary consultation with our expert providers.</div>
    <div style="font-size:1em; font-weight:600; color:#C1A890;">239.799.4866</div>
    <a href="https://booking.vonandcoaesthetics.com/webstoreNew/services?utm_source=skin-analyzer" style="display:inline-block; margin-top:12px; padding:12px 28px; background:#C1A890; color:#fff; border-radius:25px; font-size:0.95em; font-weight:600; text-decoration:none;">Book Your Complimentary Consultation</a>
    <div style="font-size:0.8em; color:rgba(255,255,255,0.5); margin-top:8px;">Naples, FL &nbsp;•&nbsp; Open 7 Days a Week</div>
  </div>

  <div style="text-align:center; margin-top:16px; font-size:0.75em; color:rgba(76,76,78,0.56);">
    This AI analysis is a preview and does not replace a professional consultation. Any concerning lesion needs an in-person medical evaluation.
  </div>
</div>
</body></html>"""

        return html, 200, {'Content-Type': 'text/html'}
    except Exception as e:
        return jsonify({"error": str(e)}), 500


BODY_AREA_PROMPTS = {
    "face": "The guest selected FACE. First identify the dominant anatomy actually visible and set observedArea honestly. Only when it is face, analyze wrinkles, redness, dark spots, texture, pore visibility, visible contour definition, visible sun-exposure signs, and uneven tone. Do not refer to another body area.",
    "neck_chest": "The guest selected NECK AND CHEST/DÉCOLLETAGE. First identify the dominant anatomy actually visible and set observedArea honestly. Only when it matches, analyze visible sun-exposure signs, visible contour definition or crepiness, redness, texture, and lines. Distinguish persistent visible lines or crepiness from a fold created by neck rotation, flexion, extension, posture, or camera angle. A single positional fold is not evidence of laxity. Score laxity or wrinkles above 40 only from a sufficiently neutral resting view when clearly visible moderate evidence remains independent of pose. For every laxity or wrinkles score above 40, state in that concern's description that the evidence is visible at rest, in a neutral resting view, or independent of pose. If the photographed view is materially turned, flexed, extended, or otherwise non-neutral, or if you cannot establish pose independence from the photo, keep laxity and wrinkles at 40 or below. Use concern keys: sunDamage, laxity, redness, texture, wrinkles. Do NOT recommend Botox or dermal fillers for this area. Focus on BBL, Halo, Moxi, RF Microneedling, Sculptra, and Microneedling. Do not refer to facial anatomy or another body area.",
    "hands": "The guest selected HANDS. First identify the dominant anatomy actually visible and set observedArea honestly. Only when it matches, analyze visible sun-exposure signs, visible contour definition or crepiness, texture, visible veins, and surface dryness. Do not infer veins from tendons, knuckles, shadows, or ordinary contour contrast. Score veins above 40 only when at least two corroborating vascular details are visible, such as blue/green/purple color plus a branching/raised network, or a branching/raised network extending across the photographed area. State those details in the veins description. Use concern keys: sunDamage, laxity, texture, veins, dryness. Do NOT recommend Botox or facial dermal fillers for hands. Do not refer to facial anatomy or another body area.",
    "back": "The guest selected BACK. First identify the dominant anatomy actually visible and set observedArea honestly. Only when it matches, analyze breakout-like surface congestion, textural marks, texture, uneven tone, and visible hair growth. Use concern keys: acne, scarring, texture, unevenTone, hairRemoval. Do NOT recommend Botox, dermal fillers, Sculptra, or facial services for the back. Focus on VI Peel, Microneedling, RF Microneedling, BBL, Halo, and Laser Hair Removal. Laser Hair Removal requires a hairRemoval score of at least 41 plus clearly visible dark or coarse body hair, stubble, distinct visible follicles, or follicular contrast. Fine natural hair and scalp hair lying across the back do not qualify. Never mention the face, jawline, or another body area.",
    "legs": "The guest selected LEGS. First identify the dominant anatomy actually visible and set observedArea honestly. Only when it matches, analyze visible vascularity, texture, visible sun-exposure signs, visible hair growth, and surface dryness. Do not infer veins from shadows or ordinary contour contrast. Score veins above 40 only when at least two corroborating vascular details are visible, such as blue/green/purple color plus a branching/raised network, or a branching/raised network extending across the photographed area. State those details in the veins description. Use concern keys: veins, texture, sunDamage, hairRemoval, dryness. Do NOT recommend Botox, dermal fillers, Sculptra, or facial services for legs. Focus on BBL, Moxi, Microneedling, and Laser Hair Removal. Laser Hair Removal requires a hairRemoval score of at least 41 plus clearly visible dark or coarse body hair, stubble, distinct visible follicles, or follicular contrast. Fine natural or vellus hair does not qualify. Never mention the face, jawline, or another body area."
}

AREA_CONCERN_KEYS = {
    "face": (
        "wrinkles",
        "redness",
        "darkSpots",
        "texture",
        "pores",
        "laxity",
        "sunDamage",
        "unevenTone",
    ),
    "neck_chest": ("sunDamage", "laxity", "redness", "texture", "wrinkles"),
    "hands": ("sunDamage", "laxity", "texture", "veins", "dryness"),
    "back": ("acne", "scarring", "texture", "unevenTone", "hairRemoval"),
    "legs": ("veins", "texture", "sunDamage", "hairRemoval", "dryness"),
}

TREATMENT_TARGETS = {
    "Botox": {"wrinkles"},
    "Dysport": {"wrinkles"},
    "Xeomin": {"wrinkles"},
    "Microneedling": {"wrinkles", "texture", "pores", "laxity", "scarring"},
    "RF Microneedling": {"wrinkles", "texture", "pores", "laxity", "scarring"},
    "Dermal Fillers": {"wrinkles", "laxity"},
    "Sculptra": {"wrinkles", "laxity"},
    "HydraFacial Clarifying": {"redness", "pores", "texture", "acne", "unevenTone"},
    "HydraFacial Customized": {"redness", "pores", "texture", "dryness", "unevenTone"},
    "HydraFacial Elite": {"texture", "dryness", "unevenTone"},
    "SaltFacial": {"pores", "texture", "dryness", "acne", "unevenTone"},
    "SkinVive": {"texture", "dryness"},
    "Sciton Moxi": {"wrinkles", "darkSpots", "texture", "sunDamage", "unevenTone"},
    "Microneedling + PRF": {"wrinkles", "texture", "pores", "laxity", "scarring"},
    "Deep Pore Facial": {"pores", "texture", "acne"},
    "Signature Facial": {"redness", "texture", "dryness", "unevenTone"},
    "Anti-Aging Facial": {"wrinkles", "texture", "dryness", "laxity"},
    "Sciton BBL": {"redness", "darkSpots", "sunDamage", "unevenTone", "veins", "acne"},
    "Sciton Halo": {"wrinkles", "darkSpots", "texture", "pores", "laxity", "sunDamage", "unevenTone", "scarring"},
    "Chemical Peels": {"wrinkles", "darkSpots", "texture", "pores", "sunDamage", "unevenTone", "acne", "scarring"},
    "Laser Hair Removal": {"hairRemoval"},
}

AREA_TREATMENTS = {
    "face": {
        "Botox", "Dysport", "Xeomin", "Microneedling", "RF Microneedling",
        "Dermal Fillers", "Sculptra", "HydraFacial Clarifying",
        "HydraFacial Customized", "HydraFacial Elite", "SaltFacial",
        "SkinVive", "Sciton Moxi", "Chemical Peels",
        "Microneedling + PRF", "Deep Pore Facial", "Signature Facial",
        "Anti-Aging Facial", "Sciton BBL", "Sciton Halo",
    },
    "neck_chest": {
        "Microneedling", "RF Microneedling", "Sculptra",
        "Sciton Moxi", "Microneedling + PRF", "Sciton BBL", "Sciton Halo",
    },
    "hands": {
        "Microneedling", "RF Microneedling",
        "Sciton Moxi", "Chemical Peels", "Microneedling + PRF",
        "Sciton BBL", "Sciton Halo",
    },
    "back": {
        "Microneedling", "RF Microneedling",
        "Chemical Peels", "Microneedling + PRF", "Sciton BBL",
        "Sciton Halo", "Laser Hair Removal",
    },
    "legs": {
        "Microneedling", "Sciton Moxi", "Sciton BBL",
        "Laser Hair Removal",
    },
}

FALLBACK_TREATMENTS_BY_AREA_CONCERN = {
    "face": {
        "wrinkles": ("Sciton Moxi",),
        "redness": ("Sciton BBL",),
        "darkSpots": ("Sciton BBL",),
        "texture": ("Sciton Moxi",),
        "pores": ("Microneedling", "HydraFacial Clarifying"),
        "laxity": ("RF Microneedling", "Sculptra"),
        "sunDamage": ("Sciton BBL",),
        "unevenTone": ("Sciton BBL",),
    },
    "neck_chest": {
        "sunDamage": ("Sciton BBL",),
        "laxity": ("RF Microneedling", "Sculptra"),
        "redness": ("Sciton BBL",),
        "texture": ("Sciton Moxi",),
        "wrinkles": ("Sciton Moxi",),
    },
    "hands": {
        "sunDamage": ("Sciton BBL",),
        "laxity": ("RF Microneedling",),
        "texture": ("Microneedling",),
        "veins": ("Sciton BBL",),
    },
    "back": {
        "acne": ("Chemical Peels",),
        "scarring": ("Microneedling",),
        "texture": ("Microneedling",),
        "unevenTone": ("Sciton BBL",),
        "hairRemoval": ("Laser Hair Removal",),
    },
    "legs": {
        "veins": ("Sciton BBL",),
        "texture": ("Microneedling",),
        "sunDamage": ("Sciton Moxi",),
        "hairRemoval": ("Laser Hair Removal",),
    },
}

_CONCERN_GOAL_LABELS = {
    "wrinkles": "visible lines",
    "redness": "visible redness",
    "darkSpots": "visible pigmentation",
    "texture": "visible texture",
    "pores": "pore visibility",
    "laxity": "visible contour softness",
    "sunDamage": "visible sun-exposure signs",
    "unevenTone": "visible tone variation",
    "acne": "visible surface congestion",
    "scarring": "visible textural marks",
    "hairRemoval": "visible hair growth",
    "veins": "visible vascularity",
    "dryness": "visible surface dryness",
}

_FALLBACK_TREATMENT_REASON_TEMPLATES = {
    "Sciton BBL": (
        "For {goal}, Sciton BBL is a focused light-based option to discuss."
    ),
    "Sciton Moxi": (
        "For {goal}, Sciton Moxi offers a gentle fractional approach to discuss."
    ),
    "Microneedling": (
        "For {goal}, Microneedling is a collagen-supporting option to discuss."
    ),
    "RF Microneedling": (
        "For {goal}, RF Microneedling is a focused collagen-remodeling option to discuss."
    ),
    "Sculptra": (
        "For {goal}, Sculptra is a gradual collagen-stimulating option to discuss."
    ),
    "Chemical Peels": (
        "For {goal}, a Chemical Peel, including VI Peel options, is a surface-renewal approach to discuss."
    ),
    "HydraFacial Clarifying": (
        "For {goal}, HydraFacial Clarifying is a cleansing and exfoliating option to discuss."
    ),
    "Laser Hair Removal": (
        "For {goal}, Laser Hair Removal is a focused reduction option to discuss."
    ),
}

_TREATMENT_REASON_GOAL_FAMILIES = (
    (re.compile(r"\b(?:redness|pinkness|flush|flushing|broken\s+capillaries|vascular\s+redness)\b", re.IGNORECASE), {"redness"}),
    (re.compile(r"\b(?:body\s+hair|hair\s+growth|hair\s+reduction|stubble|follicles?|follicular)\b", re.IGNORECASE), {"hairRemoval"}),
    (re.compile(r"\b(?:surface\s+veins?|visible\s+veins?|vascularity)\b", re.IGNORECASE), {"veins"}),
    (re.compile(r"\b(?:dryness|dry\s+skin|hydration|moisture)\b", re.IGNORECASE), {"dryness"}),
    (re.compile(r"\b(?:pores?|pore\s+visibility)\b", re.IGNORECASE), {"pores"}),
    (re.compile(r"\b(?:surface\s+congestion|breakouts?|blemishes?)\b", re.IGNORECASE), {"acne"}),
    (re.compile(r"\b(?:textural\s+marks?|scar(?:s|ring)?)\b", re.IGNORECASE), {"scarring"}),
    (re.compile(r"\b(?:laxity|contour\s+softness|crepiness|crepey)\b", re.IGNORECASE), {"laxity"}),
    (re.compile(r"\b(?:wrinkles?|fine\s+lines?|expression\s+lines?|crow'?s\s+feet|folds?)\b", re.IGNORECASE), {"wrinkles"}),
    (re.compile(r"\b(?:texture|roughness|bumpiness|skin\s+surface)\b", re.IGNORECASE), {"texture"}),
    (re.compile(r"\b(?:pigment(?:ation)?|dark\s+spots?|brown\s+spots?|discoloration|sun[- ]exposure|sun\s+signs?|uneven\s+tone|uniform\s+tone|tone\s+variation)\b", re.IGNORECASE), {"darkSpots", "sunDamage", "unevenTone"}),
)

FALLBACK_PRODUCTS_BY_CONCERN = {
    "wrinkles": ("SkinBetter AlphaRet",),
    "redness": ("Avene Cicalfate+",),
    "darkSpots": ("SkinBetter Even Tone",),
    "texture": ("ZO Complexion Renewal Pads",),
    "pores": ("ZO Complexion Renewal Pads",),
    "laxity": ("Alastin Restorative Skin Complex",),
    "sunDamage": ("Colorescience Face Shield SPF 50",),
    "unevenTone": ("SkinBetter Even Tone",),
    "acne": ("ZO Complexion Renewal Pads",),
    "dryness": ("SkinBetter Trio Moisture",),
}

PRODUCT_TARGETS = {
    "SkinBetter AlphaRet": {"wrinkles", "texture"},
    "ZO Wrinkle+Texture Repair": {"wrinkles", "texture"},
    "Avene Thermal Water": {"redness"},
    "Avene Cicalfate+": {"redness", "dryness", "texture"},
    "Alastin HydraTint": {"redness", "sunDamage"},
    "SkinBetter Even Tone": {"darkSpots", "sunDamage", "unevenTone"},
    "ISDIN Melaclear Advanced": {"darkSpots", "sunDamage", "unevenTone"},
    "ZO 10% Vitamin C": {"darkSpots", "sunDamage", "unevenTone"},
    "Hydrinity Vivid Serum": {"dryness", "unevenTone"},
    "ZO Complexion Renewal Pads": {"pores", "texture", "acne", "unevenTone"},
    "SkinBetter Peel Pads": {"pores", "texture", "acne", "unevenTone"},
    "Alastin Restorative Skin Complex": {"wrinkles", "laxity", "texture"},
    "ZO Growth Factor Serum": {"wrinkles", "laxity", "texture"},
    "Hydrinity Renewing HA Serum": {"dryness", "texture"},
    "SkinBetter Trio Moisture": {"dryness", "texture"},
    "ALASTIN Restorative Eye Treatment": {"wrinkles"},
    "ZO Growth Factor Eye": {"wrinkles"},
    "Colorescience Face Shield SPF 50": {"sunDamage"},
    "ISDIN Eryfotona Actinica": {"sunDamage"},
    "SkinBetter Sunbetter SPF 68": {"sunDamage"},
    "Alastin Skin Nectar": {"dryness", "laxity", "texture"},
    "Hydrinity Hyacyn Mist": {"redness", "dryness", "texture", "acne"},
}

SPF_PRODUCTS = {
    "Colorescience Face Shield SPF 50",
    "ISDIN Eryfotona Actinica",
    "SkinBetter Sunbetter SPF 68",
}

_COMBO_REQUIREMENTS = {
    "Anti-Aging Power": (
        {"Botox", "Dysport", "Xeomin"},
        {"Dermal Fillers"},
        {"Sciton Moxi"},
    ),
    "Glow-Up Package": ({"HydraFacial Elite"}, {"SkinVive"}),
    "Scar Reduction": (
        {"Chemical Peels"},
        {"Microneedling + PRF"},
    ),
    "Hero Combo": ({"Sciton BBL"}, {"Sciton Halo"}),
    "Full Rejuvenation": (
        {"Sciton Halo"},
        {"Sculptra"},
        {"Botox", "Dysport", "Xeomin"},
    ),
}

_ACCESSIBLE_ENTRY_TREATMENTS = {
    "face": {
        "HydraFacial Clarifying",
        "HydraFacial Customized",
        "Deep Pore Facial",
        "Signature Facial",
        "SaltFacial",
    },
}


def _product_names_for_area(body_area):
    concern_keys = set(
        AREA_CONCERN_KEYS.get(body_area, AREA_CONCERN_KEYS["face"])
    )
    return {
        product
        for product, targets in PRODUCT_TARGETS.items()
        if (product in SPF_PRODUCTS or targets.intersection(concern_keys))
        and (
            body_area == "face"
            or product
            not in {"ALASTIN Restorative Eye Treatment", "ZO Growth Factor Eye"}
        )
    }


def _normalize_suggested_combo(analysis):
    """Keep a stack only when its named treatment components are in the plan."""
    combo = analysis.get("suggestedCombo")
    if not isinstance(combo, str) or not combo.strip():
        analysis["suggestedCombo"] = None
        return analysis
    treatment_names = {
        item.get("treatment")
        for item in analysis.get("recommendations", [])
        if isinstance(item, dict)
    }
    matching_names = [
        combo_name
        for combo_name in _COMBO_REQUIREMENTS
        if combo_name.lower() in combo.lower()
    ]
    if len(matching_names) == 1:
        combo_name = matching_names[0]
        requirements = _COMBO_REQUIREMENTS[combo_name]
        if all(treatment_names.intersection(options) for options in requirements):
            analysis["suggestedCombo"] = combo_name
            return analysis
    analysis["suggestedCombo"] = None
    return analysis


def _normalize_concern_severity(analysis):
    """Keep the verbal severity aligned with the visible-concern score."""
    for concern in analysis.get("concerns", {}).values():
        if not isinstance(concern, dict) or not isinstance(concern.get("score"), int):
            continue
        score = concern["score"]
        if score <= 10:
            concern["severity"] = "none"
        elif score <= 40:
            concern["severity"] = "mild"
        elif score <= 60:
            concern["severity"] = "moderate"
        else:
            concern["severity"] = "severe"
    return analysis


_DRAPED_SCALP_HAIR_PATTERN = re.compile(
    r"\b(?:scalp\s+hair|hair\s+from\s+(?:the\s+)?head)\b|"
    r"\b(?:long\s+)?(?:dark\s+)?hair\b[^.;]{0,35}\b"
    r"(?:drapes?|draped|falls?|falling|lies|lying|rests?|resting)\b"
    r"[^.;]{0,35}\b(?:over|across|on)\b[^.;]{0,25}\b"
    r"(?:shoulders?|back)\b|"
    r"\bhair\b[^.;]{0,25}\b(?:over|across|on)\b[^.;]{0,25}\b"
    r"(?:shoulders?|back)\b[^.;]{0,35}\b"
    r"(?:drapes?|draped|falls?|falling|lies|lying|rests?|resting)\b",
    re.IGNORECASE,
)
_QUALIFYING_HAIR_EVIDENCE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bstubble\b",
        r"\b(?:visible|distinct|prominent|dark|clearly\s+visible)\b"
        r"(?:\W+\w+){0,4}\W+\b(?:hair\s+)?follicles?\b",
        r"\b(?:hair\s+)?follicles?\b(?:\W+\w+){0,4}\W+\b"
        r"(?:visible|distinct|prominent|dark|clearly\s+visible)\b",
        r"\bfollicular\s+(?:contrast|prominence|visibility|pattern)\b",
        r"\b(?:dark|coarse)\b(?:\W+\w+){0,4}\W+\b"
        r"(?:body\s+hair|hair\s+growth|hairs?)\b",
        r"\b(?:body\s+hair|hair\s+growth|hairs?)\b"
        r"(?:\W+\w+){0,4}\W+\b(?:dark|coarse)\b",
    )
)
_HAIR_EVIDENCE_NEGATION_PATTERN = re.compile(
    r"\b(?:no|not|without|lacks?|lacking|lack\s+of|hardly\s+any|"
    r"barely\s+any|cannot\s+see|can't\s+see|do(?:es)?\s+not\s+show)\b"
    r"[^.;,]{0,70}$",
    re.IGNORECASE,
)
_HAIR_EVIDENCE_POST_NEGATION_PATTERN = re.compile(
    r"^\s*(?:(?:is|are|was|were|remains?|appears?|seems?|looks?)\s+)?(?:"
    r"not\s+(?:clearly\s+)?(?:visible|present|shown|confirmed|observed|"
    r"evident|demonstrated|supported)|"
    r"absent|missing|unclear|uncertain|unconfirmed|unverified|unsupported|"
    r"indeterminate|speculative|hypothetical|alleged(?:ly)?|reported(?:ly)?|"
    r"borderline|inconclusive|debatable|theoretical(?:ly)?|purported(?:ly)?|"
    r"ostensibly|putative(?:ly)?|tentative(?:ly)?|"
    r"cannot\s+be\s+(?:clearly\s+)?(?:seen|confirmed|verified|established)|"
    r"can't\s+be\s+(?:clearly\s+)?(?:seen|confirmed|verified|established)|"
    r"could\s+not\s+be\s+(?:clearly\s+)?(?:seen|confirmed|verified|established)|"
    r"do(?:es)?\s+not\s+(?:appear\s+)?(?:visible|present|show)|"
    r"(?:is|are)\s+(?:difficult|hard)\s+to\s+"
    r"(?:see|confirm|verify|discern|detect))\b",
    re.IGNORECASE,
)
_HAIR_EVIDENCE_UNCERTAINTY_PATTERN = re.compile(
    r"\b(?:no|not|nothing|without|lacks?|lacking|lack\s+of|absent|missing|"
    r"cannot|can't|could\s+not|couldn't|do(?:es)?\s+not|don't|doesn't|"
    r"unable\s+to|fail(?:s|ed)?\s+to|hardly\s+any|barely\s+any|"
    r"unclear|unconfirmed|unverified|unsupported|uncertain|indeterminate|"
    r"speculative|hypothetical|alleged(?:ly)?|reported(?:ly)?|borderline|"
    r"inconclusive|debatable|theoretical(?:ly)?|purported(?:ly)?|ostensibly|"
    r"putative(?:ly)?|tentative(?:ly)?|hard\s+to\s+(?:discern|detect)|"
    r"questionable|possible|possibly|"
    r"potential|potentially|perhaps|maybe|may|might|could|unlikely|"
    r"doubtful|equivocal|ambiguous|invisible|imperceptible|undetectable|"
    r"indiscernible|negligible|weak|poor|zero|none|neither|"
    r"if\s+any|not\s+enough|anything\s+but\s+(?:clearly\s+)?"
    r"(?:visible|present|apparent|detectable|seen)|rules?\s+out|excludes?)\b",
    re.IGNORECASE,
)
_HAIR_EVIDENCE_CLAUSE_BOUNDARY_PATTERN = re.compile(
    r"[.;!\n]|(?:,\s*)?\b(?:(?<!anything\s)but|however|although|yet|though|whereas)\b",
    re.IGNORECASE,
)
_HAIR_EVIDENCE_POSITIVE_VISUAL_PATTERN = re.compile(
    r"\b(?:clearly\s+visible|visible|present|distinct|prominent|apparent|"
    r"noticeable|evident|clearly\s+seen|readily\s+seen|shown|strong|dense)\b",
    re.IGNORECASE,
)


def _evidence_clause_bounds(description, match):
    """Return semantic-clause bounds around one evidence match."""
    clause_start = 0
    clause_end = len(description)
    for boundary in _HAIR_EVIDENCE_CLAUSE_BOUNDARY_PATTERN.finditer(description):
        if boundary.end() <= match.start():
            clause_start = boundary.end()
            continue
        if boundary.start() >= match.end():
            clause_end = boundary.start()
            break
    return clause_start, clause_end


def _hair_evidence_clause(description, match):
    """Return the semantic clause around one candidate hair-evidence match."""
    clause_start, clause_end = _evidence_clause_bounds(description, match)
    return description[clause_start:clause_end]


def _hair_evidence_match_is_affirmative(description, match):
    """Reject negation immediately before or after a visual-hair phrase."""
    clause = _hair_evidence_clause(description, match)
    clause_prefix = re.split(
        r"[.;,]",
        description[max(0, match.start() - 90):match.start()],
    )[-1]
    clause_suffix = re.split(
        r"[.;,]",
        description[match.end():match.end() + 90],
    )[0]
    return not (
        _DRAPED_SCALP_HAIR_PATTERN.search(clause)
        or not _HAIR_EVIDENCE_POSITIVE_VISUAL_PATTERN.search(clause)
        or _HAIR_EVIDENCE_UNCERTAINTY_PATTERN.search(clause)
        or _HAIR_EVIDENCE_NEGATION_PATTERN.search(clause_prefix)
        or _HAIR_EVIDENCE_POST_NEGATION_PATTERN.search(clause_suffix)
    )


def _has_nonnegated_hair_evidence(description):
    """Return whether one qualifying visual-hair phrase is affirmative."""
    for pattern in _QUALIFYING_HAIR_EVIDENCE_PATTERNS:
        for match in pattern.finditer(description):
            if _hair_evidence_match_is_affirmative(description, match):
                return True
    return False


def _supports_laser_hair_removal(analysis):
    """Require moderate, treatment-relevant body-hair evidence from the photo."""
    concern = analysis.get("concerns", {}).get("hairRemoval", {})
    if not isinstance(concern, dict):
        return False
    score = concern.get("score")
    description = str(concern.get("description", "")).strip()
    if type(score) is not int or score < 41 or not description:
        return False
    return _has_nonnegated_hair_evidence(description)


def _fallback_treatment_for_target(area, target, concern_scores):
    """Return one conservative exact-catalog service for uncovered evidence."""
    for treatment in FALLBACK_TREATMENTS_BY_AREA_CONCERN.get(area, {}).get(
        target,
        (),
    ):
        if (
            treatment not in AREA_TREATMENTS[area]
            or target not in TREATMENT_TARGETS.get(treatment, set())
        ):
            continue
        if (
            treatment in {"Sciton Halo", "Sculptra", "Laser Hair Removal"}
            or (area == "hands" and treatment == "Sciton BBL")
        ) and concern_scores.get(target, 0) < 41:
            continue
        if (
            concern_scores.get("redness", 0) >= 41
            and treatment in {"Microneedling", "RF Microneedling"}
        ):
            continue
        return treatment
    return None


def _fallback_treatment_reason(treatment, target):
    """Create bounded guest copy for a deterministic catalog fallback."""
    template = _FALLBACK_TREATMENT_REASON_TEMPLATES.get(treatment)
    targets = [target] if isinstance(target, str) else list(target)
    goals = [
        _CONCERN_GOAL_LABELS.get(item, "the visible concern")
        for item in targets
    ]
    if len(goals) == 1:
        goal_copy = goals[0]
    elif len(goals) == 2:
        goal_copy = f"{goals[0]} and {goals[1]}"
    else:
        goal_copy = f"{', '.join(goals[:-1])}, and {goals[-1]}"
    if template is not None:
        return template.format(goal=goal_copy)
    return (
        f"For {goal_copy}, {treatment} is an option to discuss."
    )


def _treatment_reason_overreaches(reason, actual_targets):
    """Return whether prose claims a concern outside this guest's targets."""
    supported = set(actual_targets)
    return any(
        pattern.search(reason) and supported.isdisjoint(goal_family)
        for pattern, goal_family in _TREATMENT_REASON_GOAL_FAMILIES
    )


def _fallback_product_reason(product, target):
    """Create bounded guest copy for a deterministic skincare fallback."""
    goal = _CONCERN_GOAL_LABELS.get(target, "the visible concern")
    return (
        f"For {goal}, consider this formula as part of a provider-guided home "
        "routine."
    )


def _spf_product_reason(product):
    """Return a consistent, non-outcome-guaranteeing SPF rationale."""
    return (
        "Daily broad-spectrum sunscreen is the foundation of a thoughtful home "
        "routine; your provider can help confirm the right formula."
    )


def _normalize_catalog_recommendations(analysis, body_area):
    """Enforce exact catalog names and concern mappings before a result can win."""
    area = body_area if body_area in AREA_CONCERN_KEYS else "face"
    area_concerns = set(AREA_CONCERN_KEYS[area])
    allowed_treatments = AREA_TREATMENTS[area]
    _validate_score_description_coherence(analysis, area)
    _normalize_concern_severity(analysis)
    concern_scores = {}
    for key, value in analysis.get("concerns", {}).items():
        if key not in area_concerns or not isinstance(value, dict):
            continue
        score = value.get("score")
        if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 100:
            raise _GoogleResponseError(
                "Google Gemini returned an invalid visible-concern score"
            )
        concern_scores[key] = score
    if set(concern_scores) != area_concerns:
        raise _GoogleResponseError(
            "Google Gemini returned an incomplete visible-concern score set"
        )
    laser_hair_removal_supported = _supports_laser_hair_removal(analysis)
    eligible_targets = {
        key for key, score in concern_scores.items() if score > 10
        and (key != "hairRemoval" or laser_hair_removal_supported)
    }
    ranked_targets = sorted(
        eligible_targets,
        key=lambda key: (-concern_scores[key], key),
    )
    treatable_ranked_targets = [
        target
        for target in ranked_targets
        if any(
            target in TREATMENT_TARGETS.get(treatment, set())
            for treatment in allowed_treatments
        )
    ]
    moderate_targets = {
        key
        for key, score in concern_scores.items()
        if score >= 41 and key in eligible_targets
    }
    allowed_products = _product_names_for_area(area)

    def has_product_fallback(target):
        return any(
            candidate in allowed_products
            for candidate in FALLBACK_PRODUCTS_BY_CONCERN.get(target, ())
        )

    # When every visible concern is mild, cover the highest-ranked concern that
    # has a permitted catalog path. Some area-specific mappings deliberately
    # require moderate evidence (for example, BBL for visible hand vascularity).
    # Do not turn that safety threshold into a 503 merely because the blocked
    # concern happens to rank first; fall through to the next safely mappable
    # concern, or keep the baseline SPF plan when none is safely mappable.
    mild_coverage_target = None
    if not moderate_targets:
        mild_coverage_target = next(
            (
                target
                for target in ranked_targets
                if has_product_fallback(target)
                or _fallback_treatment_for_target(
                    area,
                    target,
                    concern_scores,
                )
            ),
            None,
        )
    normalized_recommendations = []
    seen_treatments = set()
    raw_recommendations = sorted(
        analysis.get("recommendations", []),
        key=lambda item: (
            item.get("priority", 99)
            if isinstance(item, dict)
            and isinstance(item.get("priority"), int)
            and not isinstance(item.get("priority"), bool)
            else 99
        ),
    )
    for item in raw_recommendations:
        if not isinstance(item, dict):
            continue
        treatment = item.get("treatment")
        if treatment in seen_treatments or treatment not in allowed_treatments:
            continue
        supported_targets = TREATMENT_TARGETS.get(treatment, set())
        model_targets = []
        for target in item.get("targets", []):
            if (
                target in area_concerns
                and target in supported_targets
                and target in eligible_targets
                and target not in model_targets
            ):
                model_targets.append(target)
        reason = str(item.get("reason", "")).strip()
        minimum_target_score = (
            41
            if (
                treatment in {"Sciton Halo", "Sculptra", "Laser Hair Removal"}
                or (area == "hands" and treatment == "Sciton BBL")
            )
            else 11
        )
        if not any(
            concern_scores.get(target, 0) >= minimum_target_score
            for target in model_targets
        ):
            continue
        if (
            concern_scores.get("redness", 0) >= 41
            and treatment in {"Microneedling", "RF Microneedling"}
        ):
            continue
        if treatment == "Laser Hair Removal" and not laser_hair_removal_supported:
            continue
        if not model_targets or not reason:
            continue
        # Gemini can select an appropriate catalog service yet omit a leading
        # concern that the same service is explicitly mapped to treat. Preserve
        # its valid targets, then add only moderate or top-two targets. Never
        # synthesize a service or silently broaden it to every mild finding.
        auto_coverage_targets = moderate_targets.union(ranked_targets[:2])
        targets = sorted(
            set(model_targets).union(
                supported_targets.intersection(auto_coverage_targets)
            ),
            key=lambda target: (-concern_scores[target], target),
        )
        reason = _fallback_treatment_reason(treatment, targets)
        seen_treatments.add(treatment)
        normalized_recommendations.append({
            "treatment": treatment,
            "reason": reason,
            "targets": targets,
            "priority": len(normalized_recommendations) + 1,
        })

    # If the model omitted a service for clearly visible moderate evidence,
    # add one conservative, area-specific exact-catalog option. This is a
    # deterministic coverage fallback, not a diagnosis or candidacy decision.
    covered_by_services = {
        target
        for item in normalized_recommendations
        for target in item["targets"]
    }
    fallback_targets = [
        target
        for target in treatable_ranked_targets
        if target in moderate_targets and target not in covered_by_services
    ]
    if (
        mild_coverage_target
        and mild_coverage_target not in covered_by_services
        and not has_product_fallback(mild_coverage_target)
        and _fallback_treatment_for_target(
            area,
            mild_coverage_target,
            concern_scores,
        )
    ):
        fallback_targets.append(mild_coverage_target)
    fallback_treatments_added = set()
    for target in fallback_targets:
        treatment = _fallback_treatment_for_target(
            area,
            target,
            concern_scores,
        )
        if not treatment:
            continue
        existing = next(
            (
                item
                for item in normalized_recommendations
                if item["treatment"] == treatment
            ),
            None,
        )
        if existing is not None:
            if target not in existing["targets"]:
                existing["targets"].append(target)
                existing["targets"].sort(
                    key=lambda key: (-concern_scores[key], key),
                )
                existing["reason"] = _fallback_treatment_reason(
                    treatment,
                    existing["targets"],
                )
        else:
            normalized_recommendations.append({
                "treatment": treatment,
                "reason": _fallback_treatment_reason(treatment, target),
                "targets": [target],
                "priority": len(normalized_recommendations) + 1,
            })
            seen_treatments.add(treatment)
            fallback_treatments_added.add(treatment)
        covered_by_services.add(target)

    # Model priority numbers can drift even when the selected treatments are
    # useful. Reorder catalog-mapped choices so leading evidence comes first.
    prioritized_recommendations = []
    remaining_recommendations = list(normalized_recommendations)
    covered_ranked_targets = [
        target
        for target in treatable_ranked_targets
        if any(
            target in item["targets"]
            for item in normalized_recommendations
        )
    ]
    for leading_target in covered_ranked_targets[:2]:
        if any(
            leading_target in item["targets"]
            for item in prioritized_recommendations
        ):
            continue
        matching_index = next(
            (
                index
                for index, item in enumerate(remaining_recommendations)
                if leading_target in item["targets"]
            ),
            None,
        )
        if matching_index is not None:
            prioritized_recommendations.append(
                remaining_recommendations.pop(matching_index)
            )
    prioritized_recommendations.extend(remaining_recommendations)
    normalized_recommendations = prioritized_recommendations
    for priority, item in enumerate(normalized_recommendations, start=1):
        item["priority"] = priority

    analysis["recommendations"] = normalized_recommendations
    if covered_ranked_targets and covered_ranked_targets[0] not in set(
        normalized_recommendations[0]["targets"]
    ):
        raise _GoogleResponseError(
            "Google Gemini did not prioritize its leading service-covered concern"
        )
    leading_targets = set(covered_ranked_targets[:2])
    leading_coverage = {
        target
        for item in normalized_recommendations[:2]
        for target in item["targets"]
    }
    if not leading_targets.issubset(leading_coverage):
        raise _GoogleResponseError(
            "Google Gemini did not prioritize its leading service-covered concerns"
        )

    normalized_products = []
    seen_products = set()
    for item in analysis.get("productRecommendations", []):
        if not isinstance(item, dict):
            continue
        product = item.get("product")
        reason = str(item.get("reason", "")).strip()
        if (
            product in seen_products
            or product not in allowed_products
            or not reason
        ):
            continue
        if (
            product not in SPF_PRODUCTS
            and not PRODUCT_TARGETS[product].intersection(eligible_targets)
        ):
            continue
        actual_product_targets = PRODUCT_TARGETS[product].intersection(
            eligible_targets
        )
        if product in SPF_PRODUCTS:
            reason = _spf_product_reason(product)
        else:
            leading_product_target = min(
                actual_product_targets,
                key=lambda target: (-concern_scores[target], target),
            )
            reason = _fallback_product_reason(product, leading_product_target)
        seen_products.add(product)
        normalized_products.append({"product": product, "reason": reason})
    if not any(item["product"] in SPF_PRODUCTS for item in normalized_products):
        normalized_products.append({
            "product": "Colorescience Face Shield SPF 50",
            "reason": _spf_product_reason("Colorescience Face Shield SPF 50"),
        })

    service_covered_targets = {
        target
        for item in normalized_recommendations
        for target in item["targets"]
    }
    product_covered_targets = {
        target
        for item in normalized_products
        for target in PRODUCT_TARGETS[item["product"]].intersection(
            eligible_targets
        )
    }
    required_coverage_targets = set(moderate_targets)
    if not required_coverage_targets and mild_coverage_target:
        required_coverage_targets.add(mild_coverage_target)
    for target in sorted(
        required_coverage_targets.difference(
            service_covered_targets.union(product_covered_targets)
        ),
        key=lambda key: (-concern_scores[key], key),
    ):
        product = next(
            (
                candidate
                for candidate in FALLBACK_PRODUCTS_BY_CONCERN.get(target, ())
                if candidate in allowed_products and candidate not in seen_products
            ),
            None,
        )
        if not product:
            continue
        normalized_products.append({
            "product": product,
            "reason": _fallback_product_reason(product, target),
        })
        seen_products.add(product)
        product_covered_targets.update(
            PRODUCT_TARGETS[product].intersection(eligible_targets)
        )
    analysis["productRecommendations"] = normalized_products

    covered_targets = {
        target
        for item in normalized_recommendations
        for target in item["targets"]
    }
    for item in normalized_products:
        covered_targets.update(
            PRODUCT_TARGETS[item["product"]].intersection(eligible_targets)
        )
    required_coverage = set(moderate_targets)
    if not required_coverage and mild_coverage_target:
        required_coverage.add(mild_coverage_target)
    if not required_coverage.issubset(covered_targets):
        raise _GoogleResponseError(
            "Google Gemini left a leading visible concern unmapped"
        )
    required_treatment_coverage = {
        target for target in moderate_targets if target in treatable_ranked_targets
    }
    if not required_treatment_coverage.issubset(
        {
            target
            for item in normalized_recommendations
            for target in item["targets"]
        }
    ):
        raise _GoogleResponseError(
            "Google Gemini left a moderate treatment-eligible concern without a service option"
        )

    if fallback_treatments_added:
        analysis["suggestedCombo"] = None
    return _normalize_suggested_combo(analysis)


def _analysis_schema_for_area(body_area="face"):
    """Constrain Gemini's concern object to the guest-selected body area."""
    schema = deepcopy(ANALYSIS_RESPONSE_SCHEMA)
    completed = schema["anyOf"][1]
    concern_value_schema = completed["properties"]["concerns"][
        "additionalProperties"
    ]
    area = body_area if body_area in AREA_CONCERN_KEYS else "face"
    concern_keys = AREA_CONCERN_KEYS[area]
    completed["properties"]["concerns"] = {
        "type": "object",
        "properties": {
            key: deepcopy(concern_value_schema) for key in concern_keys
        },
        "required": list(concern_keys),
        "additionalProperties": False,
    }
    recommendations = completed["properties"]["recommendations"]
    recommendations["minItems"] = 0
    recommendations["items"]["properties"]["treatment"]["enum"] = sorted(
        AREA_TREATMENTS[area]
    )
    recommendations["items"]["properties"]["targets"]["items"]["enum"] = list(
        concern_keys
    )
    products = completed["properties"]["productRecommendations"]
    products["minItems"] = 1
    products["items"]["properties"]["product"]["enum"] = sorted(
        _product_names_for_area(area)
    )
    return schema

def build_user_prompt(user_age=None, body_area="face"):
    """Build the user prompt for the Google Gemini vision API, including age and body area."""
    area_instruction = BODY_AREA_PROMPTS.get(body_area, BODY_AREA_PROMPTS["face"])
    base = f"Please analyze this skin image and provide a detailed assessment in the JSON format specified. The selected area is {body_area}. Do not assume the selection is correct; report the dominant visible anatomy in observedArea. {area_instruction}"
    if user_age:
        base += f" The guest entered age {user_age}. Use it only for adult eligibility and general context. Never return, infer, or compare an adult skin-age estimate."
    return base


def _apply_score_correction(analysis):
    """Recalculate the overall score transparently without altering model scores."""
    try:
        concerns = analysis.get("concerns", {})
        if concerns and isinstance(concerns, dict):
            scores = [
                value["score"]
                for value in concerns.values()
                if isinstance(value, dict)
                and isinstance(value.get("score"), int)
                and 0 <= value["score"] <= 100
            ]
            if scores:
                calculated_score = max(
                    0,
                    min(100, int(round(100 - (sum(scores) / len(scores))))),
                )
                analysis["overallScore"] = calculated_score
                print(
                    "  [Score] Overall is 100 minus mean concern severity: "
                    f"{calculated_score}"
                )

    except Exception as e:
        print(f"  [Score] Could not recalculate overall score: {e.__class__.__name__}")


_NON_COPY_RESPONSE_FIELDS = {
    "groundedIn",
    "targets",
    "treatment",
    "product",
    "observedArea",
    "reasonCode",
    "severity",
    "suggestedCombo",
}


def _sanitize_response(analysis):
    """Sanitize prohibited punctuation and categorical marketing overclaims."""

    overclaim_replacements = (
        (
            r"\bkills acne bacteria\b",
            "is used as part of a congestion-focused blue-light protocol",
        ),
        (r"\bkeratosis pilaris\b", "visible bumpiness"),
        (r"\bacne scars?\b", "textural marks"),
        (r"\bacne-focused\b", "congestion-focused"),
        (r"\bacne-prone\b", "breakout-prone"),
        (r"\brosacea\b", "visible redness"),
        (r"\bmelasma\b", "visible pigmentation"),
        (r"\bacne\b", "surface congestion"),
        (r"\b(?:dermatitis|eczema|psoriasis)\b", "visible irritation"),
        (
            r"\b(?:visible\s+)?hyper[- ]?pigmentation\b",
            "visible pigment variation",
        ),
        (
            r"\b(?:visible\s+)?photo[- ]?aging\b",
            "visible sun-exposure signs",
        ),
        (
            r"\b(?:a\s+|an\s+)?(?:visible\s+)?"
            r"(?:scarring|scars?|scarred(?:\s+surface)?)\b",
            "visible textural marks",
        ),
        (r"\bdehydration\b", "visible dryness"),
        (r"\bdehydrated\b", "visibly dry"),
        (
            r"\b(?:safe for all skin tones|all skin tones safe)\b",
            "may suit a range of skin tones after a provider confirms candidacy",
        ),
        (
            r"\b(?:safe for|suitable for) all skin types\b",
            "personalized after a provider confirms candidacy",
        ),
        (
            r"\brevers(?:e|es|ed|ing) years of sun damage\b",
            "address visible signs of sun exposure",
        ),
        (
            r"\b(?:will|can) instantly eliminat(?:e|es|ed|ing)\s+(.{1,60}?)\s+entirely\b",
            r"can help reduce \1",
        ),
        (r"\beliminates\s+(.{1,60}?)\s+entirely\b", r"helps reduce \1"),
        (r"\beliminate\s+(.{1,60}?)\s+entirely\b", r"help reduce \1"),
        (r"\beliminated\s+(.{1,60}?)\s+entirely\b", r"helped reduce \1"),
        (r"\beliminating\s+(.{1,60}?)\s+entirely\b", r"helping reduce \1"),
        (
            r"\bis permanently reducing\b",
            "supports long-term reduction of",
        ),
        (r"\bpermanently reduces\b", "supports long-term reduction of"),
        (r"\bpermanently reduce\b", "support long-term reduction of"),
        (r"\bpermanently reduced\b", "supported long-term reduction of"),
        (r"\bpermanently reducing\b", "supporting long-term reduction of"),
        (r"\bpermanent(?:ly)? reduction\b", "long-term reduction"),
        (r"\bflawless\b", "more even-looking"),
        (r"\bgold standard\b", "focused option"),
        (
            r"\b(?:a|an) (?:amazing|wonderful|fantastic|incredible|perfect|"
            r"potent|go-to|great|excellent) option\b",
            "a focused option",
        ),
        (r"\b(?:fantastic|incredible|great) for\b", "well suited to"),
        (r"\b(?:fantastic|incredible|amazing|stunning|wonderful)\b", "appealing"),
        (r"\bperfect(?:ly)?\b", "well-suited"),
        (r"\bremarkably\b", "visibly"),
        (r"\bincredibly\s+common\b", "common"),
        (r"\bincredibly\s+smooth\b", "visibly smooth"),
        (r"\bincredibly\b", ""),
        (r"\bcomplementary\s+VISIA\b", "complimentary VISIA"),
        (
            r"\b(?:is|are)\s+(?:highly|very)\s+responsive\s+to\b",
            "can often be addressed with",
        ),
        (
            r"\b(?:is|are)\s+highly\s+responsive\s+to\b",
            "can often be addressed with",
        ),
        (r"\bexceptionally\s+smooth\b", "smoother-looking"),
        (r"\b(?:an?\s+)?(?:excellent|great)\s+way\s+to\b", "an option to"),
        (r"\ban\s+excellent\s+(?:choice|option)\b", "an option"),
        (r"\bexcellent\s+for\b", "well suited to"),
        (r"\bhighly\s+appropriate\b", "potentially appropriate"),
        (r"\bhighly\s+beneficial\b", "worth discussing"),
        (r"\bgreat\s+natural\s+contours\b", "visible natural contours"),
        (r"\bgreat\s+natural\s+definition\b", "clear natural definition"),
        (
            r"\bgiving\s+you\s+long-term\s+smooth\s+results\b",
            "supporting longer-term hair reduction after a provider confirms candidacy",
        ),
        (
            r"\b(?:deliver|delivers|provides?)\s+long-term\s+smooth\s+results\b",
            "supports longer-term hair reduction after a provider confirms candidacy",
        ),
        (r"\bpremium\b", "provider-selected"),
        (
            r"\brefined(?=(?:calm|clear|even|smooth|strong)\b)",
            "refined, ",
        ),
        (
            r"\ba personalized combination of targeted treatments can be focused\b",
            "a personalized treatment plan can focus on these visible goals",
        ),
        (
            r"\bcan\s+deeply\s+exfoliate(?:\s+the\s+skin)?\b",
            "can exfoliate the skin's surface",
        ),
        (r"\bdeep\s+exfoliation\b", "controlled surface exfoliation"),
        (r"\bdynamic\s+lines\b", "visible expression lines"),
        (
            r"\brebuilds?\s+structural\s+support\s+from\s+within(?:\s+over\s+time)?\b",
            "can gradually support visible contour goals after a provider confirms candidacy",
        ),
        (r"\b(?:absolutely|truly)\b", ""),
        (r"\blooks\s+completely\s+natural\b", "has a natural appearance"),
        (r"\bcompletely\s+natural\b", "a common visible feature"),
        (r"\b(?:beautifully|wonderfully|gorgeously)\b", ""),
        (r"\b(?:beautiful|gorgeous|lovely)\b(?:,\s*)?", "refined"),
        (r"\b(?:great|excellent) foundational support\b", "visible definition"),
        (r"\bfoundational support\b", "visible definition"),
        (r"\bnatural firmness\b", "defined appearance"),
        (r"\b(?:natural )?bounce\b", "smooth-looking quality"),
        (r"\b(?:will|can) instantly\b", "can help"),
        (r"\binstantly\b", ""),
        (r"\bpotent\b", "targeted"),
        (r"\babsolute best\b", "best"),
        (r"\bgo-to\b", "focused option"),
        (
            r"\bprevent future (?:uv|sun) damage\b",
            "support daily protection from UV exposure",
        ),
        (
            r"\bprevent future (?:fine lines|wrinkles)\b",
            "soften the look of fine lines",
        ),
        (
            r"\b(?:caused by|from|due to) collagen depletion\b",
            "visible in the photographed area",
        ),
        (
            r"\b(?:natural,?\s+)?healthy reflection of light\b",
            "natural reflection of light",
        ),
        (r"\bhealthy skin\b", "fresh-looking skin"),
        (r"\bhealthy sheen\b", "natural sheen"),
        (r"\bdo not stand out prominently\b", "appear subtle"),
        (r"\bdoes not stand out prominently\b", "appears subtle"),
        (r"\bdo not stand out\b", "appear subtle"),
        (r"\bdoes not stand out\b", "appears subtle"),
        (r"\bsun damage\b", "visible sun-exposure signs"),
        (r"\blight based\b", "light-based"),
        (
            r",?\s*often\s+referred\s+to\s+as\s+strawberry\s+legs,?",
            ",",
        ),
        (
            r"\b(?:healthy|strong|intact) (?:skin )?barrier\b",
            "smooth-looking surface",
        ),
        (
            r"\b(?:excellent|strong|healthy|great) (?:natural )?(?:skin )?elasticity\b",
            "defined appearance",
        ),
        (r"\b(?:natural )?(?:skin )?elasticity\b", "defined appearance"),
        (r"\bhighly effective\b", "focused"),
        (r"\bclinically proven\b", "provider-selected"),
        (r"\bguarantees\b", "is designed to support"),
        (r"\bguarantee\b", "support"),
        (r"\bguaranteed results?\b", "results that can vary by guest"),
        (r"\bguaranteed\b", "provider-personalized"),
        (
            r"\bmakes? skin act younger at the cellular level\b",
            "supports a clearer, more even-looking complexion",
        ),
        (
            r"\b(?:(?:can|may|will)\s+)?(?:amplify|amplifies|boost|boosts) collagen(?: induction)? (?:by )?40\s*(?:-|to)\s*50%",
            "supports collagen renewal",
        ),
        (
            r"\bkills acne bacteria\b",
            "is used as part of an acne-focused blue-light protocol",
        ),
    )

    def strip_em_dashes(obj):
        if isinstance(obj, str):
            cleaned = obj.replace("\u2014", ", ").replace("\u2013", " to ")
            for pattern, replacement in overclaim_replacements:
                cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s{2,}", " ", cleaned)
            cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
            cleaned = re.sub(r",\s*,", ",", cleaned)
            cleaned = re.sub(r",\s*([.;])", r"\1", cleaned)
            cleaned = re.sub(
                r"\b(can|may|will)\s+help\s+help\s+",
                r"\1 help ",
                cleaned,
                flags=re.IGNORECASE,
            )
            cleaned = re.sub(
                r"\b([A-Za-z][A-Za-z-]*)\s+and\s+\1\b",
                r"\1",
                cleaned,
                flags=re.IGNORECASE,
            )
            cleaned = re.sub(
                r"\b(?:redness\s*(?:and|or|,)\s*visible redness|"
                r"visible redness\s*(?:and|or|,)\s*redness)\b",
                "visible redness",
                cleaned,
                flags=re.IGNORECASE,
            )
            cleaned = re.sub(
                r"\b(?:the\s+)?redness\s+(?:is|indicates|suggests|represents)\s+visible redness\b",
                "Visible redness is present",
                cleaned,
                flags=re.IGNORECASE,
            )
            cleaned = re.sub(
                r"\bvisible\s+visible\b",
                "visible",
                cleaned,
                flags=re.IGNORECASE,
            )
            cleaned = re.sub(
                r"\b(visible\s+(?:textural\s+marks|sun-exposure\s+signs))\s+"
                r"(is|was|has|appears|looks|seems|shows|indicates|suggests|represents)\b",
                lambda match: (
                    f"{match.group(1)} "
                    f"{ {'is': 'are', 'was': 'were', 'has': 'have', 'appears': 'appear', 'looks': 'look', 'seems': 'seem', 'shows': 'show', 'indicates': 'indicate', 'suggests': 'suggest', 'represents': 'represent'}[match.group(2).lower()] }"
                ),
                cleaned,
                flags=re.IGNORECASE,
            )
            cleaned = re.sub(
                r"\bthere\s+is\s+visible\s+textural\s+marks\b",
                "Visible textural marks are present",
                cleaned,
                flags=re.IGNORECASE,
            )
            cleaned = re.sub(
                r"\ba\s+visible\s+textural\s+marks\b",
                "visible textural marks",
                cleaned,
                flags=re.IGNORECASE,
            )
            cleaned = re.sub(
                r"\ba\s+(?=(?:appealing|elegant|even|excellent|option)\b)",
                "an ",
                cleaned,
                flags=re.IGNORECASE,
            )
            cleaned = re.sub(
                r"\ban\s+(?=(?:congestion|surface|visible|textural)\b)",
                "a ",
                cleaned,
                flags=re.IGNORECASE,
            )
            cleaned = re.sub(r"\b(?:a|an)\s+(?=[,.;:])", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(
                r"(^|[.!?]\s+)([a-z])",
                lambda match: f"{match.group(1)}{match.group(2).upper()}",
                cleaned,
            )
            if obj[:1].isupper() and cleaned[:1].islower():
                cleaned = cleaned[:1].upper() + cleaned[1:]
            return cleaned
        elif isinstance(obj, dict):
            return {
                key: (
                    value
                    if key in _NON_COPY_RESPONSE_FIELDS
                    else strip_em_dashes(value)
                )
                for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [strip_em_dashes(item) for item in obj]
        return obj

    return strip_em_dashes(analysis)


_ABSENCE_BASED_POSITIVE_PHRASES = (
    "does not stand out",
    "not visible",
    "no visible",
    "absence of",
    "without visible",
    "without noticeable",
    "without any",
    "free of",
    "free from",
    "little to no",
    "barely visible",
    "minimal ",
    "minimally",
    "lack of",
    "few visible",
    "skin barrier",
    "well-regulated",
    "canvas for treatment",
    "foundation for rejuvenating treatment",
    "bone and muscular structure",
    "structural integrity of your skin",
    "structurally strong",
    "structural support",
    "natural elasticity",
    "skin elasticity",
    "overall skin health",
    "indicating good hydration",
    "indicating good surface hydration",
    "healthy reflection",
    "circulation",
    "blood flow",
    "collagen level",
    "collagen density",
    "collagen stores",
    "hydration level",
    "hydration status",
    "barrier health",
    "barrier function",
    "cell turnover",
    "sebum production",
    "inflammation",
    "fantastic",
    "incredible",
    "perfect",
    "remarkably",
    "wonderfully",
    "gorgeous",
)

_DIRECT_POSITIVE_COPY = {
    "face": {
        "wrinkles": ("Soft, Rested Features", "Your features have a relaxed, rested quality that feels naturally fresh."),
        "redness": ("Calm, Balanced Complexion", "Your complexion appears calm, balanced, and composed."),
        "darkSpots": ("Clear, Luminous Finish", "Your complexion has a clear, luminous finish with an appealing glow."),
        "texture": ("Refined Texture", "Your skin has a smooth, refined surface and a naturally polished look."),
        "pores": ("Softly Diffused Finish", "Your complexion has a softly diffused finish in the photographed area."),
        "laxity": ("Natural Definition", "Your facial contours have a softly defined, balanced appearance."),
        "sunDamage": ("Fresh-Looking Skin", "Your skin has a fresh, luminous quality worth preserving."),
        "unevenTone": ("Luminous Tone", "Your complexion has an even, luminous quality with a fresh radiance."),
    },
    "neck_chest": {
        "wrinkles": ("Elegant Surface", "Your neck and chest have a smooth, elegant surface that catches the light softly."),
        "redness": ("Calm, Balanced Skin", "The skin across your neck and chest looks calm, balanced, and composed."),
        "laxity": ("Graceful Definition", "The visible contours of your neck and chest appear smooth and naturally defined."),
        "sunDamage": ("Fresh, Luminous Appearance", "Your neck and chest have a fresh, luminous appearance worth preserving."),
        "texture": ("Smooth Texture", "The visible skin has a smooth, refined texture and a polished finish."),
    },
    "hands": {
        "dryness": ("Smooth Surface Finish", "The skin on your hands has a smooth-looking, even surface finish."),
        "laxity": ("Graceful Definition", "Your hands have graceful, naturally defined contours."),
        "sunDamage": ("Warm, Even Tone", "Your hands have a warm, even tone that looks balanced."),
        "texture": ("Smooth Surface", "The skin on your hands has a smooth, refined surface."),
        "veins": ("Elegant Contours", "The visible contours of your hands have an elegant, composed look."),
    },
    "back": {
        "acne": ("Calm, Balanced Appearance", "The visible skin across your back has a calm, balanced appearance."),
        "hairRemoval": ("Well-Kept Appearance", "The photographed area has a clean, well-kept appearance."),
        "scarring": ("Cohesive Finish", "The photographed area has a cohesive visual finish."),
        "texture": ("Refined Texture", "The skin across your back has a smooth, refined texture."),
        "unevenTone": ("Even, Luminous Tone", "Your back has an even, luminous tone that catches the light softly."),
    },
    "legs": {
        "dryness": ("Smooth Surface Finish", "The skin on your legs has a smooth-looking, even surface finish."),
        "hairRemoval": ("Well-Kept Appearance", "The photographed area has a clean, well-kept appearance."),
        "sunDamage": ("Fresh, Even Tone", "Your legs have a fresh, even tone that looks balanced."),
        "texture": ("Smooth Surface", "The skin on your legs has a smooth, refined surface."),
        "veins": ("Balanced Appearance", "Your legs have a clear, balanced appearance and graceful definition."),
    },
}

_UNIVERSAL_POSITIVE_COPY = {
    "face": (
        "Your features have a natural character that is unmistakably your own.",
        "guestIdentity",
    ),
    "neck_chest": (
        "The natural character of your neck and chest is unmistakably your own.",
        "guestIdentity",
    ),
    "hands": (
        "Your hands have a natural character that is unmistakably your own.",
        "guestIdentity",
    ),
    "back": (
        "The natural character of the photographed area is unmistakably your own.",
        "guestIdentity",
    ),
    "legs": (
        "Your legs have a natural character that is unmistakably your own.",
        "guestIdentity",
    ),
}

_PHOTO_STARTING_POINT_POSITIVE = {
    "title": "A Clear Starting Point",
    "detail": (
        "This photograph gives us a clear starting point for care focused on "
        "what matters most to you."
    ),
    "groundedIn": "photoClarity",
}


def _is_absence_based_positive(highlight):
    """Return whether a purported strength is framed as a missing flaw."""
    if not isinstance(highlight, dict):
        return True
    combined = " ".join(
        str(highlight.get(field, "")).strip().lower()
        for field in ("title", "detail")
    )
    combined = combined.strip()
    if not combined:
        return True
    padded = f" {combined} "
    return " no " in padded or any(
        phrase in combined for phrase in _ABSENCE_BASED_POSITIVE_PHRASES
    )


_POSITIVE_SCORE_CONTRADICTIONS = {
    "wrinkles": re.compile(r"\b(?:soft|smooth),?\s+rested\b|\brested features?\b", re.I),
    "redness": re.compile(r"\b(?:calm|composed)\b", re.I),
    "darkSpots": re.compile(r"\bclear(?:-looking)?(?:,\s+\w+)?\s+(?:complexion|finish|skin)\b", re.I),
    "texture": re.compile(r"\b(?:smooth|refined|polished|cohesive)(?:-looking)?\b", re.I),
    "laxity": re.compile(r"\b(?:(?:softly|naturally)\s+)?(?:defined|graceful)(?:,\s+\w+)?\s+(?:appearance|contours?|definition)\b|\bnaturally defined\b", re.I),
    "unevenTone": re.compile(r"\b(?:even|harmonious|balanced)(?:-looking)?(?:,\s+\w+)?\s+(?:tone|complexion|quality)\b", re.I),
    "dryness": re.compile(r"\b(?:soft|supple|comfortable-looking|well cared for)\b", re.I),
    "acne": re.compile(r"\b(?:calm|clear)(?:,? balanced)?\s+(?:appearance|skin)\b|\bfresh-looking foundation\b", re.I),
    "hairRemoval": re.compile(r"\b(?:smooth|polished)\s+finish\b|\bclean visual (?:finish|line)\b", re.I),
}


def _positive_contradicts_scores(highlight, concerns):
    """Reject praise that directly conflicts with a severe concern score."""
    if not isinstance(highlight, dict) or not isinstance(concerns, dict):
        return True
    text = " ".join(
        str(highlight.get(field, "")).strip()
        for field in ("title", "detail")
    )
    for concern_key, pattern in _POSITIVE_SCORE_CONTRADICTIONS.items():
        concern = concerns.get(concern_key)
        score = concern.get("score") if isinstance(concern, dict) else None
        if isinstance(score, int) and score >= 61 and pattern.search(text):
            return True
    return False


def _derive_positive_highlights(concerns, body_area):
    """Return two deterministic, score-grounded strengths for a completed result."""
    if not isinstance(concerns, dict):
        return []
    area = body_area if body_area in _DIRECT_POSITIVE_COPY else "face"
    templates = _DIRECT_POSITIVE_COPY[area]
    ranked_concerns = sorted(
        (
            (key, value.get("score"))
            for key, value in concerns.items()
            if (
                key in templates
                and isinstance(value, dict)
                and type(value.get("score")) is int
                and 0 <= value["score"] <= 100
            )
        ),
        key=lambda item: (item[1], item[0]),
    )
    grounded_highlights = []
    for key, score in ranked_concerns:
        direct = {
            "title": templates[key][0],
            "detail": templates[key][1],
            "groundedIn": key,
        }
        if score <= 40 and not _positive_contradicts_scores(direct, concerns):
            grounded_highlights.append(direct)
        if len(grounded_highlights) == 2:
            return grounded_highlights

    # A standard photograph cannot support flattering claims that conflict
    # with moderate or severe concern scores. When fewer than two direct
    # appearance strengths are defensible, lead with warm human truth and the
    # value of a clear starting image instead of praising the concern itself.
    identity_detail, identity_grounding = _UNIVERSAL_POSITIVE_COPY[area]
    universal_highlights = (
        {
            "title": "Distinctly Yours",
            "detail": identity_detail,
            "groundedIn": identity_grounding,
        },
        dict(_PHOTO_STARTING_POINT_POSITIVE),
    )
    for highlight in universal_highlights:
        if len(grounded_highlights) == 2:
            break
        grounded_highlights.append(highlight)
    return grounded_highlights


def _repair_positive_highlights(analysis, body_area):
    """Replace provider-authored praise with deterministic score-grounded copy."""
    highlights = analysis.get("positiveHighlights")
    concerns = analysis.get("concerns")
    if not isinstance(concerns, dict):
        return analysis

    old_opening = ""
    if isinstance(highlights, list) and highlights and isinstance(highlights[0], dict):
        old_opening = str(highlights[0].get("detail", "")).strip()
    analysis["positiveHighlights"] = _derive_positive_highlights(
        concerns,
        body_area,
    )
    summary = str(analysis.get("summary", "")).strip()
    if old_opening and summary.lower().startswith(old_opening.lower()):
        analysis["summary"] = summary[len(old_opening):].lstrip(" .")
    return analysis


_AREA_LABELS = {
    "face": "face",
    "neck_chest": "neck and chest",
    "hands": "hands",
    "back": "back",
    "legs": "legs",
}


def _area_mismatch_result(selected_area, observed_area):
    """Return a specific, actionable mismatch without exposing model copy."""
    selected = selected_area if selected_area in _AREA_LABELS else "face"
    observed = observed_area if observed_area in _AREA_LABELS else "other"
    selected_label = _AREA_LABELS[selected]
    if observed == "other":
        return {
            "rejected": True,
            "reasonCode": "area_mismatch",
            "observedArea": "other",
            "reason": (
                f"This photo does not clearly show the selected {selected_label}. "
                "Choose the matching area and upload a clear photo of it."
            ),
        }
    observed_label = _AREA_LABELS[observed]
    return {
        "rejected": True,
        "reasonCode": "area_mismatch",
        "observedArea": observed,
        "reason": (
            f"This photo appears to show {observed_label}, not the selected "
            f"{selected_label}. Choose {observed_label.title()} and upload it again."
        ),
    }


_REJECTION_COPY = {
    "not_skin": (
        "This doesn't appear to be a photo of skin. Please upload a clear, "
        "well-lit photo of the face, neck and chest, hands, back, or legs."
    ),
    "quality": (
        "We couldn't get a clear enough read on this photo. Upload a well-lit, "
        "in-focus photo taken at arm's length in natural light."
    ),
    "filtered": (
        "This photo appears to be filtered or captured from a screen. Upload an "
        "original, unfiltered camera photo for a more reliable preview."
    ),
    "minor": (
        "Our skin analysis is designed for adults age 18 and older. If you're "
        "over 18, try a different clear photo and we'll take another look."
    ),
}

_UNDEREXPOSED_BRIGHTEST_DECILE_LIMIT = 80
_UNDEREXPOSED_95TH_PERCENTILE_LIMIT = 120
_UNDEREXPOSED_LUMINANCE_SPAN_LIMIT = 100


def _local_image_quality_rejection(image):
    """Return a bounded quality rejection for a severely underexposed image."""
    if image is None:
        return None
    preview = image.copy()
    preview.thumbnail((256, 256))
    histogram = preview.convert("L").histogram()
    pixel_count = sum(histogram)
    if pixel_count <= 0:
        return None
    def percentile(percent):
        percentile_position = pixel_count * percent
        cumulative = 0
        for luminance, count in enumerate(histogram):
            cumulative += count
            if cumulative >= percentile_position:
                return luminance
        return 255

    fifth_percentile = percentile(0.05)
    brightest_decile_floor = percentile(0.90)
    ninety_fifth_percentile = percentile(0.95)
    luminance_span = ninety_fifth_percentile - fifth_percentile
    print(
        "  [Image] Brightness preflight: "
        f"90th percentile={brightest_decile_floor}, "
        f"95th percentile={ninety_fifth_percentile}, "
        f"5th-to-95th span={luminance_span}"
    )
    if (
        brightest_decile_floor < _UNDEREXPOSED_BRIGHTEST_DECILE_LIMIT
        and ninety_fifth_percentile < _UNDEREXPOSED_95TH_PERCENTILE_LIMIT
        and luminance_span < _UNDEREXPOSED_LUMINANCE_SPAN_LIMIT
    ):
        return {
            "rejected": True,
            "reasonCode": "quality",
            "rejectionSource": "local_underexposure",
            "reason": _REJECTION_COPY["quality"],
        }
    return None


def _classify_model_rejection(reason):
    """Map provider rejection prose to a stable server-controlled reason code."""
    text = str(reason or "").strip().lower()
    if text == "area_mismatch":
        return "area_mismatch"
    if re.search(r"under\s*18|minor|child|adults?\s*\(?(?:18\+|18 and older)", text):
        return "minor"
    if re.search(r"filtered|beauty filter|screenshot|screen|watermark|original photo", text):
        return "filtered"
    if re.search(r"clear enough|too (?:blurry|dark)|low resolution|well-lit|in-focus", text):
        return "quality"
    if re.search(r"not (?:a )?(?:photo of )?(?:human )?skin|doesn'?t appear to be.*skin|non-skin", text):
        return "not_skin"
    return "other"


def _normalize_model_rejection(analysis, selected_area):
    """Return bounded, stable rejection copy while preserving observed anatomy."""
    reason = str(analysis.get("reason", "")).strip()
    if not reason:
        raise _GoogleResponseError(
            "Google Gemini returned a rejection without a reason"
        )
    observed_area = analysis.get("observedArea")
    if observed_area is not None and observed_area not in {
        *AREA_CONCERN_KEYS,
        "other",
    }:
        raise _GoogleResponseError(
            "Google Gemini returned an invalid observed area"
        )
    reason_code = _classify_model_rejection(reason)
    if reason_code == "other":
        raise _GoogleResponseError(
            "Google Gemini returned an unclassified image rejection"
        )
    if reason_code == "area_mismatch" or (
        observed_area is not None and observed_area != selected_area
    ):
        if observed_area is None or observed_area == selected_area:
            raise _GoogleResponseError(
                "Google Gemini returned an incomplete area mismatch"
            )
        return _area_mismatch_result(selected_area, observed_area)

    normalized = {
        "rejected": True,
        "reasonCode": reason_code,
        "reason": _REJECTION_COPY[reason_code],
    }
    if observed_area is not None:
        normalized["observedArea"] = observed_area
    return normalized


def _confirmed_model_rejection(candidates):
    """Require agreement from two model attempts for any image-based rejection."""
    seen = {}
    for candidate in candidates:
        reason_code = candidate.get("reasonCode")
        key = (
            reason_code,
            candidate.get("observedArea") if reason_code == "area_mismatch" else None,
        )
        seen[key] = seen.get(key, 0) + 1
        if seen[key] >= 2:
            return candidate
    return None

_ANATOMY_FAMILY_PATTERNS = {
    "face": r"\b(?:face|facial|complexion|forehead|temples?|brows?|eyes?|eyelids?|under[- ]eyes?|crow'?s[- ]feet|cheeks?|nose|nasal|lips?|mouth|chin|jaws?|jawlines?|jowls?|nasolabial|marionette|t[- ]zone)\b",
    "neck_chest": r"\b(?:neck|throat|cervical|chest|upper[- ]chest|sternum|décolletage|decolletage|décolleté|decollete|collarbones?|clavicles?)\b",
    "shoulder": r"\b(?:shoulders?|shoulder[- ]blades?)\b",
    "hands": r"\b(?:hands?|fingers?|thumbs?|knuckles?|nails?|cuticles?|wrists?|palms?)\b",
    "back": r"\b(?:back|upper[- ]back|mid[- ]back|lower[- ]back|scapula|scapulae|scapular|spine|spinal|torso|trunk)\b",
    "legs": r"\b(?:legs?|thighs?|knees?|kneecaps?|calf|calves|shins?|ankles?|feet|foot|toes?)\b",
    "unsupported": r"\b(?:arms?|upper[- ]arms?|forearms?|elbows?|abdomen|abdominal|stomach|waist|hips?|buttocks?|glutes?|scalp)\b",
}

_ALLOWED_ANATOMY_FAMILIES = {
    "face": {"face"},
    "neck_chest": {"neck_chest", "shoulder"},
    "hands": {"hands"},
    "back": {"back", "shoulder"},
    "legs": {"legs"},
}

_FORBIDDEN_ANATOMY_BY_AREA = {
    area: re.compile(
        "|".join(
            f"(?:{pattern})"
            for family, pattern in _ANATOMY_FAMILY_PATTERNS.items()
            if family not in allowed
        ),
        re.IGNORECASE,
    )
    for area, allowed in _ALLOWED_ANATOMY_FAMILIES.items()
}

_VISIBLE_ATTRIBUTE_NAMES = {
    "wrinkles": "fine lines",
    "redness": "visible redness",
    "darkSpots": "visible pigmentation",
    "texture": "surface texture",
    "pores": "pore visibility",
    "laxity": "firmness",
    "sunDamage": "visible signs of sun exposure",
    "unevenTone": "tonal variation",
    "dryness": "surface dryness",
    "veins": "visible vascularity",
    "acne": "surface congestion",
    "scarring": "textural marks",
    "hairRemoval": "visible hair growth",
}


def _has_anatomical_mismatch(value, body_area):
    pattern = _FORBIDDEN_ANATOMY_BY_AREA.get(
        body_area,
        _FORBIDDEN_ANATOMY_BY_AREA["face"],
    )
    text = str(value or "")
    determiner = r"(?:(?:the|your|his|her|their|each|my|our|a|an)\s+)?"
    if body_area == "hands":
        text = re.sub(
            rf"\bbacks?\s+of\s+{determiner}hands?\b",
            "hand surface",
            text,
            flags=re.IGNORECASE,
        )
    elif body_area == "neck_chest":
        text = re.sub(
            rf"\bback\s+of\s+{determiner}neck\b",
            "neck surface",
            text,
            flags=re.IGNORECASE,
        )
    elif body_area == "legs":
        text = re.sub(
            rf"\bbacks?\s+of\s+{determiner}(?:legs?|knees?|calf|calves)\b",
            "leg surface",
            text,
            flags=re.IGNORECASE,
        )
    return bool(pattern.search(text))


def _replacement_concern_description(body_area, concern_key, score):
    area_label = _AREA_LABELS.get(body_area, _AREA_LABELS["face"])
    attribute = _VISIBLE_ATTRIBUTE_NAMES.get(concern_key, "surface detail")
    if score <= 25:
        return f"{attribute.capitalize()} appears subtle across the photographed {area_label}."
    qualifier = "Subtle" if score <= 40 else "Visible" if score <= 65 else "More pronounced"
    return f"{qualifier} {attribute} appears across the photographed {area_label}."


def _repair_anatomical_mismatches(analysis, body_area):
    """Prevent one selected body area from receiving copy about another."""
    area = body_area if body_area in _AREA_LABELS else "face"
    area_label = _AREA_LABELS[area]

    highlights = analysis.get("positiveHighlights")
    if isinstance(highlights, list):
        for index, highlight in enumerate(highlights):
            if isinstance(highlight, dict) and _has_anatomical_mismatch(
                f"{highlight.get('title', '')} {highlight.get('detail', '')}",
                area,
            ):
                highlights[index] = {"title": "", "detail": ""}

    concerns = analysis.get("concerns")
    if isinstance(concerns, dict):
        for concern_key, concern in concerns.items():
            if not isinstance(concern, dict):
                continue
            description = concern.get("description", "")
            if _has_anatomical_mismatch(description, area):
                concern["description"] = _replacement_concern_description(
                    area,
                    concern_key,
                    concern.get("score", 50),
                )

    for recommendation in analysis.get("recommendations", []):
        if not isinstance(recommendation, dict):
            continue
        if _has_anatomical_mismatch(recommendation.get("reason", ""), area):
            target_names = [
                _VISIBLE_ATTRIBUTE_NAMES.get(target, target)
                for target in recommendation.get("targets", [])
            ]
            target_copy = " and ".join(target_names) or "visible goals"
            recommendation["reason"] = (
                f"A targeted option for {target_copy} across the photographed "
                f"{area_label}; your provider can confirm candidacy in person."
            )

    for product in analysis.get("productRecommendations", []):
        if not isinstance(product, dict):
            continue
        if _has_anatomical_mismatch(product.get("reason", ""), area):
            product["reason"] = (
                f"An at-home option selected for the photographed {area_label}; "
                "your provider can personalize how it fits your routine."
            )

    if _has_anatomical_mismatch(analysis.get("summary", ""), area):
        consultation = (
            "An in-person VISIA consultation"
            if area == "face"
            else "An in-person consultation"
        )
        analysis["summary"] = (
            f"Your results highlight visible strengths and a few opportunities "
            f"to refine the photographed {area_label}. {consultation} can confirm "
            "the best treatment and skincare plan."
        )
    return analysis


def _ensure_positive_first_summary(analysis, body_area=None):
    """Lead with a strength; optionally replace model prose with grounded copy."""
    highlights = analysis.get("positiveHighlights", [])
    summary = str(analysis.get("summary", "")).strip()
    if not highlights or not isinstance(highlights[0], dict):
        return analysis
    opening = str(highlights[0].get("detail", "")).strip()
    if body_area in AREA_CONCERN_KEYS and opening:
        concerns = analysis.get("concerns", {})
        ranked_visible = sorted(
            (
                (key, concern.get("score"))
                for key, concern in concerns.items()
                if (
                    key in AREA_CONCERN_KEYS[body_area]
                    and isinstance(concern, dict)
                    and type(concern.get("score")) is int
                    and concern["score"] > 10
                    and (
                        key != "hairRemoval"
                        or _supports_laser_hair_removal(analysis)
                    )
                )
            ),
            key=lambda item: (-item[1], item[0]),
        )
        goals = [
            _CONCERN_GOAL_LABELS.get(key, "visible surface variation")
            for key, _ in ranked_visible[:2]
        ]
        if len(goals) == 2:
            finding_copy = (
                f"This photo-based preview found the clearest opportunities in "
                f"{goals[0]} and {goals[1]}."
            )
        elif len(goals) == 1:
            finding_copy = (
                f"This photo-based preview found the clearest opportunity in {goals[0]}."
            )
        else:
            finding_copy = "Visible variation appears subtle in this photograph."
        consultation = (
            "A VISIA consultation"
            if body_area == "face"
            else "An in-person consultation"
        )
        analysis["summary"] = (
            f"{opening} {finding_copy} {consultation} can confirm what a standard "
            "photo cannot and shape a personalized treatment and skincare plan."
        )
        return analysis
    if opening and not summary.lower().startswith(opening.lower()):
        analysis["summary"] = f"{opening} {summary}".strip()
    return analysis


_DIAGNOSTIC_CONDITION = (
    r"(?:rosacea|melasma|acne|dermatitis|eczema|psoriasis|"
    r"keratosis pilaris|skin cancer|melanoma|malignan(?:t|cy)|"
    r"basal[- ]cell(?: carcinoma)?|squamous[- ]cell(?: carcinoma)?|"
    r"bcc|scc|actinic keratosis|seborrheic keratosis|lentigo maligna|"
    r"dysplastic nevus|concerning mole|telangiectasia|xerosis|"
    r"venous insufficiency|folliculitis)"
)
_PROHIBITED_MEDICAL_TERM_PATTERN = re.compile(
    _DIAGNOSTIC_CONDITION,
    re.IGNORECASE,
)
_UNSUPPORTED_APPEARANCE_TERM_PATTERN = re.compile(
    r"\b(?:hyper[- ]?pigmentation|photo[- ]?aging|scarr(?:ing|ed)?|scars?|"
    r"dehydration|dehydrated)\b",
    re.IGNORECASE,
)
_PHOTO_HISTORY_OR_CAUSE_PATTERN = re.compile(
    r"(?:\b(?:past|prior|previous|historical|chronic|cumulative|routine|"
    r"regular|frequent|repeated|prolonged|long[- ]term|lifetime|habitual|"
    r"ongoing)\s+(?:unprotected\s+)?(?:sun|uv|environmental)\s+exposure\b|"
    r"\b(?:years?\s+of|history\s+of|routine\s+of)\s+(?:unprotected\s+)?"
    r"(?:sun|uv)\s+exposure\b|"
    r"\b(?:past|prior|previous|chronic|cumulative|routine|regular|frequent|"
    r"repeated|prolonged|long[- ]term|lifetime)\s+exposure\s+to\s+"
    r"(?:the\s+)?(?:sun|uv)\b|"
    r"\b(?:sun|uv)\s+exposure\s+(?:history|over\s+time|"
    r"through(?:out)?\s+the\s+years?)\b|"
    r"\byears?\s+(?:spent\s+)?(?:in|under)\s+the\s+sun\b|"
    r"\b(?:likely\s+)?reflect(?:s|ed|ing)?\s+"
    r"(?:time\s+spent\s+)?(?:in|under)\s+(?:the\s+)?sun\b|"
    r"\breflect(?:s|ed|ing)?\s+"
    r"(?:(?:some|mild|visible|general|normal|incidental)\s+)?"
    r"(?:sun|uv|environmental)\s+exposure\b|"
    r"\b(?:often\s+)?accompan(?:y|ies|ied|ying)\s+"
    r"(?:(?:some|mild|visible|general|normal|incidental)\s+)?"
    r"(?:sun|uv|environmental)\s+exposure\b|"
    r"\b(?:suggest(?:s|ed|ing)?|indicat(?:e|es|ed|ing))\s+"
    r"(?:(?:some|mild|visible|general|normal|incidental)\s+)?"
    r"(?:sun|uv|environmental)\s+exposure\b|"
    r"\bpoint(?:s|ed|ing)?\s+to\s+"
    r"(?:(?:some|mild|visible|general|normal|incidental)\s+)?"
    r"(?:sun|uv|environmental)\s+exposure\b|"
    r"\b(?:is|are|was|were|appears?|seems?|can\s+be|may\s+be)\s+"
    r"(?:evidence|indicative|suggestive)\s+of\s+"
    r"(?:(?:some|mild|visible|general|normal|incidental)\s+)?"
    r"(?:sun|uv|environmental)\s+exposure\b|"
    r"\b(?:is|are|was|were|appears?|seems?|can\s+be|may\s+be)\s+"
    r"(?:associated|linked|related)\s+(?:with|to)\s+"
    r"(?:(?:some|mild|visible|general|normal|incidental)\s+)?"
    r"(?:sun|uv|environmental)\s+exposure\b|"
    r"\b(?:visible\s+)?signs?\s+of\s+"
    r"(?:sun|uv|environmental)\s+exposure\b|"
    r"\b(?:past|prior|previous|historical)\s+"
    r"(?:breakouts?|blemishes?|surface\s+congestion|irritation|marks?|"
    r"changes?|damage|injur(?:y|ies))\b|"
    r"\b(?:due\s+to|caused\s+by|from|likely\s+from|result(?:s|ing)?\s+from|"
    r"stemming\s+from|consistent\s+with)\s+"
    r"(?:(?:past|prior|chronic|cumulative|routine|regular|frequent|repeated|"
    r"prolonged|long[- ]term|lifetime|incidental)\s+)?"
    r"(?:sun|uv)\s+exposure\b|"
    r"\b(?:history\s+of|past|prior|previous|routine|regular|frequent|daily|"
    r"repeated|recent|ongoing|habitual)\s+"
    r"(?:shaving|razor\s+use|(?:surface\s+)?hair\s+removal)\b|"
    r"\b(?:shaving|razor\s+use|(?:surface\s+)?hair\s+removal)\s+"
    r"(?:history|routine|habits?|over\s+time)\b|"
    r"\b(?:due\s+to|caused\s+by|likely\s+from|result(?:s|ing)?\s+from|"
    r"stemming\s+from|consistent\s+with|after)\s+"
    r"(?:(?:routine|regular|frequent|daily|repeated|recent)\s+)?"
    r"(?:shaving|razor\s+use|(?:surface\s+)?hair\s+removal)\b|"
    r"\b(?:indicat(?:e|es|ed|ing)|suggest(?:s|ed|ing)?|"
    r"reflect(?:s|ed|ing)?|evidence\s+of|signs?\s+of)\s+"
    r"(?:(?:past|prior|previous|recent|regular|routine|frequent|daily|"
    r"repeated|ongoing|habitual)\s+)?(?:surface\s+)?hair\s+removal\b|"
    r"\b(?:associated|linked|related)\s+(?:with|to)\s+"
    r"(?:(?:past|prior|previous|recent|regular|routine|frequent|daily|"
    r"repeated|ongoing|habitual)\s+)?(?:surface\s+)?hair\s+removal\b|"
    r"\b(?:appears?|looks?|seems?)\s+(?:recently\s+)?shav(?:ed|en)\b|"
    r"\brecently\s+shav(?:ed|en)\b|\bpost[- ]shav(?:e|ing)\b|"
    r"\brazor\s+bumps?\b|\b(?:you|the\s+guest|they)\s+"
    r"(?:shave|shaves|use(?:s)?\s+(?:a\s+)?razor)\b|"
    r"\b(?:skin|surface|contour)\s+"
    r"(?:softens?|thins?|changes?)\s+over\s+time\b)",
    re.IGNORECASE,
)
_UNMEASURED_PHYSICAL_STATE_PATTERN = re.compile(
    r"(?:\bfirm(?:ness|er|est)?(?:[- ]looking)?\b|"
    r"\belastic(?:ity|[- ]looking)?\b|\bhydrat(?:ion|ed|ing)\b|"
    r"\bwell[- ]hydrated\b|\bmoist(?:ure|urized|urised)?\b|"
    r"\bmoistur(?:izing|ising)\b|\bsupple(?:ness)?\b|\bthickness\b|"
    r"\b(?:thin|thick)(?:ner|est)?(?:[- ]looking)?\s+(?:skin|surface)\b|"
    r"\b(?:skin|surface)\s+(?:appears?|looks?|is)\s+(?:thin|thick)\b|"
    r"\b(?:skin|surface\s+skin)\b[^.;]{0,35}\b"
    r"(?:thins?|thinned|thinning)\b|\bthinning\s+(?:surface\s+)?skin\b|"
    r"\bvolume\s+loss\b|\b(?:shifts?|changes?)\s+in\s+volume\b|"
    r"\bcollagen\s+(?:level|levels|content|density|loss|depletion|stores?)\b|"
    r"\bskin\s+barrier\b|\bunderlying\s+support\b|"
    r"\bwell[- ]supported\b|"
    r"\b(?:contours?|skin|surface)\b[^.;]{0,25}\b"
    r"(?:is|are|appears?|looks?|seems?)\s+supported\b)",
    re.IGNORECASE,
)
_OBSERVATION_RECOMMENDATION_CLAIM_PATTERN = re.compile(
    r"\b(?:ideal|excellent|good|clear)\s+(?:candidate|target)\s+for\b|"
    r"\b(?:ideal|excellent|good|clear)\s+for\s+"
    r"(?:laser|treatment|procedure)\b",
    re.IGNORECASE,
)

_OBSERVABLE_CONCERN_COPY = {
    "wrinkles": "lines and creases",
    "redness": "redness",
    "darkSpots": "pigment variation",
    "texture": "surface texture variation",
    "pores": "pore visibility",
    "laxity": "contour softness or crepiness",
    "sunDamage": "pigment variation and sun-exposure signs",
    "unevenTone": "tone variation",
    "acne": "surface congestion",
    "scarring": "textural marks",
    "hairRemoval": "body-hair growth",
    "veins": "surface veins",
    "dryness": "surface dryness",
}
_PLURAL_OBSERVABLE_CONCERNS = {"wrinkles", "sunDamage", "scarring", "veins"}


def _bounded_visible_concern_description(concern_key, concern):
    """Replace an inferred explanation with score-aligned visible wording."""
    label = _OBSERVABLE_CONCERN_COPY.get(concern_key, "surface variation")
    score = concern.get("score") if isinstance(concern, dict) else None
    plural = concern_key in _PLURAL_OBSERVABLE_CONCERNS
    if type(score) is not int:
        verb = "appear" if plural else "appears"
        return f"Visible {label} {verb} in the photographed area."
    if score <= 10:
        verb = "are" if plural else "is"
        return f"No prominent {label} {verb} visible in the photographed area."
    if score <= 40:
        verb = "are" if plural else "is"
        return f"Mild {label} {verb} visible in the photographed area."
    verb = "appear" if plural else "appears"
    return f"Clearly visible {label} {verb} in the photographed area."


def _bounded_hair_evidence_description(description, concern):
    """Preserve only affirmative treatment-relevant visual hair evidence."""
    for patterns, replacement in (
        (
            _QUALIFYING_HAIR_EVIDENCE_PATTERNS[:1],
            "Visible stubble is present in the photographed area.",
        ),
        (
            _QUALIFYING_HAIR_EVIDENCE_PATTERNS[1:4],
            "Distinct visible hair follicles and follicular contrast are present in the photographed area.",
        ),
        (
            _QUALIFYING_HAIR_EVIDENCE_PATTERNS[4:],
            "Clearly visible dark or coarse body hair is present in the photographed area.",
        ),
    ):
        for pattern in patterns:
            for match in pattern.finditer(description):
                if _hair_evidence_match_is_affirmative(description, match):
                    return replacement
    return _bounded_visible_concern_description("hairRemoval", concern)


def _bounded_neutral_rest_description(concern_key, concern):
    """Return controlled neck copy after neutral-rest evidence is validated."""
    score = concern.get("score") if isinstance(concern, dict) else None
    if type(score) is not int or score <= 40:
        return _bounded_visible_concern_description(concern_key, concern)
    label = _OBSERVABLE_CONCERN_COPY.get(concern_key, "surface variation")
    verb = "appear" if concern_key in _PLURAL_OBSERVABLE_CONCERNS else "appears"
    return (
        f"Clearly visible {label} {verb} at rest in a neutral resting view, "
        "independent of pose."
    )


def _repair_photo_observation_inferences(analysis, body_area):
    """Replace model-authored observation prose with controlled visible-only copy."""
    raw_observation_copy = [str(analysis.get("summary", ""))]
    concerns = analysis.get("concerns", {})
    if isinstance(concerns, dict):
        raw_observation_copy.extend(
            str(concern.get("description", ""))
            for concern in concerns.values()
            if isinstance(concern, dict)
        )

    # Medical labels remain a hard rejection. Other unsupported model prose is
    # recoverable because none of it is allowed to reach the guest response.
    if any(
        _PROHIBITED_MEDICAL_TERM_PATTERN.search(text)
        for text in raw_observation_copy
    ):
        raise _GoogleResponseError(
            "Google Gemini used an unsupported medical-condition label"
        )

    # Score/wording coherence and neck pose evidence must be assessed against
    # the original model evidence before that prose is discarded.
    _validate_score_description_coherence(analysis, body_area)

    if isinstance(concerns, dict):
        for concern_key, concern in concerns.items():
            if not isinstance(concern, dict):
                continue
            raw_description = str(concern.get("description", "")).strip()
            if concern_key == "hairRemoval":
                concern["description"] = _bounded_hair_evidence_description(
                    raw_description,
                    concern,
                )
            elif (
                body_area == "neck_chest"
                and concern_key in {"laxity", "wrinkles"}
            ):
                concern["description"] = _bounded_neutral_rest_description(
                    concern_key,
                    concern,
                )
            else:
                concern["description"] = _bounded_visible_concern_description(
                    concern_key,
                    concern,
                )

    # The completed summary is rebuilt later from score-grounded positives and
    # deterministic visible goals. Reset it now so no model sentence survives.
    analysis["summary"] = "A photo-based preview of visible surface features."
    return analysis
_DIAGNOSTIC_CLAIM_PATTERN = re.compile(
    rf"(?:"
    rf"\b(?:you have|we see|there (?:appears?|seems?) to be)\s+"
    rf"(?:visible\s+|possible\s+)?{_DIAGNOSTIC_CONDITION}\b|"
    rf"\b(?:i see|likely)\s+{_DIAGNOSTIC_CONDITION}\b|"
    rf"\b(?:it|that|those\s+(?:bumps?|spots?|patches?)|"
    rf"these\s+(?:bumps?|spots?|patches?)|the\s+(?:bumps?|spots?|rash|patches?))\s+"
    rf"(?:is|are|looks? like|appears? to be|seems? to be)\s+"
    rf"(?:a\s+|an\s+)?{_DIAGNOSTIC_CONDITION}\b|"
    rf"\b(?:your (?:skin|photo|appearance|complexion)|"
    rf"the (?:photo|image|appearance|area)|this(?: (?:area|appearance|photo))?|"
    rf"visible (?:skin|surface))\s+"
    rf"(?:has|shows?|reveals?|suggests?|looks? like|appears? to (?:have|be)|"
    rf"is (?:consistent with|suggestive of|likely)|may be|could be)\s+"
    rf"(?:a\s+|an\s+|visible\s+|possible\s+|likely\s+)?"
    rf"{_DIAGNOSTIC_CONDITION}\b|"
    rf"\b{_DIAGNOSTIC_CONDITION}\b.{{0,40}}\b"
    rf"(?:is|are|appears?|seems?)\s+(?:visible|present|likely|apparent|evident|shown)\b|"
    rf"\bdiagnos(?:e|es|ed|ing|is|tic)\b.{{0,80}}\b"
    rf"{_DIAGNOSTIC_CONDITION}\b"
    rf")",
    re.IGNORECASE,
)


def _guest_facing_strings(value):
    """Yield response strings without relying on a particular result field."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            if key in _NON_COPY_RESPONSE_FIELDS:
                continue
            yield from _guest_facing_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _guest_facing_strings(child)


def _validate_raw_model_observation_copy(analysis):
    """Reject labels, inferred history, and unmeasured states in photo copy."""
    observation_copy = [str(analysis.get("summary", ""))]
    for concern in analysis.get("concerns", {}).values():
        if isinstance(concern, dict):
            observation_copy.append(str(concern.get("description", "")))
    if any(
        _PROHIBITED_MEDICAL_TERM_PATTERN.search(text)
        or _UNSUPPORTED_APPEARANCE_TERM_PATTERN.search(text)
        or _PHOTO_HISTORY_OR_CAUSE_PATTERN.search(text)
        or _UNMEASURED_PHYSICAL_STATE_PATTERN.search(text)
        or _OBSERVATION_RECOMMENDATION_CLAIM_PATTERN.search(text)
        for text in observation_copy
    ):
        raise _GoogleResponseError(
            "Google Gemini used an unsupported condition, history, cause, or physical-state inference"
        )
    return analysis


_MILD_EVIDENCE_QUALIFIER = (
    r"(?:slight(?:ly)?|subtle(?:ly)?|minimal(?:ly)?|barely(?:\s+visible)?|"
    r"faint(?:ly)?|minor|mild(?:ly)?|very\s+mild|not\s+prominent)"
)
_CLEAR_EVIDENCE_QUALIFIER = (
    r"(?:moderate|severe|prominent|pronounced|significant|marked|"
    r"substantial|advanced|clearly\s+visible|widespread|diffuse|deep|"
    r"multiple|persistent)"
)
_CLEAR_NON_MILD_EVIDENCE_PATTERN = re.compile(
    rf"\b{_CLEAR_EVIDENCE_QUALIFIER}\b",
    re.IGNORECASE,
)
_CONCERN_EVIDENCE_TERMS = {
    "wrinkles": (
        r"(?:wrinkles?|lines?|creases?|bands?|folds?|rhytids?|furrows?|"
        r"crinkling|crow'?s\s+feet|neck\s+rings?)"
    ),
    "redness": (
        r"(?:redness|flushing|vascularity|vascular|warmth|pinkness|"
        r"red\s+(?:tone|hue)|blotchiness|visible\s+capillaries)"
    ),
    "darkSpots": (
        r"(?:dark\s+spots?|brown\s+(?:spots?|patches?)|dark\s+marks?|spots?|"
        r"pigment(?:ation)?\s+variation|pigment(?:ation|ed)?|discoloration|"
        r"freckl(?:e|es|ing)|mottling)"
    ),
    "texture": (
        r"(?:texture|roughness|bumpiness|surface\s+irregularity|"
        r"uneven\s+surface|coarseness|raised\s+bumps?)"
    ),
    "pores": (
        r"(?:pores?|pore\s+visibility|follicular\s+openings?|"
        r"(?:visible|enlarged)\s+openings?)"
    ),
    "laxity": (
        r"(?:laxity|crepiness|crepey|looseness|softness|folds?|contours?|"
        r"sagg(?:ing|y)?|drooping|slackness|loss\s+of\s+definition|"
        r"softened\s+definition)"
    ),
    "sunDamage": (
        r"(?:sun[- ]exposure\s+signs?|sun\s+spots?|brown\s+spots?|"
        r"mottling|uneven\s+pigment|solar\s+changes?|"
        r"pigment(?:ation)?\s+variation|pigment(?:ation|ed)?|"
        r"freckl(?:e|es|ing))"
    ),
    "unevenTone": (
        r"(?:uneven\s+tone|tone\s+variation|pigment(?:ation)?\s+variation|"
        r"discoloration|blotchiness|mottling|patchiness|color\s+variation|"
        r"uneven\s+coloration)"
    ),
    "acne": (
        r"(?:surface\s+congestion|congestion|breakouts?|blemishes?|bumps?|"
        r"comedones?|blackheads?|"
        r"whiteheads?|clogged\s+pores?|surface\s+bumps?)"
    ),
    "scarring": (
        r"(?:scarring|scars?|marks?|indentations?|depressions?|pitting|"
        r"pitted\s+texture|divots?|textural\s+(?:marks?|irregularity))"
    ),
    "hairRemoval": (
        r"(?:hair\s+growth|body\s+hair|coarse\s+hair|hairs?|stubble|"
        r"follicles?|follicular\s+(?:contrast|visibility|prominence|pattern))"
    ),
    "veins": (
        r"(?:veins?|vessels?|capillaries|vascularity|"
        r"vascular\s+(?:pattern|network)|branching\s+lines?)"
    ),
    "dryness": (
        r"(?:dryness|flaking|flakiness|dry\s+skin|dry|matte(?:ness)?|"
        r"scaling|scaliness|ashy|ashiness|peeling|parched)"
    ),
}
_EVIDENCE_PRE_ABSENCE_PATTERN = re.compile(
    r"(?:\bno(?:\s+(?:evidence|signs?))?\s+(?:of\s+)?|"
    r"\bno(?:\W+\w+){0,3}\W*|"
    r"\bnothing\s+(?:resembling|like)\s+|"
    r"\bwithout\s+|\bfree\s+of\s+|\b(?:a\s+)?lack\s+of\s+|"
    r"\bnot\s+(?!only\b)|\bpossible(?:\s+visible)?\s+|"
    r"\bpossibly(?:\s+visible)?\s+|"
    r"\bpotential(?:ly)?\s+|\bperhaps\s+|\bmaybe\s+|"
    r"\b(?:may|might|could)\s+be\s+|\bzero\s+|\bneither\s+|"
    r"\bnot\s+enough\s+to\s+(?:confirm|establish)\s+|"
    r"\bunclear\s+whether\s+)\s*$",
    re.IGNORECASE,
)
_EVIDENCE_POST_ABSENCE_PATTERN = re.compile(
    r"^\s*[:?,]*\s*(?:if\s+any\s*,?\s*)?"
    r"(?:(?:is|are|was|were|remains?|appears?|seems?|looks?)\s+)?(?:"
    r"not\s+(?:clearly\s+)?(?:visible|present|shown|apparent|detectable|seen|"
    r"observed|evident|demonstrated|supported)|"
    r"absent|missing|none|invisible|imperceptible|undetectable|indiscernible|"
    r"unclear|uncertain|unconfirmed|unverified|unsupported|indeterminate|"
    r"speculative|hypothetical|alleged(?:ly)?|reported(?:ly)?|borderline|"
    r"inconclusive|debatable|theoretical(?:ly)?|purported(?:ly)?|ostensibly|"
    r"putative(?:ly)?|tentative(?:ly)?|questionable|doubtful|equivocal|"
    r"ambiguous|unlikely|"
    r"negligible|weak|poor|"
    r"cannot\s+(?:reliably\s+)?(?:be\s+)?(?:seen|confirmed|verified|established)|"
    r"can't\s+(?:reliably\s+)?(?:be\s+)?(?:seen|confirmed|verified|established)|"
    r"could\s+not\s+(?:reliably\s+)?(?:be\s+)?(?:seen|confirmed|verified|established)|"
    r"may\s+be\s+(?:visible|present)|might\s+be\s+(?:visible|present)|"
    r"could\s+be\s+(?:visible|present)|"
    r"is\s+(?:anything\s+but|neither)\s+(?:clearly\s+)?visible|"
    r"is\s+(?:difficult|hard)\s+to\s+(?:see|confirm|verify)|"
    r"not\s+enough\s+to\s+(?:confirm|establish))\b",
    re.IGNORECASE,
)
_EVIDENCE_CLAUSE_UNCERTAINTY_PATTERN = re.compile(
    r"\b(?:nothing|unclear\s+whether|uncertain|questionable|unconfirmed|unverified|"
    r"unsupported|indeterminate|speculative|hypothetical|doubtful|equivocal|"
    r"alleged(?:ly)?|reported(?:ly)?|borderline|inconclusive|debatable|"
    r"theoretical(?:ly)?|purported(?:ly)?|ostensibly|putative(?:ly)?|"
    r"tentative(?:ly)?|hard\s+to\s+(?:discern|detect)|ambiguous|possible|"
    r"possibly|potential(?:ly)?|perhaps|maybe|unlikely|"
    r"may|might|could|if\s+any|"
    r"cannot\s+(?:reliably\s+)?(?:be\s+)?"
    r"(?:confirm(?:ed)?|see(?:n)?|establish(?:ed)?)|"
    r"can't\s+(?:reliably\s+)?(?:be\s+)?"
    r"(?:confirm(?:ed)?|see(?:n)?|establish(?:ed)?)|"
    r"couldn't\s+(?:reliably\s+)?(?:be\s+)?"
    r"(?:confirm(?:ed)?|see(?:n)?|establish(?:ed)?)|"
    r"fail(?:s|ed)?\s+to\s+(?:show|demonstrate|confirm|establish)|"
    r"not\s+(?:observed|evident|demonstrated|supported)|"
    r"does\s+not\s+(?:confirm|establish)|"
    r"do\s+not\s+(?:confirm|establish)|"
    r"not\s+enough(?:\s+evidence)?\s+to\s+"
    r"(?:confirm|establish)|anything\s+but\s+(?:clearly\s+)?"
    r"(?:visible|present|apparent|detectable|seen)|rules?\s+out|excludes?|"
    r"invisible|imperceptible|undetectable|indiscernible|zero|none|neither)\b",
    re.IGNORECASE,
)


def _observation_evidence_clause(description, match):
    """Return the semantic clause containing one concern-evidence term."""
    return _hair_evidence_clause(description, match)


def _concern_evidence_match_is_affirmative(description, match):
    """Reject absent or uncertain evidence without discarding nearby positives."""
    clause_start, clause_end = _evidence_clause_bounds(description, match)
    clause = description[clause_start:clause_end]
    prefix = description[clause_start:match.start()]
    suffix = description[match.end():clause_end]
    return not (
        _EVIDENCE_PRE_ABSENCE_PATTERN.search(prefix)
        or _EVIDENCE_POST_ABSENCE_PATTERN.search(suffix)
        or _EVIDENCE_CLAUSE_UNCERTAINTY_PATTERN.search(clause)
    )


def _affirmative_concern_evidence_matches(concern_key, description):
    """Return concern-specific visible evidence that is not absent or uncertain."""
    evidence_terms = _CONCERN_EVIDENCE_TERMS.get(concern_key)
    if not evidence_terms:
        return []
    matches = re.finditer(rf"\b{evidence_terms}\b", description, re.IGNORECASE)
    return [
        match
        for match in matches
        if _concern_evidence_match_is_affirmative(description, match)
    ]
_POSITIONAL_FOLD_PATTERN = re.compile(
    r"\b(?:single\s+(?:fold|crease|line)|natural\s+(?:fold|crease))\b|"
    r"\b(?:fold|crease|line|appearance)\b.{0,45}\b"
    r"(?:caused|created|explained|contribut(?:e|es|ed|ing))\b.{0,30}\b"
    r"(?:pose|position|posture|camera\s+angle|rotation|turned|turning|"
    r"flexion|flexed|extension|extended)\b|"
    r"\b(?:pose|position|posture|camera\s+angle|rotation|turned|turning|"
    r"flexion|flexed|extension|extended)\b.{0,30}\b"
    r"(?:causes?|creates?|explains?|contributes?)\b",
    re.IGNORECASE,
)
_NEUTRAL_REST_EVIDENCE_PATTERN = re.compile(
    r"\b(?:visible|persists?|persistent|remains?)\s+(?:clearly\s+)?at\s+rest\b|"
    r"\b(?:neutral\s+(?:resting\s+)?(?:view|position|posture)|"
    r"resting\s+(?:view|position|posture)|independent\s+of\s+"
    r"(?:pose|position|posture|rotation))\b",
    re.IGNORECASE,
)
_EXPLICIT_NON_NEUTRAL_VIEW_PATTERN = re.compile(
    r"\b(?:neck|head|view|photo|photograph|image|posture|position)\b.{0,18}"
    r"(?<!not\s)(?<!not\sa\s)(?<!isn't\s)(?<!wasn't\s)\b"
    r"(?:turned|rotated|flexed|extended|oblique|non-neutral)\b|"
    r"(?<!not\s)(?<!not\sa\s)(?<!isn't\s)(?<!wasn't\s)\b"
    r"(?:turned|rotated|flexed|extended|oblique|non-neutral)\b.{0,18}"
    r"\b(?:neck|head|view|photo|photograph|image|posture|position)\b|"
    r"\b(?:lines?|folds?|creases?|appearance)\b.{0,35}\b"
    r"(?:with|during|from)\s+(?:neck\s+)?"
    r"(?:rotation|turning|flexion|extension)\b",
    re.IGNORECASE,
)
_NEGATED_NEUTRAL_REST_EVIDENCE_PATTERN = re.compile(
    r"\b(?:no|not|without|lacks?|cannot|can't|could\s+not|couldn't|"
    r"unable\s+to|uncertain|unclear)\b[^.;,]{0,70}\b"
    r"(?:at\s+rest|neutral\s+(?:resting\s+)?"
    r"(?:view|position|posture)|resting\s+(?:view|position|posture)|"
    r"independent\s+of\s+(?:pose|position|posture|rotation))\b|"
    r"\b(?:at\s+rest|neutral\s+(?:resting\s+)?(?:view|position|posture)|"
    r"resting\s+(?:view|position|posture)|independent\s+of\s+"
    r"(?:pose|position|posture|rotation))\b[^.;,]{0,70}\b"
    r"(?:not\s+(?:shown|established|confirmed|used|available|possible)|"
    r"cannot\s+be\s+(?:confirmed|established|verified)|"
    r"can't\s+be\s+(?:confirmed|established|verified)|"
    r"is\s+uncertain|is\s+unclear)\b",
    re.IGNORECASE,
)

_VEIN_COLOR_DETAIL_PATTERN = re.compile(
    r"\b(?:blue|green|purple|bluish|greenish|purplish)\b",
    re.IGNORECASE,
)
_VEIN_STRUCTURE_DETAIL_PATTERN = re.compile(
    r"\b(?:branching|branched|network(?:ed)?|reticular|spider|raised|"
    r"bulging|rope[- ]like)\b",
    re.IGNORECASE,
)
_VEIN_EXTENT_DETAIL_PATTERN = re.compile(
    r"\b(?:multiple|numerous|clustered|widespread|diffuse)\b|"
    r"\b(?:veins?|vessels?|capillaries|vascularity)\b[^.;]{0,45}"
    r"\bacross\b|\bacross\b[^.;]{0,45}"
    r"\b(?:veins?|vessels?|capillaries|vascularity)\b",
    re.IGNORECASE,
)


def _has_corroborated_moderate_vein_evidence(description):
    """Require two independent visible details before moderate vascularity."""
    detail_categories = (
        _VEIN_COLOR_DETAIL_PATTERN.search(description),
        _VEIN_STRUCTURE_DETAIL_PATTERN.search(description),
        _VEIN_EXTENT_DETAIL_PATTERN.search(description),
    )
    return sum(bool(match) for match in detail_categories) >= 2


def _validate_score_description_coherence(analysis, body_area):
    """Conservatively align inconsistent scores with their written evidence."""
    area = body_area if body_area in AREA_CONCERN_KEYS else "face"
    concerns = analysis.get("concerns", {})
    if not isinstance(concerns, dict):
        return analysis

    for concern_key, concern in concerns.items():
        if not isinstance(concern, dict):
            continue
        score = concern.get("score")
        description = str(concern.get("description", "")).strip()
        if (
            not isinstance(score, int)
            or isinstance(score, bool)
            or not 0 <= score <= 100
        ):
            raise _GoogleResponseError(
                "Google Gemini returned an invalid visible-concern score"
            )
        if (
            concern_key == "hairRemoval"
            and score > 10
            and not _has_nonnegated_hair_evidence(description)
        ):
            concern["score"] = 10
            concern["severity"] = "none"
            print(
                "  [Score] Reset unsupported treatment-relevant body-hair "
                f"evidence from {score} to 10"
            )
            continue
        affirmative_evidence = _affirmative_concern_evidence_matches(
            concern_key,
            description,
        )
        if score > 10 and not affirmative_evidence:
            concern["score"] = 10
            concern["severity"] = "none"
            print(
                "  [Score] Reset an unsupported "
                f"{area}.{concern_key} score from {score} to 10"
            )
            continue
        if score < 41:
            continue
        if not description:
            raise _GoogleResponseError(
                "Google Gemini returned a moderate score without written evidence"
            )
        if (
            concern_key == "veins"
            and not _has_corroborated_moderate_vein_evidence(description)
        ):
            concern["score"] = 40
            concern["severity"] = "mild"
            print(
                "  [Score] Capped uncorroborated moderate "
                f"{area}.veins evidence from {score} to 40"
            )
            continue

        evidence_terms = _CONCERN_EVIDENCE_TERMS.get(concern_key)
        mild_primary_matches = []
        if evidence_terms:
            mild_primary_matches = list(re.finditer(
                rf"\b{_MILD_EVIDENCE_QUALIFIER}\b"
                rf"(?:\W+\w+){{0,3}}\W+\b{evidence_terms}\b|"
                rf"\b{evidence_terms}\b\W+"
                rf"(?:is|appears?|looks?|seems?|remains?)\W+"
                rf"(?:(?:only|quite|rather|relatively|somewhat)\W+)?"
                rf"\b{_MILD_EVIDENCE_QUALIFIER}\b",
                description,
                re.IGNORECASE,
            ))
        clear_primary_matches = []
        if evidence_terms:
            clear_primary_matches = list(re.finditer(
                rf"\b{_CLEAR_EVIDENCE_QUALIFIER}\b"
                rf"(?:\W+\w+){{0,3}}\W+\b{evidence_terms}\b|"
                rf"\b{evidence_terms}\b\W+"
                rf"(?:is|appears?|looks?|seems?|remains?)\W+"
                rf"(?:\w+\W+){{0,3}}"
                rf"\b{_CLEAR_EVIDENCE_QUALIFIER}\b",
                description,
                re.IGNORECASE,
            ))
        affirmative_clear_evidence = any(
            clear_match.start() <= evidence_match.start()
            and evidence_match.end() <= clear_match.end()
            for clear_match in clear_primary_matches
            for evidence_match in affirmative_evidence
        )
        affirmative_mild_evidence = False
        for mild_primary_match in mild_primary_matches:
            prefix = description[
                max(0, mild_primary_match.start() - 80):
                mild_primary_match.start()
            ]
            prefix = re.split(r"[.;,]", prefix)[-1]
            suffix = description[
                mild_primary_match.end():
                min(len(description), mild_primary_match.end() + 60)
            ]
            suffix = re.split(r"[.;,]", suffix)[0]
            negated_before = bool(re.search(
                r"\b(?:no(?:\s+(?:evidence|signs?)\s+of)?|never|"
                r"without|anything\s+but|far\s+from|hardly|isn't|isnt|"
                r"not(?:\s+merely|\s+even|\s+particularly|\s+prominent)?)\s*$",
                prefix,
                re.IGNORECASE,
            ))
            negated_after = bool(re.search(
                r"^\s*(?:(?:is|are)\s+)?(?:absent|not\s+present|"
                r"not\s+visible|missing)|^\s*(?:cannot|can't)\s+be\s+"
                r"(?:seen|confirmed)",
                suffix,
                re.IGNORECASE,
            ))
            if not negated_before and not negated_after:
                affirmative_mild_evidence = True
                break
        if affirmative_mild_evidence and not affirmative_clear_evidence:
            concern["score"] = 40
            concern["severity"] = "mild"
            print(
                "  [Score] Capped an internally inconsistent "
                f"{area}.{concern_key} score from {score} to 40"
            )
            continue

        if area == "neck_chest" and concern_key in {"laxity", "wrinkles"}:
            if _EXPLICIT_NON_NEUTRAL_VIEW_PATTERN.search(description):
                concern["score"] = 40
                concern["severity"] = "mild"
                print(
                    "  [Score] Capped a pose-dependent "
                    f"neck_chest.{concern_key} score from {score} to 40"
                )
                continue
            has_neutral_rest_evidence = bool(
                _NEUTRAL_REST_EVIDENCE_PATTERN.search(description)
            ) and not bool(
                _NEGATED_NEUTRAL_REST_EVIDENCE_PATTERN.search(description)
            )
            if (
                _POSITIONAL_FOLD_PATTERN.search(description)
            ):
                concern["score"] = 40
                concern["severity"] = "mild"
                print(
                    "  [Score] Capped a positional-fold "
                    f"neck_chest.{concern_key} score from {score} to 40"
                )
                continue
            if not has_neutral_rest_evidence:
                concern["score"] = 40
                concern["severity"] = "mild"
                print(
                    "  [Score] Capped an unconfirmed neutral-view "
                    f"neck_chest.{concern_key} score from {score} to 40"
                )
    return analysis


def _is_score_grounded_positive(highlight, concerns, body_area):
    """Return whether a card exactly matches deterministic score evidence."""
    if not isinstance(highlight, dict) or not isinstance(concerns, dict):
        return False
    return highlight in _derive_positive_highlights(concerns, body_area)


def _validate_final_completed_analysis(analysis, body_area):
    """Reject a completed candidate that cannot be shown safely and coherently."""
    area = body_area if body_area in AREA_CONCERN_KEYS else "face"
    if analysis.get("observedArea") != area:
        raise _GoogleResponseError(
            "Google Gemini returned a completed result for the wrong area"
        )

    highlights = analysis.get("positiveHighlights")
    expected_highlights = _derive_positive_highlights(
        analysis.get("concerns"),
        area,
    )
    if (
        not isinstance(highlights, list)
        or len(expected_highlights) != 2
        or highlights != expected_highlights
    ):
        raise _GoogleResponseError(
            "Server-derived positive highlights did not match the score evidence"
        )
    for highlight in highlights:
        if (
            not isinstance(highlight, dict)
            or not str(highlight.get("title", "")).strip()
            or not str(highlight.get("detail", "")).strip()
            or not _is_score_grounded_positive(
                highlight,
                analysis.get("concerns"),
                area,
            )
            or _is_absence_based_positive(highlight)
            or _positive_contradicts_scores(
                highlight,
                analysis.get("concerns"),
            )
            or _has_anatomical_mismatch(
                f"{highlight.get('title', '')} {highlight.get('detail', '')}",
                area,
            )
        ):
            raise _GoogleResponseError(
                "Google Gemini returned an unsupported positive highlight"
            )

    summary = str(analysis.get("summary", "")).strip()
    opening = str(highlights[0]["detail"]).strip()
    if not summary or not summary.lower().startswith(opening.lower()):
        raise _GoogleResponseError(
            "Google Gemini returned a result that did not lead with a visible strength"
        )

    guest_copy = list(_guest_facing_strings(analysis))
    if any(_PROHIBITED_MEDICAL_TERM_PATTERN.search(text) for text in guest_copy):
        raise _GoogleResponseError(
            "Google Gemini returned prohibited medical-condition language"
        )
    if any(_UNSUPPORTED_APPEARANCE_TERM_PATTERN.search(text) for text in guest_copy):
        raise _GoogleResponseError(
            "Google Gemini returned unsupported condition or inferred-state language"
        )
    if any(_DIAGNOSTIC_CLAIM_PATTERN.search(text) for text in guest_copy):
        raise _GoogleResponseError(
            "Google Gemini returned a diagnostic medical claim"
        )
    if any("\u2014" in text or "\u2013" in text for text in guest_copy):
        raise _GoogleResponseError(
            "Google Gemini returned prohibited dash punctuation"
        )
    return analysis


def _prepare_completed_analysis(analysis, body_area):
    """Repair raw model evidence before catalog mapping and final validation."""
    analysis = _repair_photo_observation_inferences(analysis, body_area)
    _validate_raw_model_observation_copy(analysis)
    _apply_score_correction(analysis)
    analysis = _sanitize_response(analysis)
    analysis = _repair_anatomical_mismatches(analysis, body_area)
    analysis = _repair_positive_highlights(analysis, body_area)
    analysis = _sanitize_response(analysis)
    analysis = _ensure_positive_first_summary(analysis, body_area)
    return analysis


def _finalize_completed_analysis(analysis, body_area):
    """Apply every guest-facing repair before a provider attempt can win."""
    analysis = _prepare_completed_analysis(analysis, body_area)
    return _validate_final_completed_analysis(analysis, body_area)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Analyze skin from uploaded image"""
    repeat_cache_key = None
    repeat_cache_owner = False

    # Check if file is in request
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files["image"]
    
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed. Accepted: jpg, jpeg, png, webp"}), 400
    
    # Check file size (10MB max)
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to start
    
    if file_size > 10 * 1024 * 1024:  # 10MB
        return jsonify({"error": "File too large. Maximum size is 10MB"}), 400
    
    # Read file bytes
    image_bytes = file.read()
    
    # Determine media type
    filename_lower = file.filename.lower()
    if filename_lower.endswith('.png'):
        media_type = "image/png"
    elif filename_lower.endswith('.webp'):
        media_type = "image/webp"
    else:  # jpg, jpeg
        media_type = "image/jpeg"
    
    body_area = request.form.get("body_area", "face")

    # Server-side age gate: reject minors
    user_age = request.form.get("age")
    if user_age:
        try:
            age_value = Decimal(user_age.strip())
            if not age_value.is_finite() or age_value < 0:
                raise InvalidOperation
            if age_value < 18:
                return jsonify({
                    "rejected": True,
                    "reason": "Our skin analysis and treatment recommendations are designed for adults (18+). Medical aesthetic treatments are not appropriate for minors."
                }), 422
        except (InvalidOperation, ValueError):
            return jsonify({
                "error": "Enter a valid age in years.",
                "code": "invalid_age",
                "retryable": False,
            }), 400

    # Demo mode
    if not LIVE_MODE:
        analysis = generate_demo_analysis(body_area)
        analysis["_isDemo"] = True
        return jsonify(analysis)

    # This fingerprint is derived only from bytes the server actually received.
    # Client-supplied hashes are intentionally ignored.
    repeat_cache_key = _analysis_repeat_key(
        image_bytes,
        body_area,
        user_age,
    )
    analysis_seed = _analysis_model_seed(
        image_bytes,
        body_area,
        user_age,
    )

    # Live mode - Google Gemini vision and analysis
    try:
        t_start = time.time()
        model_deadline = time.monotonic() + (GOOGLE_TOTAL_BUDGET_MS / 1000)

        # ── Normalize image bytes ──
        # Some JPEGs have unusual encoding (CMYK, progressive issues, EXIF
        # rotation) that can cause image-analysis APIs to reject them.
        # Re-encode through Pillow to produce clean RGB JPEG bytes.
        try:
            from PIL import Image as PILImage
            pil_img = PILImage.open(BytesIO(image_bytes))
            # Convert CMYK, RGBA, palette, etc. to RGB
            if pil_img.mode not in ("RGB",):
                pil_img = pil_img.convert("RGB")
            # Handle EXIF orientation
            try:
                from PIL import ImageOps
                pil_img = ImageOps.exif_transpose(pil_img)
            except Exception:
                pass
            # Match the former mobile canvas cap server-side so every browser
            # gets one deterministic normalization path without oversized input.
            if max(pil_img.size) > 1200:
                pil_img.thumbnail((1200, 1200), PILImage.Resampling.LANCZOS)
            # Re-encode as JPEG
            clean_buf = BytesIO()
            pil_img.save(clean_buf, format="JPEG", quality=92)
            image_bytes = clean_buf.getvalue()
            media_type = "image/jpeg"
            print(f"  [Image] Re-encoded to clean JPEG: {len(image_bytes)} bytes, {pil_img.size[0]}x{pil_img.size[1]}")
            local_quality_rejection = _local_image_quality_rejection(pil_img)
            if local_quality_rejection is not None:
                print("  [Image] Rejected locally as severely underexposed")
                return jsonify(local_quality_rejection), 422
        except Exception as img_err:
            print(f"  [Image] Could not re-encode ({img_err}), using original bytes")

        # An identical photo + selected area + age gets one canonical result.
        # The HMAC key is irreversible and the uploaded image is not retained
        # by the repeat-result store.
        repeat_state, repeated_analysis, repeated_status = (
            _claim_analysis_repeat_key(repeat_cache_key)
        )
        if repeat_state == "hit":
            print("  [Repeat] Reused the canonical result for this analysis input")
            return _analysis_json_response(
                repeated_analysis,
                repeated_status,
                "reused",
            )
        if repeat_state == "timeout":
            return jsonify({
                "error": "This analysis took longer than expected. Please try again.",
                "code": "analysis_timeout",
                "retryable": True,
            }), 504
        repeat_cache_owner = True

        # A repeat hit does not consume another analysis. New inputs are still
        # protected by the existing per-client rate limit.
        client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
        if not check_rate_limit(client_ip):
            _release_analysis_repeat_key(repeat_cache_key)
            repeat_cache_owner = False
            return jsonify({"error": "You've reached the analysis limit. Please try again in an hour, or book a consultation for a full VISIA assessment."}), 429

        # ── SINGLE-MODEL PIPELINE: Gemini 3.1 Pro (vision + JSON analysis) ──
        image_part = genai_types.Part.from_bytes(data=image_bytes, mime_type=media_type)
        user_prompt = build_user_prompt(request.form.get("age"), body_area)

        # Give the primary high-thinking request the full deadline. If it is still
        # running after the hedge delay, start one identical request and use the
        # first valid structured result. Fast requests still make only one call;
        # slow requests gain resilience without killing a response that may finish.
        def run_google_attempt(
            attempt_number,
            attempt_timeout_ms,
            rejection_review_code=None,
        ):
            attempt_started = time.time()
            attempt_seed = analysis_seed
            attempt_prompt = user_prompt
            if attempt_number == 2:
                attempt_seed = _independent_rejection_review_seed(
                    analysis_seed,
                    rejection_review_code or "hedge",
                )
            if rejection_review_code is not None:
                if rejection_review_code == "quality":
                    attempt_prompt += """

INDEPENDENT QUALITY REVIEW: The server accepted this upload for a second visual review. Reassess it independently. A clear close-up or tight crop of the selected area is valid; it does not need to show the full head, limb, or body and does not need to be taken literally at arm's length. If the selected anatomy and visible skin surface can be assessed conservatively, return the complete analysis. Use a quality rejection only if blur, darkness, obstruction, or resolution truly makes surface assessment impossible.
"""
            gemini_response = gemini_client.models.generate_content(
                model=GOOGLE_MODEL,
                contents=[image_part, attempt_prompt],
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    seed=attempt_seed,
                    max_output_tokens=GOOGLE_MAX_OUTPUT_TOKENS,
                    response_mime_type="application/json",
                    response_json_schema=_analysis_schema_for_area(body_area),
                    thinking_config=genai_types.ThinkingConfig(
                        thinking_level=genai_types.ThinkingLevel.HIGH,
                    ),
                    http_options=_google_http_options(attempt_timeout_ms),
                )
            )
            try:
                attempt_text = (gemini_response.text or "").strip()
            except ValueError as response_error:
                raise _GoogleResponseError(
                    "Google Gemini did not return a readable text response"
                ) from response_error
            if not attempt_text:
                raise _GoogleResponseError(
                    "Google Gemini returned an empty text response"
                )

            if attempt_text.startswith("```json"):
                attempt_text = attempt_text[7:]
            if attempt_text.startswith("```"):
                attempt_text = attempt_text[3:]
            if attempt_text.endswith("```"):
                attempt_text = attempt_text[:-3]
            attempt_text = attempt_text.strip()
            attempt_analysis = json.loads(attempt_text)
            usage = getattr(gemini_response, "usage_metadata", None)
            if usage is not None:
                print(
                    "  [Pipeline] Tokens: "
                    f"prompt={getattr(usage, 'prompt_token_count', None)}, "
                    f"thinking={getattr(usage, 'thoughts_token_count', None)}, "
                    f"output={getattr(usage, 'candidates_token_count', None)}"
                )
            if not isinstance(attempt_analysis, dict):
                raise _GoogleResponseError(
                    "Google Gemini returned a non-object JSON response"
                )
            if attempt_analysis.get("rejected") is True:
                attempt_analysis = _normalize_model_rejection(
                    attempt_analysis,
                    body_area,
                )
            else:
                required_keys = {
                    "overallScore",
                    "observedArea",
                    "positiveHighlights",
                    "concerns",
                    "recommendations",
                    "productRecommendations",
                    "suggestedCombo",
                    "summary",
                }
                if set(attempt_analysis) != required_keys:
                    raise _GoogleResponseError(
                        "Google Gemini returned an invalid analysis field set"
                    )
                observed_area = attempt_analysis.get("observedArea")
                if observed_area not in {*AREA_CONCERN_KEYS, "other"}:
                    raise _GoogleResponseError(
                        "Google Gemini returned an invalid observed area"
                    )
                if observed_area != body_area:
                    attempt_analysis = _area_mismatch_result(
                        body_area,
                        observed_area,
                    )
                    print(
                        "  [Image] Selected area did not match the dominant visible area: "
                        f"selected={body_area}, observed={observed_area}"
                    )
                    return attempt_analysis, attempt_text
                if not isinstance(attempt_analysis.get("overallScore"), int):
                    raise _GoogleResponseError(
                        "Google Gemini returned an invalid overall score"
                    )
                for list_key in (
                    "positiveHighlights",
                    "recommendations",
                    "productRecommendations",
                ):
                    if not isinstance(attempt_analysis.get(list_key), list):
                        raise _GoogleResponseError(
                            f"Google Gemini returned an invalid {list_key} value"
                        )
                if not str(attempt_analysis.get("summary", "")).strip():
                    raise _GoogleResponseError(
                        "Google Gemini returned an empty summary"
                    )
                candidate_concerns = attempt_analysis.get("concerns")
                if not isinstance(candidate_concerns, dict):
                    raise _GoogleResponseError(
                        "Google Gemini returned an invalid concern map"
                    )
                expected_concerns = set(
                    AREA_CONCERN_KEYS.get(body_area, AREA_CONCERN_KEYS["face"])
                )
                if set(candidate_concerns) != expected_concerns:
                    raise _GoogleResponseError(
                        "Google Gemini returned the wrong concern set"
                    )
                if any(
                    not isinstance(concern, dict)
                    or set(concern) != {"score", "severity", "description"}
                    for concern in candidate_concerns.values()
                ):
                    raise _GoogleResponseError(
                        "Google Gemini returned an invalid concern field set"
                    )
                # Repair and score raw visual evidence first so catalog mapping
                # cannot preserve a treatment whose supporting score is later
                # capped (for example, BBL after uncorroborated vascularity).
                attempt_analysis = _prepare_completed_analysis(
                    attempt_analysis,
                    body_area,
                )
                attempt_analysis = _normalize_catalog_recommendations(
                    attempt_analysis,
                    body_area,
                )
                attempt_analysis = _validate_final_completed_analysis(
                    attempt_analysis,
                    body_area,
                )
            print(
                f"  [Pipeline] {GOOGLE_MODEL} attempt {attempt_number}/2 "
                f"completed in {time.time() - attempt_started:.1f}s "
                f"({len(attempt_text)} chars)"
            )
            return attempt_analysis, attempt_text

        def classify_google_failure(model_error):
            if isinstance(model_error, httpx.TimeoutException):
                return "timeout", "transport timeout", True
            if isinstance(model_error, httpx.TransportError):
                return (
                    "unavailable",
                    f"transport {model_error.__class__.__name__}",
                    True,
                )
            if isinstance(model_error, genai_errors.APIError):
                error_code = getattr(model_error, "code", None)
                error_status = getattr(model_error, "status", None)
                detail = f"Google API {error_code} {error_status or ''}".strip()
                if error_code not in GOOGLE_TRANSIENT_STATUS_CODES:
                    return "nonretryable", detail, False
                kind = "timeout" if error_code in {408, 504} else "unavailable"
                return kind, detail, True
            if isinstance(model_error, _GoogleResponseError):
                return (
                    "invalid_response",
                    f"{model_error.__class__.__name__}: {model_error}",
                    True,
                )
            if isinstance(model_error, json.JSONDecodeError):
                return "invalid_response", model_error.__class__.__name__, True
            return None, model_error.__class__.__name__, False

        analysis = None
        response_text = ""
        failure_kinds = []
        unexpected_errors = []
        rejection_candidates = []
        pending = set()
        hedge_started = False
        attempt_number = 0
        hedge_at = min(
            model_deadline,
            time.monotonic() + (GOOGLE_HEDGE_DELAY_MS / 1000),
        )
        executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gemini")
        deadline_pending_count = 0

        def submit_google_attempt(rejection_review_code=None):
            nonlocal attempt_number, hedge_started
            remaining_budget_ms = max(
                0, int((model_deadline - time.monotonic()) * 1000)
            )
            if remaining_budget_ms <= 0 or attempt_number >= 2:
                return False
            attempt_number += 1
            if attempt_number == 2:
                hedge_started = True
                print("  [Pipeline] Started one backup Google request")
            future = executor.submit(
                run_google_attempt,
                attempt_number,
                remaining_budget_ms,
                rejection_review_code,
            )
            pending.add(future)
            return True

        submit_google_attempt()
        try:
            while pending and analysis is None:
                now = time.monotonic()
                remaining_seconds = model_deadline - now
                if remaining_seconds <= 0:
                    break
                wait_seconds = remaining_seconds
                if not hedge_started:
                    wait_seconds = min(wait_seconds, max(0, hedge_at - now))

                completed, still_pending = wait(
                    pending,
                    timeout=wait_seconds,
                    return_when=FIRST_COMPLETED,
                )
                pending = set(still_pending)

                if not completed:
                    if not hedge_started and submit_google_attempt():
                        continue
                    break

                saw_retryable_failure = False
                saw_nonretryable_failure = False
                rejection_review_code = None
                successful_candidates = []
                for future in completed:
                    try:
                        candidate_analysis, candidate_text = future.result()
                    except Exception as model_error:
                        failure_kind, failure_detail, retryable = (
                            classify_google_failure(model_error)
                        )
                        if failure_kind is None:
                            unexpected_errors.append(model_error)
                        else:
                            failure_kinds.append(failure_kind)
                            print(
                                f"  [Pipeline] {failure_detail} "
                                f"(attempt failure: {failure_kind})"
                            )
                            saw_retryable_failure = saw_retryable_failure or retryable
                            saw_nonretryable_failure = (
                                saw_nonretryable_failure or not retryable
                            )
                    else:
                        successful_candidates.append(
                            (candidate_analysis, candidate_text)
                        )

                accepted_candidate = next(
                    (
                        candidate
                        for candidate in successful_candidates
                        if not candidate[0].get("rejected")
                    ),
                    None,
                )
                if accepted_candidate is not None:
                    analysis, response_text = accepted_candidate
                else:
                    rejection_candidates.extend(
                        candidate_analysis
                        for candidate_analysis, _ in successful_candidates
                    )
                    confirmed_rejection = _confirmed_model_rejection(
                        rejection_candidates
                    )
                    if confirmed_rejection is not None:
                        if confirmed_rejection.get("reasonCode") == "quality":
                            # A model can mistake a clear close-up for an unusable
                            # photo. Only the deterministic local image gate may
                            # issue and cache a final quality rejection.
                            failure_kinds.append("invalid_response")
                        else:
                            analysis = confirmed_rejection
                    elif successful_candidates:
                        # A single image-based model rejection is not enough to
                        # deny an otherwise analyzable guest photo. Ask the identical
                        # high-thinking model for one independent validation sample
                        # within the same deadline.
                        saw_retryable_failure = True
                        rejection_review_code = successful_candidates[0][0].get(
                            "reasonCode"
                        )

                if analysis is not None:
                    break
                if (
                    not hedge_started
                    and saw_retryable_failure
                    and not saw_nonretryable_failure
                ):
                    submit_google_attempt(rejection_review_code)
                if not pending and (hedge_started or saw_nonretryable_failure):
                    break
        finally:
            if (
                analysis is None
                and pending
                and time.monotonic() >= model_deadline
            ):
                deadline_pending_count = len(pending)
            for future in pending:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)

        if deadline_pending_count:
            failure_kinds.extend(["timeout"] * deadline_pending_count)
            print(
                "  [Pipeline] Marked "
                f"{deadline_pending_count} unfinished Google request(s) as timeout"
            )

        if analysis is None and rejection_candidates:
            # Preserve one clear anatomy correction when the identical backup
            # request fails only because Google's transport is temporarily
            # unavailable. A completed analysis still wins above, and all
            # other image rejections continue to require two matching votes.
            single_rejection = (
                rejection_candidates[0]
                if len(rejection_candidates) == 1
                else None
            )
            transient_backup_only = bool(failure_kinds) and all(
                failure_kind in {"timeout", "unavailable"}
                for failure_kind in failure_kinds
            )
            if (
                single_rejection is not None
                and single_rejection.get("reasonCode") == "area_mismatch"
                and single_rejection.get("observedArea") in AREA_CONCERN_KEYS
                and transient_backup_only
            ):
                analysis = single_rejection
                print(
                    "  [Pipeline] Used one valid area-mismatch result after "
                    "the backup request failed transiently"
                )
            else:
                failure_kinds.append("invalid_response")

        if analysis is None:
            _release_analysis_repeat_key(repeat_cache_key)
            repeat_cache_owner = False
            if unexpected_errors and not failure_kinds:
                raise unexpected_errors[0]
            if "nonretryable" in failure_kinds:
                return jsonify({
                    "error": "The analysis service could not process this request.",
                    "code": "analysis_unavailable",
                    "retryable": False,
                }), 502
            if not failure_kinds or all(
                failure_kind == "timeout" for failure_kind in failure_kinds
            ):
                return jsonify({
                    "error": "This analysis took longer than expected. Please try again.",
                    "code": "analysis_timeout",
                    "retryable": True,
                }), 504
            return jsonify({
                "error": "The analysis service is temporarily unavailable. Please try again.",
                "code": "analysis_unavailable",
                "retryable": True,
            }), 503

        # Free image data from memory
        try:
            del image_bytes, image_part
        except NameError:
            pass
        import gc; gc.collect()

        print(f"  [Pipeline] Total pipeline: {time.time() - t_start:.1f}s")

        # Cache completed analyses and confirmed model rejections only. The
        # first atomic writer wins, so concurrent workers return the same JSON.
        analysis_status = 422 if analysis.get("rejected") else 200
        canonical_analysis, canonical_status = _release_analysis_repeat_key(
            repeat_cache_key,
            analysis,
            analysis_status,
        )
        repeat_cache_owner = False
        return _analysis_json_response(
            canonical_analysis,
            canonical_status,
            "generated",
        )

    except Exception:
        if repeat_cache_owner and repeat_cache_key is not None:
            _release_analysis_repeat_key(repeat_cache_key)
            repeat_cache_owner = False
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "We could not complete this analysis. Please try again.",
            "code": "analysis_failed",
            "retryable": True,
        }), 500


@app.route("/")
def serve_index():
    """Serve the main index.html"""
    return send_from_directory(str(PUBLIC_DIR), "index.html")

@app.route("/<path:path>")
def serve_static(path):
    """Serve static files from public directory"""
    file_path = PUBLIC_DIR / path

    # Security check - prevent directory traversal
    try:
        resolved = file_path.resolve()
        if not str(resolved).startswith(str(PUBLIC_DIR.resolve())):
            return "Not found", 404
    except ValueError:
        return "Not found", 404

    if file_path.is_file():
        return send_from_directory(str(PUBLIC_DIR), path)

    # SPA fallback
    return send_from_directory(str(PUBLIC_DIR), "index.html")


def print_startup_banner():
    """Print a nice startup banner"""
    banner = f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║     Von & Co Aesthetics - Skin Analyzer Backend              ║
    ║     Flask Server for AI-Powered Skin Analysis                ║
    ╚══════════════════════════════════════════════════════════════╝
    
    Mode: {MODE.upper()}
    Google API: {"Configured" if LIVE_MODE else "Not configured"}
    Gemini: {f"Enabled ({GOOGLE_MODEL}, high thinking)" if LIVE_MODE else "Not configured"}
    Pipeline: Google Gemini high-thinking vision+analysis with slow-request hedge
    Debug: {DEBUG}
    Port: {PORT}
    
    Server running at: http://localhost:{PORT}
    Health check: http://localhost:{PORT}/api/health
    
    Ready to analyze skin! 🔬
    """
    print(banner)


if __name__ == "__main__":
    print_startup_banner()
    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=DEBUG
    )
