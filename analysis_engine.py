"""Versioned, schema-validated, fail-closed visible-surface image analysis.

The model may describe only visible surface features. Versioned service and
product matches, metadata, and safety language are added and validated by the
server after model output. Images are normalized in memory and never written to
disk.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import warnings
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable, Iterable, Mapping, Sequence

from PIL import Image, ImageCms, ImageOps, ImageStat, UnidentifiedImageError

from recommendation_catalog import (
    CATALOG_VERSION,
    MAX_PRODUCT_RECOMMENDATIONS,
    MAX_SERVICE_RECOMMENDATIONS,
    build_appearance_recommendations,
)


LOGGER = logging.getLogger(__name__)

ANALYSIS_VERSION = "visible-surface-v1.5.0"
PROMPT_VERSION = "visible-surface-prompt-v1.1.0"
SCHEMA_VERSION = "visible-surface-response-schema-v1.2.0"
TOPIC_MAPPING_VERSION = CATALOG_VERSION

DEFAULT_PROVIDER_ORDER = ("gemini",)
DEFAULT_MODELS = {
    "gemini": "gemini-3.5-flash",
}
GEMINI_THINKING_LEVEL = "high"
GEMINI_MAX_OUTPUT_TOKENS = 8192
GEMINI_SEED = 20260712
GEMINI_PROVIDER_ATTEMPTS = 2

DISCLAIMER = (
    "Photo-based preview only. Service and product matches are educational "
    "options from Von & Co's current guides. It cannot diagnose or rule out "
    "disease or determine treatment or product suitability. An in-person "
    "evaluation is required before treatment, and any concerning lesion should "
    "be evaluated by a qualified medical professional."
)

OBSERVATION_LABELS: dict[str, str] = {
    "visible_lines": "Visible lines",
    "visible_redness": "Visible redness",
    "pigment_variation": "Visible pigment variation",
    "surface_texture": "Visible surface texture",
    "pore_visibility": "Visible pore appearance",
    "laxity_appearance": "Visible laxity appearance",
    "blemish_like_spots": "Blemish-like spots",
    "scar_like_texture": "Scar-like texture",
    "superficial_vessels": "Visible superficial vessels",
    "visible_flaking": "Visible flaking",
}

OBSERVATION_LEVELS = (
    "not_observed",
    "subtle",
    "visible",
    "prominent",
    "unable_to_assess",
)
IMAGE_ANGLES = ("single", "front", "left", "right")
QUALITY_LEVELS = ("suitable", "limited", "retake")
QUALITY_ISSUES = (
    "not_skin",
    "blur",
    "low_light",
    "overexposure",
    "heavy_filter",
    "obstruction",
    "framing",
    "angle_mismatch",
    "low_resolution",
    "low_contrast",
    "uneven_lighting",
    "unsupported_image",
)
QUALITY_GUIDANCE = (
    "use_natural_even_light",
    "hold_camera_steady",
    "remove_filters",
    "remove_makeup_if_possible",
    "move_camera_farther_away",
    "center_area_in_frame",
    "remove_obstructions",
    "retake_required_angles",
    "upload_a_clear_skin_photo",
    "use_a_supported_image_format",
    "use_a_higher_resolution_image",
)
MEDICAL_REASON_CODES = (
    "none",
    "open_or_broken_skin",
)

NECK_CHEST_OBSERVATIONS = (
    "visible_lines",
    "visible_redness",
    "pigment_variation",
    "surface_texture",
    "pore_visibility",
    "laxity_appearance",
    "blemish_like_spots",
    "scar_like_texture",
    "superficial_vessels",
    "visible_flaking",
)

BODY_AREA_OBSERVATIONS: dict[str, tuple[str, ...]] = {
    "face": tuple(OBSERVATION_LABELS),
    "neck": NECK_CHEST_OBSERVATIONS,
    "chest": NECK_CHEST_OBSERVATIONS,
    "hands": (
        "visible_lines",
        "visible_redness",
        "pigment_variation",
        "surface_texture",
        "laxity_appearance",
        "blemish_like_spots",
        "scar_like_texture",
        "superficial_vessels",
        "visible_flaking",
    ),
    "back": (
        "visible_redness",
        "pigment_variation",
        "surface_texture",
        "pore_visibility",
        "blemish_like_spots",
        "scar_like_texture",
        "superficial_vessels",
        "visible_flaking",
    ),
    "legs": (
        "visible_redness",
        "pigment_variation",
        "surface_texture",
        "laxity_appearance",
        "blemish_like_spots",
        "scar_like_texture",
        "superficial_vessels",
        "visible_flaking",
    ),
}

BODY_AREA_ANGLE_CONTEXT: dict[str, dict[str, str]] = {
    "face": {
        "single": "single user-selected facial view",
        "front": "front facial view",
        "left": "left facial profile",
        "right": "right facial profile",
    },
    "neck": {
        "single": "single user-selected neck view",
        "front": "center neck view",
        "left": "left oblique neck view",
        "right": "right oblique neck view",
    },
    "chest": {
        "single": "single user-selected upper-chest view",
        "front": "center upper-chest view",
        "left": "left oblique upper-chest view",
        "right": "right oblique upper-chest view",
    },
    "hands": {
        "single": "single user-selected back-of-hand view",
        "front": "back of the left hand capture slot",
        "left": "back of the right hand capture slot",
        "right": "backs of both hands capture slot",
    },
    "back": {
        "single": "single user-selected back view",
        "front": "center back capture slot",
        "left": "left-side back capture slot",
        "right": "right-side back capture slot",
    },
    "legs": {
        "single": "single user-selected leg view",
        "front": "left leg capture slot",
        "left": "right leg capture slot",
        "right": "both legs capture slot",
    },
}

BODY_AREA_QUALITY_CONTEXT: dict[str, str] = {
    "face": (
        "Accept a normal close-up when the main facial skin from forehead through "
        "chin is clear enough to review. Cropped hair, ears, shoulders, or neck do "
        "not make an otherwise clear face photo unsuitable."
    ),
    "neck": (
        "Accept the photo when the intended neck skin is clear enough to review; "
        "the upper chest does not need to be visible."
    ),
    "chest": (
        "Accept the photo when the intended upper-chest skin is clear enough to "
        "review; a full torso view is not required."
    ),
    "hands": (
        "Accept the photo only when the back of the intended hand or hands is clear "
        "enough to review. A palm-only image does not show the intended treatment "
        "area and must return retake with the framing issue and "
        "center_area_in_frame guidance. The arm and both hands do not need to be "
        "fully visible in a single upload."
    ),
    "back": (
        "Accept the photo when the intended back or shoulder skin is clear enough "
        "to review; the entire back does not need to fit in a single upload."
    ),
    "legs": (
        "Accept the photo when the intended leg skin is clear enough to review; "
        "the entire leg and both legs do not need to fit in a single upload."
    ),
}

FINAL_RESULT_KEYS = {
    "status",
    "bodyArea",
    "analysisVersion",
    "schemaVersion",
    "topicMappingVersion",
    "promptHash",
    "model",
    "imageCount",
    "quality",
    "observations",
    "strengths",
    "priorities",
    "appearanceRecommendations",
    "discussionTopics",
    "medicalReview",
    "disclaimer",
}

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_IMAGE_DIMENSION = 1600
MAX_IMAGE_PIXELS = 40_000_000
MIN_IMAGE_DIMENSION = 480
# A high-percentile edge measure keeps naturally smooth skin from dominating
# the check while still catching images with no reliably sharp detail.
SHARPNESS_PERCENTILE = 0.995
MIN_SHARPNESS_PERCENTILE = 30
SUPPORTED_PIL_FORMATS = {"JPEG", "MPO", "PNG", "WEBP"}


def _enum_schema(values: Iterable[str]) -> dict[str, Any]:
    return {"type": "string", "enum": list(values)}


MODEL_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "status",
        "quality",
        "observations",
        "strengths",
        "priorities",
        "medicalReview",
    ],
    "properties": {
        "status": _enum_schema(("complete", "retake", "medical_review")),
        "quality": {
            "type": "object",
            "additionalProperties": False,
            "required": ["overall", "issues", "guidance"],
            "properties": {
                "overall": _enum_schema(QUALITY_LEVELS),
                "issues": {
                    "type": "array",
                    "items": _enum_schema(QUALITY_ISSUES),
                },
                "guidance": {
                    "type": "array",
                    "items": _enum_schema(QUALITY_GUIDANCE),
                },
            },
        },
        "observations": {
            "type": "array",
            "minItems": 0,
            "maxItems": len(OBSERVATION_LABELS),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "label", "level", "description", "angles"],
                "properties": {
                    "id": _enum_schema(OBSERVATION_LABELS),
                    "label": _enum_schema(OBSERVATION_LABELS.values()),
                    "level": _enum_schema(OBSERVATION_LEVELS),
                    "description": {"type": "string"},
                    "angles": {
                        "type": "array",
                        "items": _enum_schema(IMAGE_ANGLES),
                    },
                },
            },
        },
        "strengths": {
            "type": "array",
            "maxItems": 2,
            "items": _enum_schema(OBSERVATION_LABELS),
        },
        "priorities": {
            "type": "array",
            "maxItems": 2,
            "items": _enum_schema(OBSERVATION_LABELS),
        },
        "medicalReview": {
            "type": "object",
            "additionalProperties": False,
            "required": ["suggested", "reason"],
            "properties": {
                "suggested": {"type": "boolean"},
                "reason": _enum_schema(MEDICAL_REASON_CODES),
            },
        },
    },
}


SYSTEM_PROMPT = """You review one or three user-submitted images for a limited, non-diagnostic visible-surface cosmetic preview.

Safety and scope:
- Describe only features directly visible in the submitted pixels.
- Never diagnose or rule out a disease or condition.
- Never infer UV damage, hydration, bacteria, ethnicity, sex, biological age, subsurface features, causes, treatment eligibility, or treatment safety.
- Do not mention treatments, products, providers, scores, percentages, grades, skin age, or an overall score.
- Do not compare the person with population norms or claim clinical-device equivalence.
- Do not identify the person or infer age. Adult access is confirmed separately by the application.
- Do not decide whether a mole, spot, or lesion is medically concerning and do not use medical_review based on its appearance; the application provides a standing in-person-evaluation disclaimer. Use medical_review only when pixels directly show open, broken, or actively bleeding-like skin that makes a cosmetic preview inappropriate. Do not name a diagnosis or infer symptoms, timing, or change from a still image.
- If the upload is not skin, is too unclear, appears heavily filtered, omits a required angle, or cannot support an honest review, use retake and return no observations, strengths, or priorities.

Output discipline:
- Return only JSON conforming exactly to the supplied schema.
- Use only the supplied IDs, labels, levels, quality codes, guidance codes, angles, and medical reason codes.
- A label must exactly match its observation ID's canonical label.
- Return an observation for every allowed appearance ID that can be honestly
  judged from the submitted views. Use not_observed only when the feature is
  clearly not seen, subtle, visible, or prominent when supported, and
  unable_to_assess when angle or photo limits prevent a judgment.
- Do not omit a category simply because it is not a priority. The complete
  profile should preserve supported strengths and neutral findings as well.
- Do not invent a finding to fill the profile.
- strengths and priorities are ordered observation IDs, zero to two each. Do not force either list.
- strengths may reference only not_observed or subtle observations.
- priorities may reference only visible or prominent observations.
- If an item cannot be judged, use unable_to_assess and do not make it a strength or priority.
- For three views, use all relevant angle evidence and do not treat the views as separate people.
- Description text must remain neutral, concise, and limited to what is visibly supported.
"""


def model_output_schema(body_area: str) -> dict[str, Any]:
    """Return a strict provider schema limited to the selected body area."""

    allowed_ids = BODY_AREA_OBSERVATIONS.get(body_area)
    if not allowed_ids:
        raise SchemaValidationError("body area is unsupported")
    schema = copy.deepcopy(MODEL_OUTPUT_SCHEMA)
    observation_properties = schema["properties"]["observations"]["items"]["properties"]
    observation_properties["id"]["enum"] = list(allowed_ids)
    observation_properties["label"]["enum"] = [OBSERVATION_LABELS[item] for item in allowed_ids]
    schema["properties"]["strengths"]["items"]["enum"] = list(allowed_ids)
    schema["properties"]["priorities"]["items"]["enum"] = list(allowed_ids)
    return schema


def gemini_output_schema(body_area: str) -> dict[str, Any]:
    """Return the strictest schema Gemini can compile for this response.

    The reduced provider grammar keeps the nested observations array from
    becoming too complex. The application always validates the original,
    stricter schema after Gemini responds.
    """

    schema = model_output_schema(body_area)

    observations_schema = schema["properties"]["observations"]
    observations_schema.pop("minItems", None)
    observations_schema.pop("maxItems", None)
    return schema


class AnalyzerError(RuntimeError):
    """Base class for controlled analyzer failures."""


class ProviderUnavailable(AnalyzerError):
    """No configured provider produced a valid response."""


class SchemaValidationError(AnalyzerError):
    """Provider output violated the analyzer contract."""


class ImageIntakeError(AnalyzerError):
    """An uploaded file cannot safely be sent for analysis."""

    def __init__(self, issue: str, guidance: str, message: str) -> None:
        super().__init__(message)
        self.issue = issue
        self.guidance = guidance
        self.public_message = message


@dataclass(frozen=True)
class NormalizedImage:
    angle: str
    data: bytes
    media_type: str
    width: int
    height: int


def _is_plain_dict(value: Any) -> bool:
    return isinstance(value, dict)


def _require_exact_keys(value: Any, keys: set[str], path: str) -> dict[str, Any]:
    if not _is_plain_dict(value):
        raise SchemaValidationError(f"{path} must be an object")
    if set(value) != keys:
        raise SchemaValidationError(f"{path} has unexpected or missing fields")
    return value


def _require_string(value: Any, path: str, *, max_length: int = 240) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > max_length:
        raise SchemaValidationError(f"{path} must be a non-empty bounded string")
    return value.strip()


def _require_string_list(
    value: Any,
    path: str,
    *,
    allowed: Iterable[str],
    max_items: int | None = None,
) -> list[str]:
    allowed_set = set(allowed)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise SchemaValidationError(f"{path} must be an array of strings")
    if max_items is not None and len(value) > max_items:
        raise SchemaValidationError(f"{path} has too many items")
    if len(value) != len(set(value)):
        raise SchemaValidationError(f"{path} must not contain duplicates")
    if any(item not in allowed_set for item in value):
        raise SchemaValidationError(f"{path} contains an unsupported value")
    return list(value)


_UNSAFE_TEXT_FRAGMENTS = (
    "acne",
    "rosacea",
    "melasma",
    "infection",
    "cancer",
    "malignant",
    "benign",
    "diagnos",
    "rule out",
    "uv damage",
    "sun damage",
    "bacteria",
    "bacterial",
    "hydration",
    "dehydrat",
    "ethnicity",
    "skin age",
    "biological age",
    "you have",
    "dermatitis",
    "eczema",
    "psoriasis",
    "melanoma",
    "carcinoma",
    "tumor",
    "lesion",
    "mole",
    "rash",
    "appears to be",
    "consistent with",
    "suggestive of",
    "indicative of",
    "likely caused",
)


def _validate_visible_text(text: str, path: str) -> None:
    normalized = " ".join(text.lower().split())
    if any(fragment in normalized for fragment in _UNSAFE_TEXT_FRAGMENTS):
        raise SchemaValidationError(f"{path} exceeds visible-surface scope")


def _deterministic_description(
    observation_id: str, level: str, angles: Sequence[str]
) -> str:
    descriptions: dict[str, dict[str, str]] = {
        "visible_lines": {
            "not_observed": "Skin looks smooth and softly defined in these photos.",
            "subtle": "Lines look soft and understated in these photos.",
            "visible": "Lines are clearly visible in these photos.",
            "prominent": "Lines are one of the clearest details in these photos.",
            "unable_to_assess": "Lines need a clearer photo to review.",
        },
        "visible_redness": {
            "not_observed": "Tone looks calm and even in these photos.",
            "subtle": "Tone looks calm overall, with subtle redness in these photos.",
            "visible": "Redness is clearly visible in these photos.",
            "prominent": "Redness is one of the clearest details in these photos.",
            "unable_to_assess": "Redness needs a clearer photo to review.",
        },
        "pigment_variation": {
            "not_observed": "Tone looks even and consistent in these photos.",
            "subtle": "Tone looks even overall, with subtle pigment variation in these photos.",
            "visible": "Pigment variation is clearly visible in these photos.",
            "prominent": "Pigment variation is one of the clearest details in these photos.",
            "unable_to_assess": "Pigment variation needs a clearer photo to review.",
        },
        "surface_texture": {
            "not_observed": "The skin surface looks smooth and even in these photos.",
            "subtle": "Texture looks smooth overall, with subtle surface variation in these photos.",
            "visible": "Surface texture is clearly visible in these photos.",
            "prominent": "Surface texture is one of the clearest details in these photos.",
            "unable_to_assess": "Surface texture needs a clearer photo to review.",
        },
        "pore_visibility": {
            "not_observed": "Pores look refined and unobtrusive in these photos.",
            "subtle": "Pores look refined overall, with subtle visibility in these photos.",
            "visible": "Pores are clearly visible in these photos.",
            "prominent": "Pore appearance is one of the clearest details in these photos.",
            "unable_to_assess": "Pore appearance needs a clearer photo to review.",
        },
        "laxity_appearance": {
            "not_observed": "The area looks firm and well supported in these photos.",
            "subtle": "The area looks firm overall, with subtle laxity in these photos.",
            "visible": "Laxity is clearly visible in these photos.",
            "prominent": "Laxity is one of the clearest details in these photos.",
            "unable_to_assess": "Laxity appearance needs a clearer photo to review.",
        },
        "blemish_like_spots": {
            "not_observed": "The skin surface looks clear and even in these photos.",
            "subtle": "The surface looks clear overall, with subtle blemish-like spots in these photos.",
            "visible": "Blemish-like spots are clearly visible in these photos.",
            "prominent": "Blemish-like spots are one of the clearest details in these photos.",
            "unable_to_assess": "Blemish-like spots need a clearer photo to review.",
        },
        "scar_like_texture": {
            "not_observed": "The skin surface looks smooth and consistent in these photos.",
            "subtle": "The surface looks smooth overall, with subtle scar-like texture in these photos.",
            "visible": "Scar-like texture is clearly visible in these photos.",
            "prominent": "Scar-like texture is one of the clearest details in these photos.",
            "unable_to_assess": "Scar-like texture needs a clearer photo to review.",
        },
        "superficial_vessels": {
            "not_observed": "Tone looks even and visually calm in these photos.",
            "subtle": "Tone looks even overall, with subtle superficial vessels in these photos.",
            "visible": "Superficial vessels are clearly visible in these photos.",
            "prominent": "Superficial vessels are one of the clearest details in these photos.",
            "unable_to_assess": "Superficial vessels need a clearer photo to review.",
        },
        "visible_flaking": {
            "not_observed": "The skin surface looks smooth and even in these photos.",
            "subtle": "The surface looks smooth overall, with subtle flaking in these photos.",
            "visible": "Flaking is clearly visible in these photos.",
            "prominent": "Flaking is one of the clearest details in these photos.",
            "unable_to_assess": "Flaking needs a clearer photo to review.",
        },
    }
    description = descriptions.get(observation_id, {}).get(level)
    if description:
        return description
    return f"{OBSERVATION_LABELS[observation_id]} needs a clearer photo to review."


def _validate_image_angle_sequence(image_angles: Sequence[str]) -> tuple[str, ...]:
    """Require the one-photo or ordered three-view intake contract exactly."""

    angles = tuple(image_angles)
    if angles not in {("single",), ("front", "left", "right")}:
        raise SchemaValidationError(
            "image angles must be one single view or ordered front, left, and right views"
        )
    return angles


def validate_model_output(
    payload: Any, image_angles: Sequence[str], body_area: str = "face"
) -> dict[str, Any]:
    """Validate model JSON with exact keys, allowlists, and cross-field rules."""

    canonical_image_angles = _validate_image_angle_sequence(image_angles)
    result = _require_exact_keys(
        payload,
        {"status", "quality", "observations", "strengths", "priorities", "medicalReview"},
        "result",
    )
    status = result["status"]
    if status not in {"complete", "retake", "medical_review"}:
        raise SchemaValidationError("status is unsupported")

    quality = _require_exact_keys(result["quality"], {"overall", "issues", "guidance"}, "quality")
    if quality["overall"] not in QUALITY_LEVELS:
        raise SchemaValidationError("quality.overall is unsupported")
    issues = _require_string_list(quality["issues"], "quality.issues", allowed=QUALITY_ISSUES)
    guidance = _require_string_list(
        quality["guidance"], "quality.guidance", allowed=QUALITY_GUIDANCE
    )
    if quality["overall"] == "suitable" and (issues or guidance):
        raise SchemaValidationError("suitable quality cannot include limitations")
    if quality["overall"] in {"limited", "retake"} and (not issues or not guidance):
        raise SchemaValidationError("limited quality requires an issue and guidance")

    allowed_observation_ids = BODY_AREA_OBSERVATIONS.get(body_area)
    if not allowed_observation_ids:
        raise SchemaValidationError("body area is unsupported")
    allowed_observation_set = set(allowed_observation_ids)
    observations_value = result["observations"]
    if not isinstance(observations_value, list) or len(observations_value) > len(allowed_observation_ids):
        raise SchemaValidationError("observations must be a bounded array")
    allowed_angles = set(canonical_image_angles)
    seen_ids: set[str] = set()
    observations: list[dict[str, Any]] = []
    levels_by_id: dict[str, str] = {}
    for index, raw_observation in enumerate(observations_value):
        observation = _require_exact_keys(
            raw_observation,
            {"id", "label", "level", "description", "angles"},
            f"observations[{index}]",
        )
        observation_id = observation["id"]
        if observation_id not in allowed_observation_set or observation_id in seen_ids:
            raise SchemaValidationError("observation IDs must be unique allowlisted values")
        if observation["label"] != OBSERVATION_LABELS[observation_id]:
            raise SchemaValidationError("observation label does not match its ID")
        level = observation["level"]
        if level not in OBSERVATION_LEVELS:
            raise SchemaValidationError("observation level is unsupported")
        model_description = _require_string(
            observation["description"], f"observations[{index}].description"
        )
        _validate_visible_text(model_description, f"observations[{index}].description")
        angles = _require_string_list(
            observation["angles"], f"observations[{index}].angles", allowed=IMAGE_ANGLES
        )
        if not angles or not set(angles).issubset(allowed_angles):
            raise SchemaValidationError("observation angles do not match submitted images")
        seen_ids.add(observation_id)
        levels_by_id[observation_id] = level
        observations.append(
            {
                "id": observation_id,
                "label": observation["label"],
                "level": level,
                "description": _deterministic_description(observation_id, level, angles),
                "angles": angles,
            }
        )

    strengths = _require_string_list(
        result["strengths"], "strengths", allowed=allowed_observation_ids, max_items=2
    )
    priorities = _require_string_list(
        result["priorities"], "priorities", allowed=allowed_observation_ids, max_items=2
    )
    if set(strengths) & set(priorities):
        raise SchemaValidationError("strengths and priorities must not overlap")
    if any(levels_by_id.get(item) not in {"not_observed", "subtle"} for item in strengths):
        raise SchemaValidationError("strength must reference a supported low-level observation")
    if any(levels_by_id.get(item) not in {"visible", "prominent"} for item in priorities):
        raise SchemaValidationError("priority must reference a supported visible observation")

    medical = _require_exact_keys(
        result["medicalReview"], {"suggested", "reason"}, "medicalReview"
    )
    if not isinstance(medical["suggested"], bool) or medical["reason"] not in MEDICAL_REASON_CODES:
        raise SchemaValidationError("medicalReview is invalid")
    if medical["suggested"] != (medical["reason"] != "none"):
        raise SchemaValidationError("medicalReview fields are inconsistent")

    if status == "retake":
        if quality["overall"] != "retake" or observations or strengths or priorities:
            raise SchemaValidationError("retake must contain no personalized findings")
        if medical["suggested"]:
            raise SchemaValidationError("retake and medical review cannot be combined")
    elif status == "medical_review":
        if not medical["suggested"] or quality["overall"] == "retake":
            raise SchemaValidationError("medical_review fields are inconsistent")
    else:
        if medical["suggested"] or quality["overall"] == "retake":
            raise SchemaValidationError("complete fields are inconsistent")
        for observation_id in allowed_observation_ids:
            if observation_id in seen_ids:
                continue
            observations.append(
                {
                    "id": observation_id,
                    "label": OBSERVATION_LABELS[observation_id],
                    "level": "unable_to_assess",
                    "description": _deterministic_description(
                        observation_id,
                        "unable_to_assess",
                        (),
                    ),
                    "angles": [],
                }
            )

    return {
        "status": status,
        "quality": {
            "overall": quality["overall"],
            "issues": issues,
            "guidance": guidance,
        },
        "observations": observations,
        "strengths": strengths,
        "priorities": priorities,
        "medicalReview": {
            "suggested": medical["suggested"],
            "reason": medical["reason"],
        },
    }


def _validate_appearance_recommendations(
    value: Any, recommendation_ids: Sequence[str]
) -> dict[str, Any]:
    """Validate the public catalog projection before deterministic comparison."""

    recommendations = _require_exact_keys(
        value, {"services", "products"}, "appearanceRecommendations"
    )
    services = recommendations["services"]
    products = recommendations["products"]
    if (
        not isinstance(services, list)
        or len(services) > MAX_SERVICE_RECOMMENDATIONS
    ):
        raise SchemaValidationError("appearanceRecommendations.services is invalid")
    if (
        not isinstance(products, list)
        or len(products) > MAX_PRODUCT_RECOMMENDATIONS
    ):
        raise SchemaValidationError("appearanceRecommendations.products is invalid")

    for index, service in enumerate(services):
        path = f"appearanceRecommendations.services[{index}]"
        item = _require_exact_keys(
            service,
            {
                "id",
                "name",
                "category",
                "matchedObservationIds",
                "why",
                "learnMoreUrl",
            },
            path,
        )
        _require_string(item["id"], f"{path}.id", max_length=80)
        _require_string(item["name"], f"{path}.name", max_length=120)
        _require_string(item["category"], f"{path}.category", max_length=80)
        matched_ids = _require_string_list(
            item["matchedObservationIds"],
            f"{path}.matchedObservationIds",
            allowed=recommendation_ids,
            max_items=len(recommendation_ids),
        )
        if not matched_ids:
            raise SchemaValidationError(f"{path} must cite an eligible observed finding")
        _require_string(item["why"], f"{path}.why", max_length=320)
        learn_more_url = _require_string(
            item["learnMoreUrl"], f"{path}.learnMoreUrl", max_length=240
        )
        if not learn_more_url.startswith(
            "https://www.vonandcoaesthetics.com/services/"
        ):
            raise SchemaValidationError(f"{path}.learnMoreUrl is invalid")

    for index, product in enumerate(products):
        path = f"appearanceRecommendations.products[{index}]"
        item = _require_exact_keys(
            product,
            {
                "id",
                "name",
                "brand",
                "category",
                "relationship",
                "matchedObservationIds",
                "why",
                "availability",
            },
            path,
        )
        _require_string(item["id"], f"{path}.id", max_length=80)
        _require_string(item["name"], f"{path}.name", max_length=120)
        _require_string(item["brand"], f"{path}.brand", max_length=80)
        _require_string(item["category"], f"{path}.category", max_length=80)
        _require_string(item["relationship"], f"{path}.relationship", max_length=80)
        matched_ids = _require_string_list(
            item["matchedObservationIds"],
            f"{path}.matchedObservationIds",
            allowed=recommendation_ids,
            max_items=len(recommendation_ids),
        )
        if not matched_ids:
            raise SchemaValidationError(f"{path} must cite an eligible observed finding")
        _require_string(item["why"], f"{path}.why", max_length=320)
        _require_string(item["availability"], f"{path}.availability", max_length=240)

    if len({item["id"] for item in services}) != len(services):
        raise SchemaValidationError("appearanceRecommendations.services has duplicates")
    if len({item["id"] for item in products}) != len(products):
        raise SchemaValidationError("appearanceRecommendations.products has duplicates")
    return recommendations


def validate_final_result(payload: Any) -> dict[str, Any]:
    """Defensively validate the public response assembled by the server."""

    result = _require_exact_keys(payload, FINAL_RESULT_KEYS, "public result")
    if result["status"] not in {"complete", "retake", "medical_review"}:
        raise SchemaValidationError("public status is unsupported")
    if result["bodyArea"] not in BODY_AREA_OBSERVATIONS:
        raise SchemaValidationError("public bodyArea is unsupported")
    if result["analysisVersion"] != ANALYSIS_VERSION:
        raise SchemaValidationError("analysisVersion is invalid")
    if result["schemaVersion"] != SCHEMA_VERSION:
        raise SchemaValidationError("schemaVersion is invalid")
    if result["topicMappingVersion"] != TOPIC_MAPPING_VERSION:
        raise SchemaValidationError("topicMappingVersion is invalid")
    if result["promptHash"] != prompt_hash():
        raise SchemaValidationError("promptHash is invalid")
    model = _require_exact_keys(result["model"], {"provider", "name", "promptVersion"}, "model")
    if model["provider"] not in {"local", *DEFAULT_PROVIDER_ORDER}:
        raise SchemaValidationError("model.provider is invalid")
    _require_string(model["name"], "model.name", max_length=120)
    if model["promptVersion"] != PROMPT_VERSION:
        raise SchemaValidationError("promptVersion is invalid")
    if type(result["imageCount"]) is not int or result["imageCount"] not in {1, 3}:
        raise SchemaValidationError("imageCount must be 1 or 3")
    quality = _require_exact_keys(result["quality"], {"overall", "issues", "guidance"}, "quality")
    if quality["overall"] not in QUALITY_LEVELS:
        raise SchemaValidationError("public quality is invalid")
    public_issues = _require_string_list(
        quality["issues"], "quality.issues", allowed=QUALITY_ISSUES
    )
    public_guidance = _require_string_list(
        quality["guidance"], "quality.guidance", allowed=QUALITY_GUIDANCE
    )
    if quality["overall"] == "suitable" and (public_issues or public_guidance):
        raise SchemaValidationError("public suitable quality has limitations")
    if quality["overall"] in {"limited", "retake"} and (
        not public_issues or not public_guidance
    ):
        raise SchemaValidationError("public limited quality lacks actionable detail")
    observations_value = result["observations"]
    if not isinstance(observations_value, list) or len(observations_value) > len(OBSERVATION_LABELS):
        raise SchemaValidationError("observations must be a bounded array")
    permitted_angles = {"single"} if result["imageCount"] == 1 else {"front", "left", "right"}
    seen_ids: set[str] = set()
    levels_by_id: dict[str, str] = {}
    for index, raw_observation in enumerate(observations_value):
        observation = _require_exact_keys(
            raw_observation,
            {"id", "label", "level", "description", "angles"},
            f"observations[{index}]",
        )
        observation_id = observation["id"]
        if (
            observation_id not in BODY_AREA_OBSERVATIONS[result["bodyArea"]]
            or observation_id in seen_ids
        ):
            raise SchemaValidationError("public observation ID is invalid")
        if observation["label"] != OBSERVATION_LABELS[observation_id]:
            raise SchemaValidationError("public observation label is invalid")
        if observation["level"] not in OBSERVATION_LEVELS:
            raise SchemaValidationError("public observation level is invalid")
        description = _require_string(
            observation["description"], f"observations[{index}].description"
        )
        _validate_visible_text(description, f"observations[{index}].description")
        angles = _require_string_list(
            observation["angles"], f"observations[{index}].angles", allowed=IMAGE_ANGLES
        )
        if (
            (not angles and observation["level"] != "unable_to_assess")
            or not set(angles).issubset(permitted_angles)
        ):
            raise SchemaValidationError("public observation angles are invalid")
        if description != _deterministic_description(
            observation_id, observation["level"], angles
        ):
            raise SchemaValidationError("public observation description is not deterministic")
        seen_ids.add(observation_id)
        levels_by_id[observation_id] = observation["level"]
    strengths = _require_string_list(
        result["strengths"], "strengths", allowed=OBSERVATION_LABELS, max_items=2
    )
    priorities = _require_string_list(
        result["priorities"], "priorities", allowed=OBSERVATION_LABELS, max_items=2
    )
    if set(strengths) & set(priorities):
        raise SchemaValidationError("public strengths and priorities overlap")
    if any(levels_by_id.get(item) not in {"not_observed", "subtle"} for item in strengths):
        raise SchemaValidationError("public strength is unsupported")
    if any(levels_by_id.get(item) not in {"visible", "prominent"} for item in priorities):
        raise SchemaValidationError("public priority is unsupported")
    recommendation_ids = _recommendation_basis(
        strengths,
        priorities,
        observations_value,
        result["bodyArea"],
    )
    recommendations = _validate_appearance_recommendations(
        result["appearanceRecommendations"], recommendation_ids
    )
    if not isinstance(result["discussionTopics"], list) or len(result["discussionTopics"]) > 2:
        raise SchemaValidationError("discussionTopics is invalid")
    for index, topic in enumerate(result["discussionTopics"]):
        _require_exact_keys(topic, {"id", "name", "why"}, f"discussionTopics[{index}]")
        _require_string(topic["id"], f"discussionTopics[{index}].id", max_length=80)
        _require_string(topic["name"], f"discussionTopics[{index}].name", max_length=120)
        _require_string(topic["why"], f"discussionTopics[{index}].why", max_length=320)
    medical = _require_exact_keys(result["medicalReview"], {"suggested", "message"}, "medicalReview")
    if not isinstance(medical["suggested"], bool):
        raise SchemaValidationError("medicalReview.suggested must be boolean")
    _require_string(medical["message"], "medicalReview.message", max_length=320)
    if medical["suggested"] != (result["status"] == "medical_review"):
        raise SchemaValidationError("public medical review fields are inconsistent")
    if result["status"] == "retake":
        if quality["overall"] != "retake" or observations_value or strengths or priorities:
            raise SchemaValidationError("public retake includes unsupported findings")
        if result["discussionTopics"]:
            raise SchemaValidationError("retake must suppress cosmetic topics")
        if recommendations != {"services": [], "products": []}:
            raise SchemaValidationError("retake must suppress recommendations")
    elif quality["overall"] == "retake":
        raise SchemaValidationError("public quality and status are inconsistent")
    if result["status"] == "medical_review" and (
        result["discussionTopics"]
        or recommendations != {"services": [], "products": []}
        or observations_value
        or strengths
        or priorities
    ):
        raise SchemaValidationError(
            "medical review must suppress cosmetic findings and recommendations"
        )
    if result["status"] == "complete":
        expected_recommendations = build_appearance_recommendations(
            recommendation_ids, result["bodyArea"], OBSERVATION_LABELS
        )
        if recommendations != expected_recommendations:
            raise SchemaValidationError("appearance recommendations are not deterministic")
        if result["discussionTopics"] != _discussion_topics(
            recommendation_ids, result["bodyArea"]
        ):
            raise SchemaValidationError("discussion topics are not deterministic")
    if result["disclaimer"] != DISCLAIMER:
        raise SchemaValidationError("disclaimer is invalid")
    return result


def _convert_embedded_profile_to_srgb(image: Image.Image, profile_bytes: Any) -> Image.Image:
    """Color-manage an embedded ICC profile in memory, falling back safely."""

    if not isinstance(profile_bytes, bytes) or not profile_bytes:
        return image
    try:
        source_profile = ImageCms.ImageCmsProfile(BytesIO(profile_bytes))
        srgb_profile = ImageCms.createProfile("sRGB")
        return ImageCms.profileToProfile(
            image,
            source_profile,
            srgb_profile,
            outputMode="RGB",
        )
    except Exception:
        # An invalid or unsupported profile must not prevent a safe ordinary-RGB
        # normalization; no profile data or exception detail is logged.
        return image


def normalize_image(file_object: Any, angle: str) -> NormalizedImage:
    """Decode and normalize one upload in memory, stripping metadata."""

    if angle not in IMAGE_ANGLES:
        raise ImageIntakeError("angle_mismatch", "retake_required_angles", "Image angle is invalid.")
    try:
        raw = file_object.read(MAX_UPLOAD_BYTES + 1)
    except Exception as exc:
        raise ImageIntakeError(
            "unsupported_image",
            "use_a_supported_image_format",
            "The image could not be read. Please upload a JPEG, PNG, or WebP image.",
        ) from exc
    if not raw:
        raise ImageIntakeError(
            "unsupported_image",
            "use_a_supported_image_format",
            "The image was empty. Please upload a JPEG, PNG, or WebP image.",
        )
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ImageIntakeError(
            "unsupported_image",
            "use_a_supported_image_format",
            "The image is too large. Please upload an image smaller than 12 MB.",
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(raw)) as opened:
                embedded_icc = opened.info.get("icc_profile")
                # Some phones and photo apps store an ordinary primary JPEG
                # inside an MPO (multi-picture) container. Browsers and macOS
                # correctly identify these files as JPEGs, while Pillow reports
                # the container as "MPO". Review only the primary frame and
                # normalize it to the same metadata-free JPEG used for every
                # other supported upload.
                if opened.format not in SUPPORTED_PIL_FORMATS:
                    raise ImageIntakeError(
                        "unsupported_image",
                        "use_a_supported_image_format",
                        "Please upload a JPEG, PNG, or WebP image.",
                    )
                if opened.format == "MPO":
                    opened.seek(0)
                if opened.width * opened.height > MAX_IMAGE_PIXELS:
                    raise ImageIntakeError(
                        "unsupported_image",
                        "use_a_supported_image_format",
                        "The image dimensions are too large. Please use a smaller image.",
                    )
                image = ImageOps.exif_transpose(opened)
                image.load()
                if min(image.size) < MIN_IMAGE_DIMENSION:
                    raise ImageIntakeError(
                        "low_resolution",
                        "use_a_higher_resolution_image",
                        "The image resolution is too low for an honest review. Please use a higher-resolution capture.",
                    )
                if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                    rgba = image.convert("RGBA")
                    alpha = rgba.getchannel("A")
                    color = _convert_embedded_profile_to_srgb(
                        rgba.convert("RGB"), embedded_icc
                    ).convert("RGBA")
                    color.putalpha(alpha)
                    background = Image.new("RGBA", rgba.size, "white")
                    background.alpha_composite(color)
                    image = background.convert("RGB")
                else:
                    # ICC conversion must see the source mode (especially CMYK)
                    # before any ordinary RGB conversion changes its meaning.
                    image = _convert_embedded_profile_to_srgb(image, embedded_icc)
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                # These deliberately conservative checks are technical capture
                # heuristics only. They are not clinical confidence measures.
                quality_sample = image.copy()
                quality_sample.thumbnail((256, 256), Image.Resampling.BILINEAR)
                grayscale = quality_sample.convert("L")
                histogram = grayscale.histogram()
                pixel_count = max(1, quality_sample.width * quality_sample.height)
                dark_fraction = sum(histogram[:8]) / pixel_count
                light_fraction = sum(histogram[248:]) / pixel_count
                contrast = float(ImageStat.Stat(grayscale).stddev[0])
                if dark_fraction >= 0.85:
                    raise ImageIntakeError(
                        "low_light",
                        "use_natural_even_light",
                        "A basic capture-quality check found the image too dark to review. Please retake it in even light.",
                    )
                if light_fraction >= 0.85:
                    raise ImageIntakeError(
                        "overexposure",
                        "use_natural_even_light",
                        "A basic capture-quality check found the image overexposed. Please retake it in even light.",
                    )
                if contrast < 3.0:
                    raise ImageIntakeError(
                        "low_contrast",
                        "use_natural_even_light",
                        "A basic capture-quality check found too little visible contrast for an honest review. Please retake the image.",
                    )
                sharpness_values: list[int] = []
                pixels = grayscale.load()
                sample_width, sample_height = grayscale.size
                for y in range(1, sample_height - 1):
                    for x in range(1, sample_width - 1):
                        center = pixels[x, y]
                        sharpness_values.append(
                            abs(
                                pixels[x - 1, y]
                                + pixels[x + 1, y]
                                + pixels[x, y - 1]
                                + pixels[x, y + 1]
                                - (4 * center)
                            )
                        )
                sharpness_values.sort()
                sharpness_index = min(
                    len(sharpness_values) - 1,
                    int((len(sharpness_values) - 1) * SHARPNESS_PERCENTILE),
                )
                if (
                    not sharpness_values
                    or sharpness_values[sharpness_index]
                    < MIN_SHARPNESS_PERCENTILE
                ):
                    raise ImageIntakeError(
                        "blur",
                        "hold_camera_steady",
                        "A basic capture-quality check found the image too soft to review. Please retake it with the camera held steady.",
                    )
                image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)
                output = BytesIO()
                image.save(output, format="JPEG", quality=92, optimize=True)
                normalized = output.getvalue()
                width, height = image.size
    except ImageIntakeError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageIntakeError(
            "unsupported_image",
            "use_a_supported_image_format",
            "The file is not a supported image. Please upload a JPEG, PNG, or WebP image.",
        ) from exc

    return NormalizedImage(
        angle=angle,
        data=normalized,
        media_type="image/jpeg",
        width=width,
        height=height,
    )


def _provider_order() -> tuple[str, ...]:
    return DEFAULT_PROVIDER_ORDER


def _provider_key(provider: str) -> str | None:
    if provider == "gemini":
        return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    return None


def model_name(provider: str) -> str:
    env_name = {"gemini": "GEMINI_MODEL"}[provider]
    return os.getenv(env_name, DEFAULT_MODELS[provider]).strip() or DEFAULT_MODELS[provider]


def provider_status() -> list[dict[str, Any]]:
    return [
        {
            "provider": provider,
            "available": bool(_provider_key(provider)),
            "model": model_name(provider),
            "thinkingLevel": GEMINI_THINKING_LEVEL,
        }
        for provider in _provider_order()
    ]


def prompt_hash() -> str:
    return hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:16]


def provider_timeout_seconds() -> float:
    try:
        configured = float(os.getenv("PROVIDER_TIMEOUT_SECONDS", "35"))
    except ValueError:
        configured = 35.0
    # Bound each Gemini request so a single same-model retry remains inside
    # Gunicorn's 120-second limit.
    return min(38.0, max(5.0, configured))


def _angle_context(body_area: str, angle: str) -> str:
    contexts = BODY_AREA_ANGLE_CONTEXT.get(body_area)
    if not contexts or angle not in contexts:
        raise SchemaValidationError("body area or angle is unsupported")
    return contexts[angle]


def _image_context(images: Sequence[NormalizedImage], body_area: str) -> str:
    labels = ", ".join(
        f"{image.angle} ({_angle_context(body_area, image.angle)})" for image in images
    )
    allowed_ids = BODY_AREA_OBSERVATIONS.get(body_area)
    if not allowed_ids:
        raise SchemaValidationError("body area is unsupported")
    quality_context = BODY_AREA_QUALITY_CONTEXT[body_area]
    return (
        f"Review {len(images)} normalized image(s) of the selected area '{body_area}'. "
        f"Image angle labels, in upload order: {labels}. Use only those angle labels. "
        f"{quality_context} Do not require studio lighting, a blank background, or "
        "perfectly centered composition when visible skin detail is still usable. "
        f"For this area, the only permitted observation IDs are: {', '.join(allowed_ids)}."
    )


def _parse_json_text(value: str) -> Any:
    text = value.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


def _call_gemini(
    images: Sequence[NormalizedImage], body_area: str, api_key: str, model: str
) -> Any:
    from google import genai
    from google.genai import types

    parts: list[Any] = [types.Part.from_text(text=_image_context(images, body_area))]
    for image in images:
        parts.append(
            types.Part.from_text(
                text=(
                    f"Angle slot: {image.angle}. Capture meaning: "
                    f"{_angle_context(body_area, image.angle)}."
                )
            )
        )
        parts.append(types.Part.from_bytes(data=image.data, mime_type=image.media_type))
    timeout_ms = int(provider_timeout_seconds() * 1000)
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=timeout_ms,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )
    response = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            thinking_config=types.ThinkingConfig(
                thinking_level=GEMINI_THINKING_LEVEL
            ),
            max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
            seed=GEMINI_SEED,
            response_mime_type="application/json",
            response_json_schema=gemini_output_schema(body_area),
        ),
    )
    output_text = getattr(response, "text", None)
    if not isinstance(output_text, str) or not output_text.strip():
        raise ProviderUnavailable("Gemini returned no structured output")
    return _parse_json_text(output_text)


PROVIDER_CALLS: Mapping[
    str, Callable[[Sequence[NormalizedImage], str, str, str], Any]
] = {
    "gemini": _call_gemini,
}


def analyze_with_providers(
    images: Sequence[NormalizedImage], body_area: str
) -> tuple[dict[str, Any], str, str]:
    """Try configured providers in order and accept only strictly valid output."""

    _validate_image_angle_sequence([image.angle for image in images])
    configured_count = 0
    for provider in _provider_order():
        api_key = _provider_key(provider)
        if not api_key:
            continue
        configured_count += 1
        selected_model = model_name(provider)
        for attempt in range(1, GEMINI_PROVIDER_ATTEMPTS + 1):
            try:
                raw = PROVIDER_CALLS[provider](images, body_area, api_key, selected_model)
                validated = validate_model_output(
                    raw, [image.angle for image in images], body_area=body_area
                )
                return validated, provider, selected_model
            except Exception as exc:
                # Never log request content, images, filenames, model text, keys, or PII.
                LOGGER.warning(
                    "Analyzer provider %s attempt %d/%d failed (%s)",
                    provider,
                    attempt,
                    GEMINI_PROVIDER_ATTEMPTS,
                    type(exc).__name__,
                )
    if configured_count == 0:
        LOGGER.warning("Analyzer has no configured providers")
    raise ProviderUnavailable("No configured provider produced a valid response")


def _recommendation_basis(
    strengths: Sequence[str],
    priorities: Sequence[str],
    observations: Sequence[Mapping[str, Any]],
    body_area: str,
) -> list[str]:
    """Return every supported visible match, or every subtle maintenance match."""

    levels_by_id = {
        str(item.get("id")): str(item.get("level")) for item in observations
    }
    ordered_visible: list[str] = []
    for observation_id in priorities:
        if (
            levels_by_id.get(observation_id) in {"visible", "prominent"}
            and observation_id not in ordered_visible
        ):
            ordered_visible.append(observation_id)
    for level in ("prominent", "visible"):
        for observation_id in BODY_AREA_OBSERVATIONS.get(body_area, ()):
            if (
                levels_by_id.get(observation_id) == level
                and observation_id not in ordered_visible
            ):
                ordered_visible.append(observation_id)
    if ordered_visible:
        return ordered_visible

    ordered_subtle: list[str] = []
    for observation_id in strengths:
        if (
            levels_by_id.get(observation_id) == "subtle"
            and observation_id not in ordered_subtle
        ):
            ordered_subtle.append(observation_id)
    for observation_id in BODY_AREA_OBSERVATIONS.get(body_area, ()):
        if (
            levels_by_id.get(observation_id) == "subtle"
            and observation_id not in ordered_subtle
        ):
            ordered_subtle.append(observation_id)
    return ordered_subtle


def _discussion_topics(
    recommendation_ids: Sequence[str], body_area: str
) -> list[dict[str, str]]:
    """Return the legacy topic alias from the first two catalog matches."""

    recommendations = build_appearance_recommendations(
        recommendation_ids, body_area, OBSERVATION_LABELS
    )
    return [
        {
            "id": service["id"],
            "name": service["name"],
            "why": service["why"],
        }
        for service in recommendations["services"][:2]
    ]


def _medical_message(status: str) -> str:
    if status == "medical_review":
        return (
            "A licensed medical professional should review this area in person before "
            "any cosmetic discussion. This preview cannot diagnose or rule out disease."
        )
    if status == "retake":
        return (
            "Image quality was not sufficient for this limited cosmetic preview. "
            "Retake the requested image or images before continuing."
        )
    return (
        "This preview does not assess medical conditions. Seek medical care for any "
        "new, changing, painful, bleeding, or otherwise concerning area."
    )


def build_final_result(
    model_result: Mapping[str, Any],
    *,
    provider: str,
    selected_model: str,
    image_count: int,
    body_area: str,
) -> dict[str, Any]:
    status = str(model_result["status"])
    medical_suggested = bool(model_result["medicalReview"]["suggested"])
    if medical_suggested:
        status = "medical_review"
    suppress_cosmetic_findings = status in {"medical_review", "retake"}
    recommendation_ids = _recommendation_basis(
        model_result["strengths"],
        model_result["priorities"],
        model_result["observations"],
        body_area,
    )
    recommendations = (
        {"services": [], "products": []}
        if suppress_cosmetic_findings
        else build_appearance_recommendations(
            recommendation_ids, body_area, OBSERVATION_LABELS
        )
    )
    topics = (
        []
        if suppress_cosmetic_findings
        else [
            {
                "id": service["id"],
                "name": service["name"],
                "why": service["why"],
            }
            for service in recommendations["services"][:2]
        ]
    )
    result: dict[str, Any] = {
        "status": status,
        "bodyArea": body_area,
        "analysisVersion": ANALYSIS_VERSION,
        "schemaVersion": SCHEMA_VERSION,
        "topicMappingVersion": TOPIC_MAPPING_VERSION,
        "promptHash": prompt_hash(),
        "model": {
            "provider": provider,
            "name": selected_model,
            "promptVersion": PROMPT_VERSION,
        },
        "imageCount": image_count,
        "quality": dict(model_result["quality"]),
        "observations": [] if suppress_cosmetic_findings else [dict(item) for item in model_result["observations"]],
        "strengths": [] if suppress_cosmetic_findings else list(model_result["strengths"]),
        "priorities": [] if suppress_cosmetic_findings else list(model_result["priorities"]),
        "appearanceRecommendations": recommendations,
        "discussionTopics": topics,
        "medicalReview": {
            "suggested": status == "medical_review",
            "message": _medical_message(status),
        },
        "disclaimer": DISCLAIMER,
    }
    return validate_final_result(result)


def build_local_retake(
    *, image_count: int, issue: str, guidance: str, message: str, body_area: str
) -> dict[str, Any]:
    """Return the same canonical schema for an intake-level retake."""

    if issue not in QUALITY_ISSUES or guidance not in QUALITY_GUIDANCE:
        raise SchemaValidationError("local retake codes are unsupported")
    result: dict[str, Any] = {
        "status": "retake",
        "bodyArea": body_area,
        "analysisVersion": ANALYSIS_VERSION,
        "schemaVersion": SCHEMA_VERSION,
        "topicMappingVersion": TOPIC_MAPPING_VERSION,
        "promptHash": prompt_hash(),
        "model": {
            "provider": "local",
            "name": "image-intake-validator",
            "promptVersion": PROMPT_VERSION,
        },
        "imageCount": image_count,
        "quality": {
            "overall": "retake",
            "issues": [issue],
            "guidance": [guidance],
        },
        "observations": [],
        "strengths": [],
        "priorities": [],
        "appearanceRecommendations": {"services": [], "products": []},
        "discussionTopics": [],
        "medicalReview": {
            "suggested": False,
            "message": message,
        },
        "disclaimer": DISCLAIMER,
    }
    return validate_final_result(result)


def analyze(images: Sequence[NormalizedImage], body_area: str) -> dict[str, Any]:
    _validate_image_angle_sequence([image.angle for image in images])
    model_result, provider, selected_model = analyze_with_providers(images, body_area)
    return build_final_result(
        model_result,
        provider=provider,
        selected_model=selected_model,
        image_count=len(images),
        body_area=body_area,
    )
