"""Run a photo intake matrix against a chosen analyzer endpoint.

The default target is the current public site. That run is a baseline intake
diagnostic, not clinical validation and not evidence that a preview branch uses
a particular model. Set REQUIRE_PREVIEW_CONTRACT=true only when a private
preview URL is available and its response contract is ready to verify.
"""

from __future__ import annotations

import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


ENDPOINT = os.getenv("ANALYZER_ENDPOINT", "https://dearvon.com/api/analyze")
REQUIRE_PREVIEW_CONTRACT = (
    os.getenv("REQUIRE_PREVIEW_CONTRACT", "false").lower() == "true"
)
ROOT = Path(__file__).resolve().parents[1]
PHOTO_DIR = ROOT / "work" / "test-images"
OUTPUT = ROOT / "work" / "qa" / (
    "preview-photo-matrix.json"
    if REQUIRE_PREVIEW_CONTRACT
    else "live-photo-matrix.json"
)
BRITTANY_PHOTO = Path(
    os.getenv(
        "BRITTANY_TEST_PHOTO",
        "/Users/macbookair2/Desktop/Von To Do/Brittany Test Photo.jpeg",
    )
)

CASES = [
    ("brittany-1", BRITTANY_PHOTO, "face", "strict_accept"),
    ("brittany-2", BRITTANY_PHOTO, "face", "strict_accept"),
    ("brittany-3", BRITTANY_PHOTO, "face", "strict_accept"),
    ("face-dark", PHOTO_DIR / "stock-face-dark.jpg", "face", "exploratory"),
    ("face-light", PHOTO_DIR / "stock-face-light.jpg", "face", "exploratory"),
    ("face-profile", PHOTO_DIR / "stock-face-dark-male.jpg", "face", "exploratory"),
    ("neck-chest", PHOTO_DIR / "stock-neck-chest.jpg", "neck_chest", "exploratory"),
    ("hands", PHOTO_DIR / "stock-hands.jpg", "hands", "exploratory"),
    ("back", PHOTO_DIR / "stock-back.jpg", "back", "exploratory"),
    ("legs", PHOTO_DIR / "stock-legs.jpg", "legs", "exploratory"),
]

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


def encode_multipart(image_path: Path, body_area: str) -> tuple[bytes, str]:
    boundary = f"----vonco-{uuid.uuid4().hex}"
    mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    chunks: list[bytes] = []

    def field(name: str, value: str) -> None:
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode(),
                b"\r\n",
            ]
        )

    field("body_area", body_area)
    field("age", "35")
    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="image"; '
                f'filename="{image_path.name}"\r\n'
            ).encode(),
            f"Content-Type: {mime}\r\n\r\n".encode(),
            image_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def request_analysis(image_path: Path, body_area: str) -> tuple[int, dict, float]:
    body, content_type = encode_multipart(image_path, body_area)
    request = urllib.request.Request(
        ENDPOINT,
        data=body,
        headers={"Content-Type": content_type, "Accept": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            status = response.status
            raw = response.read()
    except urllib.error.HTTPError as error:
        status = error.code
        raw = error.read()
    elapsed = time.monotonic() - start
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        payload = {"error": raw.decode("utf-8", errors="replace")[:500]}
    return status, payload, elapsed


def validate_preview_contract(payload: dict, area: str) -> list[str]:
    problems: list[str] = []
    required = {
        "overallScore",
        "positiveHighlights",
        "concerns",
        "recommendations",
        "productRecommendations",
        "summary",
    }
    for key in sorted(required - set(payload)):
        problems.append(f"missing {key}")
    if "skinAge" in payload:
        problems.append("removed skinAge field returned")

    highlights = payload.get("positiveHighlights", [])
    if not isinstance(highlights, list) or not 2 <= len(highlights) <= 3:
        problems.append("positiveHighlights must contain 2 to 3 items")
    elif any(
        not isinstance(item, dict)
        or not str(item.get("title", "")).strip()
        or not str(item.get("detail", "")).strip()
        for item in highlights
    ):
        problems.append("positiveHighlights contains an incomplete item")

    concerns = payload.get("concerns", {})
    keys = set(concerns) if isinstance(concerns, dict) else set()
    if keys != AREA_CONCERNS[area]:
        problems.append(
            f"concerns {sorted(keys)} != expected baseline keys "
            f"{sorted(AREA_CONCERNS[area])}"
        )
    return problems


def main() -> int:
    missing = [str(path) for _, path, _, _ in CASES if not path.exists()]
    if missing:
        print(f"Missing fixtures: {missing}", file=sys.stderr)
        return 2

    results = []
    hard_failures = []
    for index, (name, image_path, area, expectation) in enumerate(CASES, 1):
        print(f"[{index}/{len(CASES)}] {name} ({area})", flush=True)
        try:
            status, payload, elapsed = request_analysis(image_path, area)
        except Exception as error:
            status = 0
            payload = {"error": f"{type(error).__name__}: {error}"}
            elapsed = 0.0

        if status == 200 and not payload.get("error") and not payload.get("rejected"):
            outcome = "accepted"
        elif payload.get("rejected") or status == 422:
            outcome = "rejected"
        else:
            outcome = "error"

        problems = (
            validate_preview_contract(payload, area)
            if outcome == "accepted" and REQUIRE_PREVIEW_CONTRACT
            else []
        )
        record = {
            "name": name,
            "area": area,
            "expectation": expectation,
            "status": status,
            "outcome": outcome,
            "elapsedSeconds": round(elapsed, 1),
            "responseKeys": sorted(payload.keys()),
            "overallScore": payload.get("overallScore"),
            "concernKeys": (
                sorted(payload.get("concerns", {}).keys())
                if isinstance(payload.get("concerns"), dict)
                else []
            ),
            "problems": problems,
        }
        results.append(record)
        print(
            f"  {outcome} HTTP {status} in {elapsed:.1f}s, "
            f"contract problems={len(problems)}",
            flush=True,
        )

        if expectation == "strict_accept" and outcome != "accepted":
            hard_failures.append(f"{name}: exact Brittany fixture was {outcome}")
        if problems:
            hard_failures.append(f"{name}: {'; '.join(problems)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "endpoint": ENDPOINT,
                "scope": (
                    "preview response structure, not clinical accuracy"
                    if REQUIRE_PREVIEW_CONTRACT
                    else "public baseline intake acceptance only"
                ),
                "requiresPreviewContract": REQUIRE_PREVIEW_CONTRACT,
                "results": results,
                "hardFailures": hard_failures,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT}")
    if hard_failures:
        print("HARD FAILURES:")
        for failure in hard_failures:
            print(f"- {failure}")
        return 1
    print("MATRIX PASSED FOR ITS STATED SCOPE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
