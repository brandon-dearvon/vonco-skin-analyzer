"""Regression contract for the restored original app and approved deltas only."""

from __future__ import annotations

import hashlib
import re
import struct
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "public" / "index.html").read_text(encoding="utf-8")
SERVER = (ROOT / "server.py").read_text(encoding="utf-8")
REQUIREMENTS = (ROOT / "requirements.txt").read_text(encoding="utf-8")
GUNICORN = (ROOT / "gunicorn.conf.py").read_text(encoding="utf-8")
START_MAC = (ROOT / "start.command").read_text(encoding="utf-8")
START_WINDOWS = (ROOT / "start.bat").read_text(encoding="utf-8")


class RestoredOriginalContractTests(unittest.TestCase):
    def test_main_site_navigation_is_persistent_and_simple(self) -> None:
        self.assertIn('class="site-nav"', HTML)
        self.assertIn('class="site-nav-mark"', HTML)
        self.assertIn('class="site-nav-cta"', HTML)
        self.assertIn('<span>Learn More</span>', HTML)
        self.assertGreaterEqual(
            HTML.count('href="https://www.vonandcoaesthetics.com/"'), 3
        )
        nav = HTML[HTML.index('<nav class="site-nav"'):HTML.index('</nav>')]
        self.assertNotIn('<svg', nav)
        self.assertNotIn('arrow', nav.lower())

    def test_hero_and_primary_upload_copy_remain(self) -> None:
        for copy in (
            "Illuminating Results, Expertly Delivered",
            "A complimentary photo-based preview of visible skin patterns",
            "Your Skin Story Starts Here",
            "What This Photo Preview Reviews",
            "Your image is not retained after analysis. Most results are ready in under a minute.",
            "Upload one clear photo",
            "No camera capture required.",
        ):
            self.assertIn(copy, HTML)
        self.assertNotIn("Powered by the same clinical insight", HTML)
        self.assertNotIn("VISIA-Style Analysis", HTML)

    def test_canonical_hd_primary_lockups_are_direct_and_exact(self) -> None:
        expected = {
            "logo.png": "36e199cb870a2e981ab7367043365de82c661f738dd4cfd3961677add13fca01",
            "logo_white.png": "bdf8946f7a525e41266f8fdb8a2a333d3d42b7f2a671acb1e2c08475a5be5e29",
        }
        for filename, digest in expected.items():
            data = (ROOT / "public" / filename).read_bytes()
            self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(struct.unpack(">II", data[16:24]), (1549, 848))
            self.assertEqual(hashlib.sha256(data).hexdigest(), digest)

        self.assertIn('class="site-nav-logo" src="/logo.png"', HTML)
        self.assertIn('class="logo" src="/logo_white.png"', HTML)
        self.assertNotRegex(
            HTML,
            r'class="(?:site-nav-logo|logo)"[^>]+src="data:image/',
        )

    def test_header_uses_brand_fonts_and_logo_clear_space(self) -> None:
        cta_rule = re.search(r"\.site-nav-cta\s*\{([^}]+)\}", HTML, re.S)
        self.assertIsNotNone(cta_rule)
        self.assertIn("background: var(--evergreen)", cta_rule.group(1))
        self.assertIn("text-transform: uppercase", cta_rule.group(1))
        self.assertRegex(HTML, r"body\s*\{[^}]*font-family:\s*'Fira Sans'", re.S)
        logo_rule = re.search(r"\.site-nav-logo\s*\{([^}]+)\}", HTML, re.S)
        self.assertIsNotNone(logo_rule)
        self.assertIn("height: auto", logo_rule.group(1))
        self.assertNotIn("filter:", logo_rule.group(1))

    def test_original_intake_modes_and_body_areas_remain(self) -> None:
        for text in ("Upload one clear photo", "Use Camera", "Quick Snap", "Guided Capture"):
            self.assertIn(text, HTML)
        for value in ("face", "neck_chest", "hands", "back", "legs"):
            self.assertIn(f'value="{value}"', HTML)
        for marker in (
            'id="fileInput"',
            'id="webcamModal"',
            'id="modeQuick"',
            'id="modeGuided"',
            "guidedPhotos = [null, null, null]",
        ):
            self.assertIn(marker, HTML)

    def test_guided_capture_limitation_is_frozen_not_misrepresented(self) -> None:
        submit = HTML[
            HTML.index("function submitGuidedCapture"):
            HTML.index("function checkLighting")
        ]
        self.assertIn("uploadedImage = guidedPhotos[0]", submit)
        self.assertIn("window.guidedPhotos = [...guidedPhotos]", submit)
        analyze = HTML[
            HTML.index("async function analyzeImage"):
            HTML.index("function showRejectionMessage")
        ]
        self.assertEqual(analyze.count("formData.append('image'"), 1)

    def test_six_stage_progress_tracker_remains(self) -> None:
        self.assertIn('id="analysisSteps"', HTML)
        for step in range(6):
            self.assertIn(f'id="analysisStep{step}"', HTML)
        self.assertIn('id="analysisProgressFill"', HTML)
        self.assertIn('id="analysisElapsed"', HTML)

    def test_analysis_timeout_and_live_timer_are_bounded(self) -> None:
        self.assertIn("HttpOptions(", SERVER)
        self.assertIn('GOOGLE_TOTAL_BUDGET_MS', SERVER)
        self.assertIn('str(70_000)', SERVER)
        self.assertIn('GOOGLE_HEDGE_DELAY_MS', SERVER)
        self.assertIn('str(15_000)', SERVER)
        self.assertIn('GOOGLE_MAX_OUTPUT_TOKENS', SERVER)
        self.assertIn('str(32_768)', SERVER)
        self.assertIn('time.monotonic()', SERVER)
        self.assertIn('ThreadPoolExecutor(max_workers=2', SERVER)
        self.assertIn('return_when=FIRST_COMPLETED', SERVER)
        self.assertIn('shutdown(wait=False, cancel_futures=True)', SERVER)
        self.assertIn("timeout=timeout_ms", SERVER)
        self.assertIn("HttpRetryOptions(attempts=1)", SERVER)
        self.assertIn("genai_errors.APIError", SERVER)
        self.assertIn("GOOGLE_TRANSIENT_STATUS_CODES", SERVER)
        self.assertIn('"code": "analysis_timeout"', SERVER)
        self.assertIn('"retryable": True', SERVER)

        analyze = HTML[
            HTML.index("async function analyzeImage"):
            HTML.index("function showRejectionMessage")
        ]
        self.assertIn("AbortController", analyze)
        self.assertRegex(analyze, r"signal:\s*\w+\.signal")
        self.assertIn("analysis_timeout", analyze)

        progress = HTML[
            HTML.index("function startAnalysisProgress"):
            HTML.index("function stopAnalysisProgress")
        ]
        self.assertNotIn("elapsed.parentElement.innerHTML", progress)

    def test_original_result_and_recommendation_features_remain(self) -> None:
        for marker in (
            'id="overallScore"',
            'id="scoreCircle"',
            'id="scoreInterpretation"',
            'id="insightsBar"',
            'id="concernsGrid"',
            'id="recommendationCards"',
            "productRecommendations",
            "suggestedCombo",
            'id="resultsPlanTeaser"',
            'id="treatmentRecommendationCards"',
            'id="productRecommendationCards"',
            'id="findingsToggle"',
            "See My Recommendations",
        ):
            self.assertIn(marker, HTML)
        self.assertIn("View Plan", HTML)
        self.assertNotIn("Redness & Rosacea", HTML)
        self.assertNotIn("Redness & Rosacea", SERVER)
        for unsupported_label in (
            "Dark Spots & Hyperpigmentation",
            "Sun Damage & Photoaging",
            "Visible Veins & Vascularity",
            "Scarring & Marks",
            "Body Acne & Breakouts",
            "Dryness & Dehydration",
        ):
            self.assertNotIn(unsupported_label, HTML)

    def test_original_lead_offer_club_and_report_features_remain(self) -> None:
        for marker in (
            'id="leadGateOverlay"',
            'id="leadName"',
            'id="leadEmail"',
            'id="leadPhone"',
            'id="downloadReportBtn"',
            'id="reportPreviewContainer"',
            "function buildReportHTML",
            "function downloadReport",
            "15% off your first visit + The Club",
            'class="promo-banner"',
            'class="von-offer-cta"',
            'class="club-cta"',
            ">Learn More</a>",
            "arsenica-regular.otf",
            "Personalized Skin Analysis Report",
        ):
            self.assertIn(marker, HTML)
        self.assertNotIn(">Join The Club</a>", HTML)

    def test_estimated_skin_age_and_radar_are_removed_everywhere(self) -> None:
        removed = (
            "skinAge",
            "Estimated Skin Age",
            "Est. Skin Age",
            "skinAgeBadge",
            "skinAgeValue",
            "radarChart",
            "radarContainer",
            "skin-radar",
            "radar-chart",
        )
        for value in removed:
            self.assertNotIn(value, HTML)
            self.assertNotIn(value, SERVER)

    def test_completed_results_lead_with_grounded_positives(self) -> None:
        positive_index = HTML.index('id="positiveLead"')
        score_index = HTML.index('class="score-ring-container"')
        concern_index = HTML.index('id="concernsGrid"')
        self.assertLess(positive_index, score_index)
        self.assertLess(positive_index, concern_index)
        self.assertIn('id="positiveHighlights"', HTML)
        self.assertIn("positiveHighlights.length < 2", HTML)
        self.assertIn("title.textContent = highlight.title", HTML)
        self.assertIn("detail.textContent = highlight.detail", HTML)
        self.assertIn("Never phrase a positive as the absence of a concern", SERVER)
        self.assertIn("Overall View", HTML)
        self.assertNotIn("Avg Skin Health", HTML)

    def test_positive_schema_is_required_and_bounded(self) -> None:
        self.assertIn('"positiveHighlights": {', SERVER)
        self.assertIn('"minItems": 2', SERVER)
        self.assertIn('"maxItems": 3', SERVER)
        required = SERVER[SERVER.index('"required": [\n                "overallScore"'):]
        self.assertIn('"positiveHighlights"', required[:500])

    def test_take_home_report_is_positive_first_and_complete(self) -> None:
        report = HTML[HTML.index("function buildReportHTML"):]
        positive = report.index("Begin With the Positive")
        score = report.index("Overall Score:")
        concerns = report.index("Skin Analysis Results")
        self.assertLess(positive, score)
        self.assertLess(score, concerns)
        self.assertIn("new URL('/logo.png'", report)
        self.assertIn("analysis.productRecommendations || []", report)
        self.assertIn("Recommended Treatments", report)
        self.assertIn("Your Skincare Essentials", report)
        self.assertIn("Any concerning lesion needs an in-person medical evaluation.", report)

    def test_provider_is_gemini_pro_with_high_thinking_and_bounded_output(self) -> None:
        self.assertIn('GOOGLE_MODEL = "gemini-3.1-pro-preview"', SERVER)
        self.assertIn("thinking_level=genai_types.ThinkingLevel.HIGH", SERVER)
        self.assertIn('response_mime_type="application/json"', SERVER)
        self.assertIn("response_json_schema=_analysis_schema_for_area(body_area)", SERVER)
        self.assertIn("max_output_tokens=GOOGLE_MAX_OUTPUT_TOKENS", SERVER)
        self.assertEqual(REQUIREMENTS.count("google-genai==2.11.0"), 1)
        self.assertNotIn("anthropic", SERVER.lower())
        self.assertNotIn("anthropic", REQUIREMENTS.lower())
        self.assertNotIn("DiamondGlow", SERVER)

    def test_provider_setup_and_runtime_copy_are_google_only(self) -> None:
        current_provider_surface = "\n".join(
            (HTML, SERVER, REQUIREMENTS, GUNICORN, START_MAC, START_WINDOWS)
        ).lower()
        self.assertNotIn("anthropic", current_provider_surface)
        self.assertNotIn("claude", current_provider_surface)
        self.assertIn("google_api_key", START_MAC.lower())
        self.assertIn("google_api_key", START_WINDOWS.lower())
        for launcher in (START_MAC, START_WINDOWS):
            self.assertIn("m.version('google-genai') == '2.11.0'", launcher)
            self.assertIn("ThinkingLevel.HIGH.value == 'HIGH'", launcher)

        for filename in ("DEPLOY-GUIDE.docx", "SETUP-GUIDE.docx"):
            with zipfile.ZipFile(ROOT / filename) as archive:
                provider_xml = b"\n".join(
                    archive.read(name)
                    for name in archive.namelist()
                    if name.endswith((".xml", ".rels"))
                ).lower()
            self.assertNotIn(b"anthropic", provider_xml)
            self.assertNotIn(b"sk-ant", provider_xml)
            self.assertIn(b"google_api_key", provider_xml)

    def test_demo_and_rejection_results_cannot_masquerade_as_live(self) -> None:
        self.assertIn("demoMode = health.mode !== 'live'", HTML)
        self.assertIn("if (data.rejected)", HTML)
        self.assertIn("if (data._isDemo || demoMode)", HTML)
        self.assertIn('analysis["_isDemo"] = True', SERVER)
        analyze = HTML[
            HTML.index("async function analyzeImage"):
            HTML.index("function showRejectionMessage")
        ]
        self.assertNotIn("using demo fallback", analyze.lower())

    def test_model_does_not_triage_lesions_and_startup_does_not_echo_key_data(self) -> None:
        self.assertIn("Do not identify, assess, flag, rule out, or comment on lesions", SERVER)
        self.assertNotIn("suspicious mole", SERVER.lower())
        self.assertNotIn("GOOGLE_API_KEY[-", SERVER)
        self.assertIn('Google API: {"Configured" if LIVE_MODE else "Not configured"}', SERVER)

    def test_estimated_skin_age_is_removed_and_minor_gate_remains(self) -> None:
        self.assertIn("appears to be under 18 years old", SERVER)
        self.assertIn("Do not produce or compare an estimated skin age", SERVER)
        self.assertIn("Never return, infer, or compare an adult skin-age estimate", SERVER)
        self.assertNotIn("compare their estimated skin age", SERVER)

    def test_small_concerning_lesion_disclaimer_is_present(self) -> None:
        disclaimer = "Any concerning lesion needs an in-person medical evaluation."
        self.assertIn(disclaimer, HTML)
        self.assertIn(disclaimer, SERVER)

    def test_no_unapproved_behavior_modules_were_added(self) -> None:
        for abandoned_marker in (
            "PHOTO_OPTION_ALLOWLIST",
            "BODY_AREA_CONCERN_KEYS",
            "COMBO_COMPONENTS",
            "LEAD_DB_PATH",
            "analysis_response_schema_for",
        ):
            self.assertNotIn(abandoned_marker, SERVER)


if __name__ == "__main__":
    unittest.main()
