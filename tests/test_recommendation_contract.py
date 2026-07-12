"""Regression tests for server-owned appearance recommendation mapping."""

from __future__ import annotations

import unittest
from copy import deepcopy

import analysis_engine
import recommendation_catalog


def _model_result(
    priorities: list[str],
    *,
    body_area: str = "face",
    status: str = "complete",
) -> dict:
    observations = [
        {
            "id": observation_id,
            "label": analysis_engine.OBSERVATION_LABELS[observation_id],
            "level": "visible",
            "description": "This feature can be seen in the submitted view.",
            "angles": ["single"],
        }
        for observation_id in priorities
    ]
    quality = {"overall": "suitable", "issues": [], "guidance": []}
    medical_review = {"suggested": False, "reason": "none"}
    if status == "medical_review":
        quality = {
            "overall": "limited",
            "issues": ["obstruction"],
            "guidance": ["remove_obstructions"],
        }
        medical_review = {
            "suggested": True,
            "reason": "open_or_broken_skin",
        }
    elif status == "retake":
        quality = {
            "overall": "retake",
            "issues": ["blur"],
            "guidance": ["hold_camera_steady"],
        }
    raw = {
        "status": status,
        "quality": quality,
        "observations": observations,
        "strengths": [],
        "priorities": priorities,
        "medicalReview": medical_review,
    }
    return analysis_engine.validate_model_output(raw, ["single"], body_area=body_area)


def _public_result(
    priorities: list[str],
    *,
    body_area: str = "face",
    status: str = "complete",
) -> dict:
    return analysis_engine.build_final_result(
        _model_result(priorities, body_area=body_area, status=status),
        provider="gemini",
        selected_model="gemini-3.5-flash",
        image_count=1,
        body_area=body_area,
    )


class RecommendationCatalogTests(unittest.TestCase):
    def test_catalog_entries_cite_locked_source_evidence(self) -> None:
        evidence_ids = set(recommendation_catalog.SOURCE_EVIDENCE)
        for source_id, source in recommendation_catalog.SOURCE_EVIDENCE.items():
            with self.subTest(source_id=source_id):
                self.assertIn(source["reviewed"], {"2026-07-11", "2026-07-12"})
                self.assertTrue(source.get("sha256") or source.get("url"))
                if "sha256" in source:
                    self.assertEqual(len(source["sha256"]), 64)

        for catalog_name, catalog in (
            ("service", recommendation_catalog.SERVICE_CATALOG),
            ("product", recommendation_catalog.PRODUCT_CATALOG),
        ):
            for item_id, item in catalog.items():
                with self.subTest(catalog=catalog_name, item_id=item_id):
                    refs = item.get("sourceRefs")
                    self.assertIsInstance(refs, tuple)
                    self.assertTrue(refs)
                    self.assertTrue(set(refs).issubset(evidence_ids))

    def test_catalog_covers_the_full_locked_provider_guides(self) -> None:
        expected_service_ids = {
            "anti_aging_facial",
            "botox",
            "brow_lamination",
            "chemical_peels",
            "deep_pore_facial",
            "dermal_fillers",
            "dysport",
            "hair_restoration_prf",
            "hydrafacial_clarifying",
            "hydrafacial_customized",
            "hydrafacial_elite",
            "kybella",
            "laser_hair_removal",
            "lip_filler",
            "microneedling",
            "microneedling_prf",
            "rf_microneedling",
            "saltfacial",
            "sciton_bbl_photofacial",
            "sciton_halo_laser",
            "sciton_moxi_laser",
            "sculptra",
            "semaglutide_melts",
            "signature_facial",
            "skinvive",
            "tirzepatide_injections",
            "xeomin",
        }
        expected_product_ids = {
            "alastin_hydratint",
            "alastin_restorative_eye_cream",
            "alastin_restorative_skin_complex",
            "alastin_skin_nectar",
            "avene_cicalfate",
            "avene_thermal_water",
            "colorscience_brush_on_shield",
            "colorscience_face_shield",
            "colorscience_lash_mascara",
            "colorscience_lip_shine_spf_35",
            "colorscience_total_eye_spf_35",
            "hydrinity_eye_renew",
            "hydrinity_hyacyn_mist",
            "hydrinity_renewing_ha_serum",
            "hydrinity_retaxome",
            "hydrinity_vivid_serum",
            "isdin_eryfotona_actinica",
            "isdin_eryfotona_tinted",
            "isdin_melaclear_advanced",
            "isdin_retinal_advanced",
            "pavise_dynamic_age_defense",
            "revitalash_brow_conditioner",
            "revitalash_conditioner",
            "skinbetter_alpharet",
            "skinbetter_alpharet_body",
            "skinbetter_alto_defense",
            "skinbetter_even_tone",
            "skinbetter_peel_pads",
            "skinbetter_sunbetter_spf_68",
            "skinbetter_trio_moisture",
            "zo_complexion_renewal_pads",
            "zo_growth_factor_eye",
            "zo_growth_factor_serum",
            "zo_vitamin_c_10",
            "zo_wrinkle_texture_repair",
        }
        self.assertEqual(set(recommendation_catalog.SERVICE_CATALOG), expected_service_ids)
        self.assertEqual(set(recommendation_catalog.PRODUCT_CATALOG), expected_product_ids)
        self.assertEqual(recommendation_catalog.MAX_SERVICE_RECOMMENDATIONS, 27)
        self.assertEqual(recommendation_catalog.MAX_PRODUCT_RECOMMENDATIONS, 35)

    def test_source_derived_face_matrix_matches_locked_guides_and_live_pages(self) -> None:
        expected = {
            "visible_lines": (
                [
                    "botox",
                    "dysport",
                    "xeomin",
                    "microneedling",
                    "microneedling_prf",
                    "rf_microneedling",
                    "sciton_moxi_laser",
                    "sciton_halo_laser",
                    "chemical_peels",
                    "saltfacial",
                    "skinvive",
                    "anti_aging_facial",
                    "sculptra",
                ],
                [
                    "skinbetter_alpharet",
                    "zo_wrinkle_texture_repair",
                    "colorscience_face_shield",
                ],
            ),
            "visible_redness": (
                ["sciton_bbl_photofacial"],
                [
                    "avene_thermal_water",
                    "skinbetter_alto_defense",
                    "colorscience_face_shield",
                ],
            ),
            "pigment_variation": (
                [
                    "sciton_bbl_photofacial",
                    "sciton_moxi_laser",
                    "sciton_halo_laser",
                    "microneedling",
                    "microneedling_prf",
                    "rf_microneedling",
                    "chemical_peels",
                    "hydrafacial_customized",
                    "hydrafacial_elite",
                    "saltfacial",
                ],
                [
                    "skinbetter_even_tone",
                    "isdin_melaclear_advanced",
                    "zo_vitamin_c_10",
                    "hydrinity_vivid_serum",
                    "colorscience_face_shield",
                ],
            ),
            "surface_texture": (
                [
                    "sciton_moxi_laser",
                    "sciton_bbl_photofacial",
                    "hydrafacial_customized",
                    "hydrafacial_elite",
                    "saltfacial",
                    "skinvive",
                    "signature_facial",
                    "anti_aging_facial",
                    "sciton_halo_laser",
                    "microneedling",
                    "microneedling_prf",
                    "rf_microneedling",
                    "chemical_peels",
                ],
                [
                    "zo_complexion_renewal_pads",
                    "skinbetter_peel_pads",
                    "colorscience_face_shield",
                ],
            ),
            "pore_visibility": (
                [
                    "hydrafacial_clarifying",
                    "hydrafacial_customized",
                    "hydrafacial_elite",
                    "deep_pore_facial",
                    "saltfacial",
                    "microneedling",
                    "microneedling_prf",
                    "sciton_moxi_laser",
                    "rf_microneedling",
                    "sciton_halo_laser",
                    "signature_facial",
                ],
                ["zo_complexion_renewal_pads", "colorscience_face_shield"],
            ),
            "laxity_appearance": (
                ["rf_microneedling", "sculptra", "sciton_halo_laser"],
                [
                    "alastin_restorative_skin_complex",
                    "zo_growth_factor_serum",
                    "colorscience_face_shield",
                ],
            ),
            "blemish_like_spots": (
                [
                    "hydrafacial_clarifying",
                    "deep_pore_facial",
                    "saltfacial",
                    "chemical_peels",
                    "signature_facial",
                ],
                ["zo_complexion_renewal_pads", "colorscience_face_shield"],
            ),
            "scar_like_texture": (
                [
                    "microneedling",
                    "microneedling_prf",
                    "sciton_moxi_laser",
                    "rf_microneedling",
                    "chemical_peels",
                    "sciton_halo_laser",
                    "saltfacial",
                ],
                ["colorscience_face_shield"],
            ),
            "superficial_vessels": (
                ["sciton_bbl_photofacial"],
                ["colorscience_face_shield"],
            ),
            "visible_flaking": (
                [
                    "hydrafacial_customized",
                    "hydrafacial_elite",
                    "signature_facial",
                    "anti_aging_facial",
                ],
                [
                    "hydrinity_renewing_ha_serum",
                    "skinbetter_trio_moisture",
                    "colorscience_face_shield",
                ],
            ),
        }

        for observation_id, (service_ids, product_ids) in expected.items():
            with self.subTest(observation_id=observation_id):
                recommendations = recommendation_catalog.build_appearance_recommendations(
                    [observation_id], "face", analysis_engine.OBSERVATION_LABELS
                )
                self.assertEqual(
                    [item["id"] for item in recommendations["services"]], service_ids
                )
                self.assertEqual(
                    [item["id"] for item in recommendations["products"]], product_ids
                )

    def test_every_face_priority_gets_one_daily_spf_essential(self) -> None:
        for observation_id in analysis_engine.BODY_AREA_OBSERVATIONS["face"]:
            with self.subTest(observation_id=observation_id):
                recommendations = recommendation_catalog.build_appearance_recommendations(
                    [observation_id], "face", analysis_engine.OBSERVATION_LABELS
                )
                products = recommendations["products"]
                product_ids = [item["id"] for item in products]
                self.assertLessEqual(
                    len(products), recommendation_catalog.MAX_PRODUCT_RECOMMENDATIONS
                )
                self.assertEqual(product_ids[-1], "colorscience_face_shield")
                self.assertEqual(product_ids.count("colorscience_face_shield"), 1)

                protection = products[-1]
                self.assertEqual(protection["category"], "Daily protection")
                self.assertEqual(protection["relationship"], "Daily protection")
                self.assertEqual(
                    protection["matchedObservationIds"], [observation_id]
                )
                self.assertIn("daily SPF", protection["why"])

        no_priorities = recommendation_catalog.build_appearance_recommendations(
            [], "face", analysis_engine.OBSERVATION_LABELS
        )
        self.assertEqual(no_priorities["products"], [])

    def test_combined_priorities_preserve_every_supported_match(self) -> None:
        for observation_id in (
            "blemish_like_spots",
            "superficial_vessels",
            "visible_flaking",
        ):
            with self.subTest(observation_id=observation_id):
                line_only = recommendation_catalog.build_appearance_recommendations(
                    ["visible_lines"], "face", analysis_engine.OBSERVATION_LABELS
                )
                second_only = recommendation_catalog.build_appearance_recommendations(
                    [observation_id], "face", analysis_engine.OBSERVATION_LABELS
                )
                combined = recommendation_catalog.build_appearance_recommendations(
                    ["visible_lines", observation_id],
                    "face",
                    analysis_engine.OBSERVATION_LABELS,
                )
                for group in ("services", "products"):
                    combined_ids = [item["id"] for item in combined[group]]
                    expected_ids = {
                        item["id"] for item in line_only[group] + second_only[group]
                    }
                    self.assertEqual(set(combined_ids), expected_ids)
                    self.assertEqual(len(combined_ids), len(set(combined_ids)))

    def test_body_products_follow_explicit_guide_areas(self) -> None:
        cases = (
            (
                "chest",
                "visible_lines",
                ["alastin_restorative_skin_complex", "skinbetter_alpharet_body"],
            ),
            ("neck", "visible_lines", []),
            (
                "hands",
                "laxity_appearance",
                ["alastin_restorative_skin_complex", "skinbetter_alpharet_body"],
            ),
            ("back", "surface_texture", []),
            ("legs", "laxity_appearance", []),
        )
        for body_area, observation_id, expected_product_ids in cases:
            with self.subTest(body_area=body_area, observation_id=observation_id):
                recommendations = recommendation_catalog.build_appearance_recommendations(
                    [observation_id],
                    body_area,
                    analysis_engine.OBSERVATION_LABELS,
                )
                self.assertEqual(
                    [item["id"] for item in recommendations["products"]],
                    expected_product_ids,
                )

    def test_body_services_follow_explicit_area_and_appearance_gates(self) -> None:
        cases = (
            (
                "neck",
                "pigment_variation",
                [
                    "sciton_bbl_photofacial",
                    "sciton_moxi_laser",
                    "rf_microneedling",
                    "hydrafacial_customized",
                    "hydrafacial_elite",
                ],
            ),
            (
                "chest",
                "pigment_variation",
                ["sciton_bbl_photofacial", "sciton_moxi_laser"],
            ),
            (
                "legs",
                "surface_texture",
                ["sciton_moxi_laser", "sciton_bbl_photofacial"],
            ),
            ("hands", "visible_lines", ["sciton_moxi_laser"]),
        )
        for body_area, observation_id, expected_service_ids in cases:
            with self.subTest(
                body_area=body_area, observation_id=observation_id
            ):
                recommendations = recommendation_catalog.build_appearance_recommendations(
                    [observation_id], body_area, analysis_engine.OBSERVATION_LABELS
                )
                self.assertEqual(
                    [item["id"] for item in recommendations["services"]],
                    expected_service_ids,
                )

    def test_every_appearance_and_body_area_pair_has_a_locked_expectation(self) -> None:
        self.assertEqual(
            set(analysis_engine.BODY_AREA_OBSERVATIONS),
            {"face", "neck", "chest", "hands", "back", "legs"},
        )
        face_services = {
            "visible_lines": [
                "botox",
                "dysport",
                "xeomin",
                "microneedling",
                "microneedling_prf",
                "rf_microneedling",
                "sciton_moxi_laser",
                "sciton_halo_laser",
                "chemical_peels",
                "saltfacial",
                "skinvive",
                "anti_aging_facial",
                "sculptra",
            ],
            "visible_redness": ["sciton_bbl_photofacial"],
            "pigment_variation": [
                "sciton_bbl_photofacial",
                "sciton_moxi_laser",
                "sciton_halo_laser",
                "microneedling",
                "microneedling_prf",
                "rf_microneedling",
                "chemical_peels",
                "hydrafacial_customized",
                "hydrafacial_elite",
                "saltfacial",
            ],
            "surface_texture": [
                "sciton_moxi_laser",
                "sciton_bbl_photofacial",
                "hydrafacial_customized",
                "hydrafacial_elite",
                "saltfacial",
                "skinvive",
                "signature_facial",
                "anti_aging_facial",
                "sciton_halo_laser",
                "microneedling",
                "microneedling_prf",
                "rf_microneedling",
                "chemical_peels",
            ],
            "pore_visibility": [
                "hydrafacial_clarifying",
                "hydrafacial_customized",
                "hydrafacial_elite",
                "deep_pore_facial",
                "saltfacial",
                "microneedling",
                "microneedling_prf",
                "sciton_moxi_laser",
                "rf_microneedling",
                "sciton_halo_laser",
                "signature_facial",
            ],
            "laxity_appearance": [
                "rf_microneedling",
                "sculptra",
                "sciton_halo_laser",
            ],
            "blemish_like_spots": [
                "hydrafacial_clarifying",
                "deep_pore_facial",
                "saltfacial",
                "chemical_peels",
                "signature_facial",
            ],
            "scar_like_texture": [
                "microneedling",
                "microneedling_prf",
                "sciton_moxi_laser",
                "rf_microneedling",
                "chemical_peels",
                "sciton_halo_laser",
                "saltfacial",
            ],
            "superficial_vessels": ["sciton_bbl_photofacial"],
            "visible_flaking": [
                "hydrafacial_customized",
                "hydrafacial_elite",
                "signature_facial",
                "anti_aging_facial",
            ],
        }
        face_products = {
            "visible_lines": [
                "skinbetter_alpharet",
                "zo_wrinkle_texture_repair",
                "colorscience_face_shield",
            ],
            "visible_redness": [
                "avene_thermal_water",
                "skinbetter_alto_defense",
                "colorscience_face_shield",
            ],
            "pigment_variation": [
                "skinbetter_even_tone",
                "isdin_melaclear_advanced",
                "zo_vitamin_c_10",
                "hydrinity_vivid_serum",
                "colorscience_face_shield",
            ],
            "surface_texture": [
                "zo_complexion_renewal_pads",
                "skinbetter_peel_pads",
                "colorscience_face_shield",
            ],
            "pore_visibility": [
                "zo_complexion_renewal_pads",
                "colorscience_face_shield",
            ],
            "laxity_appearance": [
                "alastin_restorative_skin_complex",
                "zo_growth_factor_serum",
                "colorscience_face_shield",
            ],
            "blemish_like_spots": [
                "zo_complexion_renewal_pads",
                "colorscience_face_shield",
            ],
            "scar_like_texture": ["colorscience_face_shield"],
            "superficial_vessels": ["colorscience_face_shield"],
            "visible_flaking": [
                "hydrinity_renewing_ha_serum",
                "skinbetter_trio_moisture",
                "colorscience_face_shield",
            ],
        }
        body_services = {
            "neck": {
                "visible_lines": [
                    "botox",
                    "dysport",
                    "xeomin",
                    "rf_microneedling",
                    "sciton_moxi_laser",
                ],
                "visible_redness": ["sciton_bbl_photofacial"],
                "pigment_variation": [
                    "sciton_bbl_photofacial",
                    "sciton_moxi_laser",
                    "rf_microneedling",
                    "hydrafacial_customized",
                    "hydrafacial_elite",
                ],
                "surface_texture": [
                    "sciton_moxi_laser",
                    "sciton_bbl_photofacial",
                    "hydrafacial_customized",
                    "hydrafacial_elite",
                    "rf_microneedling",
                ],
                "pore_visibility": [
                    "hydrafacial_clarifying",
                    "hydrafacial_customized",
                    "hydrafacial_elite",
                    "sciton_moxi_laser",
                    "rf_microneedling",
                ],
                "blemish_like_spots": ["hydrafacial_clarifying"],
                "laxity_appearance": ["rf_microneedling"],
                "scar_like_texture": ["sciton_moxi_laser", "rf_microneedling"],
                "superficial_vessels": ["sciton_bbl_photofacial"],
                "visible_flaking": [
                    "hydrafacial_customized",
                    "hydrafacial_elite",
                ],
            },
            "chest": {
                "visible_lines": ["sciton_moxi_laser", "sculptra"],
                "visible_redness": ["sciton_bbl_photofacial"],
                "pigment_variation": [
                    "sciton_bbl_photofacial",
                    "sciton_moxi_laser",
                ],
                "surface_texture": [
                    "sciton_moxi_laser",
                    "sciton_bbl_photofacial",
                ],
                "pore_visibility": ["sciton_moxi_laser"],
                "laxity_appearance": ["sculptra"],
                "scar_like_texture": ["sciton_moxi_laser"],
                "superficial_vessels": ["sciton_bbl_photofacial"],
            },
            "hands": {
                "visible_lines": ["sciton_moxi_laser"],
                "visible_redness": ["sciton_bbl_photofacial"],
                "pigment_variation": [
                    "sciton_bbl_photofacial",
                    "sciton_moxi_laser",
                ],
                "surface_texture": [
                    "sciton_moxi_laser",
                    "sciton_bbl_photofacial",
                ],
                "scar_like_texture": ["sciton_moxi_laser"],
                "superficial_vessels": ["sciton_bbl_photofacial"],
            },
            "back": {
                "visible_redness": ["sciton_bbl_photofacial"],
                "pigment_variation": [
                    "sciton_bbl_photofacial",
                    "sciton_moxi_laser",
                ],
                "surface_texture": [
                    "sciton_moxi_laser",
                    "sciton_bbl_photofacial",
                ],
                "pore_visibility": ["sciton_moxi_laser"],
                "scar_like_texture": ["sciton_moxi_laser"],
                "superficial_vessels": ["sciton_bbl_photofacial"],
            },
            "legs": {
                "visible_redness": ["sciton_bbl_photofacial"],
                "pigment_variation": [
                    "sciton_bbl_photofacial",
                    "sciton_moxi_laser",
                ],
                "surface_texture": [
                    "sciton_moxi_laser",
                    "sciton_bbl_photofacial",
                ],
                "laxity_appearance": ["sculptra"],
                "scar_like_texture": ["sciton_moxi_laser"],
                "superficial_vessels": ["sciton_bbl_photofacial"],
            },
        }
        body_products = {
            "neck": {},
            "chest": {
                "visible_lines": [
                    "alastin_restorative_skin_complex",
                    "skinbetter_alpharet_body",
                ],
                "laxity_appearance": [
                    "alastin_restorative_skin_complex",
                    "skinbetter_alpharet_body",
                ],
            },
            "hands": {
                "visible_lines": [
                    "alastin_restorative_skin_complex",
                    "skinbetter_alpharet_body",
                ],
                "laxity_appearance": [
                    "alastin_restorative_skin_complex",
                    "skinbetter_alpharet_body",
                ],
            },
            "back": {},
            "legs": {},
        }

        for body_area in ("face", "neck", "chest", "hands", "back", "legs"):
            for observation_id in analysis_engine.BODY_AREA_OBSERVATIONS[body_area]:
                with self.subTest(
                    body_area=body_area, observation_id=observation_id
                ):
                    recommendations = (
                        recommendation_catalog.build_appearance_recommendations(
                            [observation_id],
                            body_area,
                            analysis_engine.OBSERVATION_LABELS,
                        )
                    )
                    expected_services = (
                        face_services[observation_id]
                        if body_area == "face"
                        else body_services[body_area].get(observation_id, [])
                    )
                    expected_products = (
                        face_products[observation_id]
                        if body_area == "face"
                        else body_products[body_area].get(observation_id, [])
                    )
                    self.assertEqual(
                        [item["id"] for item in recommendations["services"]],
                        expected_services,
                    )
                    self.assertEqual(
                        [item["id"] for item in recommendations["products"]],
                        expected_products,
                    )

    def test_round_robin_mapping_is_deterministic_complete_and_deduplicated(self) -> None:
        priorities = ["visible_lines", "surface_texture"]
        first = recommendation_catalog.build_appearance_recommendations(
            priorities, "face", analysis_engine.OBSERVATION_LABELS
        )
        second = recommendation_catalog.build_appearance_recommendations(
            priorities, "face", analysis_engine.OBSERVATION_LABELS
        )

        self.assertEqual(first, second)
        self.assertGreater(len(first["services"]), 3)
        self.assertGreater(len(first["products"]), 3)
        self.assertLessEqual(
            len(first["services"]), recommendation_catalog.MAX_SERVICE_RECOMMENDATIONS
        )
        self.assertLessEqual(
            len(first["products"]), recommendation_catalog.MAX_PRODUCT_RECOMMENDATIONS
        )
        self.assertEqual(
            [item["id"] for item in first["services"]],
            [
                "botox",
                "sciton_moxi_laser",
                "dysport",
                "sciton_bbl_photofacial",
                "xeomin",
                "hydrafacial_customized",
                "microneedling",
                "hydrafacial_elite",
                "microneedling_prf",
                "saltfacial",
                "rf_microneedling",
                "skinvive",
                "signature_facial",
                "sciton_halo_laser",
                "anti_aging_facial",
                "chemical_peels",
                "sculptra",
            ],
        )
        halo = next(item for item in first["services"] if item["id"] == "sciton_halo_laser")
        self.assertEqual(halo["matchedObservationIds"], priorities)
        self.assertEqual(
            [item["id"] for item in first["products"]],
            [
                "skinbetter_alpharet",
                "zo_complexion_renewal_pads",
                "zo_wrinkle_texture_repair",
                "skinbetter_peel_pads",
                "colorscience_face_shield",
            ],
        )
        self.assertEqual(first["products"][-1]["relationship"], "Daily protection")
        self.assertEqual(
            len({item["id"] for item in first["services"]}),
            len(first["services"]),
        )
        self.assertEqual(
            len({item["id"] for item in first["products"]}),
            len(first["products"]),
        )


class PublicRecommendationContractTests(unittest.TestCase):
    def test_model_contract_remains_appearance_only(self) -> None:
        for schema in (
            analysis_engine.MODEL_OUTPUT_SCHEMA,
            analysis_engine.model_output_schema("face"),
            analysis_engine.gemini_output_schema("face"),
        ):
            properties = schema["properties"]
            self.assertNotIn("appearanceRecommendations", properties)
            self.assertNotIn("discussionTopics", properties)

        injected = {
            "status": "complete",
            "quality": {"overall": "suitable", "issues": [], "guidance": []},
            "observations": [],
            "strengths": [],
            "priorities": [],
            "medicalReview": {"suggested": False, "reason": "none"},
            "appearanceRecommendations": {"services": [], "products": []},
        }
        with self.assertRaises(analysis_engine.SchemaValidationError):
            analysis_engine.validate_model_output(injected, ["single"], body_area="face")

    def test_combined_public_priorities_preserve_every_supported_match(self) -> None:
        for observation_id in (
            "blemish_like_spots",
            "superficial_vessels",
            "visible_flaking",
        ):
            with self.subTest(observation_id=observation_id):
                result = _public_result(["visible_lines", observation_id])
                expected = recommendation_catalog.build_appearance_recommendations(
                    ["visible_lines", observation_id],
                    "face",
                    analysis_engine.OBSERVATION_LABELS,
                )
                self.assertEqual(
                    result["appearanceRecommendations"],
                    expected,
                )
                self.assertEqual(
                    [item["id"] for item in result["discussionTopics"]],
                    [item["id"] for item in expected["services"][:2]],
                )

    def test_visible_findings_outside_priority_shortlist_still_drive_matches(self) -> None:
        raw = {
            "status": "complete",
            "quality": {"overall": "suitable", "issues": [], "guidance": []},
            "observations": [
                {
                    "id": observation_id,
                    "label": analysis_engine.OBSERVATION_LABELS[observation_id],
                    "level": "visible",
                    "description": "This feature is visible in the submitted view.",
                    "angles": ["single"],
                }
                for observation_id in (
                    "visible_lines",
                    "visible_redness",
                    "pigment_variation",
                )
            ],
            "strengths": [],
            "priorities": ["visible_lines", "visible_redness"],
            "medicalReview": {"suggested": False, "reason": "none"},
        }
        validated = analysis_engine.validate_model_output(raw, ["single"], "face")
        result = analysis_engine.build_final_result(
            validated,
            provider="gemini",
            selected_model="gemini-3.5-flash",
            image_count=1,
            body_area="face",
        )
        service_ids = {
            item["id"] for item in result["appearanceRecommendations"]["services"]
        }
        product_ids = {
            item["id"] for item in result["appearanceRecommendations"]["products"]
        }
        self.assertIn("chemical_peels", service_ids)
        self.assertIn("skinbetter_even_tone", product_ids)
        self.assertIn("hydrinity_vivid_serum", product_ids)

    def test_maximal_visible_face_profile_returns_every_eligible_match(self) -> None:
        observation_ids = analysis_engine.BODY_AREA_OBSERVATIONS["face"]
        raw = {
            "status": "complete",
            "quality": {"overall": "suitable", "issues": [], "guidance": []},
            "observations": [
                {
                    "id": observation_id,
                    "label": analysis_engine.OBSERVATION_LABELS[observation_id],
                    "level": "visible",
                    "description": "This feature is visible in the submitted view.",
                    "angles": ["single"],
                }
                for observation_id in observation_ids
            ],
            "strengths": [],
            "priorities": list(observation_ids[:2]),
            "medicalReview": {"suggested": False, "reason": "none"},
        }
        validated = analysis_engine.validate_model_output(raw, ["single"], "face")
        result = analysis_engine.build_final_result(
            validated,
            provider="gemini",
            selected_model="gemini-3.5-flash",
            image_count=1,
            body_area="face",
        )
        recommendations = result["appearanceRecommendations"]
        self.assertEqual(len(recommendations["services"]), 19)
        self.assertEqual(len(recommendations["products"]), 15)
        self.assertEqual(
            len({item["id"] for item in recommendations["services"]}), 19
        )
        self.assertEqual(
            len({item["id"] for item in recommendations["products"]}), 15
        )

    def test_disclaimer_requires_in_person_evaluation_for_concerning_lesions(self) -> None:
        self.assertIn(
            "An in-person evaluation is required before treatment",
            analysis_engine.DISCLAIMER,
        )
        self.assertIn(
            "any concerning lesion should be evaluated",
            analysis_engine.DISCLAIMER,
        )

    def test_public_shape_versions_catalog_bounds_and_compatibility_alias(self) -> None:
        result = _public_result(["visible_lines", "pigment_variation"])
        recommendations = result["appearanceRecommendations"]

        self.assertEqual(set(recommendations), {"services", "products"})
        self.assertGreater(len(recommendations["services"]), 3)
        self.assertGreater(len(recommendations["products"]), 3)
        self.assertLessEqual(
            len(recommendations["services"]),
            recommendation_catalog.MAX_SERVICE_RECOMMENDATIONS,
        )
        self.assertLessEqual(
            len(recommendations["products"]),
            recommendation_catalog.MAX_PRODUCT_RECOMMENDATIONS,
        )
        self.assertEqual(
            set(recommendations["services"][0]),
            {
                "id",
                "name",
                "category",
                "matchedObservationIds",
                "why",
                "learnMoreUrl",
            },
        )
        self.assertEqual(
            set(recommendations["products"][0]),
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
        )
        self.assertEqual(
            result["topicMappingVersion"], recommendation_catalog.CATALOG_VERSION
        )
        self.assertEqual(
            analysis_engine.TOPIC_MAPPING_VERSION,
            recommendation_catalog.CATALOG_VERSION,
        )
        self.assertEqual(
            [(item["id"], item["name"]) for item in result["discussionTopics"]],
            [
                (item["id"], item["name"])
                for item in recommendations["services"][:2]
            ],
        )

    def test_public_validation_rejects_nested_injection_and_tampering(self) -> None:
        valid = _public_result(["visible_lines", "visible_redness"])
        mutations = []

        extra_field = deepcopy(valid)
        extra_field["appearanceRecommendations"]["services"][0]["debug"] = "injected"
        mutations.append(extra_field)

        changed_product = deepcopy(valid)
        changed_product["appearanceRecommendations"]["products"][0][
            "name"
        ] = "Unapproved product"
        mutations.append(changed_product)

        changed_explanation = deepcopy(valid)
        changed_explanation["appearanceRecommendations"]["services"][0][
            "why"
        ] = "The model selected this service."
        mutations.append(changed_explanation)

        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaises(analysis_engine.SchemaValidationError):
                    analysis_engine.validate_final_result(mutation)

    def test_retake_and_medical_review_suppress_all_recommendations(self) -> None:
        local_retake = analysis_engine.build_local_retake(
            image_count=1,
            issue="blur",
            guidance="hold_camera_steady",
            message="Retake the image in steady, even light.",
            body_area="face",
        )
        model_retake = _public_result([], body_area="face", status="retake")
        medical = _public_result(
            ["visible_redness"], body_area="face", status="medical_review"
        )

        for result in (local_retake, model_retake, medical):
            with self.subTest(status=result["status"]):
                self.assertEqual(
                    result["appearanceRecommendations"],
                    {"services": [], "products": []},
                )
                self.assertEqual(result["discussionTopics"], [])

    def test_non_face_pigment_result_contains_no_unsupported_products(self) -> None:
        for body_area in ("neck", "chest", "hands", "back", "legs"):
            with self.subTest(body_area=body_area):
                result = _public_result(["pigment_variation"], body_area=body_area)
                self.assertEqual(result["appearanceRecommendations"]["products"], [])

    def test_supported_body_products_reach_the_public_result(self) -> None:
        for body_area in ("chest", "hands"):
            with self.subTest(body_area=body_area):
                result = _public_result(["visible_lines"], body_area=body_area)
                self.assertEqual(
                    [
                        item["id"]
                        for item in result["appearanceRecommendations"]["products"]
                    ],
                    [
                        "alastin_restorative_skin_complex",
                        "skinbetter_alpharet_body",
                    ],
                )


if __name__ == "__main__":
    unittest.main()
