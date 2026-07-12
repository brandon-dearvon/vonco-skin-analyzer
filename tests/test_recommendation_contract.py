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

    def test_face_mapping_matches_current_provider_and_product_guides(self) -> None:
        expected = {
            "visible_lines": (
                ["microneedling", "rf_microneedling"],
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
                ["sciton_bbl_photofacial", "sciton_halo_laser", "chemical_peels"],
                [
                    "skinbetter_even_tone",
                    "isdin_melaclear_advanced",
                    "colorscience_face_shield",
                ],
            ),
            "surface_texture": (
                ["sciton_moxi_laser", "hydrafacial_customized", "saltfacial"],
                [
                    "zo_complexion_renewal_pads",
                    "skinbetter_peel_pads",
                    "colorscience_face_shield",
                ],
            ),
            "pore_visibility": (
                ["hydrafacial_clarifying", "deep_pore_facial", "microneedling_prf"],
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
                ["hydrafacial_clarifying", "deep_pore_facial", "saltfacial"],
                ["zo_complexion_renewal_pads", "colorscience_face_shield"],
            ),
            "scar_like_texture": (
                ["microneedling_prf", "rf_microneedling", "chemical_peels"],
                ["colorscience_face_shield"],
            ),
            "superficial_vessels": (
                ["sciton_bbl_photofacial"],
                ["colorscience_face_shield"],
            ),
            "visible_flaking": (
                ["hydrafacial_customized"],
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
                self.assertLessEqual(len(products), 3)
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

    def test_combined_priorities_preserve_each_supported_shortlist(self) -> None:
        cases = {
            "blemish_like_spots": (
                ["microneedling", "hydrafacial_clarifying", "rf_microneedling"],
                [
                    "skinbetter_alpharet",
                    "zo_complexion_renewal_pads",
                    "colorscience_face_shield",
                ],
            ),
            "superficial_vessels": (
                ["microneedling", "sciton_bbl_photofacial", "rf_microneedling"],
                [
                    "skinbetter_alpharet",
                    "zo_wrinkle_texture_repair",
                    "colorscience_face_shield",
                ],
            ),
            "visible_flaking": (
                ["microneedling", "hydrafacial_customized", "rf_microneedling"],
                [
                    "skinbetter_alpharet",
                    "hydrinity_renewing_ha_serum",
                    "colorscience_face_shield",
                ],
            ),
        }
        for observation_id, (expected_services, expected_products) in cases.items():
            with self.subTest(observation_id=observation_id):
                recommendations = (
                    recommendation_catalog.build_appearance_recommendations(
                        ["visible_lines", observation_id],
                        "face",
                        analysis_engine.OBSERVATION_LABELS,
                    )
                )
                self.assertEqual(
                    [item["id"] for item in recommendations["services"]],
                    expected_services,
                )
                self.assertEqual(
                    [item["id"] for item in recommendations["products"]],
                    expected_products,
                )

    def test_products_are_suppressed_outside_face(self) -> None:
        for body_area in ("neck_chest", "hands", "back", "legs"):
            with self.subTest(body_area=body_area):
                recommendations = recommendation_catalog.build_appearance_recommendations(
                    ["pigment_variation"],
                    body_area,
                    analysis_engine.OBSERVATION_LABELS,
                )
                self.assertEqual(recommendations["products"], [])

    def test_body_services_follow_explicit_area_and_appearance_gates(self) -> None:
        cases = (
            ("neck_chest", "pigment_variation", ["sciton_bbl_photofacial"]),
            ("legs", "surface_texture", ["sciton_moxi_laser"]),
            ("hands", "visible_lines", []),
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
                self.assertEqual(recommendations["products"], [])

    def test_every_appearance_and_body_area_pair_has_a_locked_expectation(self) -> None:
        face_services = {
            "visible_lines": ["microneedling", "rf_microneedling"],
            "visible_redness": ["sciton_bbl_photofacial"],
            "pigment_variation": [
                "sciton_bbl_photofacial",
                "sciton_halo_laser",
                "chemical_peels",
            ],
            "surface_texture": [
                "sciton_moxi_laser",
                "hydrafacial_customized",
                "saltfacial",
            ],
            "pore_visibility": [
                "hydrafacial_clarifying",
                "deep_pore_facial",
                "microneedling_prf",
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
            ],
            "scar_like_texture": [
                "microneedling_prf",
                "rf_microneedling",
                "chemical_peels",
            ],
            "superficial_vessels": ["sciton_bbl_photofacial"],
            "visible_flaking": ["hydrafacial_customized"],
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
            "neck_chest": {
                "visible_redness": ["sciton_bbl_photofacial"],
                "pigment_variation": ["sciton_bbl_photofacial"],
                "surface_texture": [
                    "sciton_moxi_laser",
                    "hydrafacial_customized",
                ],
                "superficial_vessels": ["sciton_bbl_photofacial"],
                "visible_flaking": ["hydrafacial_customized"],
            },
            "hands": {
                "visible_redness": ["sciton_bbl_photofacial"],
                "pigment_variation": ["sciton_bbl_photofacial"],
                "surface_texture": ["sciton_moxi_laser"],
                "superficial_vessels": ["sciton_bbl_photofacial"],
            },
            "back": {
                "visible_redness": ["sciton_bbl_photofacial"],
                "pigment_variation": ["sciton_bbl_photofacial"],
                "surface_texture": ["sciton_moxi_laser"],
                "superficial_vessels": ["sciton_bbl_photofacial"],
            },
            "legs": {
                "visible_redness": ["sciton_bbl_photofacial"],
                "pigment_variation": ["sciton_bbl_photofacial"],
                "surface_texture": ["sciton_moxi_laser"],
                "superficial_vessels": ["sciton_bbl_photofacial"],
            },
        }

        for body_area in ("face", "neck_chest", "hands", "back", "legs"):
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
                        else []
                    )
                    self.assertEqual(
                        [item["id"] for item in recommendations["services"]],
                        expected_services,
                    )
                    self.assertEqual(
                        [item["id"] for item in recommendations["products"]],
                        expected_products,
                    )

    def test_round_robin_mapping_is_deterministic_bounded_and_deduplicated(self) -> None:
        priorities = ["visible_redness", "pigment_variation"]
        first = recommendation_catalog.build_appearance_recommendations(
            priorities, "face", analysis_engine.OBSERVATION_LABELS
        )
        second = recommendation_catalog.build_appearance_recommendations(
            priorities, "face", analysis_engine.OBSERVATION_LABELS
        )

        self.assertEqual(first, second)
        self.assertLessEqual(len(first["services"]), 3)
        self.assertLessEqual(len(first["products"]), 3)
        self.assertEqual(
            [item["id"] for item in first["services"]],
            ["sciton_bbl_photofacial", "sciton_halo_laser", "chemical_peels"],
        )
        self.assertEqual(
            first["services"][0]["matchedObservationIds"], priorities
        )
        self.assertEqual(
            [item["id"] for item in first["products"]],
            [
                "avene_thermal_water",
                "skinbetter_even_tone",
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

    def test_combined_public_priorities_preserve_supported_shortlists(self) -> None:
        cases = {
            "blemish_like_spots": (
                ["microneedling", "hydrafacial_clarifying", "rf_microneedling"],
                [
                    "skinbetter_alpharet",
                    "zo_complexion_renewal_pads",
                    "colorscience_face_shield",
                ],
            ),
            "superficial_vessels": (
                ["microneedling", "sciton_bbl_photofacial", "rf_microneedling"],
                [
                    "skinbetter_alpharet",
                    "zo_wrinkle_texture_repair",
                    "colorscience_face_shield",
                ],
            ),
            "visible_flaking": (
                ["microneedling", "hydrafacial_customized", "rf_microneedling"],
                [
                    "skinbetter_alpharet",
                    "hydrinity_renewing_ha_serum",
                    "colorscience_face_shield",
                ],
            ),
        }
        for observation_id, (expected_services, expected_products) in cases.items():
            with self.subTest(observation_id=observation_id):
                result = _public_result(["visible_lines", observation_id])
                self.assertEqual(
                    [
                        item["id"]
                        for item in result["appearanceRecommendations"]["services"]
                    ],
                    expected_services,
                )
                self.assertEqual(
                    [
                        item["id"]
                        for item in result["appearanceRecommendations"]["products"]
                    ],
                    expected_products,
                )
                self.assertEqual(
                    [item["id"] for item in result["discussionTopics"]],
                    expected_services[:2],
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

    def test_public_shape_versions_caps_and_compatibility_alias(self) -> None:
        result = _public_result(["visible_lines", "pigment_variation"])
        recommendations = result["appearanceRecommendations"]

        self.assertEqual(set(recommendations), {"services", "products"})
        self.assertLessEqual(len(recommendations["services"]), 3)
        self.assertLessEqual(len(recommendations["products"]), 3)
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

    def test_non_face_public_result_contains_no_products(self) -> None:
        for body_area in ("neck_chest", "hands", "back", "legs"):
            with self.subTest(body_area=body_area):
                result = _public_result(["pigment_variation"], body_area=body_area)
                self.assertEqual(result["appearanceRecommendations"]["products"], [])


if __name__ == "__main__":
    unittest.main()
