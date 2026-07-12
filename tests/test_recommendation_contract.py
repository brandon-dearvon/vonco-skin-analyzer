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
            "reason": "visible_concern_outside_cosmetic_scope",
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
                self.assertEqual(source["reviewed"], "2026-07-11")
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
                ["skinbetter_alpharet", "zo_wrinkle_texture_repair"],
            ),
            "visible_redness": (
                ["sciton_bbl_photofacial"],
                ["avene_thermal_water", "skinbetter_alto_defense"],
            ),
            "pigment_variation": (
                ["sciton_bbl_photofacial", "sciton_halo_laser", "chemical_peels"],
                ["skinbetter_even_tone", "isdin_melaclear_advanced"],
            ),
            "surface_texture": (
                ["sciton_moxi_laser", "rf_microneedling", "sciton_halo_laser"],
                ["zo_complexion_renewal_pads", "skinbetter_peel_pads"],
            ),
            "pore_visibility": (
                ["microneedling_prf", "sciton_halo_laser"],
                [],
            ),
            "laxity_appearance": (
                ["rf_microneedling", "sciton_halo_laser"],
                ["alastin_restorative_skin_complex", "zo_growth_factor_serum"],
            ),
            "scar_like_texture": (
                ["microneedling_prf", "rf_microneedling", "chemical_peels"],
                [],
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

    def test_unsupported_photo_only_inferences_abstain(self) -> None:
        for observation_id in (
            "blemish_like_spots",
            "superficial_vessels",
            "visible_flaking",
        ):
            with self.subTest(observation_id=observation_id):
                recommendations = recommendation_catalog.build_appearance_recommendations(
                    [observation_id], "face", analysis_engine.OBSERVATION_LABELS
                )
                self.assertEqual(recommendations, {"services": [], "products": []})

    def test_ambiguous_priority_holds_otherwise_supported_matches(self) -> None:
        for hold_id in recommendation_catalog.RECOMMENDATION_HOLD_IDS:
            with self.subTest(hold_id=hold_id):
                recommendations = (
                    recommendation_catalog.build_appearance_recommendations(
                        ["visible_lines", hold_id],
                        "face",
                        analysis_engine.OBSERVATION_LABELS,
                    )
                )
                self.assertEqual(
                    recommendations, {"services": [], "products": []}
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
                "rf_microneedling",
                "sciton_halo_laser",
            ],
            "pore_visibility": ["microneedling_prf", "sciton_halo_laser"],
            "laxity_appearance": ["rf_microneedling", "sciton_halo_laser"],
            "blemish_like_spots": [],
            "scar_like_texture": [
                "microneedling_prf",
                "rf_microneedling",
                "chemical_peels",
            ],
            "superficial_vessels": [],
            "visible_flaking": [],
        }
        face_products = {
            "visible_lines": ["skinbetter_alpharet", "zo_wrinkle_texture_repair"],
            "visible_redness": ["avene_thermal_water", "skinbetter_alto_defense"],
            "pigment_variation": [
                "skinbetter_even_tone",
                "isdin_melaclear_advanced",
            ],
            "surface_texture": [
                "zo_complexion_renewal_pads",
                "skinbetter_peel_pads",
            ],
            "pore_visibility": [],
            "laxity_appearance": [
                "alastin_restorative_skin_complex",
                "zo_growth_factor_serum",
            ],
            "blemish_like_spots": [],
            "scar_like_texture": [],
            "superficial_vessels": [],
            "visible_flaking": [],
        }
        body_services = {
            "visible_redness": ["sciton_bbl_photofacial"],
            "pigment_variation": ["sciton_bbl_photofacial"],
            "surface_texture": ["sciton_moxi_laser"],
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
                        else body_services.get(observation_id, [])
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
        self.assertLessEqual(len(first["products"]), 2)
        self.assertEqual(
            [item["id"] for item in first["services"]],
            ["sciton_bbl_photofacial", "sciton_halo_laser", "chemical_peels"],
        )
        self.assertEqual(
            first["services"][0]["matchedObservationIds"], priorities
        )
        self.assertEqual(
            [item["id"] for item in first["products"]],
            ["avene_thermal_water", "skinbetter_even_tone"],
        )
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

    def test_ambiguous_public_priority_holds_the_complete_shortlist(self) -> None:
        for hold_id in recommendation_catalog.RECOMMENDATION_HOLD_IDS:
            with self.subTest(hold_id=hold_id):
                result = _public_result(["visible_lines", hold_id])
                self.assertEqual(
                    result["appearanceRecommendations"],
                    {"services": [], "products": []},
                )
                self.assertEqual(result["discussionTopics"], [])

    def test_public_shape_versions_caps_and_compatibility_alias(self) -> None:
        result = _public_result(["visible_lines", "pigment_variation"])
        recommendations = result["appearanceRecommendations"]

        self.assertEqual(set(recommendations), {"services", "products"})
        self.assertLessEqual(len(recommendations["services"]), 3)
        self.assertLessEqual(len(recommendations["products"]), 2)
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
