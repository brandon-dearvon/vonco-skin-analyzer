"""Source-locked expected-results cases for complete recommendation profiles.

These cases exercise the public, validated result builder rather than calling the
catalog matcher directly. Expected IDs are intentionally hard-coded from the
locked Von services and product guides so a mapping change cannot silently
rewrite its own test oracle.
"""

from __future__ import annotations

import unittest

import analysis_engine


SCENARIOS = (
    {
        "name": "face_lines_returns_every_supported_option",
        "body_area": "face",
        "visible": ("visible_lines",),
        "services": (
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
        ),
        "products": (
            "skinbetter_alpharet",
            "zo_wrinkle_texture_repair",
            "colorscience_face_shield",
        ),
        "forbidden_services": ("dermal_fillers", "kybella", "lip_filler"),
        "forbidden_products": ("alastin_restorative_eye_cream",),
    },
    {
        "name": "face_redness_and_vessels_dedupe_to_bbl",
        "body_area": "face",
        "visible": ("visible_redness", "superficial_vessels"),
        "services": ("sciton_bbl_photofacial",),
        "products": (
            "avene_thermal_water",
            "skinbetter_alto_defense",
            "colorscience_face_shield",
        ),
        "forbidden_services": ("sciton_moxi_laser", "chemical_peels"),
        "forbidden_products": ("skinbetter_alpharet",),
    },
    {
        "name": "face_pigment_and_texture_union_is_complete",
        "body_area": "face",
        "visible": ("pigment_variation", "surface_texture"),
        "services": (
            "sciton_bbl_photofacial",
            "sciton_moxi_laser",
            "sciton_halo_laser",
            "hydrafacial_customized",
            "microneedling",
            "hydrafacial_elite",
            "microneedling_prf",
            "saltfacial",
            "rf_microneedling",
            "skinvive",
            "chemical_peels",
            "signature_facial",
            "anti_aging_facial",
        ),
        "products": (
            "skinbetter_even_tone",
            "zo_complexion_renewal_pads",
            "isdin_melaclear_advanced",
            "skinbetter_peel_pads",
            "zo_vitamin_c_10",
            "hydrinity_vivid_serum",
            "colorscience_face_shield",
        ),
        "forbidden_services": ("botox", "dermal_fillers", "kybella"),
        "forbidden_products": ("avene_thermal_water",),
    },
    {
        "name": "face_pores_and_blemishes_include_all_direct_matches",
        "body_area": "face",
        "visible": ("pore_visibility", "blemish_like_spots"),
        "services": (
            "hydrafacial_clarifying",
            "hydrafacial_customized",
            "deep_pore_facial",
            "hydrafacial_elite",
            "saltfacial",
            "chemical_peels",
            "signature_facial",
            "microneedling",
            "microneedling_prf",
            "sciton_moxi_laser",
            "rf_microneedling",
            "sciton_halo_laser",
        ),
        "products": (
            "zo_complexion_renewal_pads",
            "colorscience_face_shield",
        ),
        "forbidden_services": ("botox", "sciton_bbl_photofacial"),
        "forbidden_products": ("skinbetter_alpharet",),
    },
    {
        "name": "face_flaking_stays_with_hydration_and_barrier_matches",
        "body_area": "face",
        "visible": ("visible_flaking",),
        "services": (
            "hydrafacial_customized",
            "hydrafacial_elite",
            "signature_facial",
            "anti_aging_facial",
        ),
        "products": (
            "hydrinity_renewing_ha_serum",
            "skinbetter_trio_moisture",
            "colorscience_face_shield",
        ),
        "forbidden_services": ("chemical_peels", "rf_microneedling"),
        "forbidden_products": (
            "skinbetter_alpharet",
            "zo_wrinkle_texture_repair",
        ),
    },
    {
        "name": "face_maximal_profile_returns_the_full_supported_union",
        "body_area": "face",
        "visible": tuple(analysis_engine.BODY_AREA_OBSERVATIONS["face"]),
        "services": (
            "botox",
            "sciton_bbl_photofacial",
            "sciton_moxi_laser",
            "hydrafacial_clarifying",
            "rf_microneedling",
            "microneedling",
            "hydrafacial_customized",
            "dysport",
            "sculptra",
            "deep_pore_facial",
            "microneedling_prf",
            "hydrafacial_elite",
            "xeomin",
            "sciton_halo_laser",
            "saltfacial",
            "signature_facial",
            "chemical_peels",
            "anti_aging_facial",
            "skinvive",
        ),
        "products": (
            "skinbetter_alpharet",
            "avene_thermal_water",
            "skinbetter_even_tone",
            "zo_complexion_renewal_pads",
            "alastin_restorative_skin_complex",
            "hydrinity_renewing_ha_serum",
            "zo_wrinkle_texture_repair",
            "skinbetter_alto_defense",
            "isdin_melaclear_advanced",
            "skinbetter_peel_pads",
            "zo_growth_factor_serum",
            "skinbetter_trio_moisture",
            "zo_vitamin_c_10",
            "hydrinity_vivid_serum",
            "colorscience_face_shield",
        ),
        "forbidden_services": ("dermal_fillers", "kybella", "lip_filler"),
        "forbidden_products": ("colorscience_lash_mascara",),
    },
    {
        "name": "neck_lines_and_pigment_apply_neck_area_gates",
        "body_area": "neck",
        "visible": ("visible_lines", "pigment_variation"),
        "services": (
            "botox",
            "sciton_bbl_photofacial",
            "dysport",
            "sciton_moxi_laser",
            "xeomin",
            "rf_microneedling",
            "hydrafacial_customized",
            "hydrafacial_elite",
        ),
        "products": (),
        "forbidden_services": ("sciton_halo_laser", "chemical_peels"),
        "forbidden_products": ("skinbetter_alpharet_body",),
    },
    {
        "name": "chest_lines_and_laxity_include_only_chest_matches",
        "body_area": "chest",
        "visible": ("visible_lines", "laxity_appearance"),
        "services": ("sculptra", "sciton_moxi_laser"),
        "products": (
            "alastin_restorative_skin_complex",
            "skinbetter_alpharet_body",
        ),
        "forbidden_services": ("botox", "rf_microneedling"),
        "forbidden_products": ("colorscience_face_shield",),
    },
    {
        "name": "hands_lines_and_pigment_include_hand_matches",
        "body_area": "hands",
        "visible": ("visible_lines", "pigment_variation"),
        "services": ("sciton_bbl_photofacial", "sciton_moxi_laser"),
        "products": (
            "alastin_restorative_skin_complex",
            "skinbetter_alpharet_body",
        ),
        "forbidden_services": ("sculptra", "chemical_peels"),
        "forbidden_products": ("colorscience_face_shield",),
    },
    {
        "name": "back_blemish_does_not_import_face_acne_matches",
        "body_area": "back",
        "visible": ("blemish_like_spots",),
        "services": (),
        "products": (),
        "forbidden_services": (
            "hydrafacial_clarifying",
            "deep_pore_facial",
            "chemical_peels",
        ),
        "forbidden_products": ("zo_complexion_renewal_pads",),
    },
    {
        "name": "back_pigment_and_pores_keep_only_body_device_matches",
        "body_area": "back",
        "visible": ("pigment_variation", "pore_visibility"),
        "services": ("sciton_bbl_photofacial", "sciton_moxi_laser"),
        "products": (),
        "forbidden_services": (
            "hydrafacial_clarifying",
            "rf_microneedling",
            "sciton_halo_laser",
        ),
        "forbidden_products": ("zo_complexion_renewal_pads",),
    },
    {
        "name": "legs_laxity_and_scars_keep_only_leg_matches",
        "body_area": "legs",
        "visible": ("laxity_appearance", "scar_like_texture"),
        "services": ("sculptra", "sciton_moxi_laser"),
        "products": (),
        "forbidden_services": ("rf_microneedling", "chemical_peels"),
        "forbidden_products": ("skinbetter_alpharet_body",),
    },
)


def _build_result(scenario: dict) -> dict:
    visible_ids = tuple(scenario["visible"])
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
            for observation_id in visible_ids
        ],
        "strengths": [],
        "priorities": list(visible_ids[:2]),
        "medicalReview": {"suggested": False, "reason": "none"},
    }
    validated = analysis_engine.validate_model_output(
        raw, ["single"], body_area=scenario["body_area"]
    )
    return analysis_engine.build_final_result(
        validated,
        provider="gemini",
        selected_model="gemini-3.5-flash",
        image_count=1,
        body_area=scenario["body_area"],
    )


class ExpectedRecommendationScenarioTests(unittest.TestCase):
    def test_source_locked_expected_profiles(self) -> None:
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario["name"]):
                result = _build_result(scenario)
                recommendations = result["appearanceRecommendations"]
                service_ids = [item["id"] for item in recommendations["services"]]
                product_ids = [item["id"] for item in recommendations["products"]]

                self.assertEqual(service_ids, list(scenario["services"]))
                self.assertEqual(product_ids, list(scenario["products"]))
                self.assertTrue(
                    set(scenario["forbidden_services"]).isdisjoint(service_ids)
                )
                self.assertTrue(
                    set(scenario["forbidden_products"]).isdisjoint(product_ids)
                )

                visible_ids = set(scenario["visible"])
                for item in recommendations["services"] + recommendations["products"]:
                    matched_ids = item["matchedObservationIds"]
                    self.assertTrue(matched_ids)
                    self.assertTrue(set(matched_ids).issubset(visible_ids))
                    why = item["why"]
                    self.assertTrue(why)
                    self.assertNotIn("Your photos show", why)
                    self.assertNotIn("appearance concern", why)
                    self.assertNotIn("current product guide includes", why)
                    self.assertNotIn("source-supported", why)
                    self.assertNotIn("—", why)
                    self.assertNotIn("–", why)

                for product in recommendations["products"]:
                    self.assertIn(
                        product["availability"],
                        {
                            "Available at Von & Co. Please confirm current studio availability.",
                            "A Von & Co provider can help decide whether this active belongs in your routine.",
                            "Use only within the pre- or post-procedure plan provided by Von & Co.",
                        },
                    )

                self.assertEqual(len(service_ids), len(set(service_ids)))
                self.assertEqual(len(product_ids), len(set(product_ids)))

    def test_every_observation_level_uses_consumer_friendly_copy(self) -> None:
        banned = ("did not stand out", "not apparent", "not visible", "especially noticeable")
        for observation_id in analysis_engine.OBSERVATION_LABELS:
            for level in analysis_engine.OBSERVATION_LEVELS:
                with self.subTest(observation_id=observation_id, level=level):
                    angles = [] if level == "unable_to_assess" else ["single"]
                    copy = analysis_engine._deterministic_description(
                        observation_id, level, angles
                    )
                    self.assertTrue(copy.endswith("."))
                    self.assertFalse(any(fragment in copy.lower() for fragment in banned))
                    self.assertNotIn("—", copy)
                    self.assertNotIn("–", copy)


if __name__ == "__main__":
    unittest.main()
