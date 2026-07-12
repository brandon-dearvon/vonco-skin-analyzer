import unittest

from score_results import score, weighted_kappa, wilson_interval


def candidate_fields():
    return {
        "analysisVersion": "visible-surface-v1.3.1",
        "schemaVersion": "visible-surface-response-schema-v1.1.0",
        "topicMappingVersion": "naples-appearance-recommendations-v2.1.0",
        "promptHash": "locked-prompt-hash",
        "model": {
            "provider": "gemini",
            "name": "gemini-3.5-flash",
            "promptVersion": "visible-surface-prompt-v1.1.0",
        },
    }


class ValidationScoringTests(unittest.TestCase):
    def test_perfect_weighted_kappa(self):
        pairs = [("not_observed", "not_observed"), ("subtle", "subtle"), ("prominent", "prominent")]
        self.assertEqual(weighted_kappa(pairs), 1.0)

    def test_empty_interval_is_explicit(self):
        interval = wilson_interval(0, 0)
        self.assertIsNone(interval["estimate"])
        self.assertEqual(interval["denominator"], 0)

    def test_constant_pairs_have_undefined_kappa(self):
        self.assertIsNone(weighted_kappa([("subtle", "subtle"), ("subtle", "subtle")]))

    def test_scores_abstention_and_false_reassurance(self):
        rows = [
            {
                "caseId": "case-1",
                "candidate": candidate_fields(),
                "repeatGroup": "repeat-a",
                "subgroups": {"device": "phone-a"},
                "reference": {
                    "status": "medical_review",
                    "medicalReviewSuggested": True,
                    "observations": {"visible_redness": "visible"},
                },
                "prediction": {
                    "status": "complete",
                    "medicalReview": {"suggested": False},
                    "observations": [{"id": "visible_redness", "level": "unable_to_assess"}],
                },
            },
            {
                "caseId": "case-2",
                "candidate": candidate_fields(),
                "repeatGroup": "repeat-a",
                "subgroups": {"device": "phone-a"},
                "reference": {
                    "status": "complete",
                    "medicalReviewSuggested": False,
                    "observations": {"visible_redness": "subtle"},
                },
                "prediction": {
                    "status": "complete",
                    "medicalReview": {"suggested": False},
                    "observations": [{"id": "visible_redness", "level": "subtle"}],
                },
            },
        ]
        report = score(rows)
        self.assertEqual(report["medicalReview"]["falseReassurance"]["numerator"], 1)
        self.assertEqual(report["features"]["visible_redness"]["abstention"]["numerator"], 1)
        self.assertEqual(report["features"]["visible_redness"]["exactAgreement"]["estimate"], 1.0)

    def test_duplicate_cases_and_mixed_candidates_are_rejected(self):
        base = {
            "caseId": "duplicate",
            "candidate": candidate_fields(),
            "reference": {"status": "complete", "observations": {}},
            "prediction": {"status": "complete", "observations": []},
        }
        with self.assertRaisesRegex(ValueError, "Duplicate caseId"):
            score([base, dict(base)])

        changed = {
            **base,
            "caseId": "second",
            "candidate": {
                **base["candidate"],
                "analysisVersion": "visible-surface-v2.0.0",
            },
        }
        with self.assertRaisesRegex(ValueError, "mix more than one"):
            score([base, changed])


if __name__ == "__main__":
    unittest.main()
