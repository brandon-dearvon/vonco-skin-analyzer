#!/usr/bin/env python3
"""Score locked visible-surface preview results without handling photographs or PII."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


LEVELS = ("not_observed", "subtle", "visible", "prominent")
LEVEL_INDEX = {name: index for index, name in enumerate(LEVELS)}
VALID_STATUSES = {"complete", "retake", "medical_review"}
CANDIDATE_FIELDS = (
    "analysisVersion",
    "schemaVersion",
    "topicMappingVersion",
    "promptHash",
)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, float | int | None]:
    if total == 0:
        return {"numerator": successes, "denominator": total, "estimate": None, "low": None, "high": None}
    estimate = successes / total
    denominator = 1 + (z * z / total)
    center = (estimate + (z * z / (2 * total))) / denominator
    margin = (
        z
        * math.sqrt((estimate * (1 - estimate) / total) + (z * z / (4 * total * total)))
        / denominator
    )
    return {
        "numerator": successes,
        "denominator": total,
        "estimate": estimate,
        "low": max(0.0, center - margin),
        "high": min(1.0, center + margin),
    }


def weighted_kappa(pairs: Iterable[tuple[str, str]]) -> float | None:
    usable = [(LEVEL_INDEX[a], LEVEL_INDEX[b]) for a, b in pairs if a in LEVEL_INDEX and b in LEVEL_INDEX]
    total = len(usable)
    if total == 0:
        return None

    size = len(LEVELS)
    observed = [[0 for _ in range(size)] for _ in range(size)]
    reference_counts = [0 for _ in range(size)]
    prediction_counts = [0 for _ in range(size)]
    for reference, prediction in usable:
        observed[reference][prediction] += 1
        reference_counts[reference] += 1
        prediction_counts[prediction] += 1

    max_distance = (size - 1) ** 2
    observed_disagreement = 0.0
    expected_disagreement = 0.0
    for i in range(size):
        for j in range(size):
            weight = ((i - j) ** 2) / max_distance
            observed_disagreement += weight * observed[i][j] / total
            expected_disagreement += weight * (reference_counts[i] * prediction_counts[j]) / (total * total)

    if expected_disagreement == 0:
        return None
    return 1.0 - (observed_disagreement / expected_disagreement)


def candidate_metadata(candidate: Any) -> dict[str, Any]:
    if not isinstance(candidate, dict):
        raise ValueError("Validation row is missing locked candidate metadata")
    metadata: dict[str, Any] = {}
    for field in CANDIDATE_FIELDS:
        value = candidate.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Candidate is missing locked field {field}")
        metadata[field] = value.strip()

    model = candidate.get("model")
    if not isinstance(model, dict):
        raise ValueError("Candidate is missing locked model metadata")
    model_metadata: dict[str, str] = {}
    for field in ("provider", "name", "promptVersion"):
        value = model.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Candidate model is missing {field}")
        model_metadata[field] = value.strip()
    metadata["model"] = model_metadata
    return metadata


def normalize_prediction_observations(prediction: dict[str, Any]) -> dict[str, str]:
    observations = prediction.get("observations", [])
    if isinstance(observations, dict):
        return {str(key): str(value) for key, value in observations.items()}
    normalized: dict[str, str] = {}
    if isinstance(observations, list):
        for item in observations:
            if isinstance(item, dict) and isinstance(item.get("id"), str) and isinstance(item.get("level"), str):
                normalized[item["id"]] = item["level"]
    return normalized


def prediction_medical_review(prediction: dict[str, Any]) -> bool:
    value = prediction.get("medicalReview", False)
    if isinstance(value, dict):
        return bool(value.get("suggested", False))
    return bool(value)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} is not a JSON object")
            if not row.get("caseId"):
                raise ValueError(f"Line {line_number} is missing caseId")
            rows.append(row)
    if not rows:
        raise ValueError("No validation rows were found")
    return rows


def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_total = 0
    status_correct = 0
    retake_reference = 0
    retake_predicted = 0
    retake_true_positive = 0
    non_retake_reference = 0
    retake_true_negative = 0
    medical_reference = 0
    medical_false_reassurance = 0

    feature_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    feature_reference_count: Counter[str] = Counter()
    feature_abstentions: Counter[str] = Counter()
    feature_confusion: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    subgroup_counts: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    repeat_predictions: dict[str, list[str]] = defaultdict(list)
    seen_case_ids: set[str] = set()
    locked_candidate: dict[str, Any] | None = None

    for row in rows:
        case_id = row.get("caseId")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("Each validation row must have a non-empty caseId")
        case_id = case_id.strip()
        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate caseId: {case_id}")
        seen_case_ids.add(case_id)

        reference = row.get("reference", {})
        prediction = row.get("prediction", {})
        if not isinstance(reference, dict) or not isinstance(prediction, dict):
            raise ValueError(f"Case {row.get('caseId')} must contain reference and prediction objects")
        current_candidate = candidate_metadata(row.get("candidate"))
        if locked_candidate is None:
            locked_candidate = current_candidate
        elif current_candidate != locked_candidate:
            raise ValueError("Validation rows mix more than one candidate system version")

        reference_status = reference.get("status")
        prediction_status = prediction.get("status")
        if reference_status not in VALID_STATUSES or prediction_status not in VALID_STATUSES:
            raise ValueError(f"Case {row.get('caseId')} has an invalid status")
        status_total += 1
        status_correct += int(reference_status == prediction_status)

        if reference_status == "retake":
            retake_reference += 1
            retake_true_positive += int(prediction_status == "retake")
        else:
            non_retake_reference += 1
            retake_true_negative += int(prediction_status != "retake")
        retake_predicted += int(prediction_status == "retake")

        reference_medical = bool(reference.get("medicalReviewSuggested", reference_status == "medical_review"))
        predicted_medical = prediction_medical_review(prediction) or prediction_status == "medical_review"
        if reference_medical:
            medical_reference += 1
            medical_false_reassurance += int(not predicted_medical)

        reference_observations = reference.get("observations", {})
        if not isinstance(reference_observations, dict):
            raise ValueError(f"Case {row.get('caseId')} reference observations must be an object")
        predicted_observations = normalize_prediction_observations(prediction)

        case_exact = 0
        case_total = 0
        for feature_id, reference_level in reference_observations.items():
            if reference_level not in LEVEL_INDEX:
                raise ValueError(f"Case {row.get('caseId')} has invalid reference level for {feature_id}")
            feature_reference_count[feature_id] += 1
            predicted_level = predicted_observations.get(feature_id, "unable_to_assess")
            if predicted_level == "unable_to_assess":
                feature_abstentions[feature_id] += 1
                continue
            if predicted_level not in LEVEL_INDEX:
                raise ValueError(f"Case {row.get('caseId')} has invalid prediction level for {feature_id}")
            feature_pairs[feature_id].append((reference_level, predicted_level))
            feature_confusion[feature_id][(reference_level, predicted_level)] += 1
            case_total += 1
            case_exact += int(reference_level == predicted_level)

        for key, value in (row.get("subgroups") or {}).items():
            if case_total:
                subgroup_counts[str(key)][str(value)][0] += case_exact
                subgroup_counts[str(key)][str(value)][1] += case_total

        repeat_group = row.get("repeatGroup")
        if repeat_group:
            fingerprint = json.dumps(
                {
                    "status": prediction_status,
                    "medicalReview": predicted_medical,
                    "observations": predicted_observations,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            repeat_predictions[str(repeat_group)].append(fingerprint)

    feature_reports: dict[str, Any] = {}
    for feature_id in sorted(feature_reference_count):
        pairs = feature_pairs[feature_id]
        exact = sum(reference == prediction for reference, prediction in pairs)
        within_one = sum(abs(LEVEL_INDEX[reference] - LEVEL_INDEX[prediction]) <= 1 for reference, prediction in pairs)
        confusion = {
            reference: {prediction: feature_confusion[feature_id][(reference, prediction)] for prediction in LEVELS}
            for reference in LEVELS
        }
        feature_reports[feature_id] = {
            "referenceCases": feature_reference_count[feature_id],
            "assessedCases": len(pairs),
            "abstention": wilson_interval(feature_abstentions[feature_id], feature_reference_count[feature_id]),
            "exactAgreement": wilson_interval(exact, len(pairs)),
            "withinOneCategory": wilson_interval(within_one, len(pairs)),
            "quadraticWeightedKappa": weighted_kappa(pairs),
            "confusionMatrix": confusion,
        }

    repeat_group_total = 0
    repeat_group_exact = 0
    for fingerprints in repeat_predictions.values():
        if len(fingerprints) < 2:
            continue
        repeat_group_total += 1
        repeat_group_exact += int(len(set(fingerprints)) == 1)

    subgroup_report: dict[str, Any] = {}
    for subgroup_name, values in sorted(subgroup_counts.items()):
        subgroup_report[subgroup_name] = {
            value: wilson_interval(counts[0], counts[1]) for value, counts in sorted(values.items())
        }

    return {
        "candidate": locked_candidate,
        "caseCount": len(rows),
        "statusAccuracy": wilson_interval(status_correct, status_total),
        "retake": {
            "referencePositive": retake_reference,
            "predictedPositive": retake_predicted,
            "sensitivity": wilson_interval(retake_true_positive, retake_reference),
            "specificity": wilson_interval(retake_true_negative, non_retake_reference),
        },
        "medicalReview": {
            "referencePositive": medical_reference,
            "falseReassurance": wilson_interval(medical_false_reassurance, medical_reference),
        },
        "features": feature_reports,
        "repeatability": wilson_interval(repeat_group_exact, repeat_group_total),
        "subgroups": subgroup_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Locked JSONL reference and prediction file")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()

    report = score(load_rows(args.input))
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
