# Offline Validation Scoring

`score_results.py` scores a locked set of human reference labels against saved API results. It does not call an AI provider and does not accept photographs or identifying information.

Each JSONL row must contain:

```json
{
  "caseId": "opaque-study-id",
  "repeatGroup": "optional-repeat-group",
  "subgroups": {
    "skinTone": "self-reported-study-category",
    "device": "phone-model-category",
    "lighting": "standardized-or-consumer"
  },
  "candidate": {
    "analysisVersion": "visible-surface-v1.2.0",
    "schemaVersion": "visible-surface-response-schema-v1.1.0",
    "topicMappingVersion": "naples-appearance-recommendations-v2.0.1",
    "promptHash": "recorded-prompt-hash",
    "model": {
      "provider": "gemini",
      "name": "gemini-3.5-flash",
      "promptVersion": "visible-surface-prompt-v1.0.0"
    }
  },
  "reference": {
    "status": "complete",
    "medicalReviewSuggested": false,
    "observations": {
      "visible_lines": "subtle",
      "visible_redness": "not_observed"
    }
  },
  "prediction": {
    "status": "complete",
    "medicalReview": {"suggested": false},
    "observations": [
      {"id": "visible_lines", "level": "subtle"},
      {"id": "visible_redness", "level": "not_observed"}
    ]
  }
}
```

Run:

```bash
python3 validation/score_results.py locked-results.jsonl --output validation-report.json
```

The scorer rejects duplicate case IDs and any file that mixes model, prompt, schema, mapping, or analysis versions. The report includes the locked candidate metadata, status accuracy, retake performance, medical-review false-reassurance rate, feature coverage, abstention, exact agreement, within-one-category agreement, quadratic weighted kappa, confusion matrices, repeatability, and subgroup agreement. The script never decides whether a version passes. Acceptance thresholds must be written into the approved study protocol before the locked set is opened.
