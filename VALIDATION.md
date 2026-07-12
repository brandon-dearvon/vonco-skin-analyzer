# Visible-Surface Preview Validation Plan

## Current status

This application is **not clinically validated** and must not be described as diagnostic, VISIA-equivalent, or able to rule out skin disease. The production claim is limited to an AI-generated preview of visible surface features in consumer photographs. Service and product matches are deterministic educational starting points from Von & Co's current guides, not determinations of individual suitability.

The production build must remain controlled, versioned, and auditable. Model output can vary even when the photographs do not, so the surrounding application must use deterministic validation, mappings, and failure behavior:

- no random score adjustment, hidden score bands, overall score, or estimated skin age;
- no required finding or treatment quota;
- a fixed output schema with strict allowlists;
- explicit retake and unable-to-assess outcomes;
- all three guided photographs analyzed together;
- deployed model, prompt hash, response schema, and recommendation-catalog versions recorded with every result;
- a versioned server-owned recommendation catalog that the model cannot see or alter;
- no silent sample or demo result when an analysis fails.

## Intended use

The tool may describe only features visible in ordinary photographs, such as the appearance of lines, redness, pigment variation, surface texture, pore visibility, laxity, blemish-like spots, scar-like texture, superficial vessels, or flaking.

The tool must not:

- diagnose or rule out acne, rosacea, melasma, infection, skin cancer, or any other disease;
- infer UV or subsurface findings, bacterial load, hydration, ethnicity, sex, or biological age;
- determine treatment eligibility, contraindications, or medical suitability;
- reassure a guest that no concerning condition is present.

If the model suggests that professional medical review may be appropriate, the application suppresses aesthetic discussion topics and displays a non-diagnostic referral message. A lack of such a flag never implies medical clearance.

Service and product matches must be generated only after the appearance response
passes validation. The mapper must abstain when the photo vocabulary does not
support the corresponding concern—for example, blemish-like spots cannot be
converted into acne and visible flaking cannot be converted into dehydration.
If one of these ambiguous appearances is a priority, every automatic match is
held so another cosmetic feature cannot route around the abstention rule.

## Validation design

### 1. Freeze the candidate system

Before enrolling the validation set, freeze and record:

- model provider and exact model identifier;
- prompt version and cryptographic hash;
- response schema version;
- image normalization and capture-quality thresholds;
- server-side observation and recommendation mappings;
- exact service and product catalog version, source date, ordering, area gates,
  caps, and abstention rules;
- frontend wording and abstention behavior.

Any material change creates a new candidate version and requires revalidation.

### 2. Build a consented, representative image set

Collect front, left, and right photographs under the same instructions used by consumers. Include repeat captures from the same guest and deliberately varied lighting and supported phone models. Collect self-reported demographic and skin-tone information only with appropriate consent and only for prespecified subgroup analysis. Do not ask the AI to infer these attributes.

The validation set must include:

- the supported age range and all supported body areas;
- a broad range of visible-feature severity, including genuinely clear skin;
- varied skin tones, devices, lighting, makeup, facial hair, and image quality;
- invalid inputs, screenshots, filters, occlusion, non-skin images, minors, and prompt-injection text inside images;
- examples that should trigger retake, unable-to-assess, and medical-review messaging.

The final sample size and subgroup minimums must be set by a prespecified power calculation with a biostatistician. They must not be chosen after results are reviewed.

### 3. Establish blinded reference labels

At least two qualified human raters independently label each photograph set using the same categorical observation definitions available to the model. Raters must be blinded to model output. Disagreements are adjudicated under a written protocol before the locked test set is scored.

VISIA may be recorded as a separate in-studio measurement, but selfie outputs must not be benchmarked against UV, porphyrin, or other subsurface VISIA features that an ordinary photograph cannot capture.

### 4. Prespecify outcomes

Primary outcomes:

- agreement between model and adjudicated human labels for each visible feature;
- sensitivity and specificity of the retake decision for technically unusable images;
- abstention rate and the proportion of inappropriate confident outputs;
- test-retest agreement for the same unchanged images.

Secondary outcomes:

- agreement across repeat consumer captures of the same guest;
- performance by prespecified skin-tone, age, device, and lighting subgroups;
- false reassurance and inappropriate aesthetic-topic rates in medical-review cases;
- schema violations, unsafe free text, prompt-injection success, latency, and provider failure rate;
- consumer comprehension of the limitations, privacy language, and next step;
- recommendation-mapping precision against a provider-adjudicated reference,
  including unsupported-match rate, correct abstention, correct area gating, and
  catalog freshness.

Report confidence intervals for all performance estimates. Define acceptance thresholds and failure rules before the locked test set is opened. A medical director and biostatistician must approve those thresholds.

### 5. Validate the recommendation layer separately

For every supported appearance ID and body area, a qualified Von & Co provider
must approve a locked expected set or an explicit abstention. Automated contract
tests must verify exact IDs, stable ordering, deduplication, caps, source version,
area gates, and suppression for retake and medical-review states. A model response
must never be able to inject a service, product, URL, relationship label, or stock
claim.

Catalog validation is not clinical-outcome validation. Before any match is shown
as suitable for an individual, Von & Co would need a separate prospective protocol
covering contraindications, skin type, medications, pregnancy, current routine,
goals, provider judgment, and outcomes. The current application therefore routes
all matches to a consultation.

### 6. Run in stages

1. Offline technical verification with synthetic and consented test images.
2. Blinded retrospective validation on a locked set.
3. Prospective silent-mode study in which the result is not used for care.
4. Limited live pilot with provider review and documented human-factors feedback.
5. Broader release only after the prespecified safety, agreement, subgroup, and comprehension criteria pass.

## Release gates

The application cannot be called validated until all of the following exist:

- approved protocol and consent process;
- locked external test set that was not used for prompt or threshold tuning;
- blinded human reference labels and adjudication record;
- prespecified analysis plan, thresholds, and subgroup reporting;
- versioned results with confidence intervals;
- documented failure analysis and corrective actions;
- medical, privacy, security, and accessibility sign-off.

## Evidence framework

This plan follows the staged, human-factors, safety, reproducibility, and subgroup principles in [DECIDE-AI](https://www.nature.com/articles/s41591-022-01772-9) and the diagnostic-accuracy reporting framework in [STARD-AI](https://www.nature.com/articles/s41591-025-03953-8). Prospective smartphone dermatology studies show that capture success and performance can vary in real-world use and across phone models, reinforcing the need for independent prospective validation rather than model claims alone ([PMID 41701029](https://pubmed.ncbi.nlm.nih.gov/41701029/), [PMID 35124665](https://pubmed.ncbi.nlm.nih.gov/35124665/)).
