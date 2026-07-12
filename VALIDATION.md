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

The model is not allowed to decide whether a mole, spot, or lesion is medically
concerning. The medical-review state is limited to directly visible open, broken,
or actively bleeding-like skin that makes a cosmetic preview inappropriate; it
suppresses aesthetic matches and displays a non-diagnostic in-person referral.
Unclear photos route to retake. Every other result retains the same compact
in-person-evaluation disclaimer, and a completed preview never implies medical
clearance.

Service and product matches must be generated only after the appearance response
passes validation. A match must be tied to the exact visible-feature ID and body
area approved in the frozen catalog. Visible or prominent priorities lead; only
when none exist may up to two subtle findings drive maintenance-labeled matches.
Not-observed and unable-to-assess findings must never drive a match. The mapper
cannot change the meaning of that feature—for example, blemish-like spots cannot
be converted into acne and visible flaking cannot be converted into a hydration
measurement. A catalog-approved appearance match is an educational option, not a
diagnosis, suitability decision, or expected outcome. Every result retains the
small-print requirement for an in-person evaluation before treatment and medical
evaluation of any concerning lesion.

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
- examples that should trigger retake, unable-to-assess, and the narrow
  open-or-broken-skin medical-review messaging.

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

### 6. Validate the consumer presentation separately

Automated and browser-level tests must verify that the six-stage waiting tracker
does not represent estimated client-side stages as completed backend work. Steps
remain numbered while the request is pending; the interface may advance the active
stage and progress percentage with elapsed time, but it marks all stages complete
only after an accepted API response arrives.

For a complete response, tests must verify that the interface shows the quick-read
summary, visible strengths and priorities, every returned ordinal observation,
the evidence views supplied for each observation, ranked server-owned matches, and
the consultation action. The product display cap is three. A missing permitted
category may be represented only as `unable_to_assess`; the application must not
invent a positive or negative finding, score, mask, recommendation, or supporting
view for that placeholder, and it must preserve the compact in-person-evaluation
disclaimer. Responsive QA must also
verify that mobile recommendation rows remain horizontally usable, the complete
profile disclosure retains every category, and the sticky main-site navigation
does not cover the result heading.

### 7. Run in stages

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
