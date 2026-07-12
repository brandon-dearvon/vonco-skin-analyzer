# Model Strategy for the Visible Skin Preview

## Production decision

Use one provider and one model: Google Gemini `gemini-3.5-flash`.

Gemini 3.5 Flash is Google's stable, generally available production model that supports image input, structured JSON output, and controllable thinking. The analyzer sets `thinking_level` to `high`, the model's maximum reasoning level, while retaining the locked visible-surface schema and the application's stricter post-response validation.

No other AI provider or model is used as a fallback. If Gemini is unavailable, times out, or returns a response that violates the schema or safety contract, the request fails closed and the guest receives the existing unavailable state. The application never substitutes a demo or fabricated result.

## Request configuration

- Model: `gemini-3.5-flash`
- Thinking level: `high`
- Temperature: provider default; Gemini 3.x documentation advises against lowering it
- Maximum output tokens: `8192`
- Image input: one or three normalized JPEG images in a single request
- Output: `application/json` constrained by the Gemini-compatible schema
- Retry behavior: at most two bounded requests to the same Gemini model; the
  second request occurs only after an exception or schema-invalid response;
  there is no cross-provider fallback
- Post-response handling: strict application validation before any result is shown
- Analysis version: `visible-surface-v1.4.0`
- Prompt version: `visible-surface-prompt-v1.1.0`
- Response schema: `visible-surface-response-schema-v1.2.0`
- Recommendation catalog: `naples-appearance-recommendations-v3.0.0`

## Recommendation architecture

Gemini does not receive Von & Co's service or product catalog and cannot choose a
treatment or product. It returns only schema-valid observations from the locked
visible-feature vocabulary, including neutral and unable-to-assess findings it can
honestly judge. If a valid complete response omits a permitted category, the server
adds that category only as `unable_to_assess`; it never invents a positive or
negative finding or attaches photo-view evidence to the placeholder. After that
response passes validation, the server applies a
versioned, deterministic catalog to create service and skincare matches from Von
& Co's current provider guides.

- `visible` or `prominent` priority IDs lead the matches, followed by every other
  supported visible or prominent finding; when none is present, every supported
  `subtle` finding can produce clearly labeled maintenance matches so a usable
  photo does not end in an empty plan;
- `not_observed` and `unable_to_assess` findings never produce a match;
- the selected body area must be explicitly approved in the catalog;
- neck and chest are separate capture choices so neck-only and chest-only catalog
  support cannot be treated as interchangeable;
- retake results and the narrowly defined open-or-broken-skin medical-review
  state suppress every service and product; the model is not allowed to decide
  whether a mole, spot, or lesion is medically concerning;
- photo-only descriptions remain appearance language: for example,
  `blemish_like_spots` is never converted into an acne diagnosis and
  `visible_flaking` is never converted into a hydration measurement;
- an exact appearance ID may create only the service or product matches explicitly
  approved for that ID and selected body area in the frozen catalog;
- every eligible catalog match is returned, deduplicated, and labeled with all
  appearance IDs that caused the match. Response validation is bounded by the
  versioned catalog cardinality rather than an arbitrary display cap. Face
  results also include the catalog's daily SPF essential;
- the application never turns a catalog match into eligibility, safety,
  diagnosis, expected outcome, or real-time inventory language.

This separation keeps model interpretation and business-menu mapping independently
testable. Updating the model does not silently change the menu, and updating the
menu does not expand what the model is allowed to infer.

## Consumer presentation contract

The interface uses a visibly labeled six-stage estimated tracker while the single Gemini
request is running. It is a transparent waiting experience, not backend telemetry:
steps remain numbered as the interface advances, and every step becomes complete
only after a valid response actually arrives.

A completed result presents a concise quick read, visible strengths and priorities,
the full returned ordinal profile (`not_observed`, `subtle`, `visible`,
`prominent`, or `unable_to_assess`) with model-returned photo-view evidence, and then the ordered
server-owned service and skincare matches. The consultation action sits beside the
quick read, while limitations and the in-person-evaluation requirement remain
visible in compact supporting copy. On mobile, service and product cards use a
single-column layout with explicit full counts and progressive-disclosure controls;
every match remains in the document and can be expanded without a carousel. The
category-by-category profile remains available behind one disclosure control. The sticky branded header keeps the main
Von & Co site reachable throughout the result, and a completed result includes a
direct path back to analyze another photo or area.

High thinking is the deliberate quality setting, but it is not the lowest-latency setting. Google documents `medium` as the default for Gemini 3.5 Flash and warns that `high` may take significantly longer before returning an answer. Any future move to `medium` should be based on the locked validation set rather than an assumption about speed.

## Cost and stability

Google's published paid-tier price for Gemini 3.5 Flash is $1.50 per million input tokens and $9.00 per million output tokens, including thinking tokens. The stable Flash model is preferred over the preview-only Gemini 3.1 Pro model for this consumer production workflow.

## Versioning and privacy requirements

- Keep the exact model ID in `GEMINI_MODEL`; do not use a moving `latest` alias.
- Record the provider, model ID, prompt version, and schema version with every completed result.
- Record the recommendation-catalog version with every completed result.
- Keep the local capture checks and fail-closed response validation unchanged.
- Do not advertise zero provider retention until Von & Co has verified and documented the Google account's applicable data controls.
- Store `GOOGLE_API_KEY` only in the Render environment. Never place a live key in source control, documentation, chat, or browser-console code.

## Environment configuration

```text
GOOGLE_API_KEY=
GEMINI_MODEL=gemini-3.5-flash
PROVIDER_TIMEOUT_SECONDS=35
```

## Official Google sources

- [Gemini 3.5 Flash model card](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash)
- [Gemini thinking controls](https://ai.google.dev/gemini-api/docs/generate-content/thinking)
- [Gemini image understanding](https://ai.google.dev/gemini-api/docs/generate-content/image-understanding)
- [Gemini structured outputs](https://ai.google.dev/gemini-api/docs/generate-content/structured-output)
- [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing)
