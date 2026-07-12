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
- Retry behavior: one bounded provider request; no cross-provider fallback
- Post-response handling: strict application validation before any result is shown

High thinking is the deliberate quality setting, but it is not the lowest-latency setting. Google documents `medium` as the default for Gemini 3.5 Flash and warns that `high` may take significantly longer before returning an answer. Any future move to `medium` should be based on the locked validation set rather than an assumption about speed.

## Cost and stability

Google's published paid-tier price for Gemini 3.5 Flash is $1.50 per million input tokens and $9.00 per million output tokens, including thinking tokens. The stable Flash model is preferred over the preview-only Gemini 3.1 Pro model for this consumer production workflow.

## Versioning and privacy requirements

- Keep the exact model ID in `GEMINI_MODEL`; do not use a moving `latest` alias.
- Record the provider, model ID, prompt version, and schema version with every completed result.
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
