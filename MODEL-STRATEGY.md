# Model Strategy for the Visible Skin Preview

## Recommendation

Use `gpt-5.6-terra` as the primary three-image analysis model after the new OpenAI key is configured. Send the front, left, and right images together with `detail: "original"`, `store: false`, and a strict JSON Schema. Terra supports image input and Structured Outputs while costing half as much as Sol at the published standard rates.

Use `gpt-5.6-sol` only as an evaluated escalation path. An escalation should be triggered by an objective application condition, such as valid images followed by an unable-to-assess result, not by an uncalibrated model confidence score. If Sol still cannot assess the images, the application must abstain.

Use `claude-sonnet-5` as the preferred cross-provider fallback after its key and data controls are configured. Keep the existing Gemini path available during migration so production remains operational before the new keys are active.

An optional inexpensive image-quality gate may use `gemini-3.1-flash-lite` after deterministic local resolution, exposure, and file checks. Its job would be limited to framing, obstruction, angle completeness, and same-subject consistency. It must not generate the cosmetic observations or treatment discussion topics.

Do not use `gpt-image-2` or any other image-generation model for analysis. Image-generation capability is unrelated to clinical or cosmetic assessment accuracy.

## Proposed production order after key setup

1. Local deterministic capture checks.
2. `gpt-5.6-terra` for the locked visible-feature schema.
3. Optional `gpt-5.6-sol` escalation only if validation shows a measurable gain.
4. `claude-sonnet-5` as an outage fallback.
5. Honest unavailable or retake state if all configured providers fail.

Do not combine model outputs into a numerical average or call model agreement clinical validation. Provider ratings on a representative, locked image set remain the validation reference.

## Versioning and privacy requirements

- Set exact model IDs in environment variables. Avoid `latest` aliases that may change without a code release.
- Record provider, model ID, prompt version, and schema version with every result.
- Use `store: false` wherever the provider supports it.
- Do not advertise zero provider retention until Von & Co has verified and documented the account-level ZDR configuration.
- Do not place API keys in source control, local documentation, chat, or a browser console. Store them only in the Render environment settings.

## Suggested environment configuration

```text
AI_PROVIDER_ORDER=openai,anthropic,gemini
OPENAI_MODEL=gpt-5.6-terra
OPENAI_ESCALATION_MODEL=gpt-5.6-sol
ANTHROPIC_MODEL=claude-sonnet-5
GEMINI_QUALITY_MODEL=gemini-3.1-flash-lite
```

The application must continue to skip any provider whose key is absent, so these settings can be introduced without downtime.

## Official sources

- [OpenAI GPT-5.6 Terra](https://developers.openai.com/api/docs/models/gpt-5.6-terra)
- [OpenAI GPT-5.6 Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol)
- [OpenAI image inputs and multiple-image support](https://developers.openai.com/api/docs/guides/images-vision)
- [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
- [OpenAI pricing](https://developers.openai.com/api/docs/pricing)
- [OpenAI data controls](https://developers.openai.com/api/docs/guides/your-data)
- [Anthropic model overview](https://platform.claude.com/docs/en/about-claude/models/overview)
- [Anthropic vision](https://platform.claude.com/docs/en/build-with-claude/vision)
- [Anthropic API data retention](https://platform.claude.com/docs/en/manage-claude/api-and-data-retention)
- [Google Gemini 3.1 Flash-Lite](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite)
- [Google Gemini media resolution](https://ai.google.dev/gemini-api/docs/media-resolution)
- [Google Gemini zero data retention](https://ai.google.dev/gemini-api/docs/zdr)
