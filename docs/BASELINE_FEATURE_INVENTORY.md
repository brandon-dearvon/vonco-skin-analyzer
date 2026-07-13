# Von & Co Skin Analyzer baseline feature inventory

Status: frozen product contract for the narrow restoration. This document records behavior. It does not certify clinical accuracy, recommendation quality, catalog accuracy, or readiness to deploy.

Source anchors below refer to the current working tree inspected on 2026-07-13.

## Frozen reference points

| Reference | Purpose | Verified state |
|---|---|---|
| `e82ccabb451f5b88255c4c1ba98d2cd8e5a203a0` | Original product UI with persistent Von & Co navigation and the artwork present at that time | Historical UI baseline |
| `91e8d80537902329ce9d397e1ba8038a6394be4a` | Restored original UI plus the provider-only Google integration | Starting commit for this preview |
| `safety/restored-original-hd-20260713` | Safety pointer to the restored starting commit | Points to `91e8d80` |
| `preview/original-approved-updates-20260713` | Isolated working branch for only the approved changes | Local candidate commit descends from `91e8d80`; it is not pushed, merged, or deployed |
| `main` and local `origin/main` | Protected release line | Both local refs point to `91e8d80` |

At the start of the preview branch, `public/index.html`, `public/logo.png`, `public/logo_white.png`, and `public/logo_clean.png` matched `e82ccab`. The net committed provider migration from `e82ccab` to `91e8d80` touched only `.env.example`, `requirements.txt`, and `server.py`.

The preview branch is a separate Git line for review. Its local candidate commit does not move `main` or change the live site. Only a later authorized push, merge, and deployment could do that; none of those release actions is part of this inventory.

## Protected original product contract

The following product-level features existed in the restored original and must remain unless Brandon approves a separate change.

### Navigation, identity, and page framing

| Feature | Protected behavior | Current source anchor |
|---|---|---|
| Persistent main-site navigation | Fixed header with a linked brand mark and a main-site action | `public/index.html:43-119`, `public/index.html:1582-1591` |
| Multiple return paths | Header mark, hero logo, and footer link open `https://www.vonandcoaesthetics.com/` in a new tab | `public/index.html:1584-1588`, `public/index.html:1596-1597`, `public/index.html:1910-1913` |
| Hero | Evergreen hero with the full lockup, headline, and supporting copy | `public/index.html:1593-1602` |
| Responsive page shell | Desktop and mobile rules resize the fixed header, logo, action, cards, and page gutters | `public/index.html:1424-1578` |
| Footer | Studio name, main-site link, phone, Naples location, seven-day availability, small-print disclaimer, and copyright | `public/index.html:1907-1916` |

### Intake and capture

| Feature | Protected behavior | Current source anchor |
|---|---|---|
| Introductory analysis panel | Upload headline, VISIA-style explainer, and five visible concern examples | `public/index.html:1604-1635` |
| Optional entered age | Numeric age field accepts 18 to 99 and provides context to the server | `public/index.html:1637-1642`, `public/index.html:2320-2326`, `public/index.html:2375-2379`, `server.py:1092-1098`, `server.py:1230-1241` |
| Five selectable areas | Face, neck and chest, hands, back, and legs | `public/index.html:1643-1652`, `server.py:1048-1071`, `server.py:1228` |
| File upload | Tap-to-upload and drag-and-drop accept the first selected image | `public/index.html:1662-1675`, `public/index.html:1966-2010` |
| File guardrails | Client and server enforce image type checks and a 10 MB limit | `public/index.html:1993-2002`, `server.py:838-841`, `server.py:1196-1214` |
| Quick Snap | Camera modal can capture one selfie and analyze it immediately | `public/index.html:1687-1708`, `public/index.html:2034-2051`, `public/index.html:2191-2197` |
| Guided Capture interface | Front, left, and right steps, thumbnails, retakes, and review state | `public/index.html:1687-1732`, `public/index.html:2023-2120`, `public/index.html:2198-2234` |
| Camera guidance | Front-facing camera request, face oval, directional cues, and brightness-based lighting badge | `public/index.html:1697-1718`, `public/index.html:2028-2032`, `public/index.html:2122-2165` |
| Browser normalization | Canvas limits the longest side to 1200 pixels and encodes a JPEG at 90 percent quality before upload | `public/index.html:2337-2375` |
| Server normalization | Pillow converts supported input to RGB, applies EXIF orientation, and re-encodes a JPEG before Gemini | `server.py:1253-1274` |
| Adult gate | Entered ages below 18 are stopped in the browser and rejected by the server; the original model prompt also rejects a photo that appears to show a minor | `public/index.html:2320-2326`, `server.py:186-193`, `server.py:1230-1241` |
| Rejection experience | A modal explains an age or image rejection and offers Try Again | `public/index.html:2419-2486`, `server.py:172-195` |

### Analysis progress and resilience

| Feature | Protected behavior | Current source anchor |
|---|---|---|
| Six-stage tracker | Upload, profile mapping, texture and tone, sun damage and hydration, concern review, and recommendation steps | `public/index.html:1736-1776` |
| Progress feedback | Progress bar, elapsed seconds, staged completion, and a 75-second reassurance message | `public/index.html:2241-2318` |
| Health and mode check | Browser checks `/api/health`; server reports `ok` and `live` or `demo`, and the client preserves that distinction | `public/index.html:1952-1964`, `server.py:847-853` |
| Client retry and demo fallback | Browser tries `/api/analyze` twice and visibly labels generated sample results if both attempts fail; demo results cannot be marked live | `public/index.html:2381-2448`, `public/index.html:2723-2739` |
| Rejection preservation | An HTTP 422 analysis rejection opens the rejection experience and cannot fall through to demo or random recommendations | `public/index.html:2381-2486` |
| Server retry | Live Gemini generation receives two attempts before returning an analysis error | `server.py:1284-1328` |
| Demo data | Server and browser can generate area-specific sample data when live analysis is unavailable | `server.py:433-835`, `public/index.html:2521-2721` |
| Rate limit | Server tracks requests by IP within a one-hour window | `server.py:148-162`, `server.py:1191-1194` |

### Lead gate and completed results

| Feature | Protected behavior | Current source anchor |
|---|---|---|
| Lead gate | Results pause behind a form for first name, email, and optional phone; guests may also skip | `public/index.html:1888-1905`, `public/index.html:2488-2528` |
| Lead API | `/api/lead` validates name and email and stores the lead in the process memory; `/api/leads` exposes that in-memory list behind a token | `server.py:856-889` |
| Overall score | Animated circular score, label, interpretation, explainer, and contextual subtext | `public/index.html:1789-1807`, `public/index.html:2763-2808` |
| Quick insights | Visible strength, top priority, best score, and average skin-health summary | `public/index.html:1809-1810`, `public/index.html:2810-2830` |
| Concern cards | Returned concerns are sorted from highest concern score to lowest and rendered with health score, severity, description, and context | `public/index.html:2847-2893` |
| Treatment recommendations | Every item in `data.recommendations` is rendered with priority, rationale, targets, and a booking action | `public/index.html:2903-2926` |
| Product recommendations | Every item in `data.productRecommendations` is rendered as a skincare card | `public/index.html:2928-2944` |
| Suggested combination | `suggestedCombo`, when present, appears as the recommended treatment stack | `public/index.html:2946-2954` |
| Booking paths | Consultation actions appear after the score, after concerns, after recommendations, in the VISIA block, and in the closing CTA | `public/index.html:2838-2845`, `public/index.html:2895-2901`, `public/index.html:2956-2973`, `public/index.html:1877-1885` |
| New-guest offer | The page and lead gate display the currently embedded 15 percent first-visit offer | `public/index.html:1826-1832`, `public/index.html:1900-1903` |
| Club block | Current price, savings copy, discount cards, Club Funds copy, and membership link remain present | `public/index.html:1834-1856` |

### Take-home report

| Feature | Protected behavior | Current source anchor |
|---|---|---|
| Guest-facing report action | View My Treatment Plan builds a branded HTML report from the completed analysis | `public/index.html:1858-1873`, `public/index.html:2986-3137` |
| New-tab and inline fallback | The report opens in a new tab; if a popup is blocked, it opens in the embedded preview frame | `public/index.html:3139-3179` |
| Report contents | Guest name, overall score, summary, concerns, all returned services, all returned products, offer, Club copy, booking action, contact details, and small disclaimer | `public/index.html:2987-3136` |
| Server report endpoint | `/api/report` remains available as a separate server-generated HTML report route | `server.py:892-1045` |

### Provider and server contract at the restored starting commit

| Feature | Protected behavior | Current source anchor |
|---|---|---|
| Google-only live client | Live mode depends on `GOOGLE_API_KEY` and initializes `google.genai.Client` | `server.py:19-20`, `server.py:43-45`, `server.py:139-145` |
| Model | Model ID is `gemini-3.1-pro-preview` | `server.py:45` |
| High thinking | GenerateContent config explicitly uses `ThinkingLevel.HIGH` | `server.py:1287-1298` |
| Structured output | Gemini receives `application/json` plus a fresh per-area schema derived from `ANALYSIS_RESPONSE_SCHEMA`; the selected area's concern keys are required and extra concern keys are forbidden | `server.py:46-135`, `server.py:1056-1090`, `server.py:1290-1298` |
| Body-area concern contract | Face, neck and chest, hands, back, and legs each receive their exact concern family; an unknown area falls back to face | `server.py:1056-1090`, `server.py:1290-1298`, `tests/test_app_behavior.py:165-178` |
| Multimodal request | One request contains the normalized image and area-specific user prompt | `server.py:1278-1299` |
| Output families | Completed output includes score, concerns, treatments, products, optional combination, and summary; rejection output contains a reason | `server.py:46-135` |
| Score post-processing | The restored logic may spread clustered concern scores, recalculate overall score, and move scores out of its configured 63 to 73 band | `server.py:1101-1159` |
| Dependency pin | `google-genai==2.11.0` is the requested runtime dependency | `requirements.txt:5` |
| Static serving | Flask serves `public/index.html` and public assets with no-cache response headers | `server.py:33-43`, `server.py:1360-1382` |

## Approved deltas from the restored original

These are the only functional and brand changes approved for this narrow restoration:

1. Keep the Google provider migration already committed at `91e8d80`: `gemini-3.1-pro-preview`, structured JSON, and `ThinkingLevel.HIGH`. Make the runtime and setup surface Google-only: both launchers verify `google-genai==2.11.0` plus `ThinkingLevel.HIGH`, fail closed if dependency installation fails, and direct users to `GOOGLE_API_KEY`; both DOCX setup guides use the Google AI Studio key flow; startup output never prints a key suffix.
2. Remove estimated skin age from the provider contract, prompts, screen, demo data, and both report paths. Preserve the original entered-age gate and visual minor safety check; do not turn removal of the adult skin-age estimate into removal of that safeguard.
3. Remove the radar chart markup, styling, and rendering code.
4. Lead every completed on-screen result and report with two or three photo-grounded positive observations before the score and improvement areas. The schema requires two to three items, and the server makes the summary open with the first supplied positive detail. Current anchors: `server.py:61-74`, `server.py:1176-1185`, `public/index.html:1783-1787`, `public/index.html:2741-2761`, `public/index.html:3057-3095`.
5. Replace the then-current header and hero artwork with the canonical primary horizontal Evergreen and White lockups. Both tracked files are 1549 by 848 RGBA images and their rendered pixels match the corresponding canonical assets. Current tracked hashes are `36e199cb870a2e981ab7367043365de82c661f738dd4cfd3961677add13fca01` for `public/logo.png` and `bdf8946f7a525e41266f8fdb8a2a333d3d42b7f2a671acb1e2c08475a5be5e29` for `public/logo_white.png`.
6. Apply the prior explicit navigation direction: a white header, Evergreen full lockup with more clear space, and a compact `Learn More` action with no arrow. Current anchors: `public/index.html:43-119`, `public/index.html:1582-1590`.
7. Enforce the selected area's exact concern keys in Gemini's response schema. Real preview testing exposed a neck-and-chest response carrying face keys; the per-area schema correction prevents that class of cross-area response while preserving the original result families. Current anchors: `server.py:1056-1090`, `server.py:1290-1298`, `tests/test_app_behavior.py:165-178`.

One separately requested safety-copy change is retained: the small footer and report disclaimer says that any concerning lesion needs an in-person medical evaluation. This is only a fixed referral disclaimer. At Brandon's direction, the model prompt does not ask Gemini to identify, assess, flag, rule out, or comment on lesions or medical conditions. It does not mean the analyzer can reliably identify or rule out a concerning lesion. Current anchors: `public/index.html:1913`, `public/index.html:3132-3134`, `server.py:172-195`, `server.py:1037-1039`.

`public/logo_clean.png` remains unchanged and is not referenced by the current page.

## Known original limitation: Guided Capture is not three-image analysis

The interface captures and previews front, left, and right images. On submit, however, it assigns only `guidedPhotos[0]` to `uploadedImage` at `public/index.html:2115-2119`. `analyzeImage()` then appends one `image` field named `photo.jpg` at `public/index.html:2375`. The server reads one file from `request.files["image"]` at `server.py:1196-1217`.

The left and right captures are not sent to Gemini. The interface must not be described as multi-image or three-view analysis. This limitation is documented rather than fixed because multi-image inference was not approved in the narrow restoration.

## Items preserved as source, not accepted as verified truth

The original code contains treatment and product catalogs, recommendation rules, VISIA comparisons, offer details, Club pricing and benefits, operating claims, experience claims, a roughly 30-second result estimate, and an image-storage privacy statement. This inventory confirms that the copy and logic remain present. It does not confirm that they are current, complete, clinically appropriate, commercially approved, or supported by the applicable privacy terms. Primary source review by the studio and clinician-labeled recommendation validation remain release gates.

The macOS and Windows launch helpers now direct local setup to `GOOGLE_API_KEY` and verify the pinned Google SDK and HIGH-thinking capability. The two provider setup guides are Google-only. Current runtime-facing source, dependency, launcher, and retry/timeout copy contain no Anthropic or Claude provider reference, and startup output does not reveal any part of the configured key. This is provider cleanup, not a change to the protected guest flow.

## Release boundary

The independent final diff review has passed with no remaining internal code blocker. Nothing from `preview/original-approved-updates-20260713` should move to `main` or a live deployment until the dated QA report has no open release gates, the catalog and clinician-labeled gates are signed off, and Brandon gives the final release click.
