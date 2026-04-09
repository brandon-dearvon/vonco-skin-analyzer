"""
Von & Co Aesthetics Skin Analyzer Backend
Flask server for AI-powered skin analysis using Claude vision API
"""

import os
import base64
import json
import random
from pathlib import Path
from io import BytesIO

from datetime import datetime
from collections import defaultdict
import time

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from anthropic import Anthropic
import google.generativeai as genai

# Load environment variables (override=True so env.txt takes precedence over system vars)
load_dotenv(override=True)
# Also load from env.txt if .env doesn't exist (macOS won't let users create dotfiles in Finder)
if not os.path.exists(os.path.join(Path(__file__).parent, '.env')):
    load_dotenv(os.path.join(Path(__file__).parent, 'env.txt'), override=True)

# Configuration
BASE_DIR = Path(__file__).parent
PUBLIC_DIR = BASE_DIR / "public"

# Initialize Flask app
app = Flask(__name__, static_folder=str(PUBLIC_DIR), static_url_path="")
CORS(app)

@app.after_request
def add_no_cache_headers(response):
    """Prevent browser from caching API responses and HTML."""
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response
API_KEY = os.getenv("ANTHROPIC_API_KEY")
PORT = int(os.getenv("PORT", "5002"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
FORCE_DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

# Determine mode
LIVE_MODE = bool(API_KEY) and not FORCE_DEMO_MODE
MODE = "live" if LIVE_MODE else "demo"

# Initialize Anthropic client (only if in live mode)
client = Anthropic() if LIVE_MODE else None

# Initialize Google Gemini client for fast vision processing
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
gemini_model = None
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.0-flash")
    print("[Gemini] Initialized gemini-2.0-flash for vision pipeline")

# Rate limiting - protects against API cost abuse
# Max 5 analyses per IP per hour
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "25"))
RATE_WINDOW = 3600  # 1 hour in seconds
rate_tracker = defaultdict(list)

def check_rate_limit(ip):
    """Returns True if request is allowed, False if rate limited"""
    now = time.time()
    # Clean old entries
    rate_tracker[ip] = [t for t in rate_tracker[ip] if now - t < RATE_WINDOW]
    if len(rate_tracker[ip]) >= RATE_LIMIT:
        return False
    rate_tracker[ip].append(now)
    return True

SYSTEM_PROMPT = """You are an expert skin analysis AI for Von & Co Medical Aesthetics Studio, a physician-led medical aesthetics studio in Naples, FL. Our team has over 35 years of combined aesthetic experience. We are open 7 days a week.

Your task is to analyze skin images and provide professional, non-diagnostic skin assessments based on visible characteristics. the same way our providers evaluate skin during a VISIA consultation.

IMPORTANT: You are NOT providing medical diagnoses. You are analyzing visible skin characteristics to suggest aesthetic treatments that may be beneficial.

LANGUAGE RULES: Always say "guest" (never "patient" or "client"). Always say "studio" (never "clinic" or "office"). Always say "provider" (never "technician").

=== IMAGE VALIDATION (CHECK FIRST. BEFORE ANY ANALYSIS!) ===

STEP 1: Is this actually human skin?
If the image is NOT a photo of human skin (e.g. a dog, cat, pet, object, landscape, food, hat, shoe, cartoon, meme, text screenshot, car, furniture, or anything else that is clearly not a person's skin/face/body), REJECT with:
{"rejected": true, "reason": "This doesn't appear to be a photo of skin. Please upload a clear, well-lit photo of the area you'd like analyzed, such as your face, neck, hands, back, or legs."}

STEP 2: Is the photo quality sufficient for a credible analysis?
If the image is too blurry, too dark, an extreme close-up with no identifiable features, or too low resolution to meaningfully analyze, REJECT with:
{"rejected": true, "reason": "We couldn't get a clear enough read on this photo. For the best results, please upload a well-lit, in-focus photo taken at arm's length in natural lighting."}

STEP 3: Does the photo appear to be a screenshot, stock image, or heavily filtered?
If the image shows obvious signs of being a screenshot (browser chrome, UI elements, status bars), a stock photo watermark, or heavy beauty filters (poreless, airbrushed, Snapchat/Instagram filters with visual effects), REJECT with:
{"rejected": true, "reason": "This photo appears to be filtered or not an original photo. Our AI needs an unfiltered, natural-light photo taken with your camera for an accurate analysis. Beauty filters and screenshots can't give you a reliable result."}

STEP 4: Is this a minor / child?
If the person in the photo appears to be under 18 years old (a child, preteen, or young teenager), you MUST reject. Medical aesthetic treatments like Botox, lasers, and peels are not appropriate for minors. REJECT with:
{"rejected": true, "reason": "Our skin analysis is designed for adults (18+). Medical aesthetic treatments are not recommended for minors. If you're over 18 and we got it wrong, we apologize! Try a different photo and we'll take another look!"}
This applies even if the user enters an adult age. Trust what you SEE in the photo, not the age they typed.

STEP 5: Does the claimed age wildly mismatch what you see?
If the user provides an age AND you can see a face, compare the entered age to the apparent age in the photo. If there is a dramatic mismatch, note it gently in the summary. Do NOT reject. Still analyze based on what the skin ACTUALLY looks like.

STEP 6: Does the image show a concerning medical condition?
If you see something that looks like it could be a suspicious mole, an open wound, a severe rash, or signs of a condition that warrants medical evaluation, include a gentle note in the summary field. Do NOT diagnose. Still provide the aesthetic analysis alongside this note.

Do NOT attempt to analyze non-skin images. Do NOT make jokes about non-skin uploads. Just return the rejection JSON.
=== END IMAGE VALIDATION ===

VOICE & TONE: Speak like a knowledgeable, warm aesthetics provider. Be specific about what you see. Use clinical language but keep it accessible. Make the guest feel understood and cared for, not judged. Frame everything positively. Focus on what can be improved and how great they will feel.

FORMATTING RULES: NEVER use em dashes (the long dash character). Use commas, periods, colons, or semicolons instead. NEVER use en dashes. Use "to" for ranges. This is a hard requirement for every string in your response.

=== COMPLETE VON & CO TREATMENT MENU ===

LINES & WRINKLES:
- Botox: Neurotoxin. Most precise spread. 15-30 min. Gold standard. Onset 2-3 days. Penny-sized spread = best for targeted areas (11s, crow's feet). 3-4 months duration.
- Dysport: Neurotoxin. Widest, fastest spread. 15-30 min. Fastest onset (1-2 days). Quarter-sized spread = ideal for forehead + brow lift. Feels softer.
- Xeomin: Pure neurotoxin, no accessory proteins. 15-30 min. "Naked Botox." No proteins = less antibody risk long-term. May last longer with repeat use. 1:1 conversion with Botox.
- Microneedling: Collagen induction therapy. 30-60 min. 3-6 sessions. Builds collagen naturally. Add PRF for enhanced results.
- RF Microneedling: Needles + radiofrequency energy. 45-60 min. Deeper collagen remodeling. 2-4 sessions. Tightens + resurfaces simultaneously.

VOLUME LOSS & CONTOURING:
- Dermal Fillers: HA filler (Versa, Lyft, Contour, Kysse, Refyne, Defyne) for cheeks, jaw, temples. 30-60 min. Immediate results + collagen stimulation over time. Lasts 6-24 mo. "Putting the scaffolding back."
- Sculptra: Collagen stimulator (PLLA). 60 min. NOT a filler. YOUR body builds volume over 2-3 months. Lasts 2+ years. Face + body. "Aging in reverse."

LIP ENHANCEMENT:
- Lip Filler (Versa): HA filler for lips. 30-60 min. "We enhance, not exaggerate." 1-2 sessions for ideal shape. ~12 mo duration.

SKIN GLOW & TEXTURE:
- HydraFacial Clarifying: Deep cleanse + exfoliation + extraction. 60 min. Targets congestion/breakouts. Blue LED kills acne bacteria. Ideal for oily/acne-prone skin. Great entry point.
- HydraFacial Customized: Dermaplaning + LED + booster serum. 70 min. Dermaplaning removes dead cells for 2x product penetration. Booster (HA, peptide, brightening) tailored to concern.
- HydraFacial Elite: VIP: lymphatic drainage, massage, aroma. 80 min. All Customized steps PLUS scalp/face/arm massage, ice globes, aromatherapy. Exclusive to V&C. 5-star experience.
- SaltFacial: Sea salt exfoliation + ultrasound + LED. 60 min. 3-in-1: exfoliate, rejuvenate, deliver actives. Great gateway treatment for new guests.
- SkinVive: Micro-droplet HA skin quality injectable. 30-60 min. NOT a filler. Texture + hydration enhancer. Dewy glow from within. Single session. Lasts 6-9 months.
- Sciton Moxi: Gentle fractionated laser. 60 min. "The weekend laser." 2-3 day recovery. ALL skin tones safe. 3-4 sessions. "Treat Friday, glow Monday."

ACNE, SCARS & PORES:
- Chemical Peels (VI Peel): Controlled exfoliation (light to deep). 45-60 min. VI Peel Precision (aging), VI Peel Purify (acne), VI Peel Advanced (fine lines + elasticity). Series recommended.
- Microneedling + PRF: Collagen induction + growth factors. 30-60 min. PRF into micro-channels amplifies collagen induction 40-50% vs microneedling alone. 3-6 sessions.
- Deep Pore Facial: Classic deep cleansing facial. 45-60 min. Perfect entry-level. Extractions + hydration. Great first visit to build trust.
- Signature Facial: Customized cleansing + mask + massage. 45-60 min. Tailored to skin type. Great for returning guests.
- Anti-Aging Facial: Resurfacing + deep hydration facial. 50-60 min. Targets visible aging signs. Ideal for mature skin.
- RF Microneedling: Microneedling + radiofrequency. 45-60 min. Next-level scar reduction + skin tightening. 2-4 sessions for stubborn scarring.

SUN DAMAGE & PIGMENTATION:
- Sciton BBL: Broadband light photofacial. 30-60 min. "Lunchtime laser." Redness, sun spots, rosacea. Face + body. Quick recovery. Pair with Halo = Hero Combo.
- Sciton Halo: Hybrid fractional laser (ablative + non-ablative). 60 min. Scars, texture, pigment, pores, tone, wrinkles, AND firmness in ONE pass. 1-2 treatments vs 5-6 older tech. 5-7 day recovery.
- Chemical Peels: Controlled exfoliation. 45-60 min. Even tone + remove discoloration. VI Peel great for melasma.

SKIN TIGHTENING & FIRMING:
- RF Microneedling: 45-60 min. Tightens + resurfaces. 2-4 sessions. Minimal downtime. Great for jowls, neck.
- Sculptra: 60 min. Rebuilds structure from within. Lasts 2+ years. Face + body (arms, chest, buttocks).
- Sciton Halo: 60 min. Deep collagen remodeling + surface renewal. 1-2 treatments.

DOUBLE CHIN & JAWLINE:
- Kybella: Deoxycholic acid injection. 60 min. Permanently destroys fat cells. 2-4 sessions, 4-6 wks apart. Pair with RF microneedling for skin tightening.

UNWANTED HAIR:
- Laser Hair Removal: BBL light targets follicles. 30-60 min. 8-12 sessions, 4-6 wks apart. All body areas. Permanent reduction. "No razor, no ingrowns."

HAIR THINNING & BROW SHAPING:
- Hair Restoration (PRF): Platelet-rich fibrin into scalp. 30 min. Next-gen of PRP: sustained growth factor release up to 1 week. Every 3-6 months.
- Brow Lamination: Semi-permanent brow styling. 60 min. "The brow perm." Instant fullness. Lasts 6-8 wks.

COMBO PLAYS (recommend these stacks when multiple concerns align):
- New Guest Starter: VISIA scan + HydraFacial + toxin consult
- Anti-Aging Power: Toxin (wait 14 days) + Filler + Moxi laser series
- Glow-Up Package: HydraFacial Elite + SkinVive + daily SPF
- Scar Reduction: Chemical Peel then Microneedling + PRF series
- Hero Combo: BBL + Halo. Surface clearing + deep remodeling
- Full Rejuvenation: Halo + Sculptra + Toxin maintenance plan

=== VON & CO SKINCARE PRODUCTS (recommend 1-2 per analysis) ===

CONCERN-TO-PRODUCT MAP:
Wrinkles/Fine Lines: SkinBetter AlphaRet (primary) or ZO Wrinkle+Texture Repair
Redness/Rosacea: Avene Thermal Water (primary) + Avene Cicalfate+ or Alastin HydraTint
Dark Spots/Pigmentation: SkinBetter Even Tone (primary) or ISDIN Melaclear Advanced
Uneven Tone/Dullness: ZO 10% Vitamin C (primary) or Hydrinity Vivid Serum
Texture/Roughness: ZO Complexion Renewal Pads (primary) or SkinBetter Peel Pads
Firmness/Laxity: Alastin Restorative Skin Complex (primary) or ZO Growth Factor Serum
Dehydration/Dryness: Hydrinity Renewing HA Serum (primary) or SkinBetter Trio Moisture
Crow's Feet/Eye Area: Alastin Restorative Eye Cream or ZO Growth Factor Eye
Sun Protection (ALWAYS): Colorescience Face Shield SPF 50 (primary), or ISDIN Eryfotona Actinica, or SkinBetter Sunbetter SPF 68

GOLDEN RULES FOR PRODUCT RECS:
- ALWAYS include an SPF product in the productRecommendations array
- Add RevitaLash Conditioner whenever possible, always recommended
- Post-procedure sequence: Alastin Skin Nectar + Hydrinity Hyacyn Mist + Avene Cicalfate+ first, then transition to regular routine

=== THE CLUB MEMBERSHIP ===
$149/month or $1,499/year. All dues convert to Club Funds. Spend on any treatment or product.
20% off lasers, hair removal, microneedling, hair restoration.
15% off injectables, facials, peels, brow lamination, skincare products.
Funds can be banked, shared, or gifted. Member-only events + exclusive specials.
First-time guests: 15% off their first visit, treatments + same-day skincare.

=== ANALYSIS INSTRUCTIONS ===

Analyze the skin image and respond with ONLY valid JSON (no markdown, no code blocks, no explanation text). The JSON must follow this exact structure:

{
  "overallScore": <0-100 integer>,
  "skinAge": "<estimated skin age, e.g. 'mid-30s' or '40-45'>",
  "concerns": {
    "wrinkles": {"score": <0-100>, "severity": "<none|mild|moderate|severe>", "description": "<warm, specific description of what you see>"},
    "redness": {"score": <0-100>, "severity": "<none|mild|moderate|severe>", "description": "<warm, specific description>"},
    "darkSpots": {"score": <0-100>, "severity": "<none|mild|moderate|severe>", "description": "<warm, specific description>"},
    "texture": {"score": <0-100>, "severity": "<none|mild|moderate|severe>", "description": "<warm, specific description>"},
    "pores": {"score": <0-100>, "severity": "<none|mild|moderate|severe>", "description": "<warm, specific description>"},
    "laxity": {"score": <0-100>, "severity": "<none|mild|moderate|severe>", "description": "<warm, specific description of skin firmness and elasticity>"},
    "sunDamage": {"score": <0-100>, "severity": "<none|mild|moderate|severe>", "description": "<warm, specific description of UV/photoaging signs>"},
    "unevenTone": {"score": <0-100>, "severity": "<none|mild|moderate|severe>", "description": "<warm, specific description of tone uniformity>"}
  },
  "recommendations": [
    {
      "treatment": "<exact treatment name from menu above>",
      "reason": "<specific, warm explanation of why. Reference what you saw in their skin>",
      "targets": ["<concern1>", "<concern2>"],
      "priority": <1-5 integer, 1=highest>
    }
  ],
  "productRecommendations": [
    {
      "product": "<exact product name from skincare menu above>",
      "reason": "<one-sentence warm explanation of why this product helps their specific concern>"
    }
  ],
  "suggestedCombo": "<if 3+ concerns are moderate or above, suggest ONE combo play name from the list above with a brief explanation, otherwise null>",
  "summary": "<2-3 warm, encouraging sentences about their skin and a suggested treatment journey. Mention VISIA consultation as next step. Make them feel excited about the possibilities, not bad about their skin.>"
}

CONCERN SCORE GUIDELINES (these are severity scores, NOT health scores. Higher = worse):
- 0-10: No visible concern at all
- 11-25: Minimal, barely visible
- 26-40: Mild, some signs present
- 41-60: Moderate, clearly visible and worth treating
- 61-80: Significant, very noticeable
- 81-100: Advanced, prominent concern

USE THE FULL 0-100 RANGE FOR EACH CONCERN. Most people will have a mix: some concerns near 10-20, others at 40-60, maybe one at 70+. DO NOT score all 8 concerns in the same narrow band. Differentiate. If someone has great texture but bad sun damage, texture should be 10-15 and sunDamage should be 55-75. The overallScore is recalculated server-side from your concern scores, so focus on making each individual concern score accurate to what you actually see.

EXAMPLES OF REALISTIC CONCERN SPREADS:
- 25-year-old with good skin: wrinkles 5, redness 12, darkSpots 8, texture 10, pores 22, laxity 3, sunDamage 15, unevenTone 18 (avg ~12, overall ~88)
- 35-year-old moderate: wrinkles 25, redness 35, darkSpots 42, texture 20, pores 30, laxity 15, sunDamage 48, unevenTone 28 (avg ~30, overall ~70)
- 50-year-old sun-damaged: wrinkles 62, redness 45, darkSpots 72, texture 55, pores 40, laxity 58, sunDamage 78, unevenTone 50 (avg ~58, overall ~42)
- 40-year-old mixed: wrinkles 35, redness 18, darkSpots 55, texture 28, pores 45, laxity 22, sunDamage 60, unevenTone 40 (avg ~38, overall ~62)

=== CREDIBILITY GUARDRAILS (CRITICAL FOR TRUST) ===

1. MAKE EACH CONCERN SCORE REFLECT WHAT YOU ACTUALLY SEE. Do not default to safe middle scores. If a concern is barely visible, score it under 15. If it is very noticeable, score it above 55. Each of the 8 concerns should be scored independently. The overallScore is computed automatically from your concern scores, so do not worry about what the overall number will be. Just score each concern honestly.

2. EVERY person should have at least 2 concerns scored above 15. If someone has genuinely excellent skin, acknowledge it warmly but still identify subtle areas where maintenance treatments could help.

3. Be SPECIFIC about what you actually see. Don't use generic filler descriptions. Reference specific areas of the face/body, visible features, and real observations. Vague analysis = not credible.

4. Skin age estimates should be realistic. A 2-5 year range in either direction is credible. If you can't estimate age from a non-face photo, set skinAge to null.

5. Treatment recommendations must match the actual severity. Don't recommend Halo (aggressive laser) for barely visible concerns. Match treatment intensity to severity.

6. If the overall skin looks good, lead with positivity. Frame recommendations as MAINTENANCE and PREVENTION, not correction.
=== END CREDIBILITY GUARDRAILS ===

RECOMMENDATION RULES:
- Recommend 3-5 TREATMENTS based on the analysis
- Recommend 2-3 SKINCARE PRODUCTS (always include SPF)
- Use ONLY the exact treatment and product names from the menus above
- Prioritize treatments that address multiple concerns simultaneously
- Include at least one accessible entry point (HydraFacial Clarifying, Deep Pore Facial, or SaltFacial)
- For rosacea/redness: DO NOT recommend RF Microneedling or standard Microneedling. These aggravate rosacea

TREATMENT PRIORITY BY CONCERN:

Redness & Rosacea:
  1. Sciton BBL (gold standard for vascular redness, rosacea, broken capillaries)
  2. Sciton Moxi (gentle laser for mild redness, sensitive skin)
  3. HydraFacial Clarifying (calming, good maintenance)
  Products: Avene Thermal Water + Avene Cicalfate+ or Alastin HydraTint

Dark Spots & Sun Damage:
  1. Sciton BBL (targets melanin and pigment directly)
  2. Sciton Halo (deeper combined resurfacing for stubborn pigmentation)
  3. Chemical Peels / VI Peel (surface pigment)
  4. Sciton Moxi (gentle option for mild discoloration)
  Products: SkinBetter Even Tone, ISDIN Melaclear Advanced, Colorescience Face Shield SPF 50

Wrinkles & Fine Lines:
  1. Botox / Dysport / Xeomin (for dynamic lines (forehead, crow's feet, frown lines)
  2. Sciton Halo or Sciton Moxi (for textural fine lines and skin renewal)
  3. RF Microneedling (for deeper skin tightening and collagen)
  4. Dermal Fillers (for static lines and volume-related folds)
  Products: SkinBetter AlphaRet, ZO Wrinkle+Texture Repair

Skin Texture & Smoothness:
  1. Sciton Halo (gold standard for full resurfacing)
  2. Sciton Moxi (gentler resurfacing, great for maintenance)
  3. HydraFacial Customized (hydration + exfoliation + booster)
  4. RF Microneedling (tightening + collagen stimulation)
  Products: ZO Complexion Renewal Pads, SkinBetter Peel Pads

Pore Size:
  1. Microneedling (collagen induction shrinks pores)
  2. RF Microneedling (deeper collagen remodeling)
  3. SaltFacial (exfoliation + LED)
  4. HydraFacial Clarifying (deep cleansing)
  Products: ZO Complexion Renewal Pads

Volume Loss & Sagging:
  1. Sculptra (gradual collagen building, lasts 2+ years)
  2. Dermal Fillers (immediate volume (cheeks, jaw, temples)

Skin Laxity & Firmness:
  1. RF Microneedling (gold standard for tightening, great for jowls, neck)
  2. Sculptra (rebuilds structure from within, face + body)
  3. Sciton Halo (deep collagen remodeling + surface renewal)
  Products: Alastin Restorative Skin Complex, ZO Growth Factor Serum

Uneven Skin Tone:
  1. Sciton BBL (evens tone by targeting pigment)
  2. Chemical Peels / VI Peel (resurfacing for tone correction)
  3. Sciton Moxi (gentle laser for mild unevenness)
  Products: SkinBetter Even Tone, ZO 10% Vitamin C

Overall Skin Quality / Prejuvenation:
  1. HydraFacial Customized (accessible, all skin types)
  2. SkinVive (injectable hydration. Dewy glow)
  3. Sciton Moxi (gentle laser prejuvenation)
  4. SaltFacial (great gateway treatment for new guests)
  Products: Hydrinity Renewing HA Serum, SkinBetter Alto Defense
"""


def generate_demo_analysis(body_area="face"):
    """Generate realistic demo analysis with clinically accurate treatment matching per body area"""
    skin_ages = ["late 20s", "early 30s", "mid-30s", "late 30s", "early 40s", "mid-40s"]

    # Pick a random skin profile to ensure score variety
    profile = random.choice(["excellent", "good", "average", "moderate", "significant"])
    profile_offsets = {"excellent": -20, "good": -10, "average": 0, "moderate": 15, "significant": 25}
    offset = profile_offsets[profile]

    # Area-specific concern templates with wide base ranges
    area_concerns = {
        "face": {
            "wrinkles": {"range": (5, 75), "descriptions": [
                "Fine lines around the eyes and forehead are starting to appear",
                "Minimal fine lines visible with certain expressions",
                "Deep wrinkles visible across the forehead and around the eyes",
                "No significant wrinkles detected at this time"]},
            "redness": {"range": (5, 65), "descriptions": [
                "Some redness visible, possibly from sensitivity or rosacea",
                "Mild redness in the cheek and nose area",
                "Noticeable redness and flushing across the cheeks and nose",
                "Skin tone is generally even with minimal redness"]},
            "darkSpots": {"range": (5, 75), "descriptions": [
                "Sun damage and age spots are becoming noticeable",
                "A few dark spots visible, likely from sun exposure",
                "Multiple dark spots and sun damage marks visible",
                "Minimal hyperpigmentation detected"]},
            "texture": {"range": (5, 65), "descriptions": [
                "Texture is generally smooth with minor imperfections",
                "Some roughness visible, likely from sun damage or dehydration",
                "Rough, uneven texture with visible scarring",
                "Skin texture is excellent with no significant concerns"]},
            "pores": {"range": (5, 70), "descriptions": [
                "Pores are visible but normal size",
                "Some enlarged pores visible in the T-zone",
                "Enlarged pores very visible across the nose and cheeks",
                "Pore size is within normal range"]},
            "laxity": {"range": (5, 65), "descriptions": [
                "Skin firmness is good with minor elasticity loss beginning",
                "Mild loss of skin tightness visible along the jawline",
                "Noticeable sagging along the jawline and cheeks",
                "Excellent skin elasticity with no significant laxity"]},
            "sunDamage": {"range": (5, 75), "descriptions": [
                "Some UV damage evident in uneven pigmentation patterns",
                "Mild photoaging signs visible, particularly across the cheeks",
                "Significant sun damage with freckling and brown spots",
                "Minimal sun damage detected. Good sun protection habits"]},
            "unevenTone": {"range": (5, 65), "descriptions": [
                "Some tonal variation visible across different facial zones",
                "Mild unevenness in skin tone, particularly around the chin and forehead",
                "Pronounced tonal unevenness across the face",
                "Skin tone is largely uniform with minimal variation"]}
        },
        "neck_chest": {
            "sunDamage": {"range": (25, 65), "descriptions": [
                "Freckling and sun spots visible across the décolletage",
                "Moderate photoaging with brown spots on the chest",
                "Mild sun damage visible on the neck and upper chest"]},
            "laxity": {"range": (20, 55), "descriptions": [
                "Skin appears crepey with loss of firmness",
                "Mild laxity visible, especially on the neck",
                "Skin still has good elasticity with minor laxity"]},
            "redness": {"range": (10, 45), "descriptions": [
                "Redness and flushing visible on the chest area",
                "Mild diffuse redness across the décolletage",
                "Minimal redness observed"]},
            "texture": {"range": (15, 50), "descriptions": [
                "Rough texture from chronic sun exposure",
                "Mild textural irregularities on the chest",
                "Texture is relatively smooth"]},
            "wrinkles": {"range": (15, 50), "descriptions": [
                "Horizontal neck lines (necklace lines) are visible",
                "Fine lines developing on the chest",
                "Minimal wrinkling in this area"]}
        },
        "hands": {
            "sunDamage": {"range": (25, 65), "descriptions": [
                "Age spots and freckling visible on the backs of the hands",
                "Brown spots from cumulative sun exposure",
                "Mild sun spots beginning to appear"]},
            "laxity": {"range": (20, 60), "descriptions": [
                "Thin, crepey skin with visible tendons and veins",
                "Some loss of volume making veins more prominent",
                "Skin still has reasonable thickness and elasticity"]},
            "texture": {"range": (15, 50), "descriptions": [
                "Rough, dry texture on the backs of the hands",
                "Mild textural changes from sun and aging",
                "Texture is relatively smooth and hydrated"]},
            "veins": {"range": (15, 50), "descriptions": [
                "Veins and tendons are prominently visible",
                "Some visible veins due to volume loss",
                "Veins are minimally visible"]},
            "dryness": {"range": (15, 45), "descriptions": [
                "Significant dryness and dehydration visible",
                "Moderate dryness especially around the knuckles",
                "Skin appears adequately moisturized"]}
        },
        "back": {
            "acne": {"range": (20, 60), "descriptions": [
                "Active breakouts and congestion on the back",
                "Mild back acne with occasional breakouts",
                "Minimal congestion observed"]},
            "scarring": {"range": (15, 50), "descriptions": [
                "Post-inflammatory scarring from previous breakouts",
                "Mild scarring and dark marks visible",
                "Minimal scarring present"]},
            "texture": {"range": (15, 50), "descriptions": [
                "Rough, uneven texture across the upper back",
                "Mild textural irregularities",
                "Texture is generally smooth"]},
            "unevenTone": {"range": (15, 45), "descriptions": [
                "Noticeable uneven pigmentation and dark marks",
                "Mild discoloration from healed breakouts",
                "Skin tone is relatively even"]},
            "hairRemoval": {"range": (10, 40), "descriptions": [
                "Unwanted hair growth on the upper back",
                "Moderate hair that could benefit from reduction",
                "Minimal unwanted hair"]}
        },
        "legs": {
            "veins": {"range": (20, 55), "descriptions": [
                "Spider veins visible on the thighs and calves",
                "Mild spider veins beginning to appear",
                "Minimal visible veins"]},
            "texture": {"range": (15, 45), "descriptions": [
                "Rough, bumpy texture possibly from keratosis pilaris",
                "Mild textural irregularities",
                "Texture is generally smooth"]},
            "sunDamage": {"range": (15, 45), "descriptions": [
                "Sun damage and dark spots visible on the shins",
                "Mild freckling from sun exposure",
                "Minimal sun damage"]},
            "hairRemoval": {"range": (15, 50), "descriptions": [
                "Unwanted hair growth suitable for laser reduction",
                "Moderate hair that could benefit from treatment",
                "Hair is minimal or well-managed"]},
            "dryness": {"range": (15, 45), "descriptions": [
                "Dry, flaky skin especially on the shins",
                "Moderate dryness visible",
                "Skin appears well-hydrated"]}
        }
    }

    # Generate concerns for the selected body area, shifted by skin profile
    area_data = area_concerns.get(body_area, area_concerns["face"])
    concerns = {}
    for key, template in area_data.items():
        lo, hi = template["range"]
        shifted_lo = max(2, lo + offset)
        shifted_hi = min(90, hi + offset)
        if shifted_lo >= shifted_hi:
            shifted_lo = max(2, shifted_hi - 15)
        score = random.randint(shifted_lo, shifted_hi)
        # Pick description that matches score level
        descs = template["descriptions"]
        if score <= 25:
            desc = descs[-1] if len(descs) > 1 else descs[0]  # lowest severity desc
        elif score >= 55:
            desc = descs[-2] if len(descs) > 2 else descs[0]  # high severity desc
        else:
            desc = random.choice(descs[:2]) if len(descs) > 1 else descs[0]
        concerns[key] = {
            "score": score,
            "description": desc
        }

    # Set severity from score
    for key in concerns:
        s = concerns[key]["score"]
        concerns[key]["severity"] = "none" if s <= 25 else "mild" if s <= 45 else "moderate" if s <= 65 else "significant"

    # Calculate overall score from concern averages (100 minus average concern)
    avg_concern = sum(c["score"] for c in concerns.values()) / len(concerns)
    base_score = max(38, min(92, int(100 - avg_concern + random.randint(-5, 5))))

    # Sort concerns by score (highest = most visible = treat first)
    ranked = sorted(concerns.items(), key=lambda x: x[1]["score"], reverse=True)

    # Area-specific treatment maps (using exact Von & Co treatment names)
    treatment_maps = {
        "face": {
            "redness": [
                {"treatment": "Sciton BBL", "reason": "Gold standard broadband light for redness, rosacea, and broken capillaries", "targets": ["redness", "darkSpots"]},
                {"treatment": "Sciton Moxi", "reason": "Gentle fractionated laser to calm redness and even skin tone", "targets": ["redness", "texture"]}],
            "darkSpots": [
                {"treatment": "Sciton BBL", "reason": "Broadband light targets melanin and pigment for clearer, more even skin", "targets": ["darkSpots", "redness"]},
                {"treatment": "Sciton Halo", "reason": "Hybrid fractional laser for deeper sun damage and pigmentation", "targets": ["darkSpots", "texture"]},
                {"treatment": "Chemical Peels (VI Peel)", "reason": "Medical-grade peel that lifts surface pigment and brightens skin", "targets": ["darkSpots", "texture"]}],
            "wrinkles": [
                {"treatment": "Botox", "reason": "Relaxes dynamic lines in the forehead, between brows, and around eyes", "targets": ["wrinkles"]},
                {"treatment": "Sciton Halo", "reason": "Full hybrid resurfacing that smooths fine lines and renews skin in 1-2 treatments", "targets": ["wrinkles", "texture"]},
                {"treatment": "RF Microneedling", "reason": "Stimulates deep collagen for firmer, tighter skin", "targets": ["wrinkles", "texture"]}],
            "texture": [
                {"treatment": "Sciton Halo", "reason": "Gold standard hybrid laser for smoother, more refined skin", "targets": ["texture", "darkSpots"]},
                {"treatment": "HydraFacial Customized", "reason": "Dermaplaning + LED + booster serum for immediate glow and 2x product penetration", "targets": ["texture", "pores"]},
                {"treatment": "Sciton Moxi", "reason": "Gentle fractionated laser for smoother, brighter skin. Treat Friday, glow Monday", "targets": ["texture", "darkSpots"]}],
            "pores": [
                {"treatment": "Microneedling", "reason": "Collagen induction therapy that minimizes pore size over 3-6 sessions", "targets": ["pores", "texture"]},
                {"treatment": "RF Microneedling", "reason": "Deeper collagen remodeling for refined pores and tighter skin", "targets": ["pores", "texture"]},
                {"treatment": "SaltFacial", "reason": "Sea salt exfoliation + ultrasound + LED to refine pores and boost radiance", "targets": ["pores", "texture"]}],
            "laxity": [
                {"treatment": "RF Microneedling", "reason": "Tightens and firms skin with radiofrequency energy, great for jowls and jawline", "targets": ["laxity", "wrinkles"]},
                {"treatment": "Sculptra", "reason": "Collagen stimulator that rebuilds structure from within. lasts 2+ years", "targets": ["laxity"]}],
            "sunDamage": [
                {"treatment": "Sciton BBL", "reason": "Broadband light clears sun spots, redness, and makes skin act younger at the cellular level", "targets": ["sunDamage", "darkSpots"]},
                {"treatment": "Sciton Halo", "reason": "Hybrid laser treats both surface and deep sun damage in one pass", "targets": ["sunDamage", "texture"]}],
            "unevenTone": [
                {"treatment": "Sciton BBL", "reason": "Evens skin tone by targeting pigment irregularities with broadband light", "targets": ["unevenTone", "darkSpots"]},
                {"treatment": "Chemical Peels (VI Peel)", "reason": "Controlled exfoliation that brightens and evens discoloration", "targets": ["unevenTone", "texture"]}]
        },
        "neck_chest": {
            "sunDamage": [
                {"treatment": "Sciton BBL", "reason": "Clears sun spots and pigmentation on the décolletage", "targets": ["sunDamage", "redness"]},
                {"treatment": "Sciton Halo", "reason": "Deep resurfacing for stubborn chest sun damage", "targets": ["sunDamage", "texture"]}],
            "laxity": [
                {"treatment": "RF Microneedling", "reason": "Tightens and firms crepey skin on the neck and chest", "targets": ["laxity", "texture"]},
                {"treatment": "Sculptra", "reason": "Gradually rebuilds collagen for firmer skin. works on body areas", "targets": ["laxity"]}],
            "redness": [
                {"treatment": "Sciton BBL", "reason": "Reduces redness and flushing on the chest", "targets": ["redness", "sunDamage"]},
                {"treatment": "Sciton Moxi", "reason": "Gentle laser to calm redness", "targets": ["redness", "texture"]}],
            "texture": [
                {"treatment": "Sciton Moxi", "reason": "Gentle resurfacing to smooth the décolletage", "targets": ["texture", "sunDamage"]},
                {"treatment": "Microneedling", "reason": "Stimulates collagen for smoother chest skin", "targets": ["texture", "laxity"]}],
            "wrinkles": [
                {"treatment": "RF Microneedling", "reason": "Smooths neck lines and chest wrinkles with collagen stimulation", "targets": ["wrinkles", "laxity"]},
                {"treatment": "Sciton Moxi", "reason": "Gentle laser resurfacing for fine lines", "targets": ["wrinkles", "texture"]}]
        },
        "hands": {
            "sunDamage": [
                {"treatment": "Sciton BBL", "reason": "Clears age spots and sun damage on the hands", "targets": ["sunDamage"]},
                {"treatment": "Sciton Moxi", "reason": "Gentle laser to brighten and even skin tone", "targets": ["sunDamage", "texture"]}],
            "laxity": [
                {"treatment": "Sculptra", "reason": "Restores volume and collagen to thin, aging hands", "targets": ["laxity", "veins"]},
                {"treatment": "RF Microneedling", "reason": "Tightens and firms skin on the hands", "targets": ["laxity", "texture"]}],
            "texture": [
                {"treatment": "Microneedling", "reason": "Stimulates collagen for smoother skin on the hands", "targets": ["texture", "laxity"]},
                {"treatment": "HydraFacial Customized", "reason": "Deep hydration to restore moisture and glow", "targets": ["texture", "dryness"]}],
            "veins": [
                {"treatment": "Sculptra", "reason": "Adds volume to conceal visible veins and tendons", "targets": ["veins", "laxity"]}],
            "dryness": [
                {"treatment": "HydraFacial Customized", "reason": "Intensive hydration treatment for dry hands", "targets": ["dryness", "texture"]},
                {"treatment": "SkinVive", "reason": "Injectable micro-droplet HA that plumps and moisturizes from within", "targets": ["dryness", "texture"]}]
        },
        "back": {
            "acne": [
                {"treatment": "SaltFacial", "reason": "3-in-1 sea salt exfoliation + ultrasound + LED to clear back acne", "targets": ["acne", "texture"]},
                {"treatment": "Chemical Peels (VI Peel)", "reason": "VI Peel Purify. medical-grade peel to clear breakouts and decongest", "targets": ["acne", "unevenTone"]}],
            "scarring": [
                {"treatment": "Microneedling + PRF", "reason": "Collagen induction with growth factors. amplifies results 40-50%", "targets": ["scarring", "texture"]},
                {"treatment": "RF Microneedling", "reason": "Deeper remodeling for stubborn scarring", "targets": ["scarring", "texture"]},
                {"treatment": "Sciton Halo", "reason": "Resurfacing laser to reduce scar visibility", "targets": ["scarring", "texture"]}],
            "texture": [
                {"treatment": "Microneedling", "reason": "Smooths rough, uneven texture on the back", "targets": ["texture", "scarring"]},
                {"treatment": "SaltFacial", "reason": "Exfoliation and LED for smoother skin", "targets": ["texture", "acne"]}],
            "unevenTone": [
                {"treatment": "Sciton BBL", "reason": "Evens out dark marks and discoloration", "targets": ["unevenTone", "scarring"]},
                {"treatment": "Chemical Peels (VI Peel)", "reason": "Brightens and evens post-inflammatory pigmentation", "targets": ["unevenTone", "texture"]}],
            "hairRemoval": [
                {"treatment": "Laser Hair Removal", "reason": "BBL light targets follicles for permanent reduction. no razor, no ingrowns", "targets": ["hairRemoval"]}]
        },
        "legs": {
            "veins": [
                {"treatment": "Sciton BBL", "reason": "Targets and fades spider veins on the legs", "targets": ["veins"]},
                {"treatment": "Sciton Moxi", "reason": "Gentle laser treatment for vascular concerns", "targets": ["veins", "texture"]}],
            "texture": [
                {"treatment": "Microneedling", "reason": "Smooths bumpy texture and keratosis pilaris", "targets": ["texture"]},
                {"treatment": "SaltFacial", "reason": "Exfoliation to refine rough leg skin", "targets": ["texture", "dryness"]}],
            "sunDamage": [
                {"treatment": "Sciton BBL", "reason": "Clears sun spots and freckling on the legs", "targets": ["sunDamage"]},
                {"treatment": "Sciton Moxi", "reason": "Gentle laser for mild leg sun damage", "targets": ["sunDamage", "texture"]}],
            "hairRemoval": [
                {"treatment": "Laser Hair Removal", "reason": "Long-term BBL light hair reduction for smooth legs. 8-12 sessions", "targets": ["hairRemoval"]}],
            "dryness": [
                {"treatment": "HydraFacial Customized", "reason": "Intensive hydration for dry, dehydrated skin", "targets": ["dryness", "texture"]}]
        }
    }

    tx_map = treatment_maps.get(body_area, treatment_maps["face"])

    # Build recommendations. pick best treatment for top 2-3 concerns, avoid duplicates
    recs = []
    used_treatments = set()
    priority = 1

    for concern_key, concern_data in ranked:
        if priority > 3:
            break
        if concern_data["score"] < 20:
            continue
        options = tx_map.get(concern_key, [])
        for option in options:
            if option["treatment"] not in used_treatments:
                rec = {**option, "priority": priority}
                recs.append(rec)
                used_treatments.add(option["treatment"])
                priority += 1
                break

    # Accessible entry point for face only
    if body_area == "face":
        accessible = {"HydraFacial Clarifying", "HydraFacial Customized", "SaltFacial", "Deep Pore Facial"}
        if not any(t in used_treatments for t in accessible) and not any("HydraFacial" in t for t in used_treatments):
            recs.append({
                "treatment": "HydraFacial Clarifying",
                "reason": "Deep cleanse, exfoliation, and extraction. the perfect entry point to see and feel the difference",
                "targets": ["texture", "pores"],
                "priority": priority
            })

    # Product recommendations (concern-matched from Von & Co product catalog)
    product_map = {
        "wrinkles": {"product": "SkinBetter AlphaRet", "reason": "Retinoid + AHA fused in one molecule. visible results in 4 weeks with less irritation than retinol"},
        "redness": {"product": "Avene Thermal Water", "reason": "Mineral-rich spring water that calms on contact. mist anytime, before serums, after sun, or over makeup"},
        "darkSpots": {"product": "SkinBetter Even Tone", "reason": "No retinol, no hydroquinone. corrects tone in 4-6 weeks and is safe for all skin tones"},
        "texture": {"product": "ZO Complexion Renewal Pads", "reason": "Glycolic + salicylic pre-soaked pads. like a mini facial in a jar, use 2-3x per week at night"},
        "pores": {"product": "ZO Complexion Renewal Pads", "reason": "Glycolic + salicylic acid pads that unclog pores and refine skin texture"},
        "laxity": {"product": "Alastin Restorative Skin Complex", "reason": "TriHex technology clears old collagen and signals new production. like recycling for your skin"},
        "sunDamage": {"product": "SkinBetter Even Tone", "reason": "Alpha-arbutin + peptides fade sun damage spots without irritation"},
        "unevenTone": {"product": "ZO 10% Vitamin C", "reason": "L-ascorbic acid brightens overnight. your morning shield against dullness and free radicals"},
        "dryness": {"product": "Hydrinity Renewing HA Serum", "reason": "Multi-weight hyaluronic acid pulls moisture into every skin layer. apply to damp skin for max absorption"},
        "acne": {"product": "ZO Complexion Renewal Pads", "reason": "Glycolic + salicylic acid unclog pores. pair with mineral SPF since chemical SPFs can worsen breakouts"},
        "scarring": {"product": "Alastin Restorative Skin Complex", "reason": "TriHex+ technology clears damaged collagen and stimulates new production for smoother skin"},
        "veins": {"product": "Alastin Restorative Skin Complex", "reason": "TriHex technology builds collagen and helps conceal visible veins through thicker, firmer skin"},
        "hairRemoval": {"product": "Colorescience Face Shield SPF 50", "reason": "100% mineral SPF that blocks UV + visible light. essential protection for treated skin"}
    }

    product_recs = []
    used_products = set()
    for concern_key, concern_data in ranked:
        if concern_data["score"] < 15:
            continue
        if concern_key in product_map:
            pr = product_map[concern_key]
            if pr["product"] not in used_products:
                product_recs.append(pr)
                used_products.add(pr["product"])
        if len(product_recs) >= 2:
            break
    # Always add SPF
    spf = {"product": "Colorescience Face Shield SPF 50", "reason": "100% mineral SPF that blocks UV + visible light. non-negotiable for protecting your skin and treatment results"}
    if spf["product"] not in used_products:
        product_recs.append(spf)

    # Suggested combo play
    moderate_count = sum(1 for _, cd in ranked if cd["score"] >= 40)
    suggested_combo = None
    if body_area == "face":
        top_concern = ranked[0][0] if ranked else None
        if moderate_count >= 3:
            if top_concern in ("wrinkles", "laxity", "texture"):
                suggested_combo = "Full Rejuvenation: Sciton Halo + Sculptra + Toxin maintenance plan. comprehensive renewal that addresses multiple concerns simultaneously"
            elif top_concern in ("darkSpots", "sunDamage", "redness"):
                suggested_combo = "Hero Combo: Sciton BBL + Halo. Surface clearing + deep remodeling for dramatic results in fewer sessions"
            else:
                suggested_combo = "Anti-Aging Power: Toxin (wait 14 days) + Filler + Sciton Moxi laser series. the complete rejuvenation stack"
        elif moderate_count >= 1:
            suggested_combo = "Glow-Up Package: HydraFacial Elite + SkinVive + daily SPF. an accessible path to radiant, hydrated skin"

    # Build a natural summary
    area_labels = {"face": "face", "neck_chest": "neck and chest", "hands": "hands", "back": "back", "legs": "legs"}
    area_label = area_labels.get(body_area, "skin")
    concern_names_map = {
        "wrinkles": "fine lines", "redness": "redness", "darkSpots": "sun damage and pigmentation",
        "texture": "skin texture", "pores": "pore visibility", "laxity": "skin firmness",
        "sunDamage": "sun damage", "veins": "visible veins", "scarring": "scarring",
        "hairRemoval": "unwanted hair", "acne": "breakouts", "dryness": "dryness", "unevenTone": "uneven tone"
    }
    top_two = [concern_names_map.get(r[0], r[0]) for r in ranked[:2] if r[1]["score"] > 20]
    top_treatments = [r["treatment"] for r in recs[:2]]

    if top_two and top_treatments:
        summary = f"Your {area_label} shows some signs of {' and '.join(top_two)}. A combination of {' and '.join(top_treatments)} would help restore a smoother, more youthful appearance. We'd love to start with a complimentary VISIA consultation to map your skin in clinical detail."
    else:
        summary = f"Your {area_label} looks great overall! A targeted maintenance plan can help keep things looking their best. A complimentary VISIA consultation would give us even deeper insights."

    return {
        "overallScore": base_score,
        "skinAge": random.choice(skin_ages) if body_area == "face" else None,
        "concerns": concerns,
        "recommendations": recs,
        "productRecommendations": product_recs,
        "suggestedCombo": suggested_combo,
        "summary": summary
    }


def allowed_file(filename):
    """Check if file type is allowed"""
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Lead storage (in-memory for now, can upgrade to DB/CRM later)
leads = []

@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "mode": MODE
    })


@app.route("/api/lead", methods=["POST"])
def capture_lead():
    """Capture lead information from the lead gate form"""
    try:
        data = request.get_json()
        name = data.get("name", "").strip()
        email = data.get("email", "").strip()
        phone = data.get("phone", "").strip()

        if not name or not email:
            return jsonify({"error": "Name and email are required"}), 400

        lead = {
            "name": name,
            "email": email,
            "phone": phone,
            "timestamp": datetime.now().isoformat(),
            "ip": request.headers.get('X-Forwarded-For', request.remote_addr)
        }
        leads.append(lead)
        print(f"[LEAD] {lead['timestamp']} | {name} | {email} | {phone}")
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/leads", methods=["GET"])
def get_leads():
    """View captured leads (protected by simple token)"""
    token = request.args.get("token", "")
    expected = os.getenv("ADMIN_TOKEN", "vonco-admin-2026")
    if token != expected:
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify({"leads": leads, "total": len(leads)})


@app.route("/api/report", methods=["POST"])
def generate_report():
    """Generate a branded Von & Co treatment plan one-pager as HTML"""
    try:
        data = request.get_json()
        name = data.get("name", "Guest")
        analysis = data.get("analysis", {})

        overall = analysis.get("overallScore", "...")
        skin_age = analysis.get("skinAge", "...")
        summary = analysis.get("summary", "")
        concerns = analysis.get("concerns", {})
        recs = analysis.get("recommendations", [])

        concern_labels = {
            "wrinkles": "Wrinkles & Fine Lines",
            "redness": "Redness & Rosacea",
            "darkSpots": "Dark Spots & Hyperpigmentation",
            "texture": "Skin Texture & Smoothness",
            "pores": "Pore Size & Visibility"
        }

        # Build concerns HTML rows
        concern_rows = ""
        for key, label in concern_labels.items():
            c = concerns.get(key, {})
            score = c.get("score", 0)
            severity = c.get("severity", "none").upper()
            desc = c.get("description", "")
            color = "#c0c8a9" if score <= 30 else "#c68c2f" if score <= 60 else "#c58b7a"
            concern_rows += f"""
            <tr>
                <td style="padding:10px 12px; border-bottom:1px solid #ebe9e4; font-weight:500; color:#1d1d1b;">{label}</td>
                <td style="padding:10px 12px; border-bottom:1px solid #ebe9e4; text-align:center;">
                    <span style="display:inline-block; padding:3px 10px; border-radius:12px; background:{color}20; color:{color}; font-weight:600; font-size:0.85em;">{severity}</span>
                </td>
                <td style="padding:10px 12px; border-bottom:1px solid #ebe9e4; color:#555; font-size:0.9em;">{desc}</td>
            </tr>"""

        # Build recommendations HTML
        rec_items = ""
        for i, rec in enumerate(recs, 1):
            treatment = rec.get("treatment", "")
            reason = rec.get("reason", "")
            rec_items += f"""
            <div style="display:flex; gap:12px; margin-bottom:14px; align-items:flex-start;">
                <div style="min-width:28px; height:28px; background:#516b62; color:#fff; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:600; font-size:0.85em;">{i}</div>
                <div>
                    <div style="font-weight:600; color:#1d1d1b; margin-bottom:2px;">{treatment}</div>
                    <div style="color:#555; font-size:0.9em;">{reason}</div>
                </div>
            </div>"""

        html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600&family=Fira+Sans:wght@300;400;500;600&display=swap');
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Fira Sans',sans-serif; color:#1d1d1b; background:#fff; }}
  .page {{ max-width:800px; margin:0 auto; padding:40px; }}
</style>
</head><body>
<div class="page">
  <!-- Header -->
  <div style="background:#516b62; padding:30px 35px; border-radius:12px 12px 0 0; text-align:center;">
    <div style="font-family:'Cormorant Garamond',serif; font-size:1.8em; color:#fff; font-weight:500;">Von & Co Aesthetics</div>
    <div style="color:#ebe4d5; font-size:0.9em; margin-top:4px; letter-spacing:1px;">PERSONALIZED SKIN ANALYSIS REPORT</div>
  </div>

  <!-- Patient Info Bar -->
  <div style="background:#ebe9e4; padding:16px 35px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
    <div><span style="color:#888; font-size:0.85em;">Prepared for </span><strong>{name}</strong></div>
    <div style="display:flex; gap:24px;">
      <div><span style="color:#888; font-size:0.85em;">Overall Score: </span><strong style="color:#516b62; font-size:1.1em;">{overall}/100</strong></div>
      <div><span style="color:#888; font-size:0.85em;">Est. Skin Age: </span><strong style="color:#516b62;">{skin_age}</strong></div>
    </div>
  </div>

  <!-- Summary -->
  <div style="padding:25px 35px; border-left:3px solid #c1a890; margin:25px 0; background:#faf8f5; border-radius:0 8px 8px 0;">
    <div style="font-size:0.8em; text-transform:uppercase; letter-spacing:1.5px; color:#c1a890; margin-bottom:6px;">Summary</div>
    <p style="color:#333; line-height:1.6; font-size:0.95em;">{summary}</p>
  </div>

  <!-- Concerns Table -->
  <div style="margin:0 0 30px;">
    <div style="font-family:'Cormorant Garamond',serif; font-size:1.4em; color:#516b62; margin-bottom:14px; padding:0 0 8px; border-bottom:2px solid #ebe9e4;">Skin Analysis Results</div>
    <table style="width:100%; border-collapse:collapse;">
      <tr style="background:#f5f3f0;">
        <th style="padding:10px 12px; text-align:left; font-size:0.85em; text-transform:uppercase; letter-spacing:0.5px; color:#888;">Concern</th>
        <th style="padding:10px 12px; text-align:center; font-size:0.85em; text-transform:uppercase; letter-spacing:0.5px; color:#888;">Level</th>
        <th style="padding:10px 12px; text-align:left; font-size:0.85em; text-transform:uppercase; letter-spacing:0.5px; color:#888;">Assessment</th>
      </tr>
      {concern_rows}
    </table>
  </div>

  <!-- Recommendations -->
  <div style="margin:0 0 30px;">
    <div style="font-family:'Cormorant Garamond',serif; font-size:1.4em; color:#516b62; margin-bottom:14px; padding:0 0 8px; border-bottom:2px solid #ebe9e4;">Recommended Treatments</div>
    {rec_items}
  </div>

  <!-- Promo + Club -->
  <div style="display:flex; gap:16px; margin:0 0 25px; flex-wrap:wrap;">
    <div style="flex:1; min-width:200px; background:linear-gradient(135deg,rgba(198,140,47,0.08),rgba(193,168,144,0.12)); border:1px solid #c68c2f; border-radius:10px; padding:18px 20px; text-align:center;">
      <div style="font-weight:600; color:#516b62; margin-bottom:4px;">New Guest Offer</div>
      <div style="font-size:0.88em; color:#555;">15% off your first visit. including treatments and same-day skincare.</div>
    </div>
    <div style="flex:1; min-width:200px; background:#516b62; border-radius:10px; padding:18px 20px; text-align:center; color:#fff;">
      <div style="font-weight:600; margin-bottom:4px;">The Club</div>
      <div style="font-size:0.88em; color:#ebe4d5;">Members save up to 20% year-round plus exclusive perks.</div>
    </div>
  </div>

  <!-- CTA Footer -->
  <div style="background:#516b62; padding:24px 35px; border-radius:0 0 12px 12px; text-align:center; color:#fff;">
    <div style="font-family:'Cormorant Garamond',serif; font-size:1.3em; margin-bottom:6px;">Ready to Start Your Journey?</div>
    <div style="font-size:0.9em; color:#ebe4d5; margin-bottom:8px;">Book a complimentary consultation with our expert providers.</div>
    <div style="font-size:1em; font-weight:600; color:#c1a890;">239.799.4866 &nbsp;|&nbsp; vonandcoaesthetics.com</div>
    <div style="font-size:0.8em; color:rgba(255,255,255,0.5); margin-top:8px;">Naples, FL &nbsp;•&nbsp; Open 7 Days a Week</div>
  </div>

  <div style="text-align:center; margin-top:16px; font-size:0.75em; color:#aaa;">
    This AI analysis is a preview of the personalized care at Von & Co Aesthetics and does not replace a professional consultation.
  </div>
</div>
</body></html>"""

        return html, 200, {'Content-Type': 'text/html'}
    except Exception as e:
        return jsonify({"error": str(e)}), 500


BODY_AREA_PROMPTS = {
    "face": "This is a photo of the patient's FACE. Analyze for wrinkles, redness/rosacea, dark spots, texture, and pore visibility.",
    "neck_chest": "This is a photo of the patient's NECK AND CHEST/DÉCOLLETAGE. Analyze for sun damage/photoaging, skin laxity/crepiness, redness, texture, and wrinkles/lines. Use concern keys: sunDamage, laxity, redness, texture, wrinkles. Do NOT recommend Botox or dermal fillers for this area. focus on lasers (BBL, Halo, Moxi), RF Microneedling, Sculptra, and Microneedling.",
    "hands": "This is a photo of the patient's HANDS. Analyze for sun damage/age spots, skin laxity/thinning, texture, visible veins, and dryness. Use concern keys: sunDamage, laxity, texture, veins, dryness. Do NOT recommend Botox for hands. Sculptra and fillers can be appropriate for volume loss.",
    "back": "This is a photo of the patient's BACK. Analyze for acne/breakouts, scarring, texture, uneven tone, and unwanted hair. Use concern keys: acne, scarring, texture, unevenTone, hairRemoval. Do NOT recommend Botox, dermal fillers, or Sculptra for the back. Focus on SaltFacial, VI Peel, Microneedling, RF Microneedling, BBL, Halo, and Laser Hair Removal.",
    "legs": "This is a photo of the patient's LEGS. Analyze for spider veins/vascularity, texture, sun damage, unwanted hair, and dryness. Use concern keys: veins, texture, sunDamage, hairRemoval, dryness. Do NOT recommend Botox, dermal fillers, or Sculptra for legs. Focus on BBL, Moxi, Microneedling, SaltFacial, and Laser Hair Removal."
}

def build_user_prompt(user_age=None, body_area="face"):
    """Build the user prompt for the Claude vision API, including age and body area."""
    area_instruction = BODY_AREA_PROMPTS.get(body_area, BODY_AREA_PROMPTS["face"])
    base = f"Please analyze this skin image and provide a detailed assessment in the JSON format specified. {area_instruction}"
    if user_age and body_area == "face":
        base += f" The patient is {user_age} years old. compare their estimated skin age to their actual age and reference this in your summary."
    elif user_age:
        base += f" The patient is {user_age} years old."
    if body_area != "face":
        base += " Set skinAge to null since this is not a facial analysis."
    return base


class _SkipToScoreCorrection(Exception):
    """Internal: used to skip Step 2 when falling back to single-model Claude vision."""
    def __init__(self, analysis):
        self.analysis = analysis


def _apply_score_correction(analysis):
    """Server-side score correction: spread clustered concern scores and recalculate overallScore."""
    try:
        concerns = analysis.get("concerns", {})
        if concerns and isinstance(concerns, dict):
            concern_items = [(k, v) for k, v in concerns.items() if isinstance(v, dict) and "score" in v]
            raw_scores = [v["score"] for k, v in concern_items]

            if raw_scores:
                avg_raw = sum(raw_scores) / len(raw_scores)
                score_range = max(raw_scores) - min(raw_scores)

                print(f"  [Score Debug] Raw concern scores: {dict((k, v['score']) for k, v in concern_items)}")
                print(f"  [Score Debug] Avg={avg_raw:.1f}, Range={score_range}, Min={min(raw_scores)}, Max={max(raw_scores)}")

                if score_range < 25 and len(raw_scores) >= 4:
                    print(f"  [Score Fix] Scores too clustered (range={score_range}). Spreading...")
                    sorted_items = sorted(concern_items, key=lambda x: x[1]["score"])
                    n = len(sorted_items)
                    spread_low = max(3, avg_raw - 25)
                    spread_high = min(88, avg_raw + 25)
                    for i, (key, val) in enumerate(sorted_items):
                        if n > 1:
                            t = i / (n - 1)
                        else:
                            t = 0.5
                        new_score = int(spread_low + t * (spread_high - spread_low))
                        new_score = max(2, min(92, new_score + random.randint(-3, 3)))
                        concerns[key]["score"] = new_score
                        if new_score <= 25:
                            concerns[key]["severity"] = "none"
                        elif new_score <= 45:
                            concerns[key]["severity"] = "mild"
                        elif new_score <= 65:
                            concerns[key]["severity"] = "moderate"
                        else:
                            concerns[key]["severity"] = "significant"

                    print(f"  [Score Fix] Spread scores: {dict((k, concerns[k]['score']) for k, v in concern_items)}")

                final_scores = [concerns[k]["score"] for k, _ in concern_items]
                avg_concern = sum(final_scores) / len(final_scores)
                jitter = random.randint(-2, 2)
                calculated_score = max(38, min(95, int(100 - avg_concern + jitter)))

                if 63 <= calculated_score <= 73:
                    if calculated_score <= 68:
                        calculated_score = 62 - random.randint(0, 3)
                    else:
                        calculated_score = 74 + random.randint(0, 3)
                    print(f"  [Score Fix] Score was in banned zone, pushed to {calculated_score}")

                original = analysis.get("overallScore", "?")
                print(f"  [Score Fix] Model said {original}, concerns avg {avg_concern:.0f}, final overallScore = {calculated_score}")
                analysis["overallScore"] = calculated_score

    except Exception as e:
        print(f"  [Score Fix] Error in recalculation, keeping model score: {e}")


def _sanitize_response(analysis):
    """Sanitize em dashes and capitalize skinAge."""
    import re

    def strip_em_dashes(obj):
        if isinstance(obj, str):
            return obj.replace("\u2014", ", ").replace("\u2013", " to ")
        elif isinstance(obj, dict):
            return {k: strip_em_dashes(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [strip_em_dashes(item) for item in obj]
        return obj

    analysis = strip_em_dashes(analysis)

    if "skinAge" in analysis and isinstance(analysis["skinAge"], str):
        def smart_title(s):
            words = s.split()
            skip = {"to", "and", "or"}
            result = []
            for i, w in enumerate(words):
                if re.match(r'^\d', w):
                    result.append(w)
                elif w.lower() in skip and i > 0:
                    result.append(w.lower())
                else:
                    result.append(w.capitalize())
            return " ".join(result)
        analysis["skinAge"] = smart_title(analysis["skinAge"])

    return analysis

@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Analyze skin from uploaded image"""
    
    # Rate limiting
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    if not check_rate_limit(client_ip):
        return jsonify({"error": "You've reached the analysis limit. Please try again in an hour, or book a consultation for a full VISIA assessment."}), 429

    # Check if file is in request
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files["image"]
    
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed. Accepted: jpg, jpeg, png, webp"}), 400
    
    # Check file size (10MB max)
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to start
    
    if file_size > 10 * 1024 * 1024:  # 10MB
        return jsonify({"error": "File too large. Maximum size is 10MB"}), 400
    
    # Read file bytes
    image_bytes = file.read()
    
    # Determine media type
    filename_lower = file.filename.lower()
    if filename_lower.endswith('.png'):
        media_type = "image/png"
    elif filename_lower.endswith('.webp'):
        media_type = "image/webp"
    else:  # jpg, jpeg
        media_type = "image/jpeg"
    
    body_area = request.form.get("body_area", "face")

    # Server-side age gate: reject minors
    user_age = request.form.get("age")
    if user_age:
        try:
            age_int = int(user_age)
            if age_int < 18:
                return jsonify({
                    "rejected": True,
                    "reason": "Our skin analysis and treatment recommendations are designed for adults (18+). Medical aesthetic treatments are not appropriate for minors."
                }), 422
        except ValueError:
            pass  # Non-numeric age, let Claude handle it

    # Demo mode
    if not LIVE_MODE:
        analysis = generate_demo_analysis(body_area)
        return jsonify(analysis)

    # Live mode - dual-model pipeline: Gemini vision → Claude analysis
    try:
        t_start = time.time()

        # Encode image to base64
        image_base64 = base64.standard_b64encode(image_bytes).decode("utf-8")

        # ── STEP 1: Gemini Flash — fast image description ──
        if gemini_model:
            gemini_prompt = (
                "You are an expert dermatologist examining a patient photo. "
                "Describe what you see in exhaustive clinical detail. Include:\n"
                "- Skin tone, texture, and overall complexion\n"
                "- Any visible wrinkles, fine lines, and their location/depth\n"
                "- Redness, inflammation, broken capillaries, or rosacea signs\n"
                "- Dark spots, hyperpigmentation, melasma, sun damage\n"
                "- Pore size and visibility, especially on nose/cheeks/forehead\n"
                "- Skin laxity, sagging, jowling, loss of volume\n"
                "- Acne, scarring, texture irregularities\n"
                "- Under-eye concerns (dark circles, hollows, puffiness)\n"
                "- Overall skin health impression and estimated skin age\n"
                "- Any non-skin observations (this is not a skin photo, foreign object, etc.)\n\n"
                "Be extremely thorough. Your description will be used by another AI "
                "to generate a structured skin assessment, so include every relevant detail. "
                "Do NOT provide treatment recommendations — just describe what you see."
            )
            import PIL.Image
            image_pil = PIL.Image.open(BytesIO(image_bytes))
            gemini_response = gemini_model.generate_content(
                [gemini_prompt, image_pil],
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=1500,
                    temperature=0.2,
                )
            )
            skin_description = gemini_response.text.strip()
            t_gemini = time.time()
            print(f"  [Pipeline] Gemini vision completed in {t_gemini - t_start:.1f}s ({len(skin_description)} chars)")
        else:
            # Fallback: if no Gemini key, use Claude vision directly (slower)
            print("  [Pipeline] No Gemini key — falling back to Claude vision")
            fallback_response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=2500,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_base64,
                                },
                            },
                            {
                                "type": "text",
                                "text": build_user_prompt(request.form.get("age"), body_area)
                            }
                        ],
                    }
                ],
            )
            response_text = fallback_response.content[0].text.strip()
            # Skip Step 2, jump straight to JSON parsing
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            analysis = json.loads(response_text)
            # Jump past Step 2 to score correction
            raise _SkipToScoreCorrection(analysis)

        # ── STEP 2: Claude Sonnet — structured JSON analysis (text-only, fast) ──
        age_context = ""
        user_age_val = request.form.get("age")
        if user_age_val and body_area == "face":
            age_context = f" The patient is {user_age_val} years old. Compare their estimated skin age to their actual age."
        elif user_age_val:
            age_context = f" The patient is {user_age_val} years old."

        area_instruction = BODY_AREA_PROMPTS.get(body_area, BODY_AREA_PROMPTS["face"])
        skin_age_note = " Set skinAge to null since this is not a facial analysis." if body_area != "face" else ""

        claude_prompt = (
            f"A dermatologist has examined a patient's skin photo and provided this detailed description:\n\n"
            f"--- CLINICAL OBSERVATION ---\n{skin_description}\n--- END OBSERVATION ---\n\n"
            f"Based on this clinical observation, provide a detailed skin assessment in the JSON format specified. "
            f"{area_instruction}{age_context}{skin_age_note}"
        )

        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2500,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": claude_prompt
                }
            ],
        )
        t_claude = time.time()
        print(f"  [Pipeline] Claude analysis completed in {t_claude - (t_gemini if gemini_model else t_start):.1f}s")
        print(f"  [Pipeline] Total pipeline: {t_claude - t_start:.1f}s")

        # Extract response text
        response_text = response.content[0].text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Parse JSON response
        analysis = json.loads(response_text)

        # Check if image was rejected (non-skin photo)
        if analysis.get("rejected"):
            return jsonify(analysis), 422

        # Apply score correction and sanitization
        _apply_score_correction(analysis)
        analysis = _sanitize_response(analysis)

        return jsonify(analysis)
    
    except _SkipToScoreCorrection as skip:
        # Fallback path: Claude vision was used directly (no Gemini key)
        # analysis is already parsed JSON, jump to score correction + return
        analysis = skip.analysis
        if analysis.get("rejected"):
            return jsonify(analysis), 422
        # Score correction and sanitization are duplicated here for the fallback path
        # (same logic as below the normal path — kept in sync)
        _apply_score_correction(analysis)
        analysis = _sanitize_response(analysis)
        return jsonify(analysis)

    except json.JSONDecodeError as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Invalid JSON response from AI: {str(e)}"}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@app.route("/")
def serve_index():
    """Serve the main index.html"""
    return send_from_directory(str(PUBLIC_DIR), "index.html")

@app.route("/<path:path>")
def serve_static(path):
    """Serve static files from public directory"""
    file_path = PUBLIC_DIR / path

    # Security check - prevent directory traversal
    try:
        resolved = file_path.resolve()
        if not str(resolved).startswith(str(PUBLIC_DIR.resolve())):
            return "Not found", 404
    except ValueError:
        return "Not found", 404

    if file_path.is_file():
        return send_from_directory(str(PUBLIC_DIR), path)

    # SPA fallback
    return send_from_directory(str(PUBLIC_DIR), "index.html")


def print_startup_banner():
    """Print a nice startup banner"""
    banner = f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║     Von & Co Aesthetics - Skin Analyzer Backend              ║
    ║     Flask Server for AI-Powered Skin Analysis                ║
    ╚══════════════════════════════════════════════════════════════╝
    
    Mode: {MODE.upper()}
    {f"Anthropic Key: {'*' * 10}{API_KEY[-10:] if API_KEY else 'Not configured'}" if LIVE_MODE else "Demo Mode - No API Key"}
    Gemini: {"Enabled (gemini-2.0-flash)" if gemini_model else "Not configured - using Claude vision fallback"}
    Pipeline: {"Gemini vision -> Claude analysis (fast)" if gemini_model else "Claude vision + analysis (slower)"}
    Debug: {DEBUG}
    Port: {PORT}
    
    Server running at: http://localhost:{PORT}
    Health check: http://localhost:{PORT}/api/health
    
    Ready to analyze skin! 🔬
    """
    print(banner)


if __name__ == "__main__":
    print_startup_banner()
    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=DEBUG
    )
