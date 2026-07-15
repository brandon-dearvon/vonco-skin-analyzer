const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const { chromium, webkit } = require('playwright');

const baseUrl = process.env.TEST_BASE_URL || 'http://127.0.0.1:5004';
const photoDir = path.resolve(
  process.env.REAL_PHOTO_DIR || 'work/test-images/real-world-10',
);
const artifactDir = path.resolve(
  process.env.REAL_PHOTO_ARTIFACT_DIR || 'work/qa/real-photo-matrix',
);
const ledgerPath = path.join(artifactDir, 'real-photo-matrix-ledger.json');
const resultDeadlineMs = 75000;
const pexelsLicenseUrl = 'https://www.pexels.com/legal-pages/license/';
const harnessVersion = '2026-07-15-evidence-gate-v8';
const expectedBuildFingerprint = String(process.env.TEST_BUILD_FINGERPRINT || '').trim();
const runtimeSourceFiles = Object.freeze([
  'server.py',
  'public/index.html',
  'public/arsenica-regular.otf',
  'public/logo.png',
  'public/logo_clean.png',
  'public/logo_white.png',
]);
const sourceBuildFingerprint = (() => {
  const hash = crypto.createHash('sha256');
  for (const relativePath of runtimeSourceFiles) {
    const absolutePath = path.resolve(__dirname, '..', relativePath);
    hash.update(relativePath);
    hash.update('\0');
    hash.update(fs.readFileSync(absolutePath));
    hash.update('\0');
  }
  return hash.digest('hex');
})();
const expectedRuntime = Object.freeze({
  model: 'gemini-3.1-pro-preview',
  thinkingLevel: 'HIGH',
  totalBudgetMs: 70000,
  hedgeDelayMs: 15000,
  maxOutputTokens: 32768,
});
const caseFilter = new Set(
  String(process.env.REAL_CASE_FILTER || '')
    .split(',')
    .map(value => value.trim())
    .filter(Boolean),
);

const concernKeys = {
  face: ['darkSpots', 'laxity', 'pores', 'redness', 'sunDamage', 'texture', 'unevenTone', 'wrinkles'],
  neck_chest: ['laxity', 'redness', 'sunDamage', 'texture', 'wrinkles'],
  hands: ['dryness', 'laxity', 'sunDamage', 'texture', 'veins'],
  back: ['acne', 'hairRemoval', 'scarring', 'texture', 'unevenTone'],
  legs: ['dryness', 'hairRemoval', 'sunDamage', 'texture', 'veins'],
};

const concernGoalLabels = {
  wrinkles: 'visible lines',
  redness: 'visible redness',
  darkSpots: 'visible pigmentation',
  texture: 'visible texture',
  pores: 'pore visibility',
  laxity: 'visible contour softness',
  sunDamage: 'visible sun-exposure signs',
  unevenTone: 'visible tone variation',
  acne: 'visible surface congestion',
  scarring: 'visible textural marks',
  hairRemoval: 'visible hair growth',
  veins: 'visible vascularity',
  dryness: 'visible surface dryness',
};

const treatmentTargets = {
  Botox: ['wrinkles'],
  Dysport: ['wrinkles'],
  Xeomin: ['wrinkles'],
  Microneedling: ['wrinkles', 'texture', 'pores', 'laxity', 'scarring'],
  'RF Microneedling': ['wrinkles', 'texture', 'pores', 'laxity', 'scarring'],
  'Dermal Fillers': ['wrinkles', 'laxity'],
  Sculptra: ['wrinkles', 'laxity'],
  'HydraFacial Clarifying': ['redness', 'pores', 'texture', 'acne', 'unevenTone'],
  'HydraFacial Customized': ['redness', 'pores', 'texture', 'dryness', 'unevenTone'],
  'HydraFacial Elite': ['texture', 'dryness', 'unevenTone'],
  SaltFacial: ['pores', 'texture', 'dryness', 'acne', 'unevenTone'],
  SkinVive: ['texture', 'dryness'],
  'Sciton Moxi': ['wrinkles', 'darkSpots', 'texture', 'sunDamage', 'unevenTone'],
  'Microneedling + PRF': ['wrinkles', 'texture', 'pores', 'laxity', 'scarring'],
  'Deep Pore Facial': ['pores', 'texture', 'acne'],
  'Signature Facial': ['redness', 'texture', 'dryness', 'unevenTone'],
  'Anti-Aging Facial': ['wrinkles', 'texture', 'dryness', 'laxity'],
  'Sciton BBL': ['redness', 'darkSpots', 'sunDamage', 'unevenTone', 'veins', 'acne'],
  'Sciton Halo': [
    'wrinkles', 'darkSpots', 'texture', 'pores', 'laxity', 'sunDamage', 'unevenTone', 'scarring',
  ],
  'Chemical Peels': ['wrinkles', 'darkSpots', 'texture', 'pores', 'sunDamage', 'unevenTone', 'acne', 'scarring'],
  'Laser Hair Removal': ['hairRemoval'],
};

const treatmentReasonGoalFamilies = [
  [/\b(?:redness|pinkness|flush|flushing|broken\s+capillaries|vascular\s+redness)\b/i, new Set(['redness'])],
  [/\b(?:body\s+hair|hair\s+growth|hair\s+reduction|stubble|follicles?|follicular)\b/i, new Set(['hairRemoval'])],
  [/\b(?:surface\s+veins?|visible\s+veins?|vascularity)\b/i, new Set(['veins'])],
  [/\b(?:dryness|dry\s+skin|hydration|moisture)\b/i, new Set(['dryness'])],
  [/\b(?:pores?|pore\s+visibility)\b/i, new Set(['pores'])],
  [/\b(?:surface\s+congestion|breakouts?|blemishes?)\b/i, new Set(['acne'])],
  [/\b(?:textural\s+marks?|scar(?:s|ring)?)\b/i, new Set(['scarring'])],
  [/\b(?:laxity|contour\s+softness|crepiness|crepey)\b/i, new Set(['laxity'])],
  [/\b(?:wrinkles?|fine\s+lines?|expression\s+lines?|crow'?s\s+feet|folds?)\b/i, new Set(['wrinkles'])],
  [/\b(?:texture|roughness|bumpiness|skin\s+surface)\b/i, new Set(['texture'])],
  [/\b(?:pigment(?:ation)?|dark\s+spots?|brown\s+spots?|discoloration|sun[- ]exposure|sun\s+signs?|uneven\s+tone|uniform\s+tone|tone\s+variation)\b/i, new Set(['darkSpots', 'sunDamage', 'unevenTone'])],
];

function treatmentReasonOverreaches(reason, actualTargets) {
  const actual = new Set(actualTargets);
  return treatmentReasonGoalFamilies.some(([pattern, family]) => (
    pattern.test(String(reason || ''))
      && [...family].every(target => !actual.has(target))
  ));
}

const areaTreatments = {
  face: new Set([
    'Botox', 'Dysport', 'Xeomin', 'Microneedling', 'RF Microneedling', 'Dermal Fillers',
    'Sculptra', 'HydraFacial Clarifying', 'HydraFacial Customized', 'HydraFacial Elite',
    'SaltFacial', 'SkinVive', 'Sciton Moxi', 'Chemical Peels',
    'Microneedling + PRF', 'Deep Pore Facial', 'Signature Facial', 'Anti-Aging Facial',
    'Sciton BBL', 'Sciton Halo',
  ]),
  neck_chest: new Set([
    'Microneedling', 'RF Microneedling', 'Sculptra', 'Sciton Moxi',
    'Microneedling + PRF', 'Sciton BBL', 'Sciton Halo',
  ]),
  hands: new Set([
    'Microneedling', 'RF Microneedling', 'Sciton Moxi',
    'Chemical Peels', 'Microneedling + PRF', 'Sciton BBL',
    'Sciton Halo',
  ]),
  back: new Set([
    'Microneedling', 'RF Microneedling', 'Chemical Peels',
    'Microneedling + PRF', 'Sciton BBL', 'Sciton Halo',
    'Laser Hair Removal',
  ]),
  legs: new Set([
    'Microneedling', 'Sciton Moxi', 'Sciton BBL', 'Laser Hair Removal',
  ]),
};

const productTargets = {
  'SkinBetter AlphaRet': ['wrinkles', 'texture'],
  'ZO Wrinkle+Texture Repair': ['wrinkles', 'texture'],
  'Avene Thermal Water': ['redness'],
  'Avene Cicalfate+': ['redness', 'dryness', 'texture'],
  'Alastin HydraTint': ['redness', 'sunDamage'],
  'SkinBetter Even Tone': ['darkSpots', 'sunDamage', 'unevenTone'],
  'ISDIN Melaclear Advanced': ['darkSpots', 'sunDamage', 'unevenTone'],
  'ZO 10% Vitamin C': ['darkSpots', 'sunDamage', 'unevenTone'],
  'Hydrinity Vivid Serum': ['dryness', 'unevenTone'],
  'ZO Complexion Renewal Pads': ['pores', 'texture', 'acne', 'unevenTone'],
  'SkinBetter Peel Pads': ['pores', 'texture', 'acne', 'unevenTone'],
  'Alastin Restorative Skin Complex': ['wrinkles', 'laxity', 'texture'],
  'ZO Growth Factor Serum': ['wrinkles', 'laxity', 'texture'],
  'Hydrinity Renewing HA Serum': ['dryness', 'texture'],
  'SkinBetter Trio Moisture': ['dryness', 'texture'],
  'ALASTIN Restorative Eye Treatment': ['wrinkles'],
  'ZO Growth Factor Eye': ['wrinkles'],
  'Colorescience Face Shield SPF 50': ['sunDamage'],
  'ISDIN Eryfotona Actinica': ['sunDamage'],
  'SkinBetter Sunbetter SPF 68': ['sunDamage'],
  'Alastin Skin Nectar': ['dryness', 'laxity', 'texture'],
  'Hydrinity Hyacyn Mist': ['redness', 'dryness', 'texture', 'acne'],
};

const spfProducts = new Set([
  'Colorescience Face Shield SPF 50',
  'ISDIN Eryfotona Actinica',
  'SkinBetter Sunbetter SPF 68',
]);

const drapedScalpHairPattern = /\b(?:scalp\s+hair|hair\s+from\s+(?:the\s+)?head)\b|\b(?:long\s+)?(?:dark\s+)?hair\b[^.;]{0,35}\b(?:drapes?|draped|falls?|falling|lies|lying|rests?|resting)\b[^.;]{0,35}\b(?:over|across|on)\b[^.;]{0,25}\b(?:shoulders?|back)\b|\bhair\b[^.;]{0,25}\b(?:over|across|on)\b[^.;]{0,25}\b(?:shoulders?|back)\b[^.;]{0,35}\b(?:drapes?|draped|falls?|falling|lies|lying|rests?|resting)\b/i;
const qualifyingHairEvidencePatterns = [
  /\bstubble\b/gi,
  /\b(?:visible|distinct|prominent|dark|clearly\s+visible)\b(?:\W+\w+){0,4}\W+\b(?:hair\s+)?follicles?\b/gi,
  /\b(?:hair\s+)?follicles?\b(?:\W+\w+){0,4}\W+\b(?:visible|distinct|prominent|dark|clearly\s+visible)\b/gi,
  /\bfollicular\s+(?:contrast|prominence|visibility|pattern)\b/gi,
  /\b(?:dark|coarse)\b(?:\W+\w+){0,4}\W+\b(?:body\s+hair|hair\s+growth|hairs?)\b/gi,
  /\b(?:body\s+hair|hair\s+growth|hairs?)\b(?:\W+\w+){0,4}\W+\b(?:dark|coarse)\b/gi,
];
const hairEvidenceNegationPattern = /\b(?:no|not|without|lacks?|lacking|lack\s+of|hardly\s+any|barely\s+any|cannot\s+see|can't\s+see|do(?:es)?\s+not\s+show)\b[^.;,]{0,70}$/i;
const hairEvidencePostNegationPattern = /^\s*(?:(?:is|are|was|were|appears?|seems?|looks?)\s+)?(?:not\s+(?:clearly\s+)?(?:visible|present|shown|confirmed)|absent|missing|unclear|unconfirmed|cannot\s+be\s+(?:clearly\s+)?(?:seen|confirmed|verified|established)|can't\s+be\s+(?:clearly\s+)?(?:seen|confirmed|verified|established)|could\s+not\s+be\s+(?:clearly\s+)?(?:seen|confirmed|verified|established)|do(?:es)?\s+not\s+(?:appear\s+)?(?:visible|present|show)|(?:is|are)\s+(?:difficult|hard)\s+to\s+(?:see|confirm|verify))\b/i;

function hasNonnegatedHairEvidence(description) {
  for (const pattern of qualifyingHairEvidencePatterns) {
    for (const match of String(description || '').matchAll(pattern)) {
      const start = match.index || 0;
      const clausePrefix = String(description || '')
        .slice(Math.max(0, start - 90), start)
        .split(/[.;,]/)
        .pop();
      const clauseSuffix = String(description || '')
        .slice(start + match[0].length, start + match[0].length + 90)
        .split(/[.;,]/)[0];
      if (
        !hairEvidenceNegationPattern.test(clausePrefix)
        && !hairEvidencePostNegationPattern.test(clauseSuffix)
      ) return true;
    }
  }
  return false;
}

function supportsLaserHairRemoval(data) {
  const concern = data?.concerns?.hairRemoval;
  if (!concern || !Number.isInteger(concern.score) || concern.score < 41) return false;
  const description = String(concern.description || '').trim();
  return Boolean(
    description
      && !drapedScalpHairPattern.test(description)
      && hasNonnegatedHairEvidence(description)
  );
}

assert.equal(
  supportsLaserHairRemoval({ concerns: { hairRemoval: {
    score: 55,
    description: 'Some very fine natural hair is visible across the area.',
  } } }),
  false,
  'hair-evidence self-check rejects fine natural hair without treatment-relevant evidence',
);
assert.equal(
  supportsLaserHairRemoval({ concerns: { hairRemoval: {
    score: 55,
    description: 'Visible dark follicles and distinct follicular contrast are present.',
  } } }),
  true,
  'hair-evidence self-check accepts moderate visible follicular contrast',
);
for (const description of [
  'Stubble is not visible.',
  'Visible follicles are absent.',
  'Distinct follicular contrast cannot be confirmed.',
]) {
  assert.equal(
    hasNonnegatedHairEvidence(description),
    false,
    `hair-evidence self-check rejects post-phrase negation: ${description}`,
  );
}

const comboRequirements = {
  'Anti-Aging Power': [
    new Set(['Botox', 'Dysport', 'Xeomin']),
    new Set(['Dermal Fillers']),
    new Set(['Sciton Moxi']),
  ],
  'Glow-Up Package': [new Set(['HydraFacial Elite']), new Set(['SkinVive'])],
  'Scar Reduction': [
    new Set(['Chemical Peels']),
    new Set(['Microneedling + PRF']),
  ],
  'Hero Combo': [new Set(['Sciton BBL']), new Set(['Sciton Halo'])],
  'Full Rejuvenation': [
    new Set(['Sciton Halo']),
    new Set(['Sculptra']),
    new Set(['Botox', 'Dysport', 'Xeomin']),
  ],
};

const canonicalGuestCatalogNames = Object.freeze(
  [...new Set([
    ...Object.values(areaTreatments).flatMap(treatments => [...treatments]),
    ...Object.keys(productTargets),
    ...Object.keys(comboRequirements),
  ])].sort((left, right) => right.length - left.length),
);

const preAnalysisLabelPatterns = {
  face: [/^Wrinkles & Fine Lines$/i, /^Visible Redness$/i, /^Dark Spots & Pigment Variation$/i, /^Skin Texture & Smoothness$/i, /^Pore Size & Visibility$/i],
  neck_chest: [/^Visible Sun-Exposure Signs$/i, /^Skin Laxity & Crepiness$/i, /^Visible Redness$/i, /^Skin Texture & Smoothness$/i, /^Wrinkles & Fine Lines$/i],
  hands: [/^Visible Sun-Exposure Signs$/i, /^Skin Laxity & Crepiness$/i, /^Skin Texture & Smoothness$/i, /^Visible Veins$/i, /^Visible Dryness$/i],
  back: [/^Visible Breakouts & Congestion$/i, /^Visible Textural Marks$/i, /^Skin Texture & Smoothness$/i, /^Uneven Skin Tone$/i, /^Unwanted Hair$/i],
  legs: [/^Visible Veins$/i, /^Skin Texture & Smoothness$/i, /^Visible Sun-Exposure Signs$/i, /^Unwanted Hair$/i, /^Visible Dryness$/i],
};

const anatomyFamilyPatterns = {
  face: /\b(?:face|facial|complexion|forehead|temples?|brows?|eyes?|eyelids?|under[- ]eyes?|crow'?s[- ]feet|cheeks?|nose|nasal|lips?|mouth|chin|jaws?|jawlines?|jowls?|nasolabial|marionette|t[- ]zone)\b/i,
  neck_chest: /\b(?:neck|throat|cervical|chest|upper[- ]chest|sternum|décolletage|decolletage|décolleté|decollete|collarbones?|clavicles?)\b/i,
  shoulder: /\b(?:shoulders?|shoulder[- ]blades?)\b/i,
  hands: /\b(?:hands?|fingers?|thumbs?|knuckles?|nails?|cuticles?|wrists?|palms?)\b/i,
  back: /\b(?:back|upper[- ]back|mid[- ]back|lower[- ]back|scapula|scapulae|scapular|spine|spinal|torso|trunk)\b/i,
  legs: /\b(?:legs?|thighs?|knees?|kneecaps?|calf|calves|shins?|ankles?|feet|foot|toes?)\b/i,
  unsupported: /\b(?:arms?|upper[- ]arms?|forearms?|elbows?|abdomen|abdominal|stomach|waist|hips?|buttocks?|glutes?|scalp)\b/i,
};

const allowedAnatomyFamilies = {
  face: new Set(['face']),
  neck_chest: new Set(['neck_chest', 'shoulder']),
  hands: new Set(['hands']),
  back: new Set(['back', 'shoulder']),
  legs: new Set(['legs']),
};

const absenceBasedPositivePattern = /does not stand out|not visible|no visible|\bno\b|absence of|without (?:visible|noticeable|any)|free (?:of|from)|little to no|barely visible|\bminimal(?:ly)?\b|lack of|few visible|skin barrier|well-regulated|canvas for treatment|foundation for rejuvenating treatment|bone and muscular structure|structural integrity|structurally strong|structural support|natural elasticity|skin elasticity|overall skin health|indicating good (?:surface )?hydration|fantastic|incredible|perfect|remarkably|wonderfully|gorgeous|beautiful|lovely|amazing|stunning|foundational support|natural firmness|\bbounce\b/i;

const copyOverclaimPattern = /safe for all skin tones|all skin tones safe|(?:safe for|suitable for) all skin types|revers(?:e|es|ed|ing) years of sun damage|eliminat(?:e|es|ed|ing).{0,80}entirely|permanently reduc(?:e|es|ed|ing)|permanent(?:ly)? reduction|\bflawless\b|\bgold standard\b|\bfantastic\b|\bincredible\b|\bincredibly\b|\bperfect(?:ly)?\b|\bremarkably\b|\b(?:beautiful|gorgeous|lovely|wonderful|amazing|stunning)\b|\bfoundational support\b|\bnatural firmness\b|\b(?:natural )?bounce\b|\binstantly\b|\bpotent\b|\babsolute best\b|\bgo-to\b|(?:caused by|from|due to) collagen depletion|\bhealthy skin\b|(?:healthy|strong|intact) (?:skin )?barrier|(?:excellent|strong|healthy|great) (?:natural )?(?:skin )?elasticity|\bnatural (?:skin )?elasticity\b|\bclinically proven\b|\bguarantee(?:d|s)?\b|makes? skin act younger at the cellular level|(?:amplifies?|boosts?) collagen(?: induction)? (?:by )?40\s*(?:-|to)\s*50%|kills acne bacteria/i;
const malformedCopyPattern = /\brefined(?:calm|clear|even|smooth|strong)\b|\bcomplementary\s+VISIA\b/i;
const copyPolishDefectPattern = /\b(?:great|excellent)\s+way\b|\bvery\s+responsive\b|\bhealthy\s+sheen\b|\bcompletely\s+natural\b|\bsun\s+damage\b|\blight\s+based\b|\bstrawberry\s+legs\b|\bdo(?:es)? not stand out(?: prominently)?\b/i;

assert.match('The surface looks refinedstrong.', malformedCopyPattern, 'copy guard catches joined refinedstrong');
assert.match('Add a complementary VISIA scan.', malformedCopyPattern, 'copy guard catches complementary VISIA typo');
assert.match('The surface looks incredibly smooth.', copyOverclaimPattern, 'copy guard catches incredibly smooth hype');
assert.match('This is a great way to begin.', copyPolishDefectPattern, 'copy guard catches canned great-way copy');
assert.match('Fine lines do not stand out prominently.', copyPolishDefectPattern, 'copy guard catches negated stand-out copy');

const diagnosticClaimPattern = /(?:\b(?:you have|we see|there (?:appears?|seems?) to be)\s+(?:visible\s+|possible\s+)?(?:rosacea|melasma|acne|dermatitis|eczema|psoriasis|keratosis pilaris|skin cancer|melanoma|malignan(?:t|cy)|basal cell|squamous cell)\b|\b(?:i see|likely)\s+(?:rosacea|melasma|acne|dermatitis|eczema|psoriasis|keratosis pilaris|skin cancer|melanoma|malignan(?:t|cy)|basal cell|squamous cell)\b|\b(?:it|that|those\s+(?:bumps?|spots?|patches?)|these\s+(?:bumps?|spots?|patches?)|the\s+(?:bumps?|spots?|rash|patches?))\s+(?:is|are|looks? like|appears? to be|seems? to be)\s+(?:a\s+|an\s+)?(?:rosacea|melasma|acne|dermatitis|eczema|psoriasis|keratosis pilaris|skin cancer|melanoma|malignan(?:t|cy)|basal cell|squamous cell)\b|\b(?:your (?:skin|photo|appearance|complexion)|the (?:photo|image|appearance|area)|this(?: (?:area|appearance|photo))?|visible (?:skin|surface))\s+(?:has|shows?|reveals?|suggests?|looks? like|appears? to (?:have|be)|is (?:consistent with|suggestive of|likely)|may be|could be)\s+(?:a\s+|an\s+|visible\s+|possible\s+|likely\s+)?(?:rosacea|melasma|acne|dermatitis|eczema|psoriasis|keratosis pilaris|skin cancer|melanoma|malignan(?:t|cy)|basal cell|squamous cell)\b|\b(?:rosacea|melasma|acne|dermatitis|eczema|psoriasis|keratosis pilaris|skin cancer|melanoma|malignan(?:t|cy)|basal cell|squamous cell)\b.{0,40}\b(?:is|are|appears?|seems?)\s+(?:visible|present|likely|apparent|evident|shown)\b|\bdiagnos(?:e|es|ed|ing|is|tic)\b.{0,80}\b(?:rosacea|melasma|acne|dermatitis|eczema|psoriasis|keratosis pilaris|skin cancer|melanoma|malignan(?:t|cy)|basal cell|squamous cell)\b)/i;
const medicalConditionTermPattern = /\b(?:rosacea|melasma|acne|dermatitis|eczema|psoriasis|keratosis pilaris|skin cancer|melanoma|malignan(?:t|cy)|basal cell|squamous cell)\b/i;
const unsupportedAppearanceLabelPattern = /\b(?:hyper[- ]?pigmentation|photo[- ]?aging|scarr(?:ing|ed)?|scars?|dehydrat(?:ion|ed))\b/i;
const unsupportedPhotoCauseOrHistoryPattern = /(?:\b(?:past|prior|previous|historical|chronic|cumulative|routine|regular|frequent|repeated|prolonged|long[- ]term|lifetime|habitual|ongoing)\s+(?:unprotected\s+)?(?:sun|uv|environmental)\s+exposure\b|\b(?:years? of|history of|routine of)\s+(?:unprotected\s+)?(?:sun|uv)\s+exposure\b|\b(?:past|prior|previous|chronic|cumulative|routine|regular|frequent|repeated|prolonged|long[- ]term|lifetime)\s+exposure\s+to\s+(?:the\s+)?(?:sun|uv)\b|\b(?:sun|uv)\s+exposure\s+(?:history|over time|through(?:out)? the years?)\b|\byears?\s+(?:spent\s+)?(?:in|under)\s+the\s+sun\b|\b(?:likely\s+)?reflect(?:s|ed|ing)?\s+(?:time\s+spent\s+)?(?:in|under)\s+(?:the\s+)?sun\b|\breflect(?:s|ed|ing)?\s+(?:(?:some|mild|visible|general|normal|incidental)\s+)?(?:sun|uv|environmental)\s+exposure\b|\b(?:often\s+)?accompan(?:y|ies|ied|ying)\s+(?:(?:some|mild|visible|general|normal|incidental)\s+)?(?:sun|uv|environmental)\s+exposure\b|\b(?:suggest(?:s|ed|ing)?|indicat(?:e|es|ed|ing))\s+(?:(?:some|mild|visible|general|normal|incidental)\s+)?(?:sun|uv|environmental)\s+exposure\b|\bpoint(?:s|ed|ing)?\s+to\s+(?:(?:some|mild|visible|general|normal|incidental)\s+)?(?:sun|uv|environmental)\s+exposure\b|\b(?:is|are|was|were|appears?|seems?|can\s+be|may\s+be)\s+(?:evidence|indicative|suggestive)\s+of\s+(?:(?:some|mild|visible|general|normal|incidental)\s+)?(?:sun|uv|environmental)\s+exposure\b|\b(?:is|are|was|were|appears?|seems?|can\s+be|may\s+be)\s+(?:associated|linked|related)\s+(?:with|to)\s+(?:(?:some|mild|visible|general|normal|incidental)\s+)?(?:sun|uv|environmental)\s+exposure\b|\b(?:visible\s+)?signs?\s+of\s+(?:sun|uv|environmental)\s+exposure\b|\b(?:past|prior|previous|historical)\s+(?:breakouts?|blemishes?|surface\s+congestion|irritation|marks?|changes?|damage|injur(?:y|ies))\b|\b(?:due to|caused by|from|likely from|result(?:s|ing)? from|stemming from|consistent with)\s+(?:(?:past|prior|chronic|cumulative|routine|regular|frequent|repeated|prolonged|long[- ]term|lifetime|incidental)\s+)?(?:sun|uv)\s+exposure\b|\b(?:history of|past|prior|previous|routine|regular|frequent|daily|repeated|recent|ongoing|habitual)\s+(?:shaving|razor use)\b|\b(?:shaving|razor use)\s+(?:history|routine|habits?|over time)\b|\b(?:due to|caused by|likely from|result(?:s|ing)? from|stemming from|consistent with|after)\s+(?:(?:routine|regular|frequent|daily|repeated|recent)\s+)?(?:shaving|razor use)\b|\bpost[- ]shav(?:e|ing)\b|\brazor bumps?\b|\b(?:you|the guest|they)\s+(?:shave|shaves|use(?:s)? (?:a )?razor)\b|\b(?:skin|surface|contour)\s+(?:softens?|thins?|changes?)\s+over\s+time\b)/i;
const unsupportedGroomingHistoryPattern = /(?:\b(?:history of|past|prior|previous|routine|regular|frequent|daily|repeated|recent|ongoing|habitual)\s+(?:shaving|razor use|(?:surface\s+)?hair removal)\b|\b(?:shaving|razor use|(?:surface\s+)?hair removal)\s+(?:history|routine|habits?|over time)\b|\b(?:due to|caused by|likely from|result(?:s|ing)? from|stemming from|consistent with|after)\s+(?:(?:routine|regular|frequent|daily|repeated|recent)\s+)?(?:shaving|razor use|(?:surface\s+)?hair removal)\b|\b(?:indicat(?:e|es|ed|ing)|suggest(?:s|ed|ing)?|reflect(?:s|ed|ing)?|evidence\s+of|signs?\s+of)\s+(?:(?:past|prior|previous|recent|regular|routine|frequent|daily|repeated|ongoing|habitual)\s+)?(?:surface\s+)?hair removal\b|\b(?:associated|linked|related)\s+(?:with|to)\s+(?:(?:past|prior|previous|recent|regular|routine|frequent|daily|repeated|ongoing|habitual)\s+)?(?:surface\s+)?hair removal\b|\b(?:appears?|looks?|seems?)\s+(?:recently\s+)?shav(?:ed|en)\b|\brecently\s+shav(?:ed|en)\b)/i;
const unmeasuredPhotoPhysicalStatePattern = /(?:\bfirm(?:ness|er|est)?(?:[- ]looking)?\b|\belastic(?:ity|[- ]looking)?\b|\bhydrat(?:ion|ed|ing)\b|\bwell[- ]hydrated\b|\bmoist(?:ure|urized|urised)?\b|\bmoistur(?:izing|ising)\b|\bsupple(?:ness)?\b|\bthickness\b|\b(?:thin|thick)(?:ner|est)?(?:[- ]looking)?\s+(?:skin|surface)\b|\b(?:skin|surface)\s+(?:appears?|looks?|is)\s+(?:thin|thick)\b|\b(?:skin|surface\s+skin)\b[^.;]{0,35}\b(?:thins?|thinned|thinning)\b|\bthinning\s+(?:surface\s+)?skin\b|\bvolume\s+loss\b|\b(?:shifts?|changes?)\s+in\s+volume\b|\bcollagen\s+(?:level|levels|content|density|loss|depletion|stores?)\b|\bskin\s+barrier\b|\bunderlying\s+support\b|\bwell[- ]supported\b|\b(?:contours?|skin|surface)\b[^.;]{0,25}\b(?:is|are|appears?|looks?|seems?)\s+supported\b)/i;
const observationRecommendationClaimPattern = /\b(?:ideal|excellent|good|clear)\s+(?:candidate|target)\s+for\b|\b(?:ideal|excellent|good|clear)\s+for\s+(?:laser|treatment|procedure)\b/i;

const diagnosticClaimSelfCheckPhrases = Object.freeze([
  'Rosacea is visible in this photo.',
  'This looks like rosacea.',
  'This is likely rosacea.',
  'The appearance is suggestive of rosacea.',
  'There appears to be rosacea.',
  'We see eczema.',
  'Acne is present.',
  'Your complexion suggests melasma.',
  'This may be dermatitis.',
  'The photo reveals psoriasis.',
  'This looks like keratosis pilaris.',
  'The bumps are keratosis pilaris.',
  'Those bumps are acne.',
  'That is eczema.',
  'These spots look like melasma.',
  'It appears to be rosacea.',
  'I see psoriasis.',
  'The rash is dermatitis.',
  'Likely rosacea across the cheeks.',
]);
for (const phrase of diagnosticClaimSelfCheckPhrases) {
  assert.match(phrase, diagnosticClaimPattern, `diagnostic-copy guard self-check matches: ${phrase}`);
}
assert.doesNotMatch(
  'An acne-focused blue-light protocol may be discussed in person.',
  diagnosticClaimPattern,
  'diagnostic-copy guard self-check preserves benign service protocol copy',
);

function stripCanonicalGuestCatalogNames(value) {
  let copy = String(value || '');
  for (const name of canonicalGuestCatalogNames) {
    const escapedName = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    copy = copy.replace(new RegExp(escapedName, 'gi'), ' canonical offering ');
  }
  return copy;
}

function assertAppearanceOnlyGuestText(testId, surface, value) {
  const guestProse = stripCanonicalGuestCatalogNames(value);
  assert.doesNotMatch(
    guestProse,
    medicalConditionTermPattern,
    `${testId}: ${surface} contains no medical-condition labels outside exact catalog names`,
  );
  assert.doesNotMatch(
    guestProse,
    unsupportedAppearanceLabelPattern,
    `${testId}: ${surface} uses appearance-only language outside exact catalog names`,
  );
  assert.doesNotMatch(
    guestProse,
    diagnosticClaimPattern,
    `${testId}: ${surface} contains no plain-English diagnostic claim`,
  );
  return guestProse;
}

assert.doesNotMatch(
  stripCanonicalGuestCatalogNames('Recommended Treatment Stack Scar Reduction'),
  unsupportedAppearanceLabelPattern,
  'appearance-language self-check preserves the exact canonical Scar Reduction combo name',
);
assert.match(
  stripCanonicalGuestCatalogNames('The photo shows visible scarring.'),
  unsupportedAppearanceLabelPattern,
  'appearance-language self-check still rejects unsupported scarring prose',
);

const allCases = [
  {
    id: '01-face-black-woman',
    area: 'face',
    engine: 'chromium',
    filename: '01-face-black-woman.jpg',
    source: 'https://www.pexels.com/photo/close-up-photo-of-woman-s-face-8384894/',
    expectedSha256: '6f1d8a4b4348c139dea306c12d7d18b7fccfbe4f7d9020583355978ae0ac83e9',
    expectedOutcome: 'accepted',
    visualOracle: {
      anyVisibleConcern: ['darkSpots', 'unevenTone'],
      anyTreatment: ['Sciton Moxi', 'Chemical Peels', 'Sciton BBL', 'Sciton Halo'],
      reviewNotes: 'Independent review identified visible pigment/tone variation as a meaningful appearance feature.',
    },
  },
  {
    id: '02-face-older-man',
    area: 'face',
    engine: 'webkit',
    filename: '02-face-older-man.jpg',
    source: 'https://www.pexels.com/photo/close-up-photo-of-a-man-15026469/',
    expectedSha256: '6967acd069f91be26694e10cc4eefc8ab64247d85aee252955d560d3a1f1aa2f',
    expectedOutcome: 'accepted',
    visualOracle: {
      anyVisibleConcern: ['wrinkles', 'texture'],
      anyTreatment: ['Botox', 'Dysport', 'Xeomin', 'Microneedling', 'RF Microneedling', 'Sciton Moxi', 'Sciton Halo'],
      reviewNotes: 'Independent review identified lines and texture as prominent appearance features.',
    },
  },
  {
    id: '03-neck-closeup',
    area: 'neck_chest',
    engine: 'chromium',
    filename: '03-neck-closeup.jpg',
    source: 'https://www.pexels.com/photo/person-s-neck-in-close-up-photography-7479570/',
    expectedSha256: 'ca36a49ced653f78c42d238d9b31249971e0f01a7f47203d1a73f2a6a5264a5d',
    expectedOutcome: 'accepted',
    visualOracle: {
      reviewNotes: 'Anatomy-coverage fixture only; no subjective treatment outcome is forced by the harness.',
    },
  },
  {
    id: '04-neck-clavicle',
    area: 'neck_chest',
    engine: 'webkit',
    filename: '04d-neck-clavicle-7067815.jpg',
    source: 'https://www.pexels.com/photo/close-up-shot-of-a-person-s-neck-7067815/',
    expectedSha256: 'b4c2f54787e26c0325130e12b4c536680ce44698f514337d573dcff6b968761c',
    expectedOutcome: 'accepted',
    visualOracle: {
      anyVisibleConcern: ['sunDamage', 'texture'],
      anyTreatment: ['Sciton BBL', 'Sciton Moxi', 'Sciton Halo'],
      maximumScores: { laxity: 40, wrinkles: 40 },
      reviewNotes: 'Independent review confirmed anterior/lateral neck, clavicle, and upper chest with mild pigment/tone and surface-texture variation; this is non-diagnostic.',
    },
  },
  {
    id: '05-hands-older',
    area: 'hands',
    engine: 'chromium',
    filename: '05-hands-older.jpg',
    source: 'https://www.pexels.com/photo/close-up-of-hands-6878220/',
    expectedSha256: 'b77b06af0b97fd7d54769a040b921ea804b01b4dc0320c8045a49115ff1ad5ce',
    expectedOutcome: 'accepted',
    visualOracle: {
      anyVisibleConcern: ['texture', 'laxity', 'dryness', 'sunDamage'],
      anyTreatment: ['Microneedling', 'RF Microneedling', 'Sciton Moxi', 'Microneedling + PRF', 'Sciton Halo'],
      reviewNotes: 'Independent review identified visible hand surface change. Texture, visible crepiness/contour softness, dryness, or sun-exposure signs are acceptable non-diagnostic labels when at least one is visibly scored and the recommendation remains mapped.',
    },
  },
  {
    id: '06-hands-dark-skin-man',
    area: 'hands',
    engine: 'webkit',
    filename: '06b-hands-dark-skin-man-8276212.jpg',
    source: 'https://www.pexels.com/photo/close-up-shot-of-hands-8276212/',
    expectedSha256: 'c3922f4b3c0569d512559b1d3c6f44799d1aa7e21add424bb629830f22e07946',
    expectedOutcome: 'accepted',
    visualOracle: {
      maximumConcernScore: 60,
      maximumScores: { sunDamage: 40, laxity: 40, veins: 40 },
      forbiddenTreatments: ['Sculptra', 'Sciton BBL'],
      reviewNotes: 'Maintenance/negative-control fixture with adult hands clearly dominant; independent review did not identify a severe visible concern. No clinical inference is intended.',
    },
  },
  {
    id: '07-back-mature-welllit',
    area: 'back',
    engine: 'chromium',
    filename: '07b-back-mature-welllit.jpg',
    source: 'https://www.pexels.com/photo/a-person-s-back-in-close-up-photography-8624581/',
    expectedSha256: '888379ad7c782dfeaf71c940295e98419d354f09ef098130598da3e1664e6151',
    expectedOutcome: 'accepted',
    visualOracle: {
      anyVisibleConcern: ['unevenTone', 'texture', 'scarring', 'acne'],
      anyTreatment: ['Chemical Peels', 'Sciton BBL', 'Sciton Halo', 'Microneedling'],
      forbiddenTreatments: ['Laser Hair Removal'],
      reviewNotes: 'Independent review identified visible tone and surface variation. Uneven tone, texture, textural marks, or mild surface congestion are acceptable non-diagnostic label families when paired with a mapped treatment; unsupported hair-removal recommendations remain forbidden.',
    },
  },
  {
    id: '08-back-clothed-shoulders',
    area: 'back',
    engine: 'webkit',
    filename: '08-back-clothed-shoulders.jpg',
    source: 'https://www.pexels.com/photo/back-of-woman-5254325/',
    expectedSha256: 'eb53b3a043f054910676b6f4f285772635cb7ed137cfc020bfff6039cec25e1d',
    expectedOutcome: 'accepted',
    visualOracle: {
      maximumConcernScore: 60,
      forbiddenTreatments: ['Laser Hair Removal'],
      reviewNotes: 'Maintenance/negative-control fixture; independent review did not identify a severe visible concern. No clinical inference is intended.',
    },
  },
  {
    id: '09-legs-male',
    area: 'legs',
    engine: 'chromium',
    filename: '09-legs-male.jpg',
    source: 'https://www.pexels.com/photo/close-up-photo-of-a-person-s-leg-8462937/',
    expectedSha256: 'f8f61a86fb10c8d8731179c6c0fda02b5a9e4270f2f9a6c4ca881a475b45aeda',
    expectedOutcome: 'accepted',
    visualOracle: {
      anyVisibleConcern: ['texture', 'dryness'],
      anyTreatment: ['Laser Hair Removal', 'Microneedling', 'Sciton Moxi'],
      reviewNotes: 'Independent review identified surface texture/dryness and clear follicular contrast. Laser Hair Removal is an acceptable expected option only when the model maps it to moderate visible hair growth; Microneedling or Moxi remain acceptable when supported by the texture score.',
    },
  },
  {
    id: '10-legs-knees',
    area: 'legs',
    engine: 'webkit',
    filename: '10-legs-knees.jpg',
    source: 'https://www.pexels.com/photo/close-up-of-legs-and-knees-8093137/',
    expectedSha256: '25080613620cf5bb4e7449ccfbdde14ad45f25ea4f2555db1ee90465b354b0e6',
    expectedOutcome: 'accepted',
    visualOracle: {
      anyVisibleConcern: ['texture', 'dryness'],
      anyTreatment: ['Laser Hair Removal', 'Microneedling', 'Sciton Moxi'],
      reviewNotes: 'Independent review identified visible knee/leg surface texture and clearly visible hair. Laser Hair Removal is an acceptable expected option only when the model maps it to moderate treatment-relevant hair evidence; Microneedling or Moxi remain acceptable when supported by the texture score.',
    },
  },
  {
    id: '11-face-selected-as-legs-mismatch-control',
    area: 'legs',
    engine: 'chromium',
    filename: '01-face-black-woman.jpg',
    source: 'https://www.pexels.com/photo/close-up-photo-of-woman-s-face-8384894/',
    expectedSha256: '6f1d8a4b4348c139dea306c12d7d18b7fccfbe4f7d9020583355978ae0ac83e9',
    expectedOutcome: 'area-mismatch',
    reusesFixtureId: '01-face-black-woman',
    expectedReasonCode: 'area_mismatch',
    expectedObservedArea: 'face',
    expectedRejection: /appears to show face, not the selected legs/i,
  },
  {
    id: '12-shoulder-selected-as-hands-mismatch-control',
    area: 'hands',
    engine: 'webkit',
    filename: '04-neck-shoulder.jpg',
    source: 'https://www.pexels.com/photo/close-up-photo-of-woman-s-shoulder-6567936/',
    expectedSha256: '39e97eaae3c35958c9d4f949153cb1d8114a95337a7b233dd7350634e0d0ea17',
    expectedOutcome: 'area-mismatch',
    expectedReasonCode: 'area_mismatch',
    expectedObservedAreaOneOf: ['face', 'neck_chest', 'back', 'legs', 'other'],
    expectedObservedAreaNot: 'hands',
    expectedRejection: /(?:appears to show (?:face|neck and chest|back|legs), not the selected hands|does not clearly show the selected hands)/i,
  },
  {
    id: '13-dark-back-quality-control',
    area: 'back',
    engine: 'webkit',
    filename: '07-back-bare-shoulders.jpg',
    source: 'https://www.pexels.com/photo/woman-s-bare-back-633984/',
    expectedSha256: 'dd134297c5f5d75b4fbaf54ac0ababc3607a3bf2cbf0dbb93bffa628029e37b4',
    expectedOutcome: 'quality-rejection',
    expectedReasonCode: 'quality',
    expectedRejectionSource: 'local_underexposure',
    expectedRejection: /clear enough read/i,
  },
];

const cases = allCases.filter(
  testCase => caseFilter.size === 0 || caseFilter.has(testCase.id),
);
const printAreasValidated = new Set();

function launchOptions(engineName) {
  return engineName === 'chromium'
    ? {
      headless: true,
      executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    }
    : { headless: true };
}

function mobileContextOptions(width = 390, height = 844) {
  return {
    viewport: { width, height },
    screen: { width, height },
    isMobile: true,
    hasTouch: true,
    deviceScaleFactor: 3,
  };
}

function sha256File(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

function normalizeText(value) {
  return String(value ?? '').replace(/\s+/g, ' ').trim();
}

function pythonRound(value) {
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (Math.abs(fraction - 0.5) > Number.EPSILON * 8) return Math.round(value);
  return floor % 2 === 0 ? floor : floor + 1;
}

async function captureAnalyzeResponse(response) {
  const rawText = await response.text();
  let rawJson = null;
  let parseError = null;
  try {
    rawJson = JSON.parse(rawText);
  } catch (error) {
    parseError = String(error);
  }
  return {
    httpStatus: response.status(),
    receivedAt: new Date().toISOString(),
    headers: {
      contentType: response.headers()['content-type'] || null,
      analysisRepeat: response.headers()['x-von-analysis-repeat'] || null,
    },
    code: rawJson?.code || null,
    retryable: rawJson?.retryable === true,
    error: rawJson?.error || null,
    rejected: rawJson?.rejected === true,
    reasonCode: rawJson?.reasonCode || null,
    rejectionSource: rawJson?.rejectionSource || null,
    observedArea: rawJson?.observedArea || null,
    rejectionReason: rawJson?.reason || null,
    rawJson,
    rawText: rawJson === null ? rawText : null,
    parseError,
  };
}

function writeAttemptArtifact(record) {
  const attemptPath = path.join(
    artifactDir,
    `${record.id}-${record.engine}-attempt.json`,
  );
  record.attemptArtifactPath = attemptPath;
  fs.writeFileSync(attemptPath, `${JSON.stringify(record, null, 2)}\n`);
  return attemptPath;
}

function buildAndValidateManifest() {
  assert.equal(allCases.length, 13, 'matrix has exactly thirteen attempts');
  const analyzableCases = allCases.filter(testCase => testCase.expectedOutcome === 'accepted');
  const rejectionCases = allCases.filter(testCase => testCase.expectedOutcome === 'quality-rejection');
  const mismatchCases = allCases.filter(testCase => testCase.expectedOutcome === 'area-mismatch');
  assert.equal(analyzableCases.length, 10, 'matrix has exactly ten accepted-photo cases');
  assert.equal(rejectionCases.length, 1, 'matrix has exactly one image-quality rejection control');
  assert.equal(mismatchCases.length, 2, 'matrix has exactly two selected-area mismatch controls');

  const expectedAreaCounts = { face: 2, neck_chest: 2, hands: 2, back: 2, legs: 2 };
  const actualAreaCounts = Object.fromEntries(
    Object.keys(expectedAreaCounts).map(area => [
      area,
      analyzableCases.filter(testCase => testCase.area === area).length,
    ]),
  );
  assert.deepEqual(actualAreaCounts, expectedAreaCounts, 'analyzable matrix has two photos per area');
  assert.deepEqual(
    analyzableCases.map(testCase => testCase.engine),
    Array.from({ length: 10 }, (_, index) => (index % 2 === 0 ? 'chromium' : 'webkit')),
    'accepted cases alternate Chromium and WebKit',
  );

  const entries = allCases.map((testCase, index) => {
    assert.ok(concernKeys[testCase.area], `${testCase.id}: body area is supported`);
    assert.ok(['chromium', 'webkit'].includes(testCase.engine), `${testCase.id}: engine is supported`);
    assert.ok(
      ['accepted', 'quality-rejection', 'area-mismatch'].includes(testCase.expectedOutcome),
      `${testCase.id}: expected outcome is explicit`,
    );
    assert.match(testCase.source, /^https:\/\/www\.pexels\.com\/photo\//, `${testCase.id}: source is a real-photo provenance URL`);
    const sourcePath = path.join(photoDir, testCase.filename);
    assert.ok(fs.existsSync(sourcePath), `${testCase.id}: source photo exists`);
    const stat = fs.statSync(sourcePath);
    assert.equal(stat.isFile(), true, `${testCase.id}: source path is a file`);
    assert.ok(stat.size > 0, `${testCase.id}: source photo is non-empty`);
    const actualSha256 = sha256File(sourcePath);
    assert.match(testCase.expectedSha256, /^[a-f0-9]{64}$/, `${testCase.id}: expected SHA-256 is pinned`);
    assert.equal(actualSha256, testCase.expectedSha256, `${testCase.id}: fixture bytes match pinned SHA-256`);
    return {
      ordinal: index + 1,
      id: testCase.id,
      bodyArea: testCase.area,
      engine: testCase.engine,
      kind: testCase.expectedOutcome === 'accepted' ? 'accepted-real-source-photo' : 'negative-control',
      expectedOutcome: testCase.expectedOutcome,
      filename: testCase.filename,
      sourceUrl: testCase.source,
      licenseUrl: pexelsLicenseUrl,
      sourcePath,
      sha256: actualSha256,
      expectedSha256: testCase.expectedSha256,
      bytes: stat.size,
      reusesFixtureId: testCase.reusesFixtureId || null,
      expectedReasonCode: testCase.expectedReasonCode || null,
      expectedRejectionSource: testCase.expectedRejectionSource || null,
      expectedObservedArea: testCase.expectedObservedArea || null,
      expectedObservedAreaOneOf: testCase.expectedObservedAreaOneOf || null,
      expectedObservedAreaNot: testCase.expectedObservedAreaNot || null,
      identityQualification: 'Source-photo provenance only; verified human-identity uniqueness is not asserted, and one mismatch control intentionally reuses accepted bytes.',
      visualOracle: testCase.visualOracle || null,
    };
  });

  const acceptedEntries = entries.filter(entry => entry.expectedOutcome === 'accepted');
  for (const [field, values] of [
    ['id', acceptedEntries.map(entry => entry.id)],
    ['filename', acceptedEntries.map(entry => entry.filename)],
    ['source URL', acceptedEntries.map(entry => entry.sourceUrl)],
    ['SHA-256', acceptedEntries.map(entry => entry.sha256)],
  ]) {
    assert.equal(new Set(values).size, acceptedEntries.length, `all ten accepted ${field} values are unique`);
  }
  const mismatchEntry = entries.find(entry => entry.reusesFixtureId);
  const reusedEntry = entries.find(entry => entry.id === mismatchEntry.reusesFixtureId);
  assert.ok(reusedEntry, 'mismatch control names the accepted fixture it intentionally reuses');
  assert.equal(mismatchEntry.sha256, reusedEntry.sha256, 'mismatch control reuses pinned face bytes by design');
  const rejectionEntry = entries.find(entry => entry.expectedOutcome === 'quality-rejection');
  assert.equal(
    acceptedEntries.some(entry => entry.sha256 === rejectionEntry.sha256),
    false,
    'quality-rejection fixture is distinct from every accepted fixture',
  );
  return {
    entries,
    byId: new Map(entries.map(entry => [entry.id, entry])),
    analyzableAreaCounts: actualAreaCounts,
    uniqueAcceptedSha256Count: new Set(acceptedEntries.map(entry => entry.sha256)).size,
    uniqueAttemptSha256Count: new Set(entries.map(entry => entry.sha256)).size,
  };
}

function writeLedger(run) {
  fs.writeFileSync(ledgerPath, `${JSON.stringify(run, null, 2)}\n`);
}

function expectedSeverity(score) {
  if (score <= 10) return 'none';
  if (score <= 40) return 'mild';
  if (score <= 60) return 'moderate';
  return 'severe';
}

function allowedProductsForArea(area) {
  const areaKeys = new Set(concernKeys[area]);
  return new Set(
    Object.entries(productTargets)
      .filter(([product, targets]) => (
        (area === 'face' || !/\beye\b/i.test(product))
          && (spfProducts.has(product) || targets.some(target => areaKeys.has(target)))
      ))
      .map(([product]) => product),
  );
}

function collectResponseCopy(data) {
  return [
    data?.summary,
    data?.suggestedCombo,
    ...(data?.positiveHighlights || []).flatMap(item => [item?.title, item?.detail]),
    ...Object.values(data?.concerns || {}).map(item => item?.description),
    ...(data?.recommendations || []).flatMap(item => [item?.treatment, item?.reason]),
    ...(data?.productRecommendations || []).flatMap(item => [item?.product, item?.reason]),
  ].filter(value => typeof value === 'string' && value.trim()).join(' ');
}

function collectRawPhotoObservationCopy(data) {
  return [
    data?.summary,
    ...(data?.positiveHighlights || []).flatMap(item => [item?.title, item?.detail]),
    ...Object.values(data?.concerns || {}).map(item => item?.description),
  ].filter(value => typeof value === 'string' && value.trim()).join(' ');
}

function assertPhotoObservationIsVisuallyGrounded(testId, surface, value) {
  const observationProse = stripCanonicalGuestCatalogNames(value);
  assert.doesNotMatch(
    observationProse,
    unsupportedPhotoCauseOrHistoryPattern,
    `${testId}: ${surface} does not infer sun-exposure or shaving cause/history from one photo`,
  );
  assert.doesNotMatch(
    observationProse,
    unsupportedGroomingHistoryPattern,
    `${testId}: ${surface} does not infer grooming or hair-removal history from one photo`,
  );
  assert.doesNotMatch(
    observationProse,
    unmeasuredPhotoPhysicalStatePattern,
    `${testId}: ${surface} does not claim unmeasured firmness, elasticity, hydration, moisture, suppleness, thickness, barrier, collagen, volume, or underlying support`,
  );
  assert.doesNotMatch(
    observationProse,
    observationRecommendationClaimPattern,
    `${testId}: ${surface} does not turn a photo observation into a candidacy or treatment claim`,
  );
  return observationProse;
}

for (const phrase of [
  'Cumulative sun exposure is evident.',
  'The spots likely result from routine sun exposure.',
  'Years spent in the sun explain the visible variation.',
  'The pigment variation likely reflects time spent in the sun.',
  'These textural marks are from past breakouts.',
  'The bumps appear after regular shaving.',
  'The photo reveals a history of shaving.',
  'A razor bump is present.',
  'The pigment variation reflects some sun exposure.',
  'The darker areas are reflecting general sun exposure.',
  'These variations often accompany normal sun exposure.',
  'The pigment variation suggests sun exposure.',
  'This is evidence of sun exposure.',
  'The spots point to sun exposure.',
  'The variation is associated with sun exposure.',
  'The pattern indicates UV exposure.',
  'This may be related to incidental sun exposure.',
  'The photo shows visible signs of environmental exposure.',
  'The surface softens over time.',
  'The textural variation may indicate minor past marks.',
  'The pigment variation is from incidental sun exposure.',
]) {
  assert.match(
    phrase,
    unsupportedPhotoCauseOrHistoryPattern,
    `photo-history guard self-check matches: ${phrase}`,
  );
}
for (const phrase of [
  'Visible follicles are present, indicating regular surface hair removal.',
  'The pattern suggests recent hair removal.',
  'This is evidence of a hair removal routine.',
  'The area appears recently shaved.',
]) {
  assert.match(
    phrase,
    unsupportedGroomingHistoryPattern,
    `grooming-history guard self-check matches: ${phrase}`,
  );
}
assert.match(
  'The follicular pattern makes this an ideal target for laser hair reduction.',
  observationRecommendationClaimPattern,
  'observation-treatment guard self-check catches candidacy language in photo copy',
);
for (const phrase of [
  'Surface firmness looks strong.',
  'The area has natural elasticity.',
  'Hydration and moisture look balanced.',
  'The skin appears supple.',
  'Skin thickness looks even.',
  'The skin barrier appears intact.',
  'The contours appear well supported.',
  'The folds reflect shifts in volume.',
]) {
  assert.match(
    phrase,
    unmeasuredPhotoPhysicalStatePattern,
    `unmeasured-state guard self-check matches: ${phrase}`,
  );
}
for (const phrase of [
  'Visible sun-exposure signs appear across the upper back.',
  'Visible hair growth and stubble are present.',
  'Surface dryness is visible.',
]) {
  assert.doesNotMatch(
    phrase,
    unsupportedPhotoCauseOrHistoryPattern,
    `photo-history guard self-check preserves a visible observation: ${phrase}`,
  );
  assert.doesNotMatch(
    phrase,
    unsupportedGroomingHistoryPattern,
    `grooming-history guard self-check preserves a visible observation: ${phrase}`,
  );
  assert.doesNotMatch(
    phrase,
    unmeasuredPhotoPhysicalStatePattern,
    `unmeasured-state guard self-check preserves a visible observation: ${phrase}`,
  );
}
const observationScopeSelfCheck = collectRawPhotoObservationCopy({
  summary: 'Visible sun-exposure signs appear across the upper back.',
  positiveHighlights: [{ title: 'Even-looking surface', detail: 'The visible surface looks smooth.' }],
  concerns: { dryness: { description: 'Surface dryness is visible.' } },
  recommendations: [{
    treatment: 'HydraFacial Customized',
    reason: 'This service supports hydration and moisture in an appropriate treatment plan.',
  }],
  suggestedCombo: 'Scar Reduction',
});
assert.doesNotMatch(
  observationScopeSelfCheck,
  unmeasuredPhotoPhysicalStatePattern,
  'raw-observation scope self-check excludes recommendation mechanisms and combo catalog labels',
);

function hasForbiddenAnatomy(area, value) {
  let text = String(value || '');
  const determiner = '(?:(?:the|your|his|her|their|each|my|our|a|an)\\s+)?';
  if (area === 'hands') {
    text = text.replace(new RegExp(`\\bbacks?\\s+of\\s+${determiner}hands?\\b`, 'gi'), 'hand surface');
  } else if (area === 'neck_chest') {
    text = text.replace(new RegExp(`\\bback\\s+of\\s+${determiner}neck\\b`, 'gi'), 'neck surface');
  } else if (area === 'legs') {
    text = text.replace(
      new RegExp(`\\bbacks?\\s+of\\s+${determiner}(?:legs?|knees?|calf|calves)\\b`, 'gi'),
      'leg surface',
    );
  }
  const allowed = allowedAnatomyFamilies[area] || allowedAnatomyFamilies.face;
  return Object.entries(anatomyFamilyPatterns).some(
    ([family, pattern]) => !allowed.has(family) && pattern.test(text),
  );
}

async function captureGuestFacingPageText(page) {
  return page.evaluate(() => {
    const visibleText = document.body?.innerText || '';
    const accessibleText = [...document.querySelectorAll('[aria-label], [title], [placeholder], img[alt]')]
      .filter(element => !element.closest('script, style, template, noscript'))
      .flatMap(element => [
        element.getAttribute('aria-label'),
        element.getAttribute('title'),
        element.getAttribute('placeholder'),
        element.getAttribute('alt'),
      ])
      .filter(Boolean)
      .join(' ');
    return `${visibleText} ${accessibleText}`.replace(/\s+/g, ' ').trim();
  });
}

function snapshotAnalysis(data, status = 'completed') {
  if (!data || typeof data !== 'object') return null;
  return {
    status,
    isLive: data._isLive === true,
    isDemo: data._isDemo === true,
    observedArea: data.observedArea ?? null,
    overallScore: data.overallScore ?? null,
    summary: data.summary ?? null,
    positiveHighlights: Array.isArray(data.positiveHighlights)
      ? data.positiveHighlights.map(item => ({
        title: item?.title ?? null,
        detail: item?.detail ?? null,
        groundedIn: item?.groundedIn ?? null,
      }))
      : null,
    concerns: data.concerns && typeof data.concerns === 'object'
      ? Object.fromEntries(Object.entries(data.concerns).map(([key, value]) => [key, {
        score: value?.score ?? null,
        severity: value?.severity ?? null,
        description: value?.description ?? null,
      }]))
      : null,
    treatments: Array.isArray(data.recommendations)
      ? data.recommendations.map(item => ({
        treatment: item?.treatment ?? null,
        reason: item?.reason ?? null,
        targets: item?.targets ?? null,
        priority: item?.priority ?? null,
      }))
      : null,
    products: Array.isArray(data.productRecommendations)
      ? data.productRecommendations.map(item => ({
        product: item?.product ?? null,
        reason: item?.reason ?? null,
      }))
      : null,
    suggestedCombo: data.suggestedCombo ?? null,
  };
}

async function assertPreAnalysisLabels(page, testCase) {
  const labels = (await page.locator('#visiaFeatureList li').allTextContents())
    .map(value => value.replace(/\s+/g, ' ').trim())
    .filter(Boolean);
  assert.equal(labels.length, 5, `${testCase.id}: pre-analysis preview has exactly five labels`);
  for (const [index, pattern] of preAnalysisLabelPatterns[testCase.area].entries()) {
    assert.match(
      labels[index],
      pattern,
      `${testCase.id}: pre-analysis label ${index + 1} for ${testCase.area} is exact appearance-only copy`,
    );
  }
  assertAppearanceOnlyGuestText(
    testCase.id,
    'pre-analysis page, navigation, actions, and footer',
    await captureGuestFacingPageText(page),
  );
  return labels;
}

function assertStructuredResult(testCase, data) {
  assert.ok(data && typeof data === 'object', `${testCase.id}: result is an object`);
  assert.equal(data._isLive, true, `${testCase.id}: result is live`);
  assert.notEqual(data._isDemo, true, `${testCase.id}: result is not demo data`);
  assert.equal(
    data.observedArea,
    testCase.area,
    `${testCase.id}: model-observed anatomy matches the selected area`,
  );
  assert.deepEqual(
    Object.keys(data.concerns || {}).sort(),
    concernKeys[testCase.area],
    `${testCase.id}: concern keys match ${testCase.area}`,
  );
  assert.ok(
    Number.isInteger(data.overallScore) && data.overallScore >= 0 && data.overallScore <= 100,
    `${testCase.id}: overall score is an integer from 0 through 100`,
  );

  for (const [key, concern] of Object.entries(data.concerns)) {
    assert.ok(
      concern && Number.isInteger(concern.score) && concern.score >= 0 && concern.score <= 100,
      `${testCase.id}: ${key} score is an integer from 0 through 100`,
    );
    assert.equal(
      concern.severity,
      expectedSeverity(concern.score),
      `${testCase.id}: ${key} severity matches its score band`,
    );
    assert.ok(String(concern.description || '').trim(), `${testCase.id}: ${key} description is present`);
  }
  const concernScores = Object.values(data.concerns).map(concern => concern.score);
  const calculatedOverall = Math.max(
    0,
    Math.min(100, pythonRound(100 - concernScores.reduce((sum, score) => sum + score, 0) / concernScores.length)),
  );
  assert.equal(
    data.overallScore,
    calculatedOverall,
    `${testCase.id}: overall score exactly equals 100 minus the mean visible-concern score`,
  );

  assert.ok(
    Array.isArray(data.positiveHighlights)
      && data.positiveHighlights.length >= 2
      && data.positiveHighlights.length <= 3,
    `${testCase.id}: result has two to three positive highlights`,
  );
  const positiveGrounds = new Set();
  const universalPositiveGrounds = new Set(['guestIdentity', 'photoClarity']);
  for (const highlight of data.positiveHighlights) {
    assert.ok(String(highlight.title || '').trim(), `${testCase.id}: positive title is present`);
    assert.ok(String(highlight.detail || '').trim(), `${testCase.id}: positive detail is present`);
    assert.doesNotMatch(
      `${highlight.title} ${highlight.detail}`,
      absenceBasedPositivePattern,
      `${testCase.id}: positive is direct, complementary, and not absence-based`,
    );
    const concernGrounded = concernKeys[testCase.area].includes(highlight.groundedIn);
    assert.ok(
      concernGrounded || universalPositiveGrounds.has(highlight.groundedIn),
      `${testCase.id}: positive is grounded in a mild visible quality or an approved human-first truth`,
    );
    if (concernGrounded) {
      assert.ok(
        data.concerns[highlight.groundedIn].score <= 40,
        `${testCase.id}: no moderate or severe concern is repackaged as praise`,
      );
    } else if (highlight.groundedIn === 'guestIdentity') {
      assert.equal(highlight.title, 'Distinctly Yours', `${testCase.id}: identity-positive title is deterministic`);
    } else {
      assert.equal(highlight.title, 'A Clear Starting Point', `${testCase.id}: photo-positive title is deterministic`);
    }
    assert.equal(
      positiveGrounds.has(highlight.groundedIn),
      false,
      `${testCase.id}: positive grounding concerns are unique`,
    );
    positiveGrounds.add(highlight.groundedIn);
  }
  const summary = String(data.summary || '').trim();
  const firstPositiveDetail = String(data.positiveHighlights[0].detail || '').trim();
  assert.ok(summary, `${testCase.id}: summary is present`);
  assert.ok(
    summary.toLocaleLowerCase().startsWith(firstPositiveDetail.toLocaleLowerCase()),
    `${testCase.id}: summary leads verbatim with the first positive detail`,
  );

  assert.ok(
    Array.isArray(data.recommendations)
      && data.recommendations.length <= 6,
    `${testCase.id}: result has zero through six treatments`,
  );
  const allowedProducts = allowedProductsForArea(testCase.area);
  assert.ok(
    Array.isArray(data.productRecommendations)
      && data.productRecommendations.length >= 1
      && data.productRecommendations.length <= allowedProducts.size,
    `${testCase.id}: result has at least the required SPF and never exceeds the selected area's product catalog`,
  );

  const eligibleTargets = concernKeys[testCase.area]
    .filter(key => data.concerns[key].score > 10
      && (key !== 'hairRemoval' || supportsLaserHairRemoval(data)))
    .sort((left, right) => data.concerns[right].score - data.concerns[left].score
      || left.localeCompare(right));
  const eligibleTargetSet = new Set(eligibleTargets);
  const treatableRankedTargets = eligibleTargets.filter(target => (
    [...areaTreatments[testCase.area]].some(treatment => treatmentTargets[treatment].includes(target))
  ));
  const treatments = data.recommendations.map(item => item.treatment);
  const products = data.productRecommendations.map(item => item.product);
  assert.equal(new Set(treatments).size, treatments.length, `${testCase.id}: treatments are unique`);
  assert.equal(new Set(products).size, products.length, `${testCase.id}: products are unique`);
  if (eligibleTargets.length === 0) {
    assert.equal(
      data.recommendations.length,
      0,
      `${testCase.id}: no treatment is forced when every visible-concern score is 10 or below`,
    );
  }

  for (const [index, recommendation] of data.recommendations.entries()) {
    assert.ok(
      areaTreatments[testCase.area].has(recommendation.treatment),
      `${testCase.id}: ${recommendation.treatment} is an exact catalog treatment for ${testCase.area}`,
    );
    assert.ok(String(recommendation.reason || '').trim(), `${testCase.id}: treatment reason is present`);
    assert.equal(
      treatmentReasonOverreaches(recommendation.reason, recommendation.targets),
      false,
      `${testCase.id}: ${recommendation.treatment} reason is limited to this guest's actual targets`,
    );
    assert.ok(
      Array.isArray(recommendation.targets) && recommendation.targets.length > 0,
      `${testCase.id}: treatment has at least one target`,
    );
    assert.equal(
      new Set(recommendation.targets).size,
      recommendation.targets.length,
      `${testCase.id}: ${recommendation.treatment} targets are unique`,
    );
    for (const target of recommendation.targets) {
      assert.ok(
        concernKeys[testCase.area].includes(target),
        `${testCase.id}: target ${target} belongs to ${testCase.area}`,
      );
      assert.ok(
        treatmentTargets[recommendation.treatment].includes(target),
        `${testCase.id}: ${recommendation.treatment} supports ${target}`,
      );
      assert.ok(
        eligibleTargetSet.has(target),
        `${testCase.id}: ${recommendation.treatment} targets only a visible score above 10: ${target}`,
      );
      assert.ok(
        String(recommendation.reason || '').toLocaleLowerCase().includes(
          concernGoalLabels[target].toLocaleLowerCase(),
        ),
        `${testCase.id}: ${recommendation.treatment} reason explicitly names displayed target ${target}`,
      );
    }
    assert.equal(
      recommendation.priority,
      index + 1,
      `${testCase.id}: treatment priorities are exact and sequential`,
    );
    if (recommendation.treatment === 'Sciton Halo') {
      assert.ok(
        recommendation.targets.some(target => data.concerns[target].score >= 41),
        `${testCase.id}: Sciton Halo has at least one moderate-or-severe listed target`,
      );
    }
    if (recommendation.treatment === 'Sculptra') {
      assert.ok(
        recommendation.targets.some(target => data.concerns[target].score >= 41),
        `${testCase.id}: Sculptra has at least one moderate-or-severe listed target`,
      );
    }
    if (testCase.area === 'hands' && recommendation.treatment === 'Sciton BBL') {
      assert.ok(
        recommendation.targets.some(target => data.concerns[target].score >= 41),
        `${testCase.id}: hand BBL has at least one moderate-or-severe listed target`,
      );
    }
    if (recommendation.treatment === 'Laser Hair Removal') {
      assert.equal(
        supportsLaserHairRemoval(data),
        true,
        `${testCase.id}: Laser Hair Removal requires moderate visible stubble, dark/coarse hair, or follicular contrast`,
      );
    }
  }

  if (data.concerns.redness?.score >= 41) {
    assert.equal(treatments.includes('Microneedling'), false, `${testCase.id}: visible redness excludes Microneedling`);
    assert.equal(treatments.includes('RF Microneedling'), false, `${testCase.id}: visible redness excludes RF Microneedling`);
  }
  if (data.recommendations.length > 0) {
    const treatmentCoveredTargets = new Set(data.recommendations.flatMap(item => item.targets));
    const coveredRankedTargets = treatableRankedTargets.filter(target => treatmentCoveredTargets.has(target));
    assert.ok(
      coveredRankedTargets.length > 0,
      `${testCase.id}: every returned service covers a treatment-eligible visible concern`,
    );
    assert.ok(
      data.recommendations[0].targets.includes(coveredRankedTargets[0]),
      `${testCase.id}: priority one maps to the leading concern actually covered by a selected service`,
    );
    const leadingTreatmentCoverage = new Set(
      data.recommendations.slice(0, 2).flatMap(item => item.targets),
    );
    assert.ok(
      coveredRankedTargets.slice(0, 2).every(target => leadingTreatmentCoverage.has(target)),
      `${testCase.id}: the first two services cover the first two concerns actually covered by selected services`,
    );
  }
  for (const product of data.productRecommendations) {
    if (testCase.area !== 'face') {
      assert.doesNotMatch(
        product.product,
        /\beye\b/i,
        `${testCase.id}: a non-face result never recommends an eye-area product`,
      );
    }
    assert.ok(
      allowedProducts.has(product.product),
      `${testCase.id}: ${product.product} is an exact product catalog match for ${testCase.area}`,
    );
    assert.ok(String(product.reason || '').trim(), `${testCase.id}: product reason is present`);
    if (!spfProducts.has(product.product)) {
      assert.ok(
        productTargets[product.product].some(target => eligibleTargetSet.has(target)),
        `${testCase.id}: non-SPF ${product.product} maps to a visible score above 10`,
      );
      assert.ok(
        productTargets[product.product]
          .filter(target => eligibleTargetSet.has(target))
          .some(target => String(product.reason || '').toLocaleLowerCase().includes(
            concernGoalLabels[target].toLocaleLowerCase(),
          )),
        `${testCase.id}: non-SPF ${product.product} reason explicitly names an eligible visible goal`,
      );
    }
  }
  assert.ok(products.some(product => spfProducts.has(product)), `${testCase.id}: product plan includes SPF`);

  const coveredTargets = new Set(data.recommendations.flatMap(item => item.targets));
  for (const product of products) {
    for (const target of productTargets[product]) {
      if (eligibleTargetSet.has(target)) coveredTargets.add(target);
    }
  }
  const moderateConcerns = eligibleTargets.filter(key => data.concerns[key].score >= 41);
  const requiredCoverage = new Set(moderateConcerns);
  const safelyMappableMildTargets = eligibleTargets.filter(target => {
    const hasProductPath = [...allowedProducts].some(product => (
      productTargets[product].includes(target)
    ));
    const hasServicePath = [...areaTreatments[testCase.area]].some(treatment => (
      treatmentTargets[treatment].includes(target)
      && !['Sciton Halo', 'Sculptra', 'Laser Hair Removal'].includes(treatment)
      && !(testCase.area === 'hands' && treatment === 'Sciton BBL')
    ));
    return hasProductPath || hasServicePath;
  });
  if (requiredCoverage.size === 0 && eligibleTargets.length > 0) {
    if (safelyMappableMildTargets.length > 0) {
      requiredCoverage.add(safelyMappableMildTargets[0]);
    }
  }
  assert.ok(
    [...requiredCoverage].every(target => coveredTargets.has(target)),
    `${testCase.id}: treatment plus skincare covers every eligible moderate concern and the highest safely mappable mild concern`,
  );
  const treatmentCoveredTargets = new Set(data.recommendations.flatMap(item => item.targets));
  const requiredTreatmentCoverage = new Set(
    moderateConcerns.filter(target => treatableRankedTargets.includes(target)),
  );
  assert.ok(
    [...requiredTreatmentCoverage].every(target => treatmentCoveredTargets.has(target)),
    `${testCase.id}: services cover every moderate concern that has an allowed treatment match`,
  );
  const suggestedCombo = data.suggestedCombo;
  let suggestedComboValidated = null;
  if (suggestedCombo !== null && suggestedCombo !== undefined && suggestedCombo !== '') {
    assert.equal(
      Object.prototype.hasOwnProperty.call(comboRequirements, suggestedCombo),
      true,
      `${testCase.id}: suggested combo uses one exact canonical name`,
    );
    for (const componentOptions of comboRequirements[suggestedCombo]) {
      assert.ok(
        treatments.some(treatment => componentOptions.has(treatment)),
        `${testCase.id}: ${suggestedCombo} contains every required treatment component`,
      );
    }
    suggestedComboValidated = suggestedCombo;
  }

  const rawPhotoObservationCopy = collectRawPhotoObservationCopy(data);
  assertPhotoObservationIsVisuallyGrounded(
    testCase.id,
    'raw summary, positive highlights, and concern descriptions',
    rawPhotoObservationCopy,
  );
  const allCopy = collectResponseCopy(data);
  assert.doesNotMatch(allCopy, /[\u2013\u2014]/, `${testCase.id}: response copy contains no en or em dash`);
  assert.doesNotMatch(
    allCopy,
    /melanoma|malignan|skin cancer|suspicious lesion|nevus|diagnos/i,
    `${testCase.id}: model does not perform medical lesion triage`,
  );
  assert.doesNotMatch(allCopy, copyOverclaimPattern, `${testCase.id}: response copy avoids prohibited overclaims`);
  assert.doesNotMatch(allCopy, malformedCopyPattern, `${testCase.id}: response copy contains no known concatenated-word defects`);
  assert.doesNotMatch(allCopy, copyPolishDefectPattern, `${testCase.id}: response copy avoids known polish defects`);
  const guestProse = assertAppearanceOnlyGuestText(testCase.id, 'API guest-facing response copy', allCopy);
  assert.equal(
    hasForbiddenAnatomy(testCase.area, guestProse),
    false,
    `${testCase.id}: response prose outside exact catalog names stays within the selected anatomy, including compound-area exceptions`,
  );
  for (const key of ['skinAge', 'estimatedSkinAge', 'radar', 'radarChart']) {
    assert.equal(Object.prototype.hasOwnProperty.call(data, key), false, `${testCase.id}: response omits ${key}`);
  }

  return {
    rankedVisibleConcerns: eligibleTargets.map(key => ({ key, score: data.concerns[key].score })),
    moderateConcerns: moderateConcerns.map(key => ({ key, score: data.concerns[key].score })),
    requiredCoverage: [...requiredCoverage],
    skippedUnmappableMildConcerns: moderateConcerns.length === 0
      ? eligibleTargets.filter(target => !safelyMappableMildTargets.includes(target))
      : [],
    requiredTreatmentCoverage: [...requiredTreatmentCoverage],
    coveredConcerns: [...coveredTargets].sort(),
    uncoveredModerateConcerns: moderateConcerns.filter(key => !coveredTargets.has(key)),
    suggestedComboValidated,
    photoObservationInferenceGuard: {
      fields: ['summary', 'positiveHighlights.title', 'positiveHighlights.detail', 'concerns.*.description'],
      causeOrHistory: 'passed',
      unmeasuredPhysicalState: 'passed',
      recommendationMechanismsExcluded: true,
    },
  };
}

function assertNonClinicalVisualOracle(testCase, data) {
  const oracle = testCase.visualOracle;
  if (!oracle) return null;

  const evidence = {
    qualification: 'Non-clinical human visual-regression oracle; not a diagnosis, provider assessment, or clinical validation.',
    reviewNotes: oracle.reviewNotes || null,
    checked: [],
  };
  if (Array.isArray(oracle.anyVisibleConcern) && oracle.anyVisibleConcern.length > 0) {
    const matched = oracle.anyVisibleConcern.filter(key => data.concerns?.[key]?.score > 10);
    assert.ok(
      matched.length > 0,
      `${testCase.id}: non-clinical visual oracle expects at least one visibly scored concern from ${oracle.anyVisibleConcern.join(', ')}`,
    );
    evidence.checked.push({ rule: 'any-visible-concern', expected: oracle.anyVisibleConcern, matched });
  }
  if (Array.isArray(oracle.anyTreatment) && oracle.anyTreatment.length > 0) {
    const actualTreatments = data.recommendations.map(item => item.treatment);
    const matched = oracle.anyTreatment.filter(treatment => actualTreatments.includes(treatment));
    assert.ok(
      matched.length > 0,
      `${testCase.id}: non-clinical visual oracle expects at least one mapped treatment from ${oracle.anyTreatment.join(', ')}`,
    );
    evidence.checked.push({ rule: 'any-treatment', expected: oracle.anyTreatment, matched });
  }
  if (Number.isInteger(oracle.maximumConcernScore)) {
    const maximum = Math.max(...Object.values(data.concerns).map(item => item.score));
    assert.ok(
      maximum <= oracle.maximumConcernScore,
      `${testCase.id}: maintenance-control image should not produce a concern above ${oracle.maximumConcernScore}`,
    );
    evidence.checked.push({ rule: 'maximum-concern-score', expected: oracle.maximumConcernScore, actual: maximum });
  }
  if (oracle.maximumScores && typeof oracle.maximumScores === 'object') {
    for (const [key, maximum] of Object.entries(oracle.maximumScores)) {
      assert.ok(Number.isInteger(maximum) && data.concerns?.[key], `${testCase.id}: oracle score ceiling is well formed for ${key}`);
      assert.ok(
        data.concerns[key].score <= maximum,
        `${testCase.id}: non-clinical visual oracle expects ${key} at or below ${maximum}`,
      );
      evidence.checked.push({ rule: 'concern-score-ceiling', concern: key, expected: maximum, actual: data.concerns[key].score });
    }
  }
  if (Array.isArray(oracle.forbiddenTreatments) && oracle.forbiddenTreatments.length > 0) {
    const actualTreatments = data.recommendations.map(item => item.treatment);
    const unexpected = oracle.forbiddenTreatments.filter(treatment => actualTreatments.includes(treatment));
    assert.deepEqual(
      unexpected,
      [],
      `${testCase.id}: non-clinical visual oracle excludes treatments that would be forced without a visible appearance basis`,
    );
    evidence.checked.push({ rule: 'forbidden-treatments', expectedAbsent: oracle.forbiddenTreatments, actualTreatments });
  }
  return evidence;
}

async function assertRenderedDomParity(page, testCase, data) {
  const rendered = await page.evaluate(() => {
    const isRenderable = element => {
      if (!element || element.hidden) return false;
      const style = getComputedStyle(element);
      const rect = element.getBoundingClientRect();
      return style.display !== 'none'
        && style.visibility !== 'hidden'
        && Number(style.opacity) !== 0
        && rect.width > 0
        && rect.height > 0;
    };
    const text = element => (element?.textContent || '').replace(/\s+/g, ' ').trim();
    const positiveLead = document.getElementById('positiveLead');
    const concernsGrid = document.getElementById('concernsGrid');
    const recommendations = document.getElementById('recommendationsSection');
    const planButton = document.getElementById('resultsPlanButton');
    const combo = document.getElementById('comboPlay');
    return {
      horizontalOverflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
      order: {
        positiveBeforeConcerns: Boolean(
          positiveLead.compareDocumentPosition(concernsGrid) & Node.DOCUMENT_POSITION_FOLLOWING
        ),
        concernsBeforeRecommendations: Boolean(
          concernsGrid.compareDocumentPosition(recommendations) & Node.DOCUMENT_POSITION_FOLLOWING
        ),
      },
      positives: [...document.querySelectorAll('#positiveHighlights .positive-highlight')].map(card => ({
        title: text(card.querySelector('.positive-highlight-title')),
        detail: text(card.querySelector('.positive-highlight-detail')),
      })),
      concerns: [...document.querySelectorAll('#concernsGrid .concern-card')].map(card => ({
        name: text(card.querySelector('.concern-name')),
        description: text(card.querySelector('.concern-description')),
        displayedHealthScore: Number(text(card.querySelector('.concern-score-number'))),
        hidden: card.hidden,
      })),
      treatments: [...document.querySelectorAll('#treatmentRecommendationCards .recommendation-card')].map(card => ({
        name: text(card.querySelector('.rec-treatment')),
        reason: text(card.querySelector('.rec-reason')),
        priority: text(card.querySelector('.rec-priority')),
        hidden: card.hidden,
      })),
      products: [...document.querySelectorAll('#productRecommendationCards .recommendation-card')].map(card => ({
        name: text(card.querySelector('.rec-treatment')),
        reason: text(card.querySelector('.rec-reason')),
        priority: text(card.querySelector('.rec-priority')),
      })),
      resultsVisible: isRenderable(document.getElementById('resultsSection'))
        && document.getElementById('resultsSection')?.classList.contains('show'),
      recommendationsVisible: isRenderable(recommendations)
        && recommendations?.classList.contains('show'),
      treatmentGroupVisible: isRenderable(document.getElementById('treatmentRecommendationsGroup')),
      productGroupVisible: isRenderable(document.getElementById('productRecommendationsGroup')),
      planTeaserVisible: isRenderable(document.getElementById('resultsPlanTeaser')),
      planButtonVisible: isRenderable(planButton),
      planButtonHref: planButton?.getAttribute('href') || null,
      planCount: text(document.getElementById('resultsPlanCount')),
      treatmentOptionsToggle: {
        hidden: document.getElementById('treatmentOptionsToggle')?.hidden ?? true,
        expanded: document.getElementById('treatmentOptionsToggle')?.getAttribute('aria-expanded') || null,
        text: text(document.getElementById('treatmentOptionsToggle')),
      },
      reportActionVisible: isRenderable(document.getElementById('downloadReportBtn')),
      ctaVisible: isRenderable(document.getElementById('ctaSection'))
        && document.getElementById('ctaSection')?.classList.contains('show'),
      bookingActionCount: [...document.querySelectorAll('a[href*="booking.vonandcoaesthetics.com"]')]
        .filter(isRenderable).length,
      comboVisible: isRenderable(combo),
      comboTitle: text(combo?.querySelector('.combo-play-title')),
      comboText: text(combo),
      recoveryVisible: document.getElementById('analysisRecovery')?.classList.contains('show') || false,
      analysisObservationText: (document.getElementById('resultsSection')?.innerText || '')
        .replace(/\s+/g, ' ')
        .trim(),
      removedElements: document.querySelectorAll(
        '#skinAge, [id*="skinAge"], .skin-age, [class*="skin-age"], #radarChart, #radarContainer, .skin-radar'
      ).length,
    };
  });
  rendered.guestText = await captureGuestFacingPageText(page);

  assert.ok(rendered.horizontalOverflow <= 1, `${testCase.id}: mobile result has no horizontal overflow`);
  assert.equal(rendered.order.positiveBeforeConcerns, true, `${testCase.id}: positive section precedes findings in the DOM`);
  assert.equal(rendered.order.concernsBeforeRecommendations, true, `${testCase.id}: findings precede recommendations in the DOM`);
  assert.deepEqual(
    rendered.positives,
    data.positiveHighlights.map(item => ({ title: normalizeText(item.title), detail: normalizeText(item.detail) })),
    `${testCase.id}: every positive title and detail renders verbatim and in order`,
  );
  const concernNames = {
    wrinkles: 'Wrinkles & Fine Lines', redness: 'Visible Redness', darkSpots: 'Dark Spots & Pigment Variation',
    texture: 'Skin Texture & Smoothness', pores: 'Pore Size & Visibility', laxity: 'Skin Laxity & Crepiness',
    sunDamage: 'Visible Sun-Exposure Signs', veins: 'Visible Veins', scarring: 'Visible Textural Marks',
    hairRemoval: 'Unwanted Hair', acne: 'Visible Breakouts & Congestion', dryness: 'Visible Dryness',
    unevenTone: 'Uneven Skin Tone',
  };
  const expectedConcerns = Object.entries(data.concerns)
    .sort(([, left], [, right]) => right.score - left.score)
    .map(([key, concern], index) => ({
      name: concernNames[key] || key,
      description: normalizeText(concern.description),
      displayedHealthScore: 100 - concern.score,
      hidden: index >= 3,
    }));
  assert.deepEqual(rendered.concerns, expectedConcerns, `${testCase.id}: every scored finding renders verbatim in score order`);
  assert.deepEqual(
    rendered.treatments,
    data.recommendations.map((item, index) => ({
      name: normalizeText(item.treatment),
      reason: normalizeText(item.reason),
      priority: index === 0 ? 'Top Pick' : `Priority ${item.priority}`,
      hidden: index >= 3,
    })),
    `${testCase.id}: every treatment, reason, and priority renders verbatim and in order`,
  );
  assert.deepEqual(
    rendered.products,
    data.productRecommendations.map(item => ({
      name: normalizeText(item.product),
      reason: normalizeText(item.reason),
      priority: 'Skincare',
    })),
    `${testCase.id}: every product and reason renders verbatim and in order`,
  );
  assert.equal(rendered.resultsVisible, true, `${testCase.id}: results are visible`);
  assert.equal(rendered.recommendationsVisible, true, `${testCase.id}: recommendation section is visible`);
  assert.equal(
    rendered.treatmentGroupVisible,
    data.recommendations.length > 0,
    `${testCase.id}: treatment group visibility matches whether treatments were returned`,
  );
  assert.equal(rendered.productGroupVisible, true, `${testCase.id}: product group is visible`);
  assert.equal(rendered.planTeaserVisible, true, `${testCase.id}: plan teaser is visible`);
  assert.equal(rendered.planButtonVisible, true, `${testCase.id}: plan CTA is visible`);
  assert.equal(rendered.planButtonHref, '#recommendationsSection', `${testCase.id}: plan CTA targets recommendations`);
  const additionalTreatmentCount = Math.max(0, data.recommendations.length - 3);
  assert.equal(
    rendered.treatmentOptionsToggle.hidden,
    additionalTreatmentCount === 0,
    `${testCase.id}: treatment disclosure visibility matches the returned count`,
  );
  assert.equal(
    rendered.treatmentOptionsToggle.expanded,
    'false',
    `${testCase.id}: treatment disclosure starts collapsed`,
  );
  if (additionalTreatmentCount > 0) {
    assert.equal(
      rendered.treatmentOptionsToggle.text,
      `See ${additionalTreatmentCount} More Treatment ${additionalTreatmentCount === 1 ? 'Option' : 'Options'}`,
      `${testCase.id}: treatment disclosure names the exact remaining count`,
    );
    await page.locator('#treatmentOptionsToggle').click();
    const expandedTreatmentState = await page.evaluate(() => ({
      expanded: document.getElementById('treatmentOptionsToggle').getAttribute('aria-expanded'),
      hiddenCards: document.querySelectorAll('#treatmentRecommendationCards .recommendation-card[hidden]').length,
      visibleCards: [...document.querySelectorAll('#treatmentRecommendationCards .recommendation-card')]
        .filter(card => !card.hidden && getComputedStyle(card).display !== 'none').length,
    }));
    assert.deepEqual(
      expandedTreatmentState,
      {
        expanded: 'true',
        hiddenCards: 0,
        visibleCards: data.recommendations.length,
      },
      `${testCase.id}: treatment disclosure reveals every returned option`,
    );
  }
  assert.equal(rendered.reportActionVisible, true, `${testCase.id}: printable plan action is visible`);
  assert.equal(rendered.ctaVisible, true, `${testCase.id}: consultation CTA section is visible`);
  assert.ok(rendered.bookingActionCount >= 1, `${testCase.id}: at least one booking CTA is visible`);
  assert.equal(rendered.recoveryVisible, false, `${testCase.id}: recovery is hidden after success`);
  assert.equal(rendered.removedElements, 0, `${testCase.id}: skin age and radar remain removed`);
  assertAppearanceOnlyGuestText(
    testCase.id,
    'entire rendered page including preview, results, recommendations, actions, CTA, and footer',
    rendered.guestText,
  );
  assertPhotoObservationIsVisuallyGrounded(
    testCase.id,
    'rendered results analysis section with recommendation mechanisms excluded',
    rendered.analysisObservationText,
  );
  assert.equal(
    rendered.planCount,
    `(${data.recommendations.length + data.productRecommendations.length})`,
    `${testCase.id}: plan guide exposes the full recommendation count`,
  );
  if (data.suggestedCombo) {
    assert.equal(rendered.comboVisible, true, `${testCase.id}: valid suggested combo renders`);
    assert.equal(
      rendered.comboTitle,
      data.suggestedCombo,
      `${testCase.id}: rendered combo title preserves the exact canonical catalog name`,
    );
  } else {
    assert.equal(rendered.comboVisible, false, `${testCase.id}: no phantom combo renders`);
    assert.equal(rendered.comboTitle, '', `${testCase.id}: no phantom combo title renders`);
  }
  return rendered;
}

async function assertPrintableReport(context, page, testCase, data) {
  const popupPromise = context.waitForEvent('page', { timeout: 4000 }).catch(() => null);
  await page.locator('#downloadReportBtn').click();
  const popup = await popupPromise;
  let reportSurface = popup;
  let delivery = 'new-tab';
  if (popup) {
    await popup.waitForLoadState('domcontentloaded');
    await popup.setViewportSize({ width: 900, height: 1200 });
    await popup.emulateMedia({ media: 'print' });
  } else {
    delivery = 'inline-fallback';
    await page.waitForFunction(() => {
      const container = document.getElementById('reportPreviewContainer');
      const frame = document.getElementById('reportPreviewFrame');
      return getComputedStyle(container).display !== 'none' && frame?.contentDocument?.body;
    }, null, { timeout: 5000 });
    const frameHandle = await page.locator('#reportPreviewFrame').elementHandle();
    reportSurface = await frameHandle.contentFrame();
  }
  assert.ok(reportSurface, `${testCase.id}: printable report opens in a new tab or inline fallback`);

  const report = await reportSurface.evaluate(treatmentCount => {
    const normalize = value => String(value || '').replace(/\s+/g, ' ').trim();
    const accessibleText = [...document.querySelectorAll('[aria-label], [title], img[alt]')]
      .flatMap(element => [
        element.getAttribute('aria-label'),
        element.getAttribute('title'),
        element.getAttribute('alt'),
      ])
      .filter(Boolean)
      .join(' ');
    const leafWithText = expected => [...document.querySelectorAll('div')]
      .find(element => element.children.length === 0 && normalize(element.textContent) === expected);
    const positiveSection = leafWithText('Begin With the Positive')?.parentElement;
    const summarySection = leafWithText('Summary')?.parentElement;
    const findingsSection = [...document.querySelectorAll('.report-section-title')]
      .find(element => normalize(element.textContent) === 'Skin Analysis Results')
      ?.parentElement;
    return {
      text: `${document.body?.innerText || ''} ${accessibleText}`.replace(/\s+/g, ' ').trim(),
      observationText: [positiveSection, summarySection, findingsSection]
        .map(element => element?.innerText || '')
        .join(' ')
        .replace(/\s+/g, ' ')
        .trim(),
      title: document.title,
      horizontalOverflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
      hasLogo: Boolean(document.querySelector('img[alt*="Von & Co"]')),
      reportItems: document.querySelectorAll('.report-item').length,
      treatmentPriorities: [...document.querySelectorAll('.report-item')]
        .slice(0, treatmentCount)
        .map(item => (item.firstElementChild?.textContent || '').replace(/\s+/g, ' ').trim()),
    };
  }, data.recommendations.length);
  assert.match(report.title, /Von & Co Aesthetics - Treatment Plan/i, `${testCase.id}: report has the expected title`);
  assert.equal(report.hasLogo, true, `${testCase.id}: report includes the Von & Co logo`);
  assert.ok(report.horizontalOverflow <= 1, `${testCase.id}: report has no horizontal overflow at print viewport`);
  assert.match(report.text, /Begin With the Positive/i, `${testCase.id}: report leads with positive highlights`);
  assert.match(report.text, /Skin Analysis Results/i, `${testCase.id}: report includes findings`);
  assert.match(report.text, /Recommended Treatments/i, `${testCase.id}: report includes treatments`);
  assert.match(report.text, /Your Skincare Essentials/i, `${testCase.id}: report includes products`);
  assert.match(report.text, /does not replace a professional consultation/i, `${testCase.id}: report carries the consultation disclaimer`);
  assert.match(report.text, /concerning lesion needs an in-person medical evaluation/i, `${testCase.id}: report carries the lesion disclaimer`);
  assert.doesNotMatch(report.text, /estimated skin age|radar chart/i, `${testCase.id}: report omits age and radar`);
  assertAppearanceOnlyGuestText(testCase.id, 'printable take-home report', report.text);
  assert.match(
    report.observationText,
    /Begin With the Positive.*Summary.*Skin Analysis Results/i,
    `${testCase.id}: scoped report observation text includes highlights, summary, and findings in order`,
  );
  assertPhotoObservationIsVisuallyGrounded(
    testCase.id,
    'rendered report highlights, summary, and findings with recommendation mechanisms excluded',
    report.observationText,
  );
  const sectionIndexes = [
    report.text.indexOf('Begin With the Positive'),
    report.text.indexOf('Skin Analysis Results'),
    report.text.indexOf('Recommended Treatments'),
    report.text.indexOf('Your Skincare Essentials'),
  ];
  assert.ok(
    sectionIndexes.every((value, index) => value >= 0 && (index === 0 || value > sectionIndexes[index - 1])),
    `${testCase.id}: printable report keeps positives before findings before treatments before skincare`,
  );
  for (const value of [
    ...data.positiveHighlights.flatMap(item => [item.title, item.detail]),
    ...Object.values(data.concerns).map(item => item.description),
    ...data.recommendations.flatMap(item => [item.treatment, item.reason]),
    ...data.productRecommendations.flatMap(item => [item.product, item.reason]),
  ]) {
    assert.ok(report.text.includes(normalizeText(value)), `${testCase.id}: printable report preserves: ${normalizeText(value)}`);
  }
  assert.equal(
    report.reportItems,
    data.recommendations.length + data.productRecommendations.length,
    `${testCase.id}: report includes every treatment and product item`,
  );
  assert.deepEqual(
    report.treatmentPriorities,
    data.recommendations.map(item => String(item.priority)),
    `${testCase.id}: printable report preserves every treatment priority`,
  );

  const screenshotPath = path.join(artifactDir, `${testCase.id}-${testCase.area}-print-report.png`);
  if (popup) {
    await popup.screenshot({ path: screenshotPath, fullPage: true });
    await popup.close();
  } else {
    await page.locator('#reportPreviewContainer').screenshot({ path: screenshotPath });
  }
  return { checked: true, delivery, screenshotPath, snapshot: report };
}

async function runCase(browser, testCase, photo) {
  const context = await browser.newContext(mobileContextOptions());
  await context.route('https://fonts.googleapis.com/**', route => route.fulfill({
    status: 200,
    contentType: 'text/css',
    body: '',
  }));
  await context.route('https://fonts.gstatic.com/**', route => route.abort());
  const page = await context.newPage();
  const pageErrors = [];
  const consoleErrors = [];
  const requestFailures = [];
  const screenshots = [];
  let apiResponse = null;
  let apiResponsePromise = null;
  let terminal = null;
  let preAnalysisLabels = null;
  let data = null;
  let startedAtMs = null;
  let completedAtMs = null;
  page.on('pageerror', error => pageErrors.push(String(error)));
  page.on('console', message => {
    if (message.type() === 'error') consoleErrors.push(message.text());
  });
  page.on('requestfailed', request => {
    if (request.url().startsWith('https://fonts.gstatic.com/')) return;
    requestFailures.push({ url: request.url(), errorText: request.failure()?.errorText || null });
  });
  page.on('response', response => {
    if (new URL(response.url()).pathname !== '/api/analyze') return;
    if (apiResponsePromise === null) apiResponsePromise = captureAnalyzeResponse(response);
  });

  try {
    await page.goto(baseUrl, { waitUntil: 'domcontentloaded' });
    await page.locator('#bodyAreaSelect').selectOption(testCase.area);
    preAnalysisLabels = await assertPreAnalysisLabels(page, testCase);
    startedAtMs = Date.now();
    await page.locator('#fileInput').setInputFiles(photo.sourcePath);
    await page.waitForFunction(() => (
      document.getElementById('leadGateOverlay')?.classList.contains('show')
        || document.getElementById('analysisRecovery')?.classList.contains('show')
        || Boolean(document.getElementById('rejectionOverlay'))
    ), null, { timeout: resultDeadlineMs });
    completedAtMs = Date.now();
    await page.waitForTimeout(100);
    assert.ok(apiResponsePromise, `${testCase.id}: browser received an /api/analyze response`);
    apiResponse = await apiResponsePromise;

    const elapsedMs = completedAtMs - startedAtMs;
    const timing = {
      startedAt: new Date(startedAtMs).toISOString(),
      completedAt: new Date(completedAtMs).toISOString(),
      elapsedMs,
      deadlineMs: resultDeadlineMs,
      withinDeadline: elapsedMs < resultDeadlineMs,
    };
    assert.ok(timing.withinDeadline, `${testCase.id}: terminal result arrives in strictly less than 75 seconds`);
    terminal = await page.evaluate(() => ({
      lead: document.getElementById('leadGateOverlay')?.classList.contains('show') || false,
      recovery: document.getElementById('analysisRecovery')?.classList.contains('show') || false,
      recoveryTitle: document.getElementById('analysisRecoveryTitle')?.textContent?.trim() || null,
      recoveryMessage: document.getElementById('analysisRecoveryMessage')?.textContent?.trim() || null,
      rejection: Boolean(document.getElementById('rejectionOverlay')),
      rejectionTitle: document.getElementById('rejectionTitle')?.textContent?.trim() || null,
      rejectionMessage: document.getElementById('rejectionReason')?.textContent?.trim() || null,
    }));
    terminal.guestFacingPageText = await captureGuestFacingPageText(page);
    assertAppearanceOnlyGuestText(
      testCase.id,
      'terminal lead-gate or rejection page including navigation, actions, and footer',
      terminal.guestFacingPageText,
    );

    if (testCase.expectedOutcome !== 'accepted') {
      assert.equal(terminal.rejection, true, `${testCase.id}: negative control is rejected`);
      assert.equal(terminal.lead, false, `${testCase.id}: rejection does not open the result gate`);
      assert.equal(terminal.recovery, false, `${testCase.id}: rejection does not become a retryable failure`);
      assert.equal(apiResponse?.httpStatus, 422, `${testCase.id}: rejection returns HTTP 422`);
      assert.equal(apiResponse?.rejected, true, `${testCase.id}: response is marked rejected`);
      assert.ok(apiResponse?.rawJson, `${testCase.id}: full rejection JSON was captured`);
      if (testCase.expectedReasonCode) {
        assert.equal(
          apiResponse.reasonCode,
          testCase.expectedReasonCode,
          `${testCase.id}: rejection reason code is exact`,
        );
      }
      if (testCase.expectedRejectionSource) {
        assert.equal(
          apiResponse.rejectionSource,
          testCase.expectedRejectionSource,
          `${testCase.id}: rejection source is deterministic and exact`,
        );
        assert.equal(
          apiResponse.headers.analysisRepeat,
          null,
          `${testCase.id}: a local preflight rejection is not a cached model result`,
        );
      }
      if (testCase.expectedObservedArea) {
        assert.equal(
          apiResponse.observedArea,
          testCase.expectedObservedArea,
          `${testCase.id}: rejected photo reports the exact dominant observed area`,
        );
      }
      if (testCase.expectedObservedAreaOneOf) {
        assert.ok(
          testCase.expectedObservedAreaOneOf.includes(apiResponse.observedArea),
          `${testCase.id}: rejected photo reports one supported non-selected observed area`,
        );
      }
      if (testCase.expectedObservedAreaNot) {
        assert.notEqual(
          apiResponse.observedArea,
          testCase.expectedObservedAreaNot,
          `${testCase.id}: mismatch control never reports the selected area as observed`,
        );
      }
      assert.match(
        String(apiResponse?.rejectionReason || ''),
        testCase.expectedRejection,
        `${testCase.id}: rejection gives the expected correction guidance`,
      );
      assert.equal(
        normalizeText(terminal.rejectionMessage),
        normalizeText(apiResponse.rejectionReason),
        `${testCase.id}: rendered rejection guidance exactly matches the API response`,
      );
      const screenshotPath = path.join(
        artifactDir,
        `${testCase.id}-${testCase.engine}-${testCase.expectedOutcome}.png`,
      );
      await page.screenshot({ path: screenshotPath, fullPage: true });
      screenshots.push(screenshotPath);
      assert.deepEqual(pageErrors, [], `${testCase.id}: no page errors`);
      const expectedHttpErrors = consoleErrors.filter(message => (
        /^Failed to load resource: the server responded with a status of 422 \(UNPROCESSABLE ENTITY\)$/.test(message)
      ));
      assert.equal(
        expectedHttpErrors.length,
        consoleErrors.length,
        `${testCase.id}: the only allowed console error is the expected HTTP 422 resource response`,
      );
      assert.ok(
        expectedHttpErrors.length <= 1,
        `${testCase.id}: at most one expected HTTP 422 resource message is emitted`,
      );
      assert.deepEqual(requestFailures, [], `${testCase.id}: no unexpected request failures`);
      const record = {
        id: testCase.id,
        status: 'passed',
        outcome: testCase.expectedOutcome,
        engine: testCase.engine,
        bodyArea: testCase.area,
        sourceUrl: photo.sourceUrl,
        licenseUrl: photo.licenseUrl,
        sourcePath: photo.sourcePath,
        sha256: photo.sha256,
        bytes: photo.bytes,
        photo,
        timing,
        apiResponse,
        terminal,
        preAnalysisLabels,
        response: {
          status: 'expected-rejection',
          summary: null,
          positiveHighlights: null,
          concerns: null,
          treatments: null,
          products: null,
          suggestedCombo: null,
          rejection: {
            reason: apiResponse.rejectionReason,
            title: terminal.rejectionTitle,
            message: terminal.rejectionMessage,
          },
        },
        screenshots,
        pageErrors,
        consoleErrors,
        requestFailures,
      };
      record.attemptArtifactPath = writeAttemptArtifact(record);
      return record;
    }

    if (!terminal.lead) {
      throw new Error(`${testCase.id}: no completed result: ${JSON.stringify({ terminal, apiResponse })}`);
    }
    assert.equal(terminal.recovery, false, `${testCase.id}: success does not show recovery`);
    assert.equal(terminal.rejection, false, `${testCase.id}: accepted photo is not rejected`);

    data = await page.evaluate(() => pendingAnalysisData);
    assert.equal(apiResponse?.httpStatus, 200, `${testCase.id}: API returned HTTP 200`);
    assert.ok(apiResponse?.rawJson, `${testCase.id}: full successful API JSON was captured`);
    const browserStateBeforePresentationFlag = structuredClone(data);
    delete browserStateBeforePresentationFlag._isLive;
    assert.deepEqual(
      apiResponse.rawJson,
      browserStateBeforePresentationFlag,
      `${testCase.id}: browser state exactly matches full API JSON except the frontend live-result marker`,
    );
    const contractEvidence = assertStructuredResult(testCase, data);
    const visualOracleEvidence = assertNonClinicalVisualOracle(testCase, data);

    await page.getByRole('button', { name: 'Skip for now' }).click();
    await page.waitForSelector('#resultsSection.show', { timeout: 10000 });
    await page.waitForTimeout(2500);
    const rendered = await assertRenderedDomParity(page, testCase, data);
    assert.deepEqual(pageErrors, [], `${testCase.id}: no page errors`);
    assert.deepEqual(consoleErrors, [], `${testCase.id}: no console errors`);
    assert.deepEqual(requestFailures, [], `${testCase.id}: no unexpected request failures`);

    let reportEvidence = { checked: false, reason: 'Another accepted photo already covered this area.' };
    if (!printAreasValidated.has(testCase.area)) {
      reportEvidence = await assertPrintableReport(context, page, testCase, data);
      printAreasValidated.add(testCase.area);
      screenshots.push(reportEvidence.screenshotPath);
    }

    const screenshotPath = path.join(artifactDir, `${testCase.id}-${testCase.engine}-results.png`);
    await page.screenshot({ path: screenshotPath, fullPage: true });
    screenshots.push(screenshotPath);
    const record = {
      id: testCase.id,
      status: 'passed',
      outcome: 'completed-live-result',
      engine: testCase.engine,
      bodyArea: testCase.area,
      sourceUrl: photo.sourceUrl,
      licenseUrl: photo.licenseUrl,
      sourcePath: photo.sourcePath,
      sha256: photo.sha256,
      bytes: photo.bytes,
      photo,
      timing,
      apiResponse,
      terminal,
      preAnalysisLabels,
      response: snapshotAnalysis(data),
      contractEvidence,
      visualOracleEvidence,
      rendered,
      reportEvidence,
      screenshots,
      pageErrors,
      consoleErrors,
      requestFailures,
    };
    record.attemptArtifactPath = writeAttemptArtifact(record);
    return record;
  } catch (error) {
    if (completedAtMs === null && startedAtMs !== null) completedAtMs = Date.now();
    if (apiResponse === null && apiResponsePromise !== null) {
      try {
        apiResponse = await apiResponsePromise;
      } catch (captureError) {
        apiResponse = {
          httpStatus: null,
          rawJson: null,
          rawText: null,
          parseError: `Response capture failed: ${String(captureError)}`,
        };
      }
    }
    try {
      if (!page.isClosed()) {
        const screenshotPath = path.join(artifactDir, `${testCase.id}-${testCase.engine}-failure.png`);
        await page.screenshot({ path: screenshotPath, fullPage: true });
        screenshots.push(screenshotPath);
      }
    } catch (_) {
      // Preserve the original failure when a diagnostic screenshot is unavailable.
    }
    error.qa = {
      id: testCase.id,
      status: 'failed',
      outcome: testCase.expectedOutcome === 'accepted' ? 'failed-live-result' : 'unexpected-control-outcome',
      engine: testCase.engine,
      bodyArea: testCase.area,
      sourceUrl: photo.sourceUrl,
      licenseUrl: photo.licenseUrl,
      sourcePath: photo.sourcePath,
      sha256: photo.sha256,
      bytes: photo.bytes,
      photo,
      timing: startedAtMs === null ? null : {
        startedAt: new Date(startedAtMs).toISOString(),
        completedAt: new Date(completedAtMs).toISOString(),
        elapsedMs: completedAtMs - startedAtMs,
        deadlineMs: resultDeadlineMs,
        withinDeadline: completedAtMs - startedAtMs < resultDeadlineMs,
      },
      apiResponse,
      terminal,
      preAnalysisLabels,
      response: snapshotAnalysis(data, 'failed-contract-validation'),
      screenshots,
      pageErrors,
      consoleErrors,
      requestFailures,
      assertionError: {
        message: String(error),
        stack: error.stack || null,
      },
    };
    writeAttemptArtifact(error.qa);
    throw error;
  } finally {
    await context.close();
  }
}

function createRunLedger(manifest) {
  return {
    schemaVersion: 4,
    harnessVersion,
    suite: 'strict-live-real-photo-matrix',
    status: 'running',
    acceptanceRun: caseFilter.size === 0,
    startedAt: new Date().toISOString(),
    completedAt: null,
    baseUrl,
    photoDir,
    artifactDir,
    ledgerPath,
    resultDeadlineMs,
    expectedBuildFingerprint,
    sourceBuildFingerprint,
    runtimeSourceFiles,
    expectedRuntime,
    qualification: {
      validates: 'Live functional reliability, selected-area handling, response-contract enforcement, catalog mapping, non-clinical appearance regression oracles, and browser/report presentation parity.',
      doesNotValidate: 'Clinical accuracy, medical diagnosis, treatment candidacy, or ten verified unique human identities. Ten distinct pinned source photos are tested.',
      browserEngines: 'Automated local Google Chrome (Chromium) and Playwright WebKit. WebKit is not a claim of testing physical Mobile Safari on an iPhone.',
      demoModeOwnership: 'Deterministic demo API mocking and sample-result disclosure are owned by tests/browser_regression.cjs. This suite intentionally rejects demo data so mocked presentation evidence cannot be mistaken for live-model acceptance.',
    },
    selectedCaseIds: cases.map(testCase => testCase.id),
    contract: {
      manifest: '10 distinct accepted Pexels source photos, exactly 2 per area and alternating Chromium/WebKit, plus deterministic face-as-legs and preserved shoulder-as-hands mismatch controls, and 1 distinct dark quality-rejection attempt',
      uniqueness: 'all 10 accepted filenames, provenance URLs, and pinned SHA-256 hashes are unique; no unique-person identity claim is made',
      scoreMeaning: 'visible-concern severity from 0 through 100; higher is worse',
      severityBands: { none: '0-10', mild: '11-40', moderate: '41-60', severe: '61-100' },
      targetEligibility: 'treatment and non-SPF product targets require concern score > 10',
      halo: 'Sciton Halo requires at least one listed target score >= 41',
      treatmentIntensity: 'Sculptra requires at least one listed target score >= 41 and is not cataloged for hands or veins; Sciton BBL on hands also requires at least one listed target score >= 41',
      redness: 'when redness >= 41, exclude exact standard Microneedling and RF Microneedling; Microneedling + PRF remains independently catalog-mapped',
      recommendationCount: 'zero through six; no minimum, no padding, and no treatment is forced when every concern score is 10 or below',
      priority: 'returned treatment priorities are unique, sequential, and preserve API order in both the result page and printable report; priority one maps to the leading concern actually covered by a selected service, and the first two services cover the first two service-covered concerns when present',
      coverage: 'treatment plus skincare covers every moderate visible concern; when all findings are mild, it covers the highest concern with a permitted catalog path and records any higher unmappable mild finding without forcing a contraindicated option; services cover every moderate concern with an allowed treatment match; a sparse all-10-or-below result remains valid with zero services',
      products: 'one through the selected area catalog maximum, including at least one SPF; every non-SPF product maps to a visible score above 10; no artificial 2-3 quota',
      combo: 'suggestedCombo is null or one exact canonical Von catalog name whose required component groups all appear; exact official treatment, product, and combo names are isolated before prose-language scans and are never renamed',
      copy: 'positive-first, exact score-grounded strengths using direct copy only when supported and explicitly comparative copy otherwise; all guest-facing prose outside exact catalog names has no medical-condition labels, unsupported finding terminology, medical lesion triage, prohibited overclaims, absence framing, anatomy mismatch, or en/em dash',
      photoObservation: 'raw summary, positive highlights, and finding descriptions plus their rendered result/report surfaces may describe only visible appearance; they cannot infer sun-exposure or shaving cause/history or unmeasured firmness, elasticity, hydration, moisture, suppleness, thickness, barrier, collagen, volume, or underlying support; treatment and product mechanisms are deliberately outside this scoped assertion',
      timing: 'measure from file input assignment through terminal result and require strictly less than 75,000 ms',
      rawEvidence: 'capture the complete parsed /api/analyze JSON, or the unparsed response text and parse error, for every attempt',
      interface: 'area-specific exact appearance-only five-label preview; positive-before-findings DOM order; verbatim parity for all findings, recommendations, reasons, priorities, and canonical combo titles; visible plan/report/consultation actions; appearance-only scans cover all visible and accessible page copy from navigation through footer without scanning internal scripts',
      mobileEmulation: 'every live case uses a 390x844 touch-enabled mobile context with a 390x844 screen, device scale factor 3, and isMobile=true in Chrome or WebKit',
      report: 'open and inspect one printable report for each of the five selected areas, with full content parity, appearance-only guest prose, disclaimers, section order, and no horizontal overflow',
      controls: 'a face uploaded with legs selected and the preserved shoulder/back-dominant crop uploaded with hands selected must return area_mismatch; a pinned dark back image must return a deterministic local_underexposure quality rejection with no model-repeat header',
      runtime: 'start and end health checks must match the deterministic SHA-256 of server.py, index.html, and all three logo assets; both must also report Gemini 3.1 Pro Preview, HIGH thinking, 70,000 ms total budget, 15,000 ms hedge delay, and 32,768 output tokens',
      visualOracle: 'limited non-clinical human visual-regression expectations are explicit per fixture and must never be described as provider-labeled or clinically validated',
    },
    manifestSummary: {
      total: manifest.entries.length,
      acceptedRealSourcePhotos: manifest.entries.filter(entry => entry.expectedOutcome === 'accepted').length,
      qualityRejectionControls: manifest.entries.filter(entry => entry.expectedOutcome === 'quality-rejection').length,
      areaMismatchControls: manifest.entries.filter(entry => entry.expectedOutcome === 'area-mismatch').length,
      analyzableAreaCounts: manifest.analyzableAreaCounts,
      uniqueAcceptedSha256Count: manifest.uniqueAcceptedSha256Count,
      uniqueAttemptSha256Count: manifest.uniqueAttemptSha256Count,
    },
    manifest: manifest.entries,
    health: null,
    healthAfter: null,
    summary: {
      selected: cases.length,
      completedLiveResults: 0,
      qualityRejections: 0,
      areaMismatches: 0,
      printAreasValidated: [],
      failed: 0,
    },
    records: [],
    harnessError: null,
  };
}

async function main() {
  fs.mkdirSync(artifactDir, { recursive: true });
  const manifest = buildAndValidateManifest();
  assert.ok(cases.length > 0, 'REAL_CASE_FILTER must match at least one case');
  assert.ok(expectedBuildFingerprint, 'TEST_BUILD_FINGERPRINT is required to prevent certifying a stale listener');
  assert.equal(
    expectedBuildFingerprint,
    sourceBuildFingerprint,
    'TEST_BUILD_FINGERPRINT must be the SHA-256 fingerprint of the current runtime source files',
  );
  const run = createRunLedger(manifest);
  writeLedger(run);
  const browsers = {};
  const failures = [];
  try {
    const healthResponse = await fetch(`${baseUrl}/api/health`, {
      signal: AbortSignal.timeout(10000),
    });
    const health = await healthResponse.json();
    run.health = { httpStatus: healthResponse.status, ...health };
    assert.equal(healthResponse.status, 200, 'health endpoint returns HTTP 200');
    assert.equal(health.mode, 'live', 'real-photo matrix requires live mode');
    assert.equal(health.buildFingerprint, expectedBuildFingerprint, 'health fingerprint matches the restarted build under test');
    for (const [key, expectedValue] of Object.entries(expectedRuntime)) {
      assert.equal(health[key], expectedValue, `health reports exact ${key}`);
    }

    browsers.chromium = await chromium.launch(launchOptions('chromium'));
    browsers.webkit = await webkit.launch(launchOptions('webkit'));
    for (const testCase of cases) {
      try {
        const record = await runCase(browsers[testCase.engine], testCase, manifest.byId.get(testCase.id));
        run.records.push(record);
        if (record.outcome === 'completed-live-result') run.summary.completedLiveResults += 1;
        if (record.outcome === 'quality-rejection') run.summary.qualityRejections += 1;
        if (record.outcome === 'area-mismatch') run.summary.areaMismatches += 1;
        run.summary.printAreasValidated = [...printAreasValidated].sort();
        process.stdout.write(
          `PASS ${testCase.id} ${testCase.area} ${testCase.engine} ${record.outcome} in ${record.timing.elapsedMs}ms\n`,
        );
      } catch (error) {
        const record = { ...error.qa, error: String(error), stack: error.stack || null };
        run.records.push(record);
        run.summary.failed += 1;
        failures.push(record);
        process.stderr.write(`FAIL ${testCase.id}: ${String(error)}\n`);
      }
      writeLedger(run);
    }

    const finalHealthResponse = await fetch(`${baseUrl}/api/health`, {
      signal: AbortSignal.timeout(10000),
    });
    const finalHealth = await finalHealthResponse.json();
    run.healthAfter = { httpStatus: finalHealthResponse.status, ...finalHealth };
    assert.equal(finalHealthResponse.status, 200, 'final health endpoint returns HTTP 200');
    assert.equal(finalHealth.mode, 'live', 'final health remains in live mode');
    assert.equal(finalHealth.buildFingerprint, expectedBuildFingerprint, 'final health fingerprint still matches the build under test');
    for (const [key, expectedValue] of Object.entries(expectedRuntime)) {
      assert.equal(finalHealth[key], expectedValue, `final health still reports exact ${key}`);
    }

    if (run.acceptanceRun) {
      assert.equal(run.records.length, 13, 'full acceptance run records all thirteen attempts');
      assert.equal(run.summary.completedLiveResults, 10, 'all ten accepted real-source photos complete live');
      assert.equal(run.summary.qualityRejections, 1, 'dark quality control is rejected as expected');
      assert.equal(run.summary.areaMismatches, 2, 'both selected-area mismatch controls are rejected as expected');
      assert.deepEqual(
        [...printAreasValidated].sort(),
        Object.keys(concernKeys).sort(),
        'one printable report is validated for each of the five selected areas',
      );
    }
    assert.deepEqual(failures, [], `Real-photo failures:\n${JSON.stringify(failures, null, 2)}`);
    run.status = run.acceptanceRun ? 'passed' : 'filtered-passed';
    process.stdout.write(
      run.acceptanceRun
        ? 'ALL 10 ACCEPTED REAL-SOURCE PHOTOS + 2 AREA-MISMATCH CONTROLS + DARK QUALITY CONTROL PASSED\n'
        : `FILTERED RUN PASSED ${run.records.length} CASE(S); THIS IS NOT A FULL ACCEPTANCE RUN\n`,
    );
  } catch (error) {
    run.status = 'failed';
    run.harnessError = { error: String(error), stack: error.stack || null };
    throw error;
  } finally {
    await Promise.all(Object.values(browsers).map(browser => browser.close()));
    run.summary.printAreasValidated = [...printAreasValidated].sort();
    run.completedAt = new Date().toISOString();
    writeLedger(run);
  }
}

module.exports = {
  assertNonClinicalVisualOracle,
  assertPrintableReport,
  assertRenderedDomParity,
  assertStructuredResult,
  buildAndValidateManifest,
  captureAnalyzeResponse,
  concernKeys,
  launchOptions,
  mobileContextOptions,
  normalizeText,
  resultDeadlineMs,
  runtimeSourceFiles,
  sourceBuildFingerprint,
  writeAttemptArtifact,
};

if (require.main === module) {
  main().catch(error => {
    console.error(error);
    process.exitCode = 1;
  });
}
