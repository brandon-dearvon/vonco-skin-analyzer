const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium, webkit } = require('playwright');

const baseUrl = process.env.TEST_BASE_URL || 'http://127.0.0.1:5003';
const requestedEngine = process.env.TEST_ENGINE || '';
const requestedViewport = process.env.TEST_VIEWPORT || '';
const brittanyPhoto = process.env.BRITTANY_TEST_PHOTO
  || path.resolve('work/test-images/brittany-test-photo.jpeg');
const artifactDir = path.resolve(process.env.BROWSER_ARTIFACT_DIR || 'work/qa/browser-approved');
const pdfOutputDir = path.resolve(process.env.BROWSER_PDF_DIR || 'output/pdf');

// Deterministic presentation fixture only. It intentionally contains no skin-age field.
const fixture = {
  visibleBodyArea: 'face',
  overallScore: 76,
  positiveHighlights: [
    { title: 'Luminous complexion', detail: 'Your complexion has a fresh, naturally luminous quality.' },
    { title: 'Refined texture', detail: 'The visible skin surface appears smooth and polished.' },
    { title: 'Balanced tone', detail: 'Your overall tone looks harmonious and beautifully composed.' },
  ],
  concerns: {
    wrinkles: { score: 28, severity: 'mild', description: 'Fine expression lines are visible around the eyes.' },
    redness: { score: 42, severity: 'moderate', description: 'A soft flush is visible through the central face.' },
    darkSpots: { score: 51, severity: 'moderate', description: 'A few areas of visible pigmentation appear across the cheeks.' },
    texture: { score: 19, severity: 'none', description: 'The visible skin surface appears smooth overall.' },
  },
  recommendations: [
    { treatment: 'Sciton BBL', reason: 'An in-person consultation can determine whether this option fits your goals.', targets: ['darkSpots', 'redness'], priority: 1 },
    { treatment: 'Sciton Halo', reason: 'An in-person consultation can determine whether this option fits your goals.', targets: ['darkSpots', 'texture'], priority: 2 },
    { treatment: 'Chemical Peels', reason: 'An in-person consultation can determine whether this option fits your goals.', targets: ['darkSpots'], priority: 3 },
    { treatment: 'HydraFacial Customized', reason: 'An in-person consultation can determine whether this option fits your goals.', targets: ['texture'], priority: 4 },
    { treatment: 'Sciton Moxi', reason: 'An in-person consultation can determine whether this option fits your goals.', targets: ['darkSpots', 'texture'], priority: 5 },
    { treatment: 'Signature Facial', reason: 'An in-person consultation can determine whether this option fits your goals.', targets: ['redness'], priority: 6 },
  ],
  productRecommendations: [
    { product: 'SkinBetter Even Tone', reason: 'A provider can discuss whether this option fits your routine.' },
    { product: 'Colorescience Face Shield SPF 50', reason: 'A provider can discuss whether this option fits your routine.' },
  ],
  suggestedCombo: 'Hero Combo',
  summary: 'Your complexion has a fresh, naturally luminous quality. A tailored plan can build on that foundation while refining visible pigment and redness.',
};

const cases = [
  { name: 'desktop', width: 1280, height: 900 },
  { name: 'desktop-125-scale', width: 1024, height: 720 },
  { name: 'desktop-150-scale', width: 853, height: 600 },
  { name: 'mobile', width: 390, height: 844 },
  { name: 'small-mobile', width: 320, height: 568 },
];

function launchOptions(engineName) {
  return engineName === 'chromium'
    ? { headless: true, executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' }
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

function viewportContextOptions(viewport) {
  return viewport.name.includes('mobile')
    ? mobileContextOptions(viewport.width, viewport.height)
    : { viewport: { width: viewport.width, height: viewport.height } };
}

function countImageParts(request) {
  const body = request.postDataBuffer();
  return body ? (body.toString('latin1').match(/name="image"/g) || []).length : 0;
}

async function assertLandingContract(page, engineName, viewport) {
  assert.equal(
    await page.locator('#analysisSteps .step').count(),
    6,
    `${engineName}/${viewport.name}: original six-stage progress tracker remains`,
  );

  const navLogo = page.locator('.site-nav-logo');
  const heroLogo = page.locator('.logo');
  await Promise.all([
    navLogo.evaluate(image => image.decode()),
    heroLogo.evaluate(image => image.decode()),
  ]);

  const landing = await page.evaluate(() => {
    const nav = document.querySelector('.site-nav');
    const navInner = document.querySelector('.site-nav-inner');
    const navMark = document.querySelector('.site-nav-mark');
    const navLogoEl = document.querySelector('.site-nav-logo');
    const heroLogoEl = document.querySelector('.logo');
    const cta = document.querySelector('.site-nav-cta');
    const navRect = nav.getBoundingClientRect();
    const markRect = navMark.getBoundingClientRect();
    const logoRect = navLogoEl.getBoundingClientRect();
    const heroRect = heroLogoEl.getBoundingClientRect();
    const ctaRect = cta.getBoundingClientRect();
    const expectedRatio = 1549 / 848;
    const pseudoBefore = getComputedStyle(cta, '::before').content;
    const pseudoAfter = getComputedStyle(cta, '::after').content;
    return {
      navSource: navLogoEl.getAttribute('src'),
      heroSource: heroLogoEl.getAttribute('src'),
      navSourceSize: [navLogoEl.naturalWidth, navLogoEl.naturalHeight],
      heroSourceSize: [heroLogoEl.naturalWidth, heroLogoEl.naturalHeight],
      navRenderedRatioError: Math.abs(logoRect.width / logoRect.height - expectedRatio),
      heroRenderedRatioError: Math.abs(heroRect.width / heroRect.height - expectedRatio),
      heroVisible: heroRect.width > 0 && heroRect.height > 0,
      navFilter: getComputedStyle(navLogoEl).filter,
      heroFilter: getComputedStyle(heroLogoEl).filter,
      navBackground: getComputedStyle(nav).backgroundColor,
      navBackdropFilter: getComputedStyle(nav).backdropFilter
        || getComputedStyle(nav).webkitBackdropFilter
        || 'none',
      ctaBackground: getComputedStyle(cta).backgroundColor,
      ctaFontFamily: getComputedStyle(cta).fontFamily,
      ctaText: cta.textContent.trim(),
      ctaHasArrowElement: Boolean(cta.querySelector('svg, img, [aria-hidden="true"]')),
      ctaHasPseudoArrow: !['none', 'normal', '""', "''"].includes(pseudoBefore)
        || !['none', 'normal', '""', "''"].includes(pseudoAfter),
      ctaSize: [ctaRect.width, ctaRect.height],
      logoVerticalBuffer: navRect.height - logoRect.height,
      navSideBuffer: [markRect.left, innerWidth - ctaRect.right],
      navInsideViewport: navInner.getBoundingClientRect().left >= -1
        && navInner.getBoundingClientRect().right <= innerWidth + 1,
      horizontalOverflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
      intakeControlHeights: [
        document.getElementById('ageInput').getBoundingClientRect().height,
        document.getElementById('bodyAreaSelect').getBoundingClientRect().height,
      ],
      intakeControlFontSizes: [
        parseFloat(getComputedStyle(document.getElementById('ageInput')).fontSize),
        parseFloat(getComputedStyle(document.getElementById('bodyAreaSelect')).fontSize),
      ],
      cameraClose: {
        label: document.getElementById('closeWebcam').getAttribute('aria-label'),
        width: parseFloat(getComputedStyle(document.getElementById('closeWebcam')).width),
        height: parseFloat(getComputedStyle(document.getElementById('closeWebcam')).height),
      },
      removedResultElements: document.querySelectorAll(
        '#skinAge, [id*="skinAge"], .skin-age, [class*="skin-age"], #radarChart, #radarContainer, .skin-radar',
      ).length,
      mainSiteHref: cta.href,
      uploadBeforeCamera: Boolean(
        document.getElementById('dropZone').compareDocumentPosition(document.getElementById('webcamBtn'))
          & Node.DOCUMENT_POSITION_FOLLOWING
      ),
      uploadText: document.getElementById('dropZone').textContent.replace(/\s+/g, ' ').trim(),
      heroUploadHref: document.querySelector('.hero-upload-cta').getAttribute('href'),
      intakeBeforeUpload: Boolean(
        document.getElementById('intakeFields').compareDocumentPosition(document.getElementById('dropZone'))
          & Node.DOCUMENT_POSITION_FOLLOWING
      ),
      heroUploadInFirstViewport: (() => {
        const rect = document.querySelector('.hero-upload-cta').getBoundingClientRect();
        return rect.width > 0 && rect.height > 0 && rect.top >= navRect.bottom && rect.bottom <= innerHeight;
      })(),
      preAnalysisUsesUnsupportedConditionLabel: /rosacea|body acne|scarring & marks|dryness & dehydration|photoaging|hyperpigmentation/i.test(document.body.innerText),
    };
  });

  assert.equal(landing.navSource, '/logo.png', `${engineName}/${viewport.name}: evergreen logo is a direct asset`);
  assert.equal(landing.heroSource, '/logo_white.png', `${engineName}/${viewport.name}: white logo is a direct asset`);
  assert.deepEqual(landing.navSourceSize, [1549, 848], `${engineName}/${viewport.name}: evergreen logo is canonical HD`);
  assert.deepEqual(landing.heroSourceSize, [1549, 848], `${engineName}/${viewport.name}: white logo is canonical HD`);
  assert.ok(landing.navRenderedRatioError < 0.01, `${engineName}/${viewport.name}: header logo preserves aspect ratio`);
  if (viewport.name.includes('mobile')) {
    assert.equal(landing.heroVisible, false, `${engineName}/${viewport.name}: redundant hero logo is removed from the short mobile path`);
  } else {
    assert.ok(landing.heroRenderedRatioError < 0.01, `${engineName}/${viewport.name}: hero logo preserves aspect ratio`);
  }
  assert.equal(landing.navFilter, 'none', `${engineName}/${viewport.name}: evergreen logo is not recolored by CSS`);
  assert.equal(landing.heroFilter, 'none', `${engineName}/${viewport.name}: white logo is not recolored by CSS`);
  assert.equal(landing.navBackground, 'rgb(255, 255, 255)', `${engineName}/${viewport.name}: green logo sits on a solid white header`);
  assert.equal(landing.navBackdropFilter, 'none', `${engineName}/${viewport.name}: fixed header never ghosts page text`);
  assert.equal(landing.ctaBackground, 'rgb(81, 104, 98)', `${engineName}/${viewport.name}: Learn More uses evergreen`);
  assert.match(landing.ctaFontFamily, /Fira Sans/i, `${engineName}/${viewport.name}: Learn More uses the brand sans serif`);
  assert.equal(landing.ctaText, 'Learn More', `${engineName}/${viewport.name}: main-site action is Learn More`);
  assert.equal(landing.ctaHasArrowElement, false, `${engineName}/${viewport.name}: Learn More has no arrow element`);
  assert.equal(landing.ctaHasPseudoArrow, false, `${engineName}/${viewport.name}: Learn More has no CSS arrow`);
  assert.ok(landing.ctaSize[0] <= 125 && landing.ctaSize[1] <= 44, `${engineName}/${viewport.name}: Learn More remains compact (${landing.ctaSize.join('x')})`);
  assert.ok(landing.logoVerticalBuffer >= 14, `${engineName}/${viewport.name}: header logo has vertical clear space`);
  assert.ok(landing.navSideBuffer[0] >= 16 && landing.navSideBuffer[1] >= 16, `${engineName}/${viewport.name}: header controls retain side buffer`);
  assert.equal(landing.navInsideViewport, true, `${engineName}/${viewport.name}: navigation remains inside viewport`);
  assert.ok(landing.horizontalOverflow <= 1, `${engineName}/${viewport.name}: landing overflow ${landing.horizontalOverflow}px`);
  assert.ok(
    landing.intakeControlHeights.every(height => height >= 44),
    `${engineName}/${viewport.name}: intake controls retain 44px touch targets (${landing.intakeControlHeights.join('px/') }px)`,
  );
  assert.ok(
    Math.abs(landing.intakeControlHeights[0] - landing.intakeControlHeights[1]) <= 1,
    `${engineName}/${viewport.name}: age and area controls align (${landing.intakeControlHeights.join('px/') }px)`,
  );
  assert.ok(
    landing.intakeControlFontSizes.every(size => size >= 16),
    `${engineName}/${viewport.name}: intake controls avoid Safari auto-zoom (${landing.intakeControlFontSizes.join('px/') }px)`,
  );
  assert.deepEqual(
    landing.cameraClose,
    { label: 'Close camera', width: 44, height: 44 },
    `${engineName}/${viewport.name}: camera close control is labeled and touch sized`,
  );
  assert.equal(landing.removedResultElements, 0, `${engineName}/${viewport.name}: removed age/radar elements are absent`);
  assert.equal(landing.preAnalysisUsesUnsupportedConditionLabel, false, `${engineName}/${viewport.name}: pre-analysis copy uses appearance-only labels`);
  assert.equal(landing.mainSiteHref, 'https://www.vonandcoaesthetics.com/', `${engineName}/${viewport.name}: Learn More reaches the main site`);
  assert.equal(landing.uploadBeforeCamera, true, `${engineName}/${viewport.name}: upload is the primary choice before camera capture`);
  assert.match(landing.uploadText, /Upload one clear photo/i, `${engineName}/${viewport.name}: upload card names the one-photo path`);
  assert.match(landing.uploadText, /No camera capture required\./i, `${engineName}/${viewport.name}: upload card makes camera capture optional`);
  assert.equal(landing.heroUploadHref, '#intakeFields', `${engineName}/${viewport.name}: primary upload CTA begins with area selection`);
  assert.equal(landing.intakeBeforeUpload, true, `${engineName}/${viewport.name}: area selection remains before file upload`);
  if (viewport.name === 'small-mobile') {
    assert.equal(landing.heroUploadInFirstViewport, true, `${engineName}/${viewport.name}: primary upload action is visible without scrolling`);
  }
}

async function assertResultContract(page, engineName, viewport, uploadImageParts) {
  await page.evaluate(async () => {
    document.getElementById('offersMembership').open = true;
    await document.fonts.ready;
  });
  const result = await page.evaluate(expectedCardCount => {
    const positives = document.getElementById('positiveLead');
    const score = document.querySelector('.score-ring-container');
    const improvements = document.getElementById('concernsGrid');
    const report = buildReportHTML('Brittany', window.lastAnalysis);
    const positions = {
      positive: report.indexOf('Begin With the Positive'),
      score: report.indexOf('Overall Score:'),
      summary: report.indexOf('>Summary<'),
      improvements: report.indexOf('Skin Analysis Results'),
    };
    const visibleSelectors = 'img, svg, button, a, input, select, .positive-highlight, .recommendation-card';
    const horizontalOffenders = [...document.querySelectorAll(visibleSelectors)]
      .filter(element => {
        const style = getComputedStyle(element);
        const rect = element.getBoundingClientRect();
        if (style.display === 'none' || style.visibility === 'hidden' || rect.width === 0 || rect.height === 0) return false;
        return rect.left < -1 || rect.right > innerWidth + 1;
      })
      .map(element => `${element.tagName.toLowerCase()}#${element.id}.${element.className}`);
    const contentMargins = ['.results-container', '.recommendations-container', '.cta-content']
      .map(selector => {
        const rect = document.querySelector(selector).getBoundingClientRect();
        return [selector, rect.left, innerWidth - rect.right];
      });
    const resultText = document.getElementById('resultsSection').innerText;
    const escapeProbe = document.createElement('div');
    escapeProbe.innerHTML = escapeHTML('<img src=x onerror="window.__unsafe=1">');
    const escapedReport = buildReportHTML('<img src=x onerror="window.__unsafe=1">', window.lastAnalysis);
    const offerSummaryTitle = document.querySelector('.offers-membership-title');
    const promoBanner = document.getElementById('promoBanner');
    const promoTitle = document.querySelector('.promo-banner-title');
    const newGuestCta = document.querySelector('.von-offer-cta');
    const clubUpsell = document.getElementById('clubUpsell');
    const clubTitle = document.querySelector('.club-title');
    const clubCta = document.querySelector('.club-cta');
    const styleSnapshot = element => {
      const style = getComputedStyle(element);
      const rect = element.getBoundingClientRect();
      return {
        backgroundColor: style.backgroundColor,
        backgroundImage: style.backgroundImage,
        borderColor: style.borderColor,
        borderRadius: style.borderRadius,
        borderWidth: style.borderWidth,
        color: style.color,
        fontFamily: style.fontFamily,
        fontSize: parseFloat(style.fontSize),
        height: rect.height,
        letterSpacing: style.letterSpacing,
        textTransform: style.textTransform,
      };
    };
    return {
      positiveCount: document.querySelectorAll('#positiveHighlights .positive-highlight').length,
      firstPositiveTitle: document.querySelector('.positive-highlight-title')?.textContent,
      positivesBeforeScore: Boolean(positives.compareDocumentPosition(score) & Node.DOCUMENT_POSITION_FOLLOWING),
      positivesBeforeImprovements: Boolean(positives.compareDocumentPosition(improvements) & Node.DOCUMENT_POSITION_FOLLOWING),
      recommendationCardCount: document.querySelectorAll('#recommendationCards .recommendation-card').length,
      expectedCardCount,
      reportPositiveFirst: positions.positive >= 0
        && positions.positive < positions.score
        && positions.score < positions.summary
        && positions.summary < positions.improvements,
      reportHasCanonicalLogo: report.includes('/logo.png') && !report.includes('data:image'),
      reportHasLesionDisclaimer: report.includes('Any concerning lesion needs an in-person medical evaluation.'),
      reportHasNoAgeOrRadar: !/Estimated Skin Age|Skin Age|radarChart|radarContainer|skin-radar|<canvas/i.test(report),
      resultHasNoAgeOrRadar: !/Estimated Skin Age|Skin Age/i.test(resultText)
        && !document.querySelector('#radarChart, #radarContainer, .skin-radar, canvas'),
      baselineReportButton: document.getElementById('downloadReportBtn')?.textContent.includes('View & Print My Treatment Plan'),
      baselineOfferPresent: document.getElementById('promoBanner')?.textContent.includes('15% off your first visit'),
      baselineClubPresent: document.getElementById('clubUpsell')?.textContent.includes('The Club'),
      offersOpen: document.getElementById('offersMembership')?.open,
      offerSummaryText: offerSummaryTitle?.textContent.trim(),
      offerSummaryStyle: styleSnapshot(offerSummaryTitle),
      displayFontLoaded: document.fonts.check('24px Arsenica'),
      promoStyle: styleSnapshot(promoBanner),
      promoTitleStyle: styleSnapshot(promoTitle),
      newGuestCtaText: newGuestCta?.textContent.trim(),
      newGuestCtaStyle: styleSnapshot(newGuestCta),
      clubStyle: styleSnapshot(clubUpsell),
      clubTitleStyle: styleSnapshot(clubTitle),
      clubCtaText: clubCta?.textContent.trim(),
      clubCtaStyle: styleSnapshot(clubCta),
      planTeaserVisible: document.getElementById('resultsPlanTeaser')?.getBoundingClientRect().height > 0,
      planGuideText: document.querySelector('.results-guide-primary')?.textContent.replace(/\s+/g, ' ').trim(),
      planGuideGap: parseFloat(getComputedStyle(document.querySelector('.results-guide-primary')).columnGap),
      planCount: document.getElementById('resultsPlanCount')?.textContent.trim(),
      planSummary: document.getElementById('resultsPlanSummary')?.textContent.trim(),
      resultUsesUnsupportedConditionLabel: /rosacea|body acne|scarring & marks|dryness & dehydration|photoaging|hyperpigmentation/i.test(resultText),
      reportUsesUnsupportedConditionLabel: /rosacea|body acne|scarring & marks|dryness & dehydration|photoaging|hyperpigmentation/i.test(report),
      treatmentGroupTitle: document.querySelector('#treatmentRecommendationsGroup .recommendation-group-title')?.textContent.replace(/\s+/g, ' ').trim(),
      productGroupTitle: document.querySelector('#productRecommendationsGroup .recommendation-group-title')?.textContent.replace(/\s+/g, ' ').trim(),
      treatmentCardCount: document.querySelectorAll('#treatmentRecommendationCards .recommendation-card').length,
      productCardCount: document.querySelectorAll('#productRecommendationCards .recommendation-card').length,
      initiallyVisibleTreatmentCount: [...document.querySelectorAll('#treatmentRecommendationCards .recommendation-card')]
        .filter(card => !card.hidden && getComputedStyle(card).display !== 'none').length,
      initiallyHiddenTreatmentCount: document.querySelectorAll('#treatmentRecommendationCards .recommendation-card[hidden]').length,
      treatmentOptionsToggleVisible: !document.getElementById('treatmentOptionsToggle')?.hidden,
      treatmentOptionsToggleExpanded: document.getElementById('treatmentOptionsToggle')?.getAttribute('aria-expanded'),
      treatmentOptionsToggleText: document.getElementById('treatmentOptionsToggle')?.textContent.trim(),
      initiallyVisibleFindingCount: [...document.querySelectorAll('#concernsGrid .concern-card')]
        .filter(card => !card.hidden && getComputedStyle(card).display !== 'none').length,
      initiallyHiddenFindingCount: document.querySelectorAll('#concernsGrid .concern-card[hidden]').length,
      findingsToggleVisible: !document.getElementById('findingsToggle')?.hidden,
      findingsToggleExpanded: document.getElementById('findingsToggle')?.getAttribute('aria-expanded'),
      scoreInterpretation: document.getElementById('scoreInterpretation')?.textContent.trim(),
      scoreLegend: document.getElementById('scoreExplainer')?.textContent.replace(/\s+/g, ' ').trim(),
      scoreBoundaryLabels: [59, 60, 74, 75, 89, 90].map(score => getScoreBand(score).label),
      comboTitle: document.querySelector('#comboPlay .combo-play-title')?.textContent.trim(),
      comboAfterProducts: Boolean(
        document.getElementById('productRecommendationsGroup')?.compareDocumentPosition(document.getElementById('comboPlay'))
          & Node.DOCUMENT_POSITION_FOLLOWING
      ),
      duplicateDynamicBlocks: document.querySelectorAll('#bottomBookingCTA, #visiaUpsell').length,
      reportActionCount: document.querySelectorAll('button[onclick="downloadReport()"], button[onclick="downloadReport();"]').length,
      productHeadersUseBrandClass: [...document.querySelectorAll('#productRecommendationCards .rec-header')]
        .every(header => header.classList.contains('rec-header-product')),
      productHeaderColors: [...document.querySelectorAll('#productRecommendationCards .rec-header')]
        .map(header => [getComputedStyle(header).color, getComputedStyle(header).backgroundColor]),
      productBadgeColors: [...document.querySelectorAll('#productRecommendationCards .rec-priority')]
        .map(badge => [getComputedStyle(badge).color, getComputedStyle(badge).backgroundColor]),
      escapeProbe: {
        childElements: escapeProbe.childElementCount,
        text: escapeProbe.textContent,
        reportContainsRawProbe: escapedReport.includes('<img src=x onerror="window.__unsafe=1">'),
        reportContainsEscapedProbe: escapedReport.includes('&lt;img src=x onerror=&quot;window.__unsafe=1&quot;&gt;'),
      },
      horizontalOverflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
      horizontalOffenders,
      contentMargins,
    };
  }, fixture.recommendations.length + fixture.productRecommendations.length);

  assert.equal(uploadImageParts, 1, `${engineName}/${viewport.name}: original upload sends one image`);
  assert.equal(result.positiveCount, 3, `${engineName}/${viewport.name}: positive highlights render`);
  assert.equal(result.firstPositiveTitle, fixture.positiveHighlights[0].title, `${engineName}/${viewport.name}: result opens with the first positive`);
  assert.equal(result.positivesBeforeScore, true, `${engineName}/${viewport.name}: positives precede the score`);
  assert.equal(result.positivesBeforeImprovements, true, `${engineName}/${viewport.name}: positives precede improvement areas`);
  assert.equal(result.recommendationCardCount, result.expectedCardCount, `${engineName}/${viewport.name}: baseline service/product cards render`);
  assert.equal(result.reportPositiveFirst, true, `${engineName}/${viewport.name}: take-home report is positive-first`);
  assert.equal(result.reportHasCanonicalLogo, true, `${engineName}/${viewport.name}: report uses the direct canonical logo`);
  assert.equal(result.reportHasLesionDisclaimer, true, `${engineName}/${viewport.name}: report keeps the lesion disclaimer`);
  assert.equal(result.reportHasNoAgeOrRadar, true, `${engineName}/${viewport.name}: report excludes age and radar`);
  assert.equal(result.resultHasNoAgeOrRadar, true, `${engineName}/${viewport.name}: completed visit excludes age and radar`);
  assert.equal(result.baselineReportButton, true, `${engineName}/${viewport.name}: original treatment-plan action remains`);
  assert.equal(result.baselineOfferPresent, true, `${engineName}/${viewport.name}: original new-guest offer remains`);
  assert.equal(result.baselineClubPresent, true, `${engineName}/${viewport.name}: original Club block remains`);
  assert.equal(result.offersOpen, true, `${engineName}/${viewport.name}: offer and membership details open for full layout QA`);
  assert.equal(result.offerSummaryText, '15% off your first visit + The Club', `${engineName}/${viewport.name}: offer headline remains complete`);
  assert.equal(result.displayFontLoaded, true, `${engineName}/${viewport.name}: local Arsenica font bytes are loaded`);
  assert.match(result.offerSummaryStyle.fontFamily, /Arsenica/i, `${engineName}/${viewport.name}: offer headline uses Arsenica`);
  assert.ok(result.offerSummaryStyle.fontSize >= 27, `${engineName}/${viewport.name}: offer headline is prominent at ${result.offerSummaryStyle.fontSize}px`);
  assert.equal(result.promoStyle.backgroundColor, 'rgb(255, 255, 255)', `${engineName}/${viewport.name}: new-guest card uses the live-site white treatment`);
  assert.equal(result.promoStyle.backgroundImage, 'none', `${engineName}/${viewport.name}: new-guest card has no orange gradient`);
  assert.equal(result.promoStyle.borderRadius, '20px', `${engineName}/${viewport.name}: new-guest card uses the live-site radius`);
  assert.match(result.promoStyle.borderColor, /81, 104, 98/, `${engineName}/${viewport.name}: new-guest card uses an evergreen border`);
  assert.match(result.promoTitleStyle.fontFamily, /Arsenica/i, `${engineName}/${viewport.name}: new-guest title uses Arsenica`);
  assert.equal(result.newGuestCtaText, 'Book My Appointment', `${engineName}/${viewport.name}: new-guest action matches the main site`);
  assert.ok(result.newGuestCtaStyle.height >= 48, `${engineName}/${viewport.name}: new-guest action is touch sized`);
  assert.equal(result.newGuestCtaStyle.backgroundColor, 'rgba(0, 0, 0, 0)', `${engineName}/${viewport.name}: new-guest action uses the live-site outline treatment`);
  assert.match(result.newGuestCtaStyle.borderColor, /81, 104, 98/, `${engineName}/${viewport.name}: new-guest action keeps an evergreen outline`);
  assert.equal(result.newGuestCtaStyle.borderRadius, '9999px', `${engineName}/${viewport.name}: new-guest action uses the live-site pill radius`);
  assert.equal(result.newGuestCtaStyle.textTransform, 'none', `${engineName}/${viewport.name}: new-guest action stays in live-site title case`);
  assert.match(result.newGuestCtaStyle.fontFamily, /Fira Sans/i, `${engineName}/${viewport.name}: new-guest action uses Fira Sans`);
  assert.equal(result.clubStyle.backgroundColor, 'rgb(81, 104, 98)', `${engineName}/${viewport.name}: Club card uses evergreen`);
  assert.match(result.clubStyle.backgroundImage, /linear-gradient/i, `${engineName}/${viewport.name}: Club card uses the live-site green gradient`);
  assert.doesNotMatch(result.clubStyle.backgroundImage, /198, 141, 47|197, 139, 116/, `${engineName}/${viewport.name}: Club gradient contains no gold or orange`);
  assert.equal(result.clubStyle.borderRadius, '20px', `${engineName}/${viewport.name}: Club card uses the live-site radius`);
  assert.match(result.clubTitleStyle.fontFamily, /Arsenica/i, `${engineName}/${viewport.name}: Club title uses Arsenica`);
  assert.equal(result.clubCtaText, 'Learn More', `${engineName}/${viewport.name}: Club action is Learn More`);
  assert.ok(result.clubCtaStyle.height >= 48, `${engineName}/${viewport.name}: Club action is touch sized`);
  assert.equal(result.clubCtaStyle.borderWidth, '2px', `${engineName}/${viewport.name}: Club action uses the live-site outline treatment`);
  assert.equal(result.clubCtaStyle.borderRadius, '9999px', `${engineName}/${viewport.name}: Club action uses the live-site pill radius`);
  assert.equal(result.clubCtaStyle.color, 'rgb(255, 255, 255)', `${engineName}/${viewport.name}: Club action is white on evergreen`);
  assert.equal(result.clubCtaStyle.textTransform, 'none', `${engineName}/${viewport.name}: Club action stays in live-site title case`);
  assert.match(result.clubCtaStyle.fontFamily, /Fira Sans/i, `${engineName}/${viewport.name}: Club action uses Fira Sans`);
  assert.equal(result.planTeaserVisible, true, `${engineName}/${viewport.name}: recommendation teaser is visible in the result overview`);
  assert.equal(result.planGuideText, 'View Plan (8)', `${engineName}/${viewport.name}: highlighted plan control reads as an action, not a selected tab`);
  assert.ok(result.planGuideGap >= 3, `${engineName}/${viewport.name}: plan label and count retain visible spacing`);
  assert.equal(result.planCount, '(8)', `${engineName}/${viewport.name}: plan guide shows the total recommendation count`);
  assert.match(result.planSummary, /6 treatment options and 2 at-home skincare picks/i, `${engineName}/${viewport.name}: plan teaser separates treatment and skincare counts`);
  assert.equal(result.resultUsesUnsupportedConditionLabel, false, `${engineName}/${viewport.name}: result uses appearance-only finding labels`);
  assert.equal(result.reportUsesUnsupportedConditionLabel, false, `${engineName}/${viewport.name}: report uses appearance-only finding labels`);
  assert.equal(result.treatmentGroupTitle, 'Treatment Options (6)', `${engineName}/${viewport.name}: treatment recommendations have a labeled count`);
  assert.equal(result.productGroupTitle, 'At-Home Skincare (2)', `${engineName}/${viewport.name}: product recommendations have a labeled count`);
  assert.equal(result.treatmentCardCount, 6, `${engineName}/${viewport.name}: treatment cards render in the treatment group`);
  assert.equal(result.productCardCount, 2, `${engineName}/${viewport.name}: skincare cards render in the skincare group`);
  assert.equal(result.initiallyVisibleTreatmentCount, 3, `${engineName}/${viewport.name}: the top three treatments show first`);
  assert.equal(result.initiallyHiddenTreatmentCount, 3, `${engineName}/${viewport.name}: three additional treatments remain available`);
  assert.equal(result.treatmentOptionsToggleVisible, true, `${engineName}/${viewport.name}: additional treatments have a disclosure`);
  assert.equal(result.treatmentOptionsToggleExpanded, 'false', `${engineName}/${viewport.name}: treatment disclosure starts collapsed`);
  assert.equal(result.treatmentOptionsToggleText, 'See 3 More Treatment Options', `${engineName}/${viewport.name}: treatment disclosure names the exact additional count`);
  assert.equal(result.initiallyVisibleFindingCount, 3, `${engineName}/${viewport.name}: only three findings show initially`);
  assert.equal(result.initiallyHiddenFindingCount, 1, `${engineName}/${viewport.name}: remaining findings are preserved behind disclosure`);
  assert.equal(result.findingsToggleVisible, true, `${engineName}/${viewport.name}: findings disclosure is available`);
  assert.equal(result.findingsToggleExpanded, 'false', `${engineName}/${viewport.name}: findings disclosure starts collapsed`);
  assert.equal(result.scoreInterpretation, 'Very Good', `${engineName}/${viewport.name}: score 76 uses the shared Very Good band`);
  assert.equal(
    result.scoreLegend,
    'All scores: higher is better. 90+ Excellent, 75+ Very Good, 60+ Good, below 60 Room to Refine.',
    `${engineName}/${viewport.name}: visible score legend matches the shared thresholds`,
  );
  assert.deepEqual(
    result.scoreBoundaryLabels,
    ['Room to Refine', 'Good', 'Good', 'Very Good', 'Very Good', 'Excellent'],
    `${engineName}/${viewport.name}: score boundary helper is internally consistent`,
  );
  assert.equal(result.comboTitle, 'Hero Combo', `${engineName}/${viewport.name}: canonical combo name renders`);
  assert.equal(result.comboAfterProducts, true, `${engineName}/${viewport.name}: individual treatment and skincare cards appear before the combo`);
  assert.equal(result.duplicateDynamicBlocks, 0, `${engineName}/${viewport.name}: duplicate dynamic booking and VISIA blocks are absent`);
  assert.equal(result.reportActionCount, 1, `${engineName}/${viewport.name}: one treatment-plan action remains`);
  assert.equal(result.productHeadersUseBrandClass, true, `${engineName}/${viewport.name}: product headers use the accessible brand class`);
  assert.deepEqual(
    result.productHeaderColors,
    Array(fixture.productRecommendations.length).fill(['rgb(81, 104, 98)', 'rgb(235, 234, 229)']),
    `${engineName}/${viewport.name}: product headers use evergreen on Von White`,
  );
  assert.deepEqual(
    result.productBadgeColors,
    Array(fixture.productRecommendations.length).fill(['rgb(255, 255, 255)', 'rgb(81, 104, 98)']),
    `${engineName}/${viewport.name}: product badges use white on evergreen`,
  );
  assert.deepEqual(
    result.escapeProbe,
    {
      childElements: 0,
      text: '<img src=x onerror="window.__unsafe=1">',
      reportContainsRawProbe: false,
      reportContainsEscapedProbe: true,
    },
    `${engineName}/${viewport.name}: model and guest copy is escaped in results and printable reports`,
  );

  await page.locator('#treatmentOptionsToggle').click();
  const expandedTreatments = await page.evaluate(() => ({
    visibleCount: [...document.querySelectorAll('#treatmentRecommendationCards .recommendation-card')]
      .filter(card => !card.hidden && getComputedStyle(card).display !== 'none').length,
    hiddenCount: document.querySelectorAll('#treatmentRecommendationCards .recommendation-card[hidden]').length,
    expanded: document.getElementById('treatmentOptionsToggle').getAttribute('aria-expanded'),
    text: document.getElementById('treatmentOptionsToggle').textContent.trim(),
  }));
  assert.equal(expandedTreatments.visibleCount, 6, `${engineName}/${viewport.name}: expanding reveals every supported treatment`);
  assert.equal(expandedTreatments.hiddenCount, 0, `${engineName}/${viewport.name}: no treatment remains hidden after expansion`);
  assert.equal(expandedTreatments.expanded, 'true', `${engineName}/${viewport.name}: treatment disclosure exposes its expanded state`);
  assert.equal(expandedTreatments.text, 'Show Top 3', `${engineName}/${viewport.name}: expanded treatment disclosure offers a compact return`);
  await page.locator('#treatmentOptionsToggle').click();

  for (const count of [0, 1, 2, 3, 4, 5]) {
    const disclosure = await page.evaluate(({ presentationFixture, recommendationCount }) => {
      displayResults({
        ...presentationFixture,
        recommendations: presentationFixture.recommendations.slice(0, recommendationCount),
      });
      const button = document.getElementById('treatmentOptionsToggle');
      return {
        buttonHidden: button.hidden,
        expanded: button.getAttribute('aria-expanded'),
        text: button.textContent.trim(),
        visibleCards: [...document.querySelectorAll('#treatmentRecommendationCards .recommendation-card')]
          .filter(card => !card.hidden && getComputedStyle(card).display !== 'none').length,
        hiddenCards: document.querySelectorAll('#treatmentRecommendationCards .recommendation-card[hidden]').length,
      };
    }, { presentationFixture: fixture, recommendationCount: count });
    assert.equal(disclosure.buttonHidden, count <= 3, `${engineName}/${viewport.name}: ${count} treatments use the correct disclosure state`);
    assert.equal(disclosure.expanded, 'false', `${engineName}/${viewport.name}: ${count} treatments reset the disclosure`);
    assert.equal(disclosure.visibleCards, Math.min(count, 3), `${engineName}/${viewport.name}: ${count} treatments show at most the top three`);
    assert.equal(disclosure.hiddenCards, Math.max(0, count - 3), `${engineName}/${viewport.name}: ${count} treatments preserve every additional card`);
    if (count === 4) {
      assert.equal(disclosure.text, 'See 1 More Treatment Option', `${engineName}/${viewport.name}: singular treatment disclosure is exact`);
    }
    if (count === 5) {
      assert.equal(disclosure.text, 'See 2 More Treatment Options', `${engineName}/${viewport.name}: plural treatment disclosure is exact`);
    }
  }

  await page.evaluate(presentationFixture => displayResults(presentationFixture), fixture);
  await page.locator('#treatmentOptionsToggle').click();
  await page.evaluate(presentationFixture => displayResults({
    ...presentationFixture,
    recommendations: presentationFixture.recommendations.slice(0, 5),
  }), fixture);
  const resetDisclosure = await page.evaluate(() => ({
    expanded: document.getElementById('treatmentOptionsToggle').getAttribute('aria-expanded'),
    hiddenCards: document.querySelectorAll('#treatmentRecommendationCards .recommendation-card[hidden]').length,
  }));
  assert.deepEqual(
    resetDisclosure,
    { expanded: 'false', hiddenCards: 2 },
    `${engineName}/${viewport.name}: a new analysis collapses previously expanded treatment options`,
  );
  await page.evaluate(presentationFixture => displayResults(presentationFixture), fixture);

  await page.locator('#findingsToggle').click();
  const expandedFindings = await page.evaluate(() => ({
    visibleCount: [...document.querySelectorAll('#concernsGrid .concern-card')]
      .filter(card => !card.hidden && getComputedStyle(card).display !== 'none').length,
    hiddenCount: document.querySelectorAll('#concernsGrid .concern-card[hidden]').length,
    expanded: document.getElementById('findingsToggle').getAttribute('aria-expanded'),
  }));
  assert.equal(expandedFindings.visibleCount, 4, `${engineName}/${viewport.name}: expanding reveals every finding`);
  assert.equal(expandedFindings.hiddenCount, 0, `${engineName}/${viewport.name}: no finding remains hidden after expansion`);
  assert.equal(expandedFindings.expanded, 'true', `${engineName}/${viewport.name}: disclosure exposes its expanded state`);
  await page.locator('#findingsToggle').click();
  assert.ok(result.horizontalOverflow <= 1, `${engineName}/${viewport.name}: result overflow ${result.horizontalOverflow}px`);
  assert.deepEqual(result.horizontalOffenders, [], `${engineName}/${viewport.name}: elements outside viewport: ${result.horizontalOffenders.join(' | ')}`);
  for (const [selector, left, right] of result.contentMargins) {
    assert.ok(left >= 14 && right >= 14, `${engineName}/${viewport.name}: ${selector} side buffer is ${left}px/${right}px`);
  }
}

async function assertPrintableReport(page, context, engineName, viewport) {
  const reportHtml = await page.evaluate(() => buildReportHTML('Brittany', window.lastAnalysis));
  assert.ok(
    reportHtml.indexOf('Begin With the Positive') < reportHtml.indexOf('Overall Score:'),
    `${engineName}: report source puts positives before score`,
  );
  assert.equal(
    /Estimated Skin Age|Skin Age|radarChart|radarContainer|skin-radar|<canvas/i.test(reportHtml),
    false,
    `${engineName}: report source excludes estimated age and radar`,
  );
  assert.match(reportHtml, /Personalized Skin Analysis Report/, `${engineName}/${viewport.name}: report uses the branded title case`);
  assert.match(reportHtml, /font-family:'Arsenica'/, `${engineName}/${viewport.name}: report embeds the Arsenica display face`);
  assert.match(reportHtml, /name="viewport"/, `${engineName}/${viewport.name}: report declares a mobile viewport`);
  for (const recommendation of fixture.recommendations) {
    assert.equal(
      reportHtml.includes(recommendation.treatment),
      true,
      `${engineName}/${viewport.name}: client report preserves ${recommendation.treatment}`,
    );
  }

  const serverReportResponse = await context.request.post(`${baseUrl}/api/report`, {
    data: { name: 'Brittany', analysis: fixture },
  });
  assert.equal(serverReportResponse.ok(), true, `${engineName}/${viewport.name}: fallback report endpoint succeeds`);
  const serverReportHtml = await serverReportResponse.text();
  for (const requiredCopy of [
    'Personalized Skin Analysis Report',
    'Begin With the Positive',
    'Your Skincare Essentials',
    'New Guest Offer',
    'The Club. $149/month',
    'Book Your Complimentary Consultation',
    'Any concerning lesion needs an in-person medical evaluation.',
  ]) {
    assert.equal(
      serverReportHtml.includes(requiredCopy),
      true,
      `${engineName}/${viewport.name}: fallback report preserves ${requiredCopy}`,
    );
  }
  assert.match(serverReportHtml, /https:\/\/booking\.vonandcoaesthetics\.com\/webstoreNew\/services\?utm_source=skin-analyzer/, `${engineName}/${viewport.name}: fallback report uses the same booking action`);
  assert.match(serverReportHtml, /https?:\/\/[^"']+\/arsenica-regular\.otf/, `${engineName}/${viewport.name}: fallback report uses an absolute local Arsenica asset`);
  for (const recommendation of fixture.recommendations) {
    assert.equal(
      serverReportHtml.includes(recommendation.treatment),
      true,
      `${engineName}/${viewport.name}: fallback report preserves ${recommendation.treatment}`,
    );
  }

  const serverReportPage = await context.newPage();
  await serverReportPage.setContent(serverReportHtml, { waitUntil: 'domcontentloaded' });
  await serverReportPage.evaluate(() => document.fonts.ready);
  const serverReportLayout = await serverReportPage.evaluate(() => {
    const title = document.querySelector('.report-title');
    const row = document.querySelector('.report-result-row');
    const cell = row?.querySelector('td');
    return {
      overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
      title: title?.textContent.trim(),
      titleFont: title ? getComputedStyle(title).fontFamily : '',
      titleFontLoaded: document.fonts.check('24px Arsenica'),
      rowDisplay: row ? getComputedStyle(row).display : '',
      cellDisplay: cell ? getComputedStyle(cell).display : '',
      cellFontSize: cell ? parseFloat(getComputedStyle(cell).fontSize) : 0,
    };
  });
  assert.ok(serverReportLayout.overflow <= 1, `${engineName}/${viewport.name}: fallback report overflow ${serverReportLayout.overflow}px`);
  assert.equal(serverReportLayout.title, 'Personalized Skin Analysis Report', `${engineName}/${viewport.name}: fallback report title matches`);
  assert.match(serverReportLayout.titleFont, /Arsenica/i, `${engineName}/${viewport.name}: fallback report title uses Arsenica`);
  assert.equal(serverReportLayout.titleFontLoaded, true, `${engineName}/${viewport.name}: fallback report loads Arsenica bytes`);
  if (viewport.name.includes('mobile')) {
    assert.equal(serverReportLayout.rowDisplay, 'block', `${engineName}/${viewport.name}: fallback report findings stack into cards`);
    assert.equal(serverReportLayout.cellDisplay, 'grid', `${engineName}/${viewport.name}: fallback report fields retain labels`);
    assert.ok(serverReportLayout.cellFontSize >= 15, `${engineName}/${viewport.name}: fallback report finding copy stays legible`);
  }
  await serverReportPage.screenshot({
    path: path.join(artifactDir, `${engineName}-${viewport.name}-server-take-home-report-screen.png`),
    fullPage: true,
  });
  await serverReportPage.close();

  const reportPage = await context.newPage();
  await reportPage.setContent(reportHtml, { waitUntil: 'domcontentloaded' });
  await reportPage.evaluate(() => document.fonts.ready);
  const reportLogo = reportPage.locator('img[alt^="Von & Co"]');
  await reportLogo.evaluate(image => image.decode());
  const logoContract = await reportLogo.evaluate(image => ({
    source: image.getAttribute('src'),
    size: [image.naturalWidth, image.naturalHeight],
    filter: getComputedStyle(image).filter,
    renderedRatio: image.getBoundingClientRect().width / image.getBoundingClientRect().height,
  }));
  assert.match(logoContract.source, /\/logo\.png$/, `${engineName}: report logo is a direct asset`);
  assert.equal(logoContract.source.startsWith('data:'), false, `${engineName}: report logo is not embedded low-resolution art`);
  assert.deepEqual(logoContract.size, [1549, 848], `${engineName}: report renders canonical HD logo`);
  assert.equal(logoContract.filter, 'none', `${engineName}: report logo has no effect`);
  assert.ok(Math.abs(logoContract.renderedRatio - 1549 / 848) < 0.01, `${engineName}: report logo preserves aspect ratio`);

  const screenLayout = await reportPage.evaluate(() => {
    const title = document.querySelector('.report-title');
    const titleStyle = getComputedStyle(title);
    const resultRow = document.querySelector('.report-result-row');
    const resultCell = resultRow?.querySelector('td');
    return {
      overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
      title: title.textContent.trim(),
      titleFont: titleStyle.fontFamily,
      titleSize: parseFloat(titleStyle.fontSize),
      titleFontLoaded: document.fonts.check('24px Arsenica'),
      resultRowDisplay: resultRow ? getComputedStyle(resultRow).display : '',
      resultCellDisplay: resultCell ? getComputedStyle(resultCell).display : '',
      resultCellFontSize: resultCell ? parseFloat(getComputedStyle(resultCell).fontSize) : 0,
      resultCellLabel: resultCell ? getComputedStyle(resultCell, '::before').content : '',
    };
  });
  assert.ok(screenLayout.overflow <= 1, `${engineName}/${viewport.name}: on-screen report overflow ${screenLayout.overflow}px`);
  assert.equal(screenLayout.title, 'Personalized Skin Analysis Report', `${engineName}/${viewport.name}: report title is correct`);
  assert.match(screenLayout.titleFont, /Arsenica/i, `${engineName}/${viewport.name}: rendered report title uses Arsenica`);
  assert.equal(screenLayout.titleFontLoaded, true, `${engineName}/${viewport.name}: rendered report loads the Arsenica font asset`);
  assert.ok(screenLayout.titleSize >= 24, `${engineName}/${viewport.name}: rendered report title is prominent at ${screenLayout.titleSize}px`);
  if (viewport.name.includes('mobile')) {
    assert.equal(screenLayout.resultRowDisplay, 'block', `${engineName}/${viewport.name}: report findings stack into mobile cards`);
    assert.equal(screenLayout.resultCellDisplay, 'grid', `${engineName}/${viewport.name}: mobile finding fields use labeled rows`);
    assert.ok(screenLayout.resultCellFontSize >= 15, `${engineName}/${viewport.name}: mobile finding text remains legible at ${screenLayout.resultCellFontSize}px`);
    assert.match(screenLayout.resultCellLabel, /Concern/, `${engineName}/${viewport.name}: stacked finding retains its Concern label`);
  }

  await reportPage.screenshot({
    path: path.join(artifactDir, `${engineName}-${viewport.name}-take-home-report-screen.png`),
    fullPage: true,
  });

  await reportPage.emulateMedia({ media: 'print' });
  const reportLayout = await reportPage.evaluate(() => ({
    overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    text: document.body.innerText,
    hasCanvas: Boolean(document.querySelector('canvas')),
  }));
  assert.ok(reportLayout.overflow <= 1, `${engineName}: printable report overflow ${reportLayout.overflow}px`);
  assert.match(reportLayout.text, /Begin With the Positive/, `${engineName}: printable report includes positives`);
  assert.match(reportLayout.text, /Any concerning lesion needs an in-person medical evaluation\./, `${engineName}: printable report includes disclaimer`);
  assert.doesNotMatch(reportLayout.text, /Estimated Skin Age|Skin Age/i, `${engineName}: printable report has no age estimate`);
  assert.equal(reportLayout.hasCanvas, false, `${engineName}: printable report has no radar canvas`);

  await reportPage.screenshot({
    path: path.join(artifactDir, `${engineName}-${viewport.name}-take-home-report.png`),
    fullPage: true,
  });
  if (engineName === 'chromium' && viewport.name === 'desktop') {
    await reportPage.pdf({
      path: path.join(pdfOutputDir, 'von-co-take-home-report-approved-preview.pdf'),
      format: 'Letter',
      printBackground: true,
      margin: { top: '0.25in', right: '0.25in', bottom: '0.25in', left: '0.25in' },
    });
  }
  await reportPage.close();
}

async function runCase(browserType, engineName, viewport) {
  const browser = await browserType.launch(launchOptions(engineName));
  const context = await browser.newContext(viewportContextOptions(viewport));
  await context.route('https://fonts.googleapis.com/**', route => route.fulfill({
    status: 200,
    contentType: 'text/css',
    body: '',
  }));
  await context.route('https://fonts.gstatic.com/**', route => route.abort());
  const page = await context.newPage();
  const pageErrors = [];
  let uploadImageParts = 0;
  page.on('pageerror', error => pageErrors.push(String(error)));

  await page.route('**/api/health', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ status: 'ok', mode: 'live' }),
  }));

  // Health mode and analysis output are deterministic; the page itself comes from the local preview.
  await page.route('**/api/analyze', async route => {
    uploadImageParts = countImageParts(route.request());
    await new Promise(resolve => setTimeout(resolve, 300));
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(fixture) });
  });

  await page.goto(baseUrl, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1100);
  await assertLandingContract(page, engineName, viewport);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: path.join(artifactDir, `${engineName}-${viewport.name}-landing.png`) });
  if (viewport.name.includes('mobile')) {
    await page.getByRole('link', { name: 'Upload a Photo' }).click();
    await page.waitForFunction(() => {
      const intake = document.getElementById('intakeFields');
      const area = document.getElementById('bodyAreaSelect');
      const nav = document.querySelector('.site-nav');
      if (!intake || !area || !nav) return false;
      const intakeRect = intake.getBoundingClientRect();
      const areaRect = area.getBoundingClientRect();
      const navRect = nav.getBoundingClientRect();
      return intakeRect.top >= navRect.bottom + 12 && areaRect.bottom <= innerHeight;
    }, undefined, { timeout: 2000 });
    const anchorPosition = await page.evaluate(() => ({
      intakeTop: document.getElementById('intakeFields').getBoundingClientRect().top,
      areaBottom: document.getElementById('bodyAreaSelect').getBoundingClientRect().bottom,
      navBottom: document.querySelector('.site-nav').getBoundingClientRect().bottom,
      viewportHeight: innerHeight,
    }));
    assert.ok(
      anchorPosition.intakeTop >= anchorPosition.navBottom + 12,
      `${engineName}/${viewport.name}: intake target clears fixed header (${anchorPosition.intakeTop}px vs ${anchorPosition.navBottom}px)`,
    );
    assert.ok(
      anchorPosition.areaBottom <= anchorPosition.viewportHeight,
      `${engineName}/${viewport.name}: body-area selector is visible before upload (${JSON.stringify(anchorPosition)})`,
    );
  } else {
    await page.locator('#dropZone').scrollIntoViewIfNeeded();
  }
  await page.waitForTimeout(100);
  await page.screenshot({ path: path.join(artifactDir, `${engineName}-${viewport.name}-upload.png`) });

  await page.locator('#fileInput').setInputFiles(brittanyPhoto);
  await page.waitForSelector('#analysisSection.show', { timeout: 10000 });
  await page.waitForSelector('#leadGateOverlay.show', { timeout: 10000 });
  if (viewport.name.includes('mobile')) {
    const leadGateTop = await page.evaluate(() => {
      const overlay = document.getElementById('leadGateOverlay');
      const card = overlay.querySelector('.lead-gate-card');
      const cardRect = card.getBoundingClientRect();
      return {
        cardTop: cardRect.top,
        overlayOverflowY: getComputedStyle(overlay).overflowY,
        overlayScrollable: overlay.scrollHeight > overlay.clientHeight,
      };
    });
    assert.ok(leadGateTop.cardTop >= 0, `${engineName}/${viewport.name}: lead card begins inside the viewport (${leadGateTop.cardTop}px)`);
    assert.match(leadGateTop.overlayOverflowY, /auto|scroll/, `${engineName}/${viewport.name}: tall lead card can scroll vertically`);
    await page.evaluate(() => {
      const overlay = document.getElementById('leadGateOverlay');
      overlay.scrollTop = overlay.scrollHeight;
    });
    await page.waitForTimeout(50);
    const leadGateBottom = await page.locator('.lead-gate-card').evaluate(card => card.getBoundingClientRect().bottom);
    assert.ok(leadGateBottom <= viewport.height + 1, `${engineName}/${viewport.name}: lead card bottom remains reachable (${leadGateBottom}px)`);
  }
  await page.getByRole('button', { name: 'Skip for now' }).click();
  await page.waitForSelector('#resultsSection.show', { timeout: 10000 });
  await page.waitForTimeout(2400);
  await assertResultContract(page, engineName, viewport, uploadImageParts);
  if (viewport.name.includes('mobile')) {
    const stickyGuide = await page.evaluate(async () => {
      const guide = document.querySelector('.results-guide');
      window.scrollTo(0, 0);
      await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      const documentTop = guide.getBoundingClientRect().top + window.scrollY;
      const requestedScrollY = documentTop + 220;
      window.scrollTo(0, requestedScrollY);
      await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      return {
        position: getComputedStyle(guide).position,
        top: guide.getBoundingClientRect().top,
        configuredTop: getComputedStyle(guide).top,
        requestedScrollY,
        actualScrollY: window.scrollY,
        maxScrollY: document.documentElement.scrollHeight - window.innerHeight,
        htmlOverflow: [getComputedStyle(document.documentElement).overflowX, getComputedStyle(document.documentElement).overflowY],
        bodyOverflow: [getComputedStyle(document.body).overflowX, getComputedStyle(document.body).overflowY],
        container: (() => {
          const rect = document.querySelector('.results-container').getBoundingClientRect();
          return { top: rect.top, bottom: rect.bottom, height: rect.height };
        })(),
      };
    });
    assert.equal(stickyGuide.position, 'sticky', `${engineName}/${viewport.name}: results jump guide uses sticky positioning`);
    assert.ok(
      stickyGuide.top >= 79 && stickyGuide.top <= 81,
      `${engineName}/${viewport.name}: results jump guide remains visible at 80px (${JSON.stringify(stickyGuide)})`,
    );
  }
  assert.deepEqual(pageErrors, [], `${engineName}/${viewport.name}: page errors: ${pageErrors.join('; ')}`);

  await page.evaluate(() => window.scrollTo(0, document.getElementById('resultsSection').offsetTop - 50));
  await page.waitForTimeout(150);
  await page.screenshot({ path: path.join(artifactDir, `${engineName}-${viewport.name}-results.png`) });
  await page.evaluate(() => document.getElementById('recommendationsSection').scrollIntoView({ block: 'start' }));
  await page.waitForTimeout(150);
  if (viewport.name === 'small-mobile') {
    const firstTreatmentCard = await page.evaluate(() => {
      const navBottom = document.querySelector('.site-nav').getBoundingClientRect().bottom;
      const rect = document.querySelector('#treatmentRecommendationCards .recommendation-card').getBoundingClientRect();
      return {
        top: rect.top,
        bottom: rect.bottom,
        navBottom,
        viewportHeight: innerHeight,
      };
    });
    assert.ok(
      firstTreatmentCard.top < firstTreatmentCard.viewportHeight && firstTreatmentCard.bottom > firstTreatmentCard.navBottom,
      `${engineName}/${viewport.name}: the first service card is visible after opening the plan (${JSON.stringify(firstTreatmentCard)})`,
    );
  }
  await page.screenshot({ path: path.join(artifactDir, `${engineName}-${viewport.name}-recommendations.png`) });
  await page.locator('#offersMembership').scrollIntoViewIfNeeded();
  await page.waitForTimeout(100);
  await page.screenshot({ path: path.join(artifactDir, `${engineName}-${viewport.name}-offers.png`) });
  await page.locator('#offersMembership > summary').screenshot({
    path: path.join(artifactDir, `${engineName}-${viewport.name}-offer-summary.png`),
  });
  await page.locator('#promoBanner').screenshot({
    path: path.join(artifactDir, `${engineName}-${viewport.name}-new-guest-card.png`),
  });
  const screenshotNavOverride = await page.addStyleTag({ content: '.site-nav { visibility: hidden !important; }' });
  await page.locator('#clubUpsell').screenshot({
    path: path.join(artifactDir, `${engineName}-${viewport.name}-club-card.png`),
  });
  await screenshotNavOverride.evaluate(style => style.remove());
  await page.screenshot({ path: path.join(artifactDir, `${engineName}-${viewport.name}-full.png`), fullPage: true });

  if (['desktop', 'mobile', 'small-mobile'].includes(viewport.name)) {
    await assertPrintableReport(page, context, engineName, viewport);
  }
  await browser.close();
}

async function runOriginalGuidedOneImageContract(browserType, engineName) {
  const browser = await browserType.launch(launchOptions(engineName));
  const context = await browser.newContext(mobileContextOptions());
  const page = await context.newPage();
  let imagePartCount = 0;
  await page.route('**/api/health', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ status: 'ok', mode: 'live' }),
  }));
  await page.route('**/api/analyze', route => {
    imagePartCount = countImageParts(route.request());
    return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(fixture) });
  });
  await page.goto(baseUrl, { waitUntil: 'domcontentloaded' });
  const source = `data:image/jpeg;base64,${fs.readFileSync(brittanyPhoto).toString('base64')}`;
  await page.evaluate(dataUrl => {
    guidedPhotos = [dataUrl, dataUrl, dataUrl];
    submitGuidedCapture();
  }, source);
  await page.waitForSelector('#leadGateOverlay.show', { timeout: 10000 });
  assert.equal(imagePartCount, 1, `${engineName}: restored guided flow keeps original one-image submission`);
  await browser.close();
}

async function runBodyAreaCameraContract(browserType, engineName) {
  const browser = await browserType.launch(launchOptions(engineName));
  const context = await browser.newContext(mobileContextOptions());
  const page = await context.newPage();
  await page.route('**/api/health', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ status: 'ok', mode: 'live' }),
  }));
  await page.goto(baseUrl, { waitUntil: 'domcontentloaded' });

  const cameraContract = await page.evaluate(() => {
    const area = document.getElementById('bodyAreaSelect');
    area.value = 'legs';
    area.dispatchEvent(new Event('change', { bubbles: true }));
    setCaptureMode('guided');

    return {
      mode: captureMode,
      pickerHidden: document.getElementById('captureModePicker').hidden,
      guidedButtonHidden: document.getElementById('modeGuided').hidden,
      ovalHidden: document.getElementById('guidedOval').hidden,
      stepsDisplay: getComputedStyle(document.getElementById('guidedSteps')).display,
      thumbsDisplay: getComputedStyle(document.getElementById('guidedThumbs')).display,
      title: document.getElementById('webcamTitle').textContent,
      instruction: document.getElementById('webcamInstruction').textContent,
      optionNote: document.getElementById('cameraOptionNote').textContent,
    };
  });

  assert.equal(cameraContract.mode, 'quick', `${engineName}: non-face capture cannot enter guided facial mode`);
  assert.equal(cameraContract.pickerHidden, true, `${engineName}: non-face capture hides the facial mode picker`);
  assert.equal(cameraContract.guidedButtonHidden, true, `${engineName}: non-face capture hides Guided Capture`);
  assert.equal(cameraContract.ovalHidden, true, `${engineName}: non-face capture hides the facial oval`);
  assert.equal(cameraContract.stepsDisplay, 'none', `${engineName}: non-face capture hides facial pose steps`);
  assert.equal(cameraContract.thumbsDisplay, 'none', `${engineName}: non-face capture hides facial pose thumbnails`);
  assert.equal(cameraContract.title, 'Capture Legs Photo', `${engineName}: non-face camera title names the selected area`);
  assert.equal(cameraContract.instruction, 'Position the selected area clearly in the frame', `${engineName}: non-face instruction is anatomy-neutral`);
  assert.equal(cameraContract.optionNote, 'Quick Snap takes one clear photo of the selected area.', `${engineName}: non-face camera copy explains the one-photo path`);
  await browser.close();
}

async function runDemoDisclosureContract(browserType, engineName) {
  const browser = await browserType.launch(launchOptions(engineName));
  const context = await browser.newContext(mobileContextOptions());
  const page = await context.newPage();
  await page.route('**/api/health', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ status: 'ok', mode: 'demo' }),
  }));
  await page.route('**/api/analyze', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify(fixture),
  }));

  await page.goto(baseUrl, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => demoMode === true);
  await page.locator('#fileInput').setInputFiles(brittanyPhoto);
  await page.waitForSelector('#leadGateOverlay.show', { timeout: 10000 });
  const pendingMode = await page.evaluate(() => ({
    demo: pendingAnalysisData?._isDemo === true,
    live: pendingAnalysisData?._isLive === true,
  }));
  assert.equal(pendingMode.demo, true, `${engineName}: health demo mode marks server results as sample data`);
  assert.equal(pendingMode.live, false, `${engineName}: health demo mode never labels sample data live`);

  await page.getByRole('button', { name: 'Skip for now' }).click();
  await page.waitForSelector('#resultsSection.show', { timeout: 10000 });
  const disclosure = await page.locator('#demoBanner').evaluate(element => ({
    display: getComputedStyle(element).display,
    text: element.textContent,
  }));
  assert.notEqual(disclosure.display, 'none', `${engineName}: sample-result disclosure is visible`);
  assert.match(disclosure.text, /sample results, not a real analysis/i, `${engineName}: disclosure is explicit`);

  const demoAudit = await page.evaluate(() => {
    const areas = ['face', 'neck_chest', 'hands', 'back', 'legs'];
    const patterns = [
      ['medical condition', /\b(?:rosacea|melasma|acne|dermatitis|eczema|psoriasis|keratosis pilaris|skin cancer|melanoma|malignan(?:t|cy)|basal cell|squamous cell)\b/i],
      ['unsupported inferred state', /\b(?:hyperpigmentation|photoaging|scarring|scars?|dehydration|dehydrated)\b/i],
      ['marketing overclaim', /\b(?:gold standard|permanent(?:ly)? reduction|perfect(?:ly)?|flawless|safe for all skin tones|all skin tones safe|makes? skin act younger)\b/i],
      ['unsupported cause', /\b(?:due to|caused by|likely from|from (?:chronic|cumulative) sun exposure|from volume loss)\b/i],
      ['unmeasured property', /\b(?:good|reasonable|strong|healthy) (?:elasticity|thickness)|\b(?:well-hydrated|adequately moisturized)\b/i],
    ];
    const violations = [];
    const sampleCounts = {};
    const select = document.getElementById('bodyAreaSelect');
    const collectGuestCopy = data => [
      data?.summary,
      data?.suggestedCombo,
      ...(data?.positiveHighlights || []).flatMap(item => [item?.title, item?.detail]),
      ...Object.values(data?.concerns || {}).map(item => item?.description),
      ...(data?.recommendations || []).flatMap(item => [item?.treatment, item?.reason]),
      ...(data?.productRecommendations || []).flatMap(item => [item?.product, item?.reason]),
    ].filter(value => typeof value === 'string').join(' ');
    const scan = (area, surface, copy) => {
      for (const [label, pattern] of patterns) {
        const match = String(copy || '').match(pattern);
        if (match && violations.length < 25) {
          violations.push({ area, surface, label, match: match[0], copy: String(copy).slice(0, 500) });
        }
      }
    };

    for (const area of areas) {
      select.value = area;
      let representative = null;
      for (let index = 0; index < 500; index += 1) {
        const sample = generateDemoResults();
        representative ||= sample;
        scan(area, 'generated guest copy', collectGuestCopy(sample));
      }
      sampleCounts[area] = 500;
      displayResults({ ...representative, _isDemo: true, _isLive: false });
      const renderedText = [
        document.body.innerText,
        new DOMParser().parseFromString(buildReportHTML('Demo Guest', representative), 'text/html').body.innerText,
      ].join(' ');
      scan(area, 'rendered page and report', renderedText);
    }
    return { violations, sampleCounts };
  });
  assert.deepEqual(
    demoAudit.sampleCounts,
    { face: 500, neck_chest: 500, hands: 500, back: 500, legs: 500 },
    `${engineName}: frontend demo generator is sampled 500 times for every body area`,
  );
  assert.deepEqual(demoAudit.violations, [], `${engineName}: demo results and reports use supported appearance-only copy`);
  await page.screenshot({ path: path.join(artifactDir, `${engineName}-demo-disclosure.png`) });
  await browser.close();
}

async function runRejectionContract(browserType, engineName) {
  const browser = await browserType.launch(launchOptions(engineName));
  const context = await browser.newContext(mobileContextOptions());
  const page = await context.newPage();
  let analyzeRequests = 0;
  const rejectionReason = 'Our skin analysis is designed for adults (18+).';
  await page.route('**/api/health', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ status: 'ok', mode: 'live' }),
  }));
  await page.route('**/api/analyze', route => {
    analyzeRequests += 1;
    return route.fulfill({
      status: 422,
      contentType: 'application/json',
      body: JSON.stringify({ rejected: true, reason: rejectionReason }),
    });
  });

  await page.goto(baseUrl, { waitUntil: 'domcontentloaded' });
  await page.locator('#fileInput').setInputFiles(brittanyPhoto);
  await page.waitForSelector('#rejectionOverlay', { state: 'visible', timeout: 10000 });
  const rejection = await page.evaluate(() => ({
    reason: document.getElementById('rejectionReason')?.textContent,
    title: document.getElementById('rejectionTitle')?.textContent,
    leadGateOpen: document.getElementById('leadGateOverlay')?.classList.contains('show'),
    pendingData: pendingAnalysisData,
  }));
  assert.equal(analyzeRequests, 1, `${engineName}: a definitive rejection is not retried`);
  assert.equal(rejection.reason, rejectionReason, `${engineName}: the real rejection reason reaches the guest`);
  assert.equal(rejection.title, 'Adults Only (18+)', `${engineName}: minor rejection keeps its contextual title`);
  assert.equal(rejection.leadGateOpen, false, `${engineName}: a rejected photo never opens the lead gate`);
  assert.equal(rejection.pendingData, null, `${engineName}: a rejected photo never becomes recommendation data`);
  await page.screenshot({ path: path.join(artifactDir, `${engineName}-rejection.png`) });
  await browser.close();
}

async function runServerTimeoutRecoveryContract(browserType, engineName) {
  const browser = await browserType.launch(launchOptions(engineName));
  const context = await browser.newContext(mobileContextOptions());
  const page = await context.newPage();
  let analyzeRequests = 0;

  await page.route('**/api/health', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ status: 'ok', mode: 'live' }),
  }));
  await page.route('**/api/analyze', route => {
    analyzeRequests += 1;
    if (analyzeRequests > 1) {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(fixture),
      });
    }
    return route.fulfill({
      status: 504,
      contentType: 'application/json',
      body: JSON.stringify({
        error: 'The analysis did not finish within the expected time.',
        code: 'analysis_timeout',
        retryable: true,
      }),
    });
  });

  await page.goto(baseUrl, { waitUntil: 'domcontentloaded' });
  await page.locator('#fileInput').setInputFiles(brittanyPhoto);
  await page.waitForSelector('#analysisRecovery.show', { timeout: 10000 });
  const recovery = await page.evaluate(() => ({
    title: document.getElementById('analysisRecoveryTitle')?.textContent,
    message: document.getElementById('analysisRecoveryMessage')?.textContent,
    analysisOpen: document.getElementById('analysisSection')?.classList.contains('show'),
    leadGateOpen: document.getElementById('leadGateOverlay')?.classList.contains('show'),
    resultsOpen: document.getElementById('resultsSection')?.classList.contains('show'),
    recommendationsOpen: document.getElementById('recommendationsSection')?.classList.contains('show'),
    demoVisible: Boolean(document.getElementById('demoBanner'))
      && getComputedStyle(document.getElementById('demoBanner')).display !== 'none',
    pendingData: pendingAnalysisData,
    lastAnalysisPresent: Boolean(window.lastAnalysis),
    retryClass: document.getElementById('retryAnalysisBtn')?.className,
    chooseClass: document.getElementById('chooseAnotherPhotoBtn')?.className,
  }));
  assert.equal(analyzeRequests, 1, `${engineName}: server timeout receives exactly one analysis POST`);
  assert.equal(recovery.title, 'The analysis needs one more try', `${engineName}: server timeout opens the recovery state`);
  assert.match(recovery.message, /photo uploaded successfully/i, `${engineName}: recovery confirms that re-uploading is unnecessary`);
  assert.match(recovery.retryClass, /btn-primary/, `${engineName}: retrying the selected photo is the primary recovery action`);
  assert.match(recovery.chooseClass, /btn-secondary/, `${engineName}: choosing another photo is secondary`);
  assert.equal(recovery.analysisOpen, false, `${engineName}: server timeout closes the progress state`);
  assert.equal(recovery.leadGateOpen, false, `${engineName}: server timeout never opens the lead gate`);
  assert.equal(recovery.resultsOpen, false, `${engineName}: server timeout never shows results`);
  assert.equal(recovery.recommendationsOpen, false, `${engineName}: server timeout never shows recommendations`);
  assert.equal(recovery.demoVisible, false, `${engineName}: server timeout never falls back to demo data`);
  assert.equal(recovery.pendingData, null, `${engineName}: server timeout creates no pending result`);
  assert.equal(recovery.lastAnalysisPresent, false, `${engineName}: server timeout creates no reportable result`);
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(artifactDir, `${engineName}-timeout-recovery.png`) });
  await page.locator('#retryAnalysisBtn').click();
  await page.waitForSelector('#leadGateOverlay.show', { timeout: 10000 });
  assert.equal(analyzeRequests, 2, `${engineName}: an explicit retry sends one new analysis POST`);
  assert.equal(
    await page.locator('#analysisRecovery').evaluate(element => element.classList.contains('show')),
    false,
    `${engineName}: an explicit retry closes the recovery state`,
  );
  await browser.close();
}

async function runClientAbortRecoveryContract(browserType, engineName) {
  const browser = await browserType.launch(launchOptions(engineName));
  const context = await browser.newContext(mobileContextOptions());
  const page = await context.newPage();
  let analyzeRequests = 0;
  await page.addInitScript(() => {
    window.__ANALYSIS_TIMEOUT_MS__ = 60;
  });

  await page.route('**/api/health', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ status: 'ok', mode: 'live' }),
  }));
  await page.route('**/api/analyze', async route => {
    analyzeRequests += 1;
    await new Promise(resolve => setTimeout(resolve, 250));
    try {
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(fixture) });
    } catch (_) {
      // The browser is expected to cancel this delayed response through AbortController.
    }
  });

  await page.goto(baseUrl, { waitUntil: 'domcontentloaded' });
  await page.locator('#fileInput').setInputFiles(brittanyPhoto);
  await page.waitForSelector('#analysisRecovery.show', { timeout: 10000 });
  const recovery = await page.evaluate(() => ({
    title: document.getElementById('analysisRecoveryTitle')?.textContent,
    leadGateOpen: document.getElementById('leadGateOverlay')?.classList.contains('show'),
    resultsOpen: document.getElementById('resultsSection')?.classList.contains('show'),
    demoVisible: Boolean(document.getElementById('demoBanner'))
      && getComputedStyle(document.getElementById('demoBanner')).display !== 'none',
    pendingData: pendingAnalysisData,
  }));
  assert.equal(analyzeRequests, 1, `${engineName}: client deadline sends exactly one analysis POST`);
  assert.equal(recovery.title, 'The analysis needs one more try', `${engineName}: AbortController opens the same recovery state`);
  assert.equal(recovery.leadGateOpen, false, `${engineName}: client deadline never opens the lead gate`);
  assert.equal(recovery.resultsOpen, false, `${engineName}: client deadline never shows results`);
  assert.equal(recovery.demoVisible, false, `${engineName}: client deadline never falls back to demo data`);
  assert.equal(recovery.pendingData, null, `${engineName}: client deadline creates no pending result`);
  await browser.close();
}

async function runNonretryableProviderContract(browserType, engineName) {
  const browser = await browserType.launch(launchOptions(engineName));
  const context = await browser.newContext(mobileContextOptions());
  const page = await context.newPage();
  let analyzeRequests = 0;

  await page.route('**/api/health', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ status: 'ok', mode: 'live' }),
  }));
  await page.route('**/api/analyze', route => {
    analyzeRequests += 1;
    return route.fulfill({
      status: 502,
      contentType: 'application/json',
      body: JSON.stringify({
        error: 'The analysis service could not process this request.',
        code: 'analysis_unavailable',
        retryable: false,
      }),
    });
  });

  await page.goto(baseUrl, { waitUntil: 'domcontentloaded' });
  await page.locator('#fileInput').setInputFiles(brittanyPhoto);
  await page.waitForSelector('#analysisRecovery.show', { timeout: 10000 });
  const recovery = await page.evaluate(() => ({
    title: document.getElementById('analysisRecoveryTitle')?.textContent,
    message: document.getElementById('analysisRecoveryMessage')?.textContent,
    retryHidden: document.getElementById('retryAnalysisBtn')?.hidden,
    chooseClass: document.getElementById('chooseAnotherPhotoBtn')?.className,
  }));
  assert.equal(analyzeRequests, 1, `${engineName}: a nonretryable provider error sends one POST`);
  assert.equal(recovery.title, 'We couldn\'t complete your analysis', `${engineName}: permanent failure is not described as temporary`);
  assert.match(recovery.message, /choose another clear, well-lit photo or return later/i, `${engineName}: permanent failure does not invite a futile same-photo retry`);
  assert.equal(recovery.retryHidden, true, `${engineName}: same-photo retry is hidden for a nonretryable failure`);
  assert.match(recovery.chooseClass, /btn-primary/, `${engineName}: choosing another photo becomes the primary action`);
  await browser.close();
}

async function runAcceleratedTimerContract(browserType, engineName) {
  const browser = await browserType.launch(launchOptions(engineName));
  const context = await browser.newContext(mobileContextOptions());
  const page = await context.newPage();
  await page.addInitScript(() => {
    const nativeSetInterval = window.setInterval.bind(window);
    window.setInterval = (callback, delay, ...args) => nativeSetInterval(
      callback,
      delay === 1000 ? 1 : delay,
      ...args,
    );
  });
  await page.route('**/api/health', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ status: 'ok', mode: 'live' }),
  }));

  await page.goto(baseUrl, { waitUntil: 'domcontentloaded' });
  const elapsedHandle = await page.locator('#analysisElapsed').elementHandle();
  assert.ok(elapsedHandle, `${engineName}: elapsed timer node exists before progress starts`);
  await page.evaluate(() => {
    document.getElementById('analysisSection').classList.add('show');
    startAnalysisProgress();
  });
  await page.waitForFunction(() => parseInt(document.getElementById('analysisElapsed')?.textContent, 10) > 75, null, { timeout: 3000 });
  const timer = await elapsedHandle.evaluate(node => ({
    connected: node.isConnected,
    sameNode: node === document.getElementById('analysisElapsed'),
    visible: getComputedStyle(node).display !== 'none' && node.getBoundingClientRect().height > 0,
    seconds: parseInt(node.textContent, 10),
    status: document.getElementById('analysisTimerStatus')?.textContent,
  }));
  assert.equal(timer.connected, true, `${engineName}: elapsed timer node remains connected after simulated 75 seconds`);
  assert.equal(timer.sameNode, true, `${engineName}: elapsed timer is not replaced after simulated 75 seconds`);
  assert.equal(timer.visible, true, `${engineName}: elapsed timer remains visible after simulated 75 seconds`);
  assert.ok(timer.seconds > 75, `${engineName}: visible timer advances past 75 seconds (${timer.seconds}s)`);
  assert.match(timer.status, /still working on your personalized plan/i, `${engineName}: long-wait reassurance remains visible`);
  await page.evaluate(() => stopAnalysisProgress(false));
  await browser.close();
}

(async () => {
  assert.ok(fs.existsSync(brittanyPhoto), `Brittany photo must be present: ${brittanyPhoto}`);
  fs.mkdirSync(artifactDir, { recursive: true });
  fs.mkdirSync(pdfOutputDir, { recursive: true });

  for (const [engineName, browserType] of [['chromium', chromium], ['webkit', webkit]]) {
    if (requestedEngine && requestedEngine !== engineName) continue;
    for (const viewport of cases) {
      if (requestedViewport && requestedViewport !== viewport.name) continue;
      await runCase(browserType, engineName, viewport);
      process.stdout.write(`PASS ${engineName} ${viewport.name} ${viewport.width}x${viewport.height}\n`);
    }
    if (!requestedViewport || requestedViewport === 'desktop') {
      await runOriginalGuidedOneImageContract(browserType, engineName);
      process.stdout.write(`PASS ${engineName} original guided one-image contract\n`);
      await runBodyAreaCameraContract(browserType, engineName);
      process.stdout.write(`PASS ${engineName} non-face Quick Snap contract\n`);
      await runDemoDisclosureContract(browserType, engineName);
      process.stdout.write(`PASS ${engineName} demo disclosure contract\n`);
      await runRejectionContract(browserType, engineName);
      process.stdout.write(`PASS ${engineName} 422 rejection contract\n`);
      await runServerTimeoutRecoveryContract(browserType, engineName);
      process.stdout.write(`PASS ${engineName} 504 timeout recovery contract\n`);
      await runClientAbortRecoveryContract(browserType, engineName);
      process.stdout.write(`PASS ${engineName} client AbortController recovery contract\n`);
      await runNonretryableProviderContract(browserType, engineName);
      process.stdout.write(`PASS ${engineName} nonretryable provider recovery contract\n`);
      await runAcceleratedTimerContract(browserType, engineName);
      process.stdout.write(`PASS ${engineName} accelerated 75-second timer contract\n`);
    }
  }
  process.stdout.write('ALL REQUESTED BROWSER CASES PASSED\n');
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
