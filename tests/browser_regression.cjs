const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium, webkit } = require('playwright');

const baseUrl = process.env.TEST_BASE_URL || 'http://127.0.0.1:5003';
const requestedEngine = process.env.TEST_ENGINE || '';
const requestedViewport = process.env.TEST_VIEWPORT || '';
const brittanyPhoto = process.env.BRITTANY_TEST_PHOTO
  || path.resolve('work/test-images/brittany-test-photo.jpeg');
const artifactDir = path.resolve('work/qa/browser-approved');
const pdfOutputDir = path.resolve('output/pdf');

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
    { treatment: 'Sciton Moxi', reason: 'An in-person consultation can determine whether this option fits your goals.', targets: ['darkSpots', 'texture'], priority: 2 },
  ],
  productRecommendations: [
    { product: 'SkinBetter Even Tone', reason: 'A provider can discuss whether this option fits your routine.' },
    { product: 'Colorescience Face Shield SPF 50', reason: 'A provider can discuss whether this option fits your routine.' },
  ],
  suggestedCombo: 'Discuss a tailored treatment sequence during an in-person consultation.',
  summary: 'Your complexion has a fresh, naturally luminous quality. A tailored plan can build on that foundation while refining visible pigment and redness.',
};

const cases = [
  { name: 'desktop', width: 1280, height: 900 },
  { name: 'mobile', width: 390, height: 844 },
  { name: 'small-mobile', width: 320, height: 568 },
];

function launchOptions(engineName) {
  return engineName === 'chromium'
    ? { headless: true, executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' }
    : { headless: true };
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
      navFilter: getComputedStyle(navLogoEl).filter,
      heroFilter: getComputedStyle(heroLogoEl).filter,
      navBackground: getComputedStyle(nav).backgroundColor,
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
      removedResultElements: document.querySelectorAll(
        '#skinAge, [id*="skinAge"], .skin-age, [class*="skin-age"], #radarChart, #radarContainer, .skin-radar',
      ).length,
      mainSiteHref: cta.href,
    };
  });

  assert.equal(landing.navSource, '/logo.png', `${engineName}/${viewport.name}: evergreen logo is a direct asset`);
  assert.equal(landing.heroSource, '/logo_white.png', `${engineName}/${viewport.name}: white logo is a direct asset`);
  assert.deepEqual(landing.navSourceSize, [1549, 848], `${engineName}/${viewport.name}: evergreen logo is canonical HD`);
  assert.deepEqual(landing.heroSourceSize, [1549, 848], `${engineName}/${viewport.name}: white logo is canonical HD`);
  assert.ok(landing.navRenderedRatioError < 0.01, `${engineName}/${viewport.name}: header logo preserves aspect ratio`);
  assert.ok(landing.heroRenderedRatioError < 0.01, `${engineName}/${viewport.name}: hero logo preserves aspect ratio`);
  assert.equal(landing.navFilter, 'none', `${engineName}/${viewport.name}: evergreen logo is not recolored by CSS`);
  assert.equal(landing.heroFilter, 'none', `${engineName}/${viewport.name}: white logo is not recolored by CSS`);
  assert.match(landing.navBackground, /rgba?\(255, 255, 255/, `${engineName}/${viewport.name}: green logo sits on a light header`);
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
  assert.equal(landing.removedResultElements, 0, `${engineName}/${viewport.name}: removed age/radar elements are absent`);
  assert.equal(landing.mainSiteHref, 'https://www.vonandcoaesthetics.com/', `${engineName}/${viewport.name}: Learn More reaches the main site`);
}

async function assertResultContract(page, engineName, viewport, uploadImageParts) {
  const result = await page.evaluate(expectedCardCount => {
    const positives = document.getElementById('positiveLead');
    const score = document.querySelector('.score-ring-container');
    const improvements = document.getElementById('concernsGrid');
    const report = buildReportHTML('Brittany', window.lastAnalysis);
    const positions = {
      positive: report.indexOf('What Looks Especially Good'),
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
      baselineReportButton: document.getElementById('downloadReportBtn')?.textContent.includes('View My Treatment Plan'),
      baselineOfferPresent: document.getElementById('promoBanner')?.textContent.includes('15% Off Your First Visit'),
      baselineClubPresent: document.getElementById('clubUpsell')?.textContent.includes('The Club'),
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
  assert.ok(result.horizontalOverflow <= 1, `${engineName}/${viewport.name}: result overflow ${result.horizontalOverflow}px`);
  assert.deepEqual(result.horizontalOffenders, [], `${engineName}/${viewport.name}: elements outside viewport: ${result.horizontalOffenders.join(' | ')}`);
  for (const [selector, left, right] of result.contentMargins) {
    assert.ok(left >= 14 && right >= 14, `${engineName}/${viewport.name}: ${selector} side buffer is ${left}px/${right}px`);
  }
}

async function assertPrintableReport(page, context, engineName) {
  const reportHtml = await page.evaluate(() => buildReportHTML('Brittany', window.lastAnalysis));
  assert.ok(
    reportHtml.indexOf('What Looks Especially Good') < reportHtml.indexOf('Overall Score:'),
    `${engineName}: report source puts positives before score`,
  );
  assert.equal(
    /Estimated Skin Age|Skin Age|radarChart|radarContainer|skin-radar|<canvas/i.test(reportHtml),
    false,
    `${engineName}: report source excludes estimated age and radar`,
  );

  const reportPage = await context.newPage();
  await reportPage.setContent(reportHtml, { waitUntil: 'domcontentloaded' });
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

  await reportPage.emulateMedia({ media: 'print' });
  const reportLayout = await reportPage.evaluate(() => ({
    overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    text: document.body.innerText,
    hasCanvas: Boolean(document.querySelector('canvas')),
  }));
  assert.ok(reportLayout.overflow <= 1, `${engineName}: printable report overflow ${reportLayout.overflow}px`);
  assert.match(reportLayout.text, /What Looks Especially Good/, `${engineName}: printable report includes positives`);
  assert.match(reportLayout.text, /Any concerning lesion needs an in-person medical evaluation\./, `${engineName}: printable report includes disclaimer`);
  assert.doesNotMatch(reportLayout.text, /Estimated Skin Age|Skin Age/i, `${engineName}: printable report has no age estimate`);
  assert.equal(reportLayout.hasCanvas, false, `${engineName}: printable report has no radar canvas`);

  await reportPage.screenshot({
    path: path.join(artifactDir, `${engineName}-take-home-report.png`),
    fullPage: true,
  });
  if (engineName === 'chromium') {
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
  const context = await browser.newContext({ viewport: { width: viewport.width, height: viewport.height } });
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

  await page.locator('#fileInput').setInputFiles(brittanyPhoto);
  await page.waitForSelector('#analysisSection.show', { timeout: 10000 });
  await page.waitForSelector('#leadGateOverlay.show', { timeout: 10000 });
  await page.getByRole('button', { name: 'Skip for now' }).click();
  await page.waitForSelector('#resultsSection.show', { timeout: 10000 });
  await page.waitForTimeout(2400);
  await assertResultContract(page, engineName, viewport, uploadImageParts);
  assert.deepEqual(pageErrors, [], `${engineName}/${viewport.name}: page errors: ${pageErrors.join('; ')}`);

  await page.evaluate(() => window.scrollTo(0, document.getElementById('resultsSection').offsetTop - 50));
  await page.waitForTimeout(150);
  await page.screenshot({ path: path.join(artifactDir, `${engineName}-${viewport.name}-results.png`) });
  await page.screenshot({ path: path.join(artifactDir, `${engineName}-${viewport.name}-full.png`), fullPage: true });

  if (viewport.name === 'desktop') await assertPrintableReport(page, context, engineName);
  await browser.close();
}

async function runOriginalGuidedOneImageContract(browserType, engineName) {
  const browser = await browserType.launch(launchOptions(engineName));
  const context = await browser.newContext({ viewport: { width: 390, height: 844 } });
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

async function runDemoDisclosureContract(browserType, engineName) {
  const browser = await browserType.launch(launchOptions(engineName));
  const context = await browser.newContext({ viewport: { width: 390, height: 844 } });
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
  await page.screenshot({ path: path.join(artifactDir, `${engineName}-demo-disclosure.png`) });
  await browser.close();
}

async function runRejectionContract(browserType, engineName) {
  const browser = await browserType.launch(launchOptions(engineName));
  const context = await browser.newContext({ viewport: { width: 390, height: 844 } });
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
      await runDemoDisclosureContract(browserType, engineName);
      process.stdout.write(`PASS ${engineName} demo disclosure contract\n`);
      await runRejectionContract(browserType, engineName);
      process.stdout.write(`PASS ${engineName} 422 rejection contract\n`);
    }
  }
  process.stdout.write('ALL REQUESTED BROWSER CASES PASSED\n');
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
