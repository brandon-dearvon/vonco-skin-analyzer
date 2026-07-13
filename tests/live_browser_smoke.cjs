const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium, webkit } = require('playwright');

const baseUrl = process.env.TEST_BASE_URL || 'http://127.0.0.1:5004';
const brittanyPhoto = process.env.BRITTANY_TEST_PHOTO
  || path.resolve('work/test-images/brittany-test-photo.jpeg');
const artifactDir = path.resolve('work/qa/live-preview');
const expectedFaceConcerns = [
  'darkSpots', 'laxity', 'pores', 'redness',
  'sunDamage', 'texture', 'unevenTone', 'wrinkles',
];

function launchOptions(engineName) {
  return engineName === 'chromium'
    ? { headless: true, executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' }
    : { headless: true };
}

async function runLiveUpload(browserType, engineName) {
  const browser = await browserType.launch(launchOptions(engineName));
  const context = await browser.newContext({ viewport: { width: 390, height: 844 } });
  const page = await context.newPage();
  const pageErrors = [];
  page.on('pageerror', error => pageErrors.push(String(error)));

  const healthResponse = await context.request.get(`${baseUrl}/api/health`);
  assert.equal(healthResponse.ok(), true, `${engineName}: preview health request succeeds`);
  assert.equal((await healthResponse.json()).mode, 'live', `${engineName}: preview is in live mode`);

  await page.goto(baseUrl, { waitUntil: 'domcontentloaded' });
  await page.locator('#fileInput').setInputFiles(brittanyPhoto);
  await page.waitForSelector('#leadGateOverlay.show', { timeout: 120000 });

  const pending = await page.evaluate(() => ({
    live: pendingAnalysisData?._isLive === true,
    demo: pendingAnalysisData?._isDemo === true,
    selectedBodyArea: document.getElementById('bodyAreaSelect')?.value,
    concernKeys: Object.keys(pendingAnalysisData?.concerns || {}).sort(),
    positives: pendingAnalysisData?.positiveHighlights || [],
    summary: pendingAnalysisData?.summary || '',
    recommendations: pendingAnalysisData?.recommendations || [],
    products: pendingAnalysisData?.productRecommendations || [],
    hasSkinAge: Object.prototype.hasOwnProperty.call(pendingAnalysisData || {}, 'skinAge'),
  }));

  assert.equal(pending.live, true, `${engineName}: Brittany result is marked live`);
  assert.equal(pending.demo, false, `${engineName}: Brittany result is not demo fallback`);
  assert.equal(pending.selectedBodyArea, 'face', `${engineName}: face remains selected`);
  assert.deepEqual(pending.concernKeys, expectedFaceConcerns, `${engineName}: face concern contract is exact`);
  assert.ok(pending.positives.length >= 2 && pending.positives.length <= 3, `${engineName}: result has 2-3 positives`);
  assert.ok(
    pending.summary.startsWith(pending.positives[0].detail),
    `${engineName}: summary leads with the first grounded positive`,
  );
  assert.ok(
    pending.recommendations.length >= 3 && pending.recommendations.length <= 5,
    `${engineName}: result has 3-5 treatment recommendations`,
  );
  assert.ok(pending.products.length >= 2 && pending.products.length <= 3, `${engineName}: result has 2-3 products`);
  assert.equal(pending.hasSkinAge, false, `${engineName}: result has no estimated skin age`);

  await page.getByRole('button', { name: 'Skip for now' }).click();
  await page.waitForSelector('#resultsSection.show', { timeout: 10000 });
  await page.waitForTimeout(2600);

  const rendered = await page.evaluate(() => ({
    horizontalOverflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    demoBannerVisible: getComputedStyle(document.getElementById('demoBanner')).display !== 'none',
    removedElements: document.querySelectorAll(
      '#skinAge, [id*="skinAge"], .skin-age, [class*="skin-age"], #radarChart, #radarContainer, .skin-radar',
    ).length,
    positiveText: document.getElementById('positiveLead')?.innerText || '',
    recommendationsText: document.getElementById('recommendationsSection')?.innerText || '',
    planSummary: document.getElementById('resultsPlanSummary')?.textContent || '',
    visibleFindings: [...document.querySelectorAll('#concernsGrid .concern-card')]
      .filter(card => !card.hidden && getComputedStyle(card).display !== 'none').length,
    hiddenFindings: document.querySelectorAll('#concernsGrid .concern-card[hidden]').length,
    treatmentCards: document.querySelectorAll('#treatmentRecommendationCards .recommendation-card').length,
    productCards: document.querySelectorAll('#productRecommendationCards .recommendation-card').length,
    recoveryVisible: document.getElementById('analysisRecovery')?.classList.contains('show'),
  }));

  assert.ok(rendered.horizontalOverflow <= 1, `${engineName}: mobile result has no horizontal overflow`);
  assert.equal(rendered.demoBannerVisible, false, `${engineName}: live result has no demo banner`);
  assert.equal(rendered.removedElements, 0, `${engineName}: age and radar remain removed`);
  assert.match(rendered.positiveText, /What Looks Especially Good/i, `${engineName}: positive lead renders first`);
  assert.match(rendered.recommendationsText, /Your Personalized Plan/i, `${engineName}: personalized plan renders`);
  assert.match(rendered.planSummary, /treatment options? and .*at-home skincare/i, `${engineName}: result overview points to treatments and skincare`);
  assert.equal(rendered.visibleFindings, 3, `${engineName}: mobile result opens with three priority findings`);
  assert.equal(rendered.hiddenFindings, 5, `${engineName}: five additional face findings remain available behind disclosure`);
  assert.equal(rendered.treatmentCards, pending.recommendations.length, `${engineName}: every treatment recommendation renders`);
  assert.equal(rendered.productCards, pending.products.length, `${engineName}: every skincare recommendation renders`);
  assert.equal(rendered.recoveryVisible, false, `${engineName}: successful live analysis does not show timeout recovery`);
  assert.deepEqual(pageErrors, [], `${engineName}: no page errors`);

  await page.screenshot({
    path: path.join(artifactDir, `${engineName}-mobile-live-brittany.png`),
    fullPage: true,
  });
  await browser.close();
  process.stdout.write(`PASS ${engineName} live Brittany upload at 390x844\n`);
}

(async () => {
  assert.ok(fs.existsSync(brittanyPhoto), `Brittany photo must be present: ${brittanyPhoto}`);
  fs.mkdirSync(artifactDir, { recursive: true });
  for (const [engineName, browserType] of [['chromium', chromium], ['webkit', webkit]]) {
    await runLiveUpload(browserType, engineName);
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
