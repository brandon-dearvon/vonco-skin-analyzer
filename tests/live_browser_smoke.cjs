const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const { chromium, webkit } = require('playwright');
const {
  assertRenderedDomParity,
  assertStructuredResult,
  captureAnalyzeResponse,
  launchOptions,
  mobileContextOptions,
  resultDeadlineMs,
  runtimeSourceFiles,
  sourceBuildFingerprint,
} = require('./live_real_photo_matrix.cjs');

const baseUrl = process.env.TEST_BASE_URL || 'http://127.0.0.1:5004';
const brittanyPhoto = process.env.BRITTANY_TEST_PHOTO
  || path.resolve('work/test-images/brittany-test-photo.jpeg');
const acceptanceMode = process.env.LIVE_ACCEPTANCE_MODE !== 'false';
const artifactDir = path.resolve(
  process.env.LIVE_ARTIFACT_DIR
    || (acceptanceMode ? 'work/qa/live-preview' : 'work/qa/live-preview/diagnostic'),
);
const chromiumRuns = Number(process.env.LIVE_CHROMIUM_RUNS || (acceptanceMode ? 3 : 1));
const webkitRuns = Number(process.env.LIVE_WEBKIT_RUNS || (acceptanceMode ? 2 : 1));
const expectedBrittanySha256 = 'df1d305937419e60fadacaf120365162a80a828cda379aea90507a1de430132a';
const ledgerPath = path.join(artifactDir, 'brittany-reliability-ledger.json');
const expectedBuildFingerprint = String(process.env.TEST_BUILD_FINGERPRINT || '').trim();
const expectedRuntime = Object.freeze({
  model: 'gemini-3.1-pro-preview',
  thinkingLevel: 'HIGH',
  totalBudgetMs: 70000,
  hedgeDelayMs: 15000,
  maxOutputTokens: 32768,
});

function sha256File(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

function writeLedger(ledger) {
  fs.writeFileSync(ledgerPath, `${JSON.stringify(ledger, null, 2)}\n`);
}

async function runLiveUpload(browserType, engineName, runNumber) {
  let browser = null;
  let context = null;
  let page = null;
  let startedAt = null;
  let apiResponse = null;
  let apiResponsePromise = null;
  let terminalState = null;
  let analysis = null;
  const pageErrors = [];
  const consoleErrors = [];
  const requestFailures = [];
  const screenshots = [];
  const attemptPath = path.join(artifactDir, `brittany-${engineName}-run-${runNumber}-attempt.json`);
  try {
    browser = await browserType.launch(launchOptions(engineName));
    context = await browser.newContext(mobileContextOptions());
    await context.route('https://fonts.googleapis.com/**', route => route.fulfill({
      status: 200,
      contentType: 'text/css',
      body: '',
    }));
    await context.route('https://fonts.gstatic.com/**', route => route.abort());
    page = await context.newPage();
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
    const healthResponse = await context.request.get(`${baseUrl}/api/health`);
    const health = await healthResponse.json();
    assert.equal(healthResponse.ok(), true, `${engineName}: preview health request succeeds`);
    assert.equal(health.mode, 'live', `${engineName}: preview is in live mode`);
    assert.equal(
      health.buildFingerprint,
      expectedBuildFingerprint,
      `${engineName}: health fingerprint matches the restarted build under test`,
    );
    for (const [key, expectedValue] of Object.entries(expectedRuntime)) {
      assert.equal(health[key], expectedValue, `${engineName}: health reports exact ${key}`);
    }

    await page.goto(baseUrl, { waitUntil: 'domcontentloaded' });
    await page.locator('#ageInput').fill('35');
    startedAt = Date.now();
    await page.locator('#fileInput').setInputFiles(brittanyPhoto);
    await page.waitForFunction(() => (
      document.getElementById('leadGateOverlay')?.classList.contains('show')
        || document.getElementById('analysisRecovery')?.classList.contains('show')
        || Boolean(document.getElementById('rejectionOverlay'))
    ), null, { timeout: resultDeadlineMs });

    terminalState = await page.evaluate(() => ({
      lead: document.getElementById('leadGateOverlay')?.classList.contains('show') || false,
      recovery: document.getElementById('analysisRecovery')?.classList.contains('show') || false,
      recoveryTitle: document.getElementById('analysisRecoveryTitle')?.textContent?.trim() || null,
      recoveryMessage: document.getElementById('analysisRecoveryMessage')?.textContent?.trim() || null,
      rejection: Boolean(document.getElementById('rejectionOverlay')),
      rejectionTitle: document.getElementById('rejectionTitle')?.textContent?.trim() || null,
      rejectionMessage: document.getElementById('rejectionReason')?.textContent?.trim() || null,
    }));
    const analysisElapsedMs = Date.now() - startedAt;
    if (apiResponsePromise) apiResponse = await apiResponsePromise;
    assert.ok(apiResponse, `${engineName} run ${runNumber}: full /api/analyze response was captured`);
    assert.equal(terminalState.lead, true, `${engineName} run ${runNumber}: Brittany reaches completed results`);
    assert.equal(terminalState.recovery, false, `${engineName} run ${runNumber}: Brittany does not reach recovery`);
    assert.equal(terminalState.rejection, false, `${engineName} run ${runNumber}: Brittany is not rejected`);
    assert.ok(
      analysisElapsedMs < resultDeadlineMs,
      `${engineName} run ${runNumber}: Brittany completes in strictly less than 75 seconds`,
    );

    analysis = await page.evaluate(() => pendingAnalysisData);
    assert.equal(apiResponse.httpStatus, 200, `${engineName} run ${runNumber}: API returns HTTP 200`);
    assert.ok(
      ['generated', 'reused'].includes(apiResponse.headers.analysisRepeat),
      `${engineName} run ${runNumber}: API identifies canonical-result generation or reuse`,
    );
    assert.ok(apiResponse.rawJson, `${engineName} run ${runNumber}: full successful JSON is present`);
    const browserStateBeforePresentationFlag = structuredClone(analysis);
    delete browserStateBeforePresentationFlag._isLive;
    assert.deepEqual(
      apiResponse.rawJson,
      browserStateBeforePresentationFlag,
      `${engineName} run ${runNumber}: browser state matches API JSON except the frontend live marker`,
    );
    const contractEvidence = assertStructuredResult(
      { id: `brittany-${engineName}-run-${runNumber}`, area: 'face' },
      analysis,
    );

    await page.getByRole('button', { name: 'Skip for now' }).click();
    await page.waitForSelector('#resultsSection.show', { timeout: 10000 });
    await page.waitForTimeout(2600);
    const rendered = await assertRenderedDomParity(
      page,
      { id: `brittany-${engineName}-run-${runNumber}`, area: 'face' },
      analysis,
    );
    const presentation = await page.evaluate(() => ({
      demoBannerVisible: getComputedStyle(document.getElementById('demoBanner')).display !== 'none',
      visibleFindings: [...document.querySelectorAll('#concernsGrid .concern-card')]
        .filter(card => !card.hidden && getComputedStyle(card).display !== 'none').length,
      hiddenFindings: document.querySelectorAll('#concernsGrid .concern-card[hidden]').length,
    }));
    assert.equal(presentation.demoBannerVisible, false, `${engineName}: live result has no demo banner`);
    assert.equal(presentation.visibleFindings, 3, `${engineName}: result opens with three priority findings`);
    assert.equal(presentation.hiddenFindings, 5, `${engineName}: five additional face findings remain behind disclosure`);
    assert.deepEqual(pageErrors, [], `${engineName}: no page errors`);
    assert.deepEqual(consoleErrors, [], `${engineName}: no console errors`);
    assert.deepEqual(requestFailures, [], `${engineName}: no unexpected request failures`);

    const screenshotPath = path.join(
      artifactDir,
      `${engineName}-run-${runNumber}-mobile-live-brittany.png`,
    );
    await page.screenshot({ path: screenshotPath, fullPage: true });
    screenshots.push(screenshotPath);
    const record = {
      engine: engineName,
      run: runNumber,
      outcome: 'pass',
      elapsedMs: analysisElapsedMs,
      strictDeadlineMs: resultDeadlineMs,
      source: {
        provenance: 'User-provided Brittany test fixture; no public source URL is asserted.',
        sourcePath: brittanyPhoto,
        sha256: expectedBrittanySha256,
      },
      health,
      apiResponse,
      terminalState,
      analysis,
      contractEvidence,
      rendered,
      presentation,
      screenshots,
      pageErrors,
      consoleErrors,
      requestFailures,
      attemptArtifactPath: attemptPath,
    };
    fs.writeFileSync(attemptPath, `${JSON.stringify(record, null, 2)}\n`);
    process.stdout.write(
      `PASS ${engineName} run ${runNumber} live Brittany upload in ${analysisElapsedMs}ms at 390x844\n`,
    );
    return record;
  } catch (error) {
    if (apiResponse === null && apiResponsePromise !== null) {
      try { apiResponse = await apiResponsePromise; } catch (_) { apiResponse = null; }
    }
    if (page && !page.isClosed()) {
      try {
        const screenshotPath = path.join(artifactDir, `${engineName}-run-${runNumber}-failure.png`);
        await page.screenshot({ path: screenshotPath, fullPage: true });
        screenshots.push(screenshotPath);
      } catch (_) {
        // Preserve the original assertion or transport failure.
      }
    }
    const record = {
      engine: engineName,
      run: runNumber,
      outcome: 'fail',
      elapsedMs: startedAt === null ? null : Date.now() - startedAt,
      strictDeadlineMs: resultDeadlineMs,
      source: {
        provenance: 'User-provided Brittany test fixture; no public source URL is asserted.',
        sourcePath: brittanyPhoto,
        sha256: expectedBrittanySha256,
      },
      apiResponse,
      terminalState,
      analysis,
      screenshots,
      pageErrors,
      consoleErrors,
      requestFailures,
      error: String(error),
      stack: error.stack || null,
      attemptArtifactPath: attemptPath,
    };
    fs.writeFileSync(attemptPath, `${JSON.stringify(record, null, 2)}\n`);
    error.qaRecord = record;
    throw error;
  } finally {
    if (context) await context.close();
    if (browser) await browser.close();
  }
}

async function main() {
  assert.ok(fs.existsSync(brittanyPhoto), `Brittany photo must be present: ${brittanyPhoto}`);
  fs.mkdirSync(artifactDir, { recursive: true });
  assert.ok(expectedBuildFingerprint, 'TEST_BUILD_FINGERPRINT is required to prevent certifying a stale listener');
  assert.equal(
    expectedBuildFingerprint,
    sourceBuildFingerprint,
    'TEST_BUILD_FINGERPRINT must be the SHA-256 fingerprint of the current runtime source files',
  );
  assert.ok(Number.isInteger(chromiumRuns) && chromiumRuns > 0, 'Chromium run count must be a positive integer');
  assert.ok(Number.isInteger(webkitRuns) && webkitRuns > 0, 'WebKit run count must be a positive integer');
  if (acceptanceMode) {
    assert.equal(chromiumRuns, 3, 'acceptance mode requires exactly three Chromium passes');
    assert.equal(webkitRuns, 2, 'acceptance mode requires exactly two WebKit passes');
  }
  const actualSha256 = sha256File(brittanyPhoto);
  assert.equal(actualSha256, expectedBrittanySha256, 'Brittany fixture bytes match the pinned SHA-256');

  const ledger = {
    schemaVersion: 3,
    suite: 'brittany-live-reliability-gate',
    status: 'running',
    acceptanceRun: acceptanceMode,
    startedAt: new Date().toISOString(),
    completedAt: null,
    baseUrl,
    artifactDir,
    ledgerPath,
    expectedBuildFingerprint,
    sourceBuildFingerprint,
    runtimeSourceFiles,
    expectedRuntime,
    contract: {
      acceptanceSequence: 'Exactly 3 local headless Google Chrome (Chromium) attempts and 2 local headless Playwright WebKit attempts; any failure invalidates the run.',
      timing: 'Each attempt must reach completed results in strictly less than 75,000 ms from file assignment.',
      response: 'Capture full raw API JSON and enforce the deep face response/catalog/copy/anatomy contract.',
      repeatability: 'Every run sends the same Brittany photo, face selection, and age 35. All five completed API JSON results must have one canonical SHA-256 hash.',
      presentation: 'Every positive, finding, treatment, product, reason, and priority must match the API response in a 390x844 touch-enabled mobile DOM with a 390x844 screen, device scale factor 3, and isMobile=true.',
      runtime: 'The caller-supplied and health build fingerprints must equal the deterministic SHA-256 of server.py, index.html, and all three logo assets; every health check must also report Gemini 3.1 Pro Preview, HIGH thinking, 70,000 ms total budget, 15,000 ms hedge delay, and 32,768 output tokens.',
    },
    qualification: {
      validates: 'Repeated live functional reliability and presentation parity for one exact user-provided photo.',
      doesNotValidate: 'Clinical accuracy or medical diagnosis.',
      webkit: 'Automated Playwright WebKit is not a claim of a physical iPhone Mobile Safari test.',
    },
    source: {
      provenance: 'User-provided Brittany test fixture; no public source URL is asserted.',
      sourcePath: brittanyPhoto,
      sha256: actualSha256,
      expectedSha256: expectedBrittanySha256,
    },
    requestedRuns: { chromium: chromiumRuns, webkit: webkitRuns },
    records: [],
    summary: null,
  };
  writeLedger(ledger);
  const failures = [];
  for (const [engineName, browserType, runs] of [
    ['chromium', chromium, chromiumRuns],
    ['webkit', webkit, webkitRuns],
  ]) {
    for (let runNumber = 1; runNumber <= runs; runNumber += 1) {
      try {
        ledger.records.push(await runLiveUpload(browserType, engineName, runNumber));
      } catch (error) {
        failures.push(String(error));
        ledger.records.push(error.qaRecord || {
          engine: engineName,
          run: runNumber,
          outcome: 'fail',
          error: String(error),
        });
        process.stderr.write(`FAIL ${String(error)}\n`);
      }
      writeLedger(ledger);
    }
  }
  ledger.summary = {
    total: ledger.records.length,
    passed: ledger.records.filter(record => record.outcome === 'pass').length,
    failed: ledger.records.filter(record => record.outcome === 'fail').length,
    chromiumPassed: ledger.records.filter(record => record.engine === 'chromium' && record.outcome === 'pass').length,
    webkitPassed: ledger.records.filter(record => record.engine === 'webkit' && record.outcome === 'pass').length,
    allStrictlyUnder75Seconds: ledger.records
      .filter(record => record.outcome === 'pass')
      .every(record => record.elapsedMs < resultDeadlineMs),
    canonicalResultHashes: [...new Set(
      ledger.records
        .filter(record => record.outcome === 'pass' && record.apiResponse?.rawJson)
        .map(record => crypto.createHash('sha256')
          .update(JSON.stringify(record.apiResponse.rawJson))
          .digest('hex')),
    )],
    reusedCanonicalResults: ledger.records.filter(
      record => record.outcome === 'pass'
        && record.apiResponse?.headers?.analysisRepeat === 'reused',
    ).length,
  };
  try {
    assert.deepEqual(failures, [], `Live Brittany reliability failures:\n${failures.join('\n')}`);
    if (acceptanceMode) {
      assert.equal(ledger.summary.total, 5, 'acceptance run records all five attempts');
      assert.equal(ledger.summary.chromiumPassed, 3, 'all three Chromium attempts pass');
      assert.equal(ledger.summary.webkitPassed, 2, 'both WebKit attempts pass');
      assert.equal(ledger.summary.allStrictlyUnder75Seconds, true, 'all five passes are strictly under 75 seconds');
      assert.equal(ledger.summary.canonicalResultHashes.length, 1, 'all five runs return one canonical result');
      assert.ok(ledger.summary.reusedCanonicalResults >= 4, 'at least four runs reuse the canonical result');
    }
    ledger.status = acceptanceMode ? 'passed' : 'diagnostic-passed';
  } catch (error) {
    ledger.status = 'failed';
    ledger.harnessError = { error: String(error), stack: error.stack || null };
    throw error;
  } finally {
    ledger.completedAt = new Date().toISOString();
    writeLedger(ledger);
  }
}

if (require.main === module) {
  main().catch(error => {
    console.error(error);
    process.exitCode = 1;
  });
}
