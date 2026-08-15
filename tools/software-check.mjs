#!/usr/bin/env node
/**
 * The software-rasterizer warning.
 *
 * Chromium fell back to SwiftShader mid-session on the machine this project is built on, and
 * decode measured 0.2 tok/s against 34–38 an hour earlier. Nothing on the page said so, and the
 * only conclusion available to someone watching was that the engine is slow. This checks that
 * the warning appears when it should — and, just as importantly, that it stays away when it
 * should not, because a warning that fires on a healthy machine is worse than none.
 *
 * The fault is injected rather than waited for: `?forceSoftware` makes the device layer report
 * a software adapter whatever the hardware is, the same way `?clampStorage` exercises the
 * split-binding path. A check that has only ever been run against a passing case has not been
 * shown to detect anything.
 *
 *   node tools/software-check.mjs
 *   URL=https://enargeia.dev/ node tools/software-check.mjs
 *   ENGINE=webkit node tools/software-check.mjs
 */

import { chromium, webkit } from 'playwright';

const TARGET = process.env.URL ?? 'http://localhost:4180/';
const ENGINE = process.env.ENGINE === 'webkit' ? webkit : chromium;

const browser = await ENGINE.launch(
  ENGINE === chromium ? { args: ['--enable-unsafe-webgpu'] } : {},
);

/** Load the app and report what the warning surfaces say. */
async function inspect(url) {
  const page = await (await browser.newContext({ viewport: { width: 1440, height: 900 } })).newPage();
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.locator('#app button.primary').click();
  await page.waitForSelector('.composer textarea', { timeout: 600_000 });
  await page.evaluate(() => document.getElementById('demo')?.scrollIntoView());
  await page.waitForTimeout(400);

  const state = await page.evaluate(() => {
    const banner = document.querySelector('#app > .notice');
    const panel = document.querySelector('.inspector .notice');
    // Read the architecture out of its own row rather than the whole panel's text.
    let architecture = '';
    for (const list of document.querySelectorAll('.inspector .kv')) {
      const terms = [...list.querySelectorAll('dt')];
      const index = terms.findIndex((dt) => dt.textContent === 'architecture');
      if (index >= 0) architecture = list.querySelectorAll('dd')[index]?.textContent ?? '';
    }
    return {
      banner: banner ? (banner.querySelector('strong')?.textContent ?? '') : null,
      bannerBody: banner ? (banner.querySelector('.prose')?.textContent ?? '') : null,
      panel: panel ? (panel.querySelector('strong')?.textContent ?? '') : null,
      throughputNote: document.querySelector('.gauge .warn-text')?.textContent || null,
      throughputHidden: document.querySelector('.gauge .warn-text')?.hidden ?? null,
      degraded: !!document.querySelector('.readout .big.degraded'),
      architecture,
    };
  });
  await page.close();
  return { ...state, errors };
}

// The negative cases — a real GPU that merely lacks f16, a redacted adapter that has it — are
// unit tests over `classifySoftware`, in test/gpu/software.test.ts. They cannot be asserted
// here: this machine is *actually* on SwiftShader today, so there is no healthy adapter to
// point the browser at. What can be asserted end to end is that the warning agrees with the
// adapter the browser reports, whichever that is.
const SOFTWARE_NAMES = /swiftshader|llvmpipe|lavapipe|warp/i;

const cases = [
  {
    name: 'forced software',
    url: `${TARGET}${TARGET.includes('?') ? '&' : '?'}forceSoftware`,
    expect: () => ({
      warn: true,
      // Every claim the warning is supposed to make.
      says: [/software/i, /orders of magnitude/i, /restart/i, /hardware acceleration/i],
    }),
  },
  {
    name: 'adapter as reported',
    url: TARGET,
    // Whatever this machine has: the warning must be present exactly when the adapter names a
    // software rasterizer, and absent otherwise.
    expect: (found) => ({ warn: SOFTWARE_NAMES.test(found.architecture), says: [] }),
  },
];

let failures = 0;
for (const testCase of cases) {
  const found = await inspect(testCase.url);
  const expected = testCase.expect(found);
  const problems = [];

  const has = (value) => value !== null && value !== undefined && value !== '';
  const throughputShown = has(found.throughputNote) && found.throughputHidden === false;
  // All three surfaces, together: a warning in one place and not the others is the failure
  // mode that lets a bad number look like a result.
  for (const [label, shown] of [
    ['banner above the chat', has(found.banner)],
    ['device panel notice', has(found.panel)],
    ['throughput note', throughputShown],
    ['degraded throughput number', found.degraded],
  ]) {
    if (shown !== expected.warn) {
      problems.push(`${label} ${shown ? 'shown' : 'missing'}, expected ${expected.warn ? 'shown' : 'absent'}`);
    }
  }
  for (const pattern of expected.says) {
    if (!pattern.test(found.bannerBody ?? '')) problems.push(`banner does not mention ${pattern}`);
  }
  if (found.errors.length) problems.push(`page errors: ${found.errors[0]}`);

  const ok = problems.length === 0;
  if (!ok) failures++;
  console.log(`${ok ? 'pass' : 'FAIL'}  ${testCase.name.padEnd(18)} ` +
    `banner=${has(found.banner)} panel=${has(found.panel)} ` +
    `throughput=${throughputShown} degraded=${found.degraded} arch="${found.architecture}"`);
  for (const problem of problems) console.log(`        ${problem}`);
  if (ok && found.banner) console.log(`        "${found.banner}"`);
}

await browser.close();
console.log(failures === 0 ? '\nsoftware warning behaves' : `\n${failures} failing`);
process.exit(failures === 0 ? 0 : 1);
