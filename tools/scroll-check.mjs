#!/usr/bin/env node
/**
 * Wheel-scrolling over every nested scroll container.
 *
 * Lenis handles wheel events on the window and applies them to the page, so a nested
 * `overflow-y: auto` container never receives them unless it carries `data-lenis-prevent`.
 * Without it the container can only be scrolled by dragging its scrollbar — which no screenshot
 * can show, and which survived a full five-width visual sweep in two engines.
 *
 * The test uses `page.mouse.wheel`, which sends input through the browser's real event pipeline.
 * A `new WheelEvent(...)` dispatched from JavaScript would prove nothing: untrusted events do not
 * trigger the default scrolling action, so the container would never move even when correct,
 * while Lenis — being ordinary JavaScript — would still react to it. Only trusted input
 * distinguishes a working container from a broken one.
 *
 * For each container: put the pointer over it, wheel down, and assert that the container scrolled
 * and the page did not.
 *
 *   node tools/scroll-check.mjs                          # local dev server, local weights
 *   URL=https://enargeia.dev/ node tools/scroll-check.mjs # the deployed site
 *   ENGINE=webkit node tools/scroll-check.mjs
 *   STRIP=1 node tools/scroll-check.mjs                   # inject the fault; this must FAIL
 *
 * `STRIP=1` removes every `data-lenis-prevent` before testing, which reproduces the original
 * bug. Run it once after changing this file: a check that has never failed has not been shown
 * to detect anything.
 */

import { launchChecked, adapterOf } from './launch.mjs';

const TARGET = process.env.URL ?? 'http://localhost:5179/';

/** Every element that can actually scroll, whether or not anyone remembered to tag it. */
const FIND_SCROLLERS = `(() => {
  const found = [];
  for (const el of document.querySelectorAll('body *')) {
    const style = getComputedStyle(el);
    const scrollsY = /auto|scroll/.test(style.overflowY);
    const overflows = el.scrollHeight > el.clientHeight + 2;
    if (!scrollsY || !overflows) continue;
    const box = el.getBoundingClientRect();
    if (box.width < 8 || box.height < 8) continue;
    found.push({
      selector: el.tagName.toLowerCase() + (el.className ? '.' + String(el.className).trim().split(/\\s+/).join('.') : ''),
      prevented: el.hasAttribute('data-lenis-prevent') ||
                 el.hasAttribute('data-lenis-prevent-wheel') ||
                 !!el.closest('[data-lenis-prevent]'),
      x: Math.round(box.left + box.width / 2),
      y: Math.round(box.top + Math.min(box.height / 2, 200)),
      scrollTop: el.scrollTop,
      scrollHeight: el.scrollHeight,
      clientHeight: el.clientHeight,
    });
  }
  return found;
})()`;

const { browser, label } = await launchChecked();
const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const page = await context.newPage();
page.on('pageerror', (error) => console.log(`[pageerror] ${error.message}`));

await page.goto(TARGET, { waitUntil: 'domcontentloaded' });
// Reported alongside the result: a check that ran on a software adapter is still a valid check
// of event routing, but it should say which browser and which adapter produced it.
console.log(`${label} — adapter ${JSON.stringify(await adapterOf(page))}`);

// The chat transcript and the inspector only exist once a session does.
const loadButton = page.locator('#app button.primary');
if (await loadButton.count() === 0) {
  const shown = (await page.locator('#app').textContent())?.slice(0, 160);
  console.error(`no load button — the app did not mount. #app says: ${shown}`);
  process.exit(1);
}
console.log('loading the model…');
await loadButton.click();
await page.waitForSelector('.composer textarea', { timeout: 600_000 });
console.log('session ready');

// Both panels need enough content to have something to scroll — but not real content. What is
// under test is where a wheel event is routed, which is a property of the DOM, the CSS and
// Lenis, and has nothing to do with the model. Generating a couple of replies took a minute on
// a healthy GPU and did not finish at all the day Chromium fell back to SwiftShader, so the
// transcript is filled directly instead.
await page.evaluate(() => {
  const transcript = document.querySelector('.transcript');
  if (!transcript) return;
  for (let i = 0; i < 30; i++) {
    const turn = document.createElement('div');
    turn.className = i % 2 === 0 ? 'turn user' : 'turn assistant';
    turn.innerHTML =
      `<span class="who">${i % 2 === 0 ? 'you' : 'qwen2.5-0.5b'}</span>` +
      `<div class="body">Filler turn ${i} — present so the transcript overflows and can be scrolled.</div>`;
    transcript.append(turn);
  }
});
await page.evaluate(() => document.getElementById('demo')?.scrollIntoView());
await page.waitForTimeout(600);

if (process.env.STRIP) {
  const removed = await page.evaluate(() => {
    const targets = document.querySelectorAll('[data-lenis-prevent]');
    for (const el of targets) el.removeAttribute('data-lenis-prevent');
    return targets.length;
  });
  console.log(`STRIP: removed data-lenis-prevent from ${removed} elements — these must now fail`);
}

const scrollers = await page.evaluate(FIND_SCROLLERS);
if (scrollers.length === 0) {
  console.error('no scrollable containers found — the page did not reach a state worth testing');
  process.exit(1);
}

let failures = 0;
for (const scroller of scrollers) {
  // Reset both the container and the page so each case starts from a known position.
  await page.evaluate((sel) => {
    const el = document.querySelector(sel);
    if (el) el.scrollTop = 0;
  }, scroller.selector);
  await page.waitForTimeout(150);

  const before = await page.evaluate((sel) => ({
    top: document.querySelector(sel)?.scrollTop ?? -1,
    page: Math.round(window.scrollY),
  }), scroller.selector);

  await page.mouse.move(scroller.x, scroller.y);
  await page.mouse.wheel(0, 240);
  // Lenis animates, so give the page a few frames to move if it is going to.
  await page.waitForTimeout(700);

  const after = await page.evaluate((sel) => ({
    top: document.querySelector(sel)?.scrollTop ?? -1,
    page: Math.round(window.scrollY),
  }), scroller.selector);

  const scrolled = after.top > before.top;
  const pageMoved = Math.abs(after.page - before.page) > 2;
  const ok = scrolled && !pageMoved;
  if (!ok) failures++;

  console.log(
    `${ok ? 'pass' : 'FAIL'}  ${scroller.selector.padEnd(28)} ` +
    `container ${before.top}→${after.top}  page ${before.page}→${after.page}` +
    `${scroller.prevented ? '' : '  [no data-lenis-prevent]'}`,
  );
  if (!ok && !scrolled) console.log('        the wheel did not scroll the container');
  if (!ok && pageMoved) console.log('        the wheel scrolled the page instead');
}

// The pinned section is the opposite assertion: wheeling over it *must* move the page, because
// that is what drives the scrub. A stray data-lenis-prevent there would freeze it.
//
// The scroll position has to land inside the pin's range. `scrollIntoView` on a pinned element
// does not: the element is `position: fixed` and its scroll length lives in the spacer around
// it, so the range is measured from the spacer instead.
// Getting into the pin's range has to be done with real wheel input: Lenis replaces
// `window.scrollTo`, so a scripted jump either animates somewhere else or is undone.
const isPinned = () =>
  page.evaluate(() => getComputedStyle(document.getElementById('how')).position === 'fixed');

await page.evaluate(() => document.getElementById('how')?.scrollIntoView());
await page.waitForTimeout(700);
await page.mouse.move(720, 500);
let pinned = await isPinned();
for (let i = 0; i < 40 && !pinned; i++) {
  await page.mouse.wheel(0, 300);
  await page.waitForTimeout(160);
  pinned = await isPinned();
}

let pinOk = true;
if (!pinned) {
  console.log('skip  #how — never entered the pinned state at this width');
} else {
  const pinBefore = await page.evaluate(() => ({
    page: Math.round(window.scrollY),
    stage: document.querySelector('.stage-detail h3')?.textContent,
    position: getComputedStyle(document.getElementById('how')).position,
  }));
  await page.mouse.wheel(0, 500);
  await page.waitForTimeout(900);
  const pinAfter = await page.evaluate(() => ({
    page: Math.round(window.scrollY),
    stage: document.querySelector('.stage-detail h3')?.textContent,
    position: getComputedStyle(document.getElementById('how')).position,
  }));
  // Wheeling over the pinned section must move the page — that is what drives the scrub — and
  // must advance the stage. A stray data-lenis-prevent here would freeze both.
  pinOk = pinAfter.page > pinBefore.page && pinAfter.stage !== pinBefore.stage;
  if (!pinOk) failures++;
  console.log(
    `${pinOk ? 'pass' : 'FAIL'}  ${'#how (pinned, must scrub)'.padEnd(28)} ` +
    `page ${pinBefore.page}→${pinAfter.page}  stage ${pinBefore.stage}→${pinAfter.stage}  ` +
    `position ${pinBefore.position}→${pinAfter.position}`,
  );
}

await browser.close();
console.log(failures === 0 ? '\nall scroll containers behave' : `\n${failures} failing`);
process.exit(failures === 0 ? 0 : 1);
