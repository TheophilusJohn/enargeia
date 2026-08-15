#!/usr/bin/env node
// `npm run sweep` — every section, five widths, in WebKit (Safari's engine).
//
// This exists because the site was built and checked entirely in headless Chromium, and the
// first person to open it in Safari found a broken section immediately. The automatic checks
// below catch the classes of fault a screenshot review misses: horizontal overflow, undersized
// tap targets, console errors. The screenshots are for everything else.
//
//   node tools/browser-sweep.mjs                 # local preview
//   URL=https://enargeia.dev/ node tools/browser-sweep.mjs
//   REDUCED=1 node tools/browser-sweep.mjs       # prefers-reduced-motion
import { webkit } from 'playwright';
const TARGET = process.env.URL ?? 'http://localhost:4180/';
const OUT = process.env.OUT ?? '.sweep';
await (await import('node:fs/promises')).mkdir(OUT, { recursive: true });

const VIEWS = [
  { label: 'desktop', width: 1440, height: 900 },
  { label: 'laptop', width: 1180, height: 800 },
  { label: 'tablet', width: 820, height: 1100 },
  { label: 'mobile', width: 393, height: 852 },
  { label: 'mobile-small', width: 320, height: 700 },
];

// Preflight: every module the document references must actually be served as JavaScript.
//
// A missing asset comes back as the SPA fallback — HTML with a 200 — and Cloudflare will cache
// that response against the asset's URL. Requesting an asset during the window between a deploy
// completing and it propagating is enough to poison the edge for that URL, which happened once
// and cost an hour. So: check with plain fetches first, and refuse to open a browser until the
// deployment is actually serving what the HTML asks for.
const document_ = await (await fetch(TARGET)).text();
const refs = [...document_.matchAll(/(?:src|href)="(\/assets\/[^"]+\.js)"/g)].map((m) => m[1]);
const bad = [];
for (const ref of refs) {
  const response = await fetch(new URL(ref, TARGET));
  const type = response.headers.get('content-type') ?? '';
  if (!response.ok || !/javascript|ecmascript/.test(type)) {
    bad.push(`${ref} -> ${response.status} ${type}`);
  }
}
if (bad.length) {
  console.error('preflight failed — these are not being served as JavaScript:');
  for (const line of bad) console.error(`  ${line}`);
  process.exit(1);
}
console.log(`preflight: ${refs.length} module${refs.length === 1 ? '' : 's'} served correctly`);

for (const view of VIEWS) {
  const browser = await webkit.launch();
  const context = await browser.newContext({
    viewport: { width: view.width, height: view.height },
    deviceScaleFactor: 2,
    reducedMotion: process.env.REDUCED ? 'reduce' : 'no-preference',
  });
  const page = await context.newPage();
  const problems = [];
  page.on('pageerror', (e) => problems.push(`pageerror: ${e.message}`));
  page.on('console', (m) => { if (m.type() === 'error') problems.push(`console: ${m.text().slice(0, 140)}`); });
  await page.goto(TARGET, { waitUntil: 'networkidle' });
  await page.waitForTimeout(2500);

  const checks = await page.evaluate(() => {
    const out = [];
    const doc = document.documentElement;
    if (doc.scrollWidth > doc.clientWidth) {
      out.push(`horizontal overflow: scrollWidth ${doc.scrollWidth} > clientWidth ${doc.clientWidth}`);
      for (const el of document.querySelectorAll('body *')) {
        const b = el.getBoundingClientRect();
        if (b.right > doc.clientWidth + 1 && b.width > 0) {
          out.push(`  overflows: <${el.tagName.toLowerCase()} class="${el.className}"> right=${Math.round(b.right)}`);
          if (out.length > 8) break;
        }
      }
    }
    // Any interactive target smaller than 24px in either direction.
    for (const el of document.querySelectorAll('button, a')) {
      const b = el.getBoundingClientRect();
      if (b.width > 0 && (b.height < 24 || b.width < 24)) {
        out.push(`small target: <${el.tagName.toLowerCase()}> "${el.textContent?.trim().slice(0, 24)}" ${Math.round(b.width)}x${Math.round(b.height)}`);
      }
    }
    return out;
  });
  if (checks.length) problems.push(...checks);

  const sections = await page.evaluate(() =>
    [...document.querySelectorAll('main section, header, footer')].map((el, i) => ({
      i, id: el.id || el.tagName.toLowerCase(),
      top: Math.round(el.getBoundingClientRect().top + scrollY),
      height: Math.round(el.getBoundingClientRect().height),
    })));

  for (const s of sections) {
    await page.evaluate((y) => scrollTo(0, y), Math.max(0, s.top - 8));
    await page.waitForTimeout(500);
    await page.screenshot({ path: `${OUT}/sweep-${view.label}-${s.i}-${s.id}.png` });
  }
  console.log(`\n${view.label} ${view.width}x${view.height}: ${sections.length} sections`);
  console.log(problems.length ? problems.map((p) => '  ' + p).join('\n') : '  (no automatic problems)');
  await browser.close();
}
