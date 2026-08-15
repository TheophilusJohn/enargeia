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

// Preflight: everything the document references must be served as the type it claims to be.
//
// A missing file comes back as the SPA fallback — HTML with a 200 — and Cloudflare caches that
// response against the file's URL. Requesting one in the window between a deploy completing and
// the file propagating is enough to poison the edge for that path. It happened once with a
// JavaScript chunk and once with `/apple-touch-icon.png`, which Safari probes on its own without
// anything on the page asking it to. So every reference is checked with a plain fetch before a
// browser is opened, and a mismatch is reported as a deployment problem rather than being
// requested, cached, and then puzzled over.
const EXPECTED = [
  [/\.js$/, /javascript|ecmascript/],
  [/\.css$/, /text\/css/],
  [/\.svg$/, /image\/svg/],
  [/\.png$/, /image\/png/],
  [/\.woff2$/, /font\/woff2|application\/octet-stream/],
  [/\.webmanifest$/, /manifest\+json|application\/json/],
];

const documentText = await (await fetch(TARGET)).text();
const referenced = new Set(
  [...documentText.matchAll(/(?:src|href)="(\/[^"]+\.(?:js|css|svg|png|woff2|webmanifest))"/g)].map((m) => m[1]),
);
// Icons the manifest names, which the document does not reference directly.
if (referenced.has('/site.webmanifest')) {
  const manifest = await (await fetch(new URL('/site.webmanifest', TARGET))).json().catch(() => null);
  for (const icon of manifest?.icons ?? []) referenced.add(icon.src);
}

const bad = [];
for (const ref of referenced) {
  const response = await fetch(new URL(ref, TARGET));
  const type = (response.headers.get('content-type') ?? '').split(';')[0];
  const rule = EXPECTED.find(([pattern]) => pattern.test(ref));
  if (!response.ok || (rule && !rule[1].test(type))) {
    bad.push(`${ref} -> ${response.status} ${type || '(no type)'}`);
  }
}
if (bad.length) {
  console.error('preflight failed — these are not served as the type they claim:');
  for (const line of bad) console.error(`  ${line}`);
  process.exit(1);
}
console.log(`preflight: ${referenced.size} referenced files served correctly`);

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
