#!/usr/bin/env node
/**
 * The favicon and its raster derivatives.
 *
 * The mark is the residual stream falling through layers — the project's central image, reduced
 * until it survives 16 pixels. Three horizontal bars are the layer boundaries; the vertical
 * stroke is the stream passing through them. It carries exactly two hues, and they are the two
 * ends of the kernel spectrum used for what they actually mean: `--k0` at the top, where a token
 * enters at the embedding, and `--k6` below, where it leaves through the sampler. The hue change
 * happens behind the middle bar, because that is where the stream is inside a layer.
 *
 * Everything is drawn on a 16-unit grid so that at 16 CSS pixels every edge lands on a device
 * pixel and nothing is antialiased into mush. Candidates were compared as 16×16 rasters in a
 * mock tab strip, light and dark, which is the only size that decides a favicon: three of the
 * eight first-round designs were legible at 512 and unreadable at 16.
 *
 *   node tools/make-icons.mjs
 */

import { writeFile } from 'node:fs/promises';
import { chromium } from 'playwright';

const OUT = 'public';

const BG = '#05050A';   // --bg
const LAYER = '#4E4E62'; // structural, between --line and --muted; present but never competing
const K0 = '#FF4757';   // --k0 embedding — where a token enters
const K6 = '#B45BFF';   // --k6 sample — where it leaves

/**
 * @param rx corner radius. The browser favicon rounds its own corners so it carries a small
 *   one; the iOS icon must be square and full-bleed because iOS applies the mask itself.
 * @param inset padding around the mark, in grid units. iOS crops, so its icon is drawn smaller.
 */
function mark({ rx = 3, inset = 0 } = {}) {
  const scale = (16 - inset * 2) / 16;
  const at = (v) => +(inset + v * scale).toFixed(3);
  const size = (v) => +(v * scale).toFixed(3);
  const bars = [3, 7, 11]
    .map((y) => `<rect x="${at(1)}" y="${at(y)}" width="${size(14)}" height="${size(2)}" fill="${LAYER}"/>`)
    .join('');
  return (
    `<rect width="16" height="16"${rx ? ` rx="${rx}"` : ''} fill="${BG}"/>` +
    bars +
    `<rect x="${at(6)}" y="${at(0)}" width="${size(4)}" height="${size(8)}" fill="${K0}"/>` +
    `<rect x="${at(6)}" y="${at(8)}" width="${size(4)}" height="${size(8)}" fill="${K6}"/>`
  );
}

const svg = (inner, extra = '') =>
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"${extra} shape-rendering="crispEdges">` +
  `${inner}</svg>\n`;

const FAVICON = svg(mark({ rx: 3 }));
/** iOS crops to its own rounded square, so: square, full-bleed, mark inset from the edges. */
const APPLE = svg(mark({ rx: 0, inset: 2 }));
/** Maskable icons can be cropped to a circle of 80% diameter; the mark stays inside it. */
const MASKABLE = svg(mark({ rx: 0, inset: 2.5 }));

async function raster(browser, source, size, path) {
  const page = await browser.newPage({ viewport: { width: size, height: size }, deviceScaleFactor: 1 });
  const uri = `data:image/svg+xml;base64,${Buffer.from(source).toString('base64')}`;
  await page.setContent(
    `<style>html,body{margin:0;padding:0;overflow:hidden}img{display:block}</style>` +
    `<img src="${uri}" width="${size}" height="${size}">`,
    { waitUntil: 'load' },
  );
  await page.waitForFunction(() => document.images[0]?.complete === true);
  await page.screenshot({ path, omitBackground: false });
  await page.close();
}

const browser = await chromium.launch();
await writeFile(`${OUT}/favicon.svg`, FAVICON);
await raster(browser, APPLE, 180, `${OUT}/apple-touch-icon.png`);
await raster(browser, FAVICON, 192, `${OUT}/icon-192.png`);
await raster(browser, FAVICON, 512, `${OUT}/icon-512.png`);
await raster(browser, MASKABLE, 512, `${OUT}/icon-maskable-512.png`);
// A 32px ICO-substitute PNG for the handful of places that still ignore SVG favicons.
await raster(browser, FAVICON, 32, `${OUT}/favicon-32.png`);
await browser.close();

console.log('wrote favicon.svg, apple-touch-icon.png, icon-192/512, icon-maskable-512, favicon-32');
