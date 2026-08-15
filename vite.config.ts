import { readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { defineConfig } from 'vite';

/**
 * The model files live in `public/models/` so the dev server can range-request them, and they
 * are gitignored. Vite copies `public/` into the build wholesale, which puts up to 2 GB of
 * weights into `dist/` — and Cloudflare Pages rejects any file over 25 MiB. Production fetches
 * the weights from R2 (see `.env.production`), so they have no business in the bundle.
 */
function excludeModels() {
  return {
    name: 'enargeia:exclude-models',
    async closeBundle() {
      await rm('dist/models', { recursive: true, force: true });
    },
  };
}

/**
 * Inline the stylesheet into the document.
 *
 * It is 3 kB gzipped, and as a separate file it is a render-blocking request on a fresh
 * connection — Lighthouse measured 150 ms of the live site's first paint against it. Inlining
 * costs that 3 kB on every load instead of caching it, which is the right trade for a page
 * whose whole job is to look immediate before a 335 MiB download starts.
 */
function inlineStylesheet() {
  return {
    name: 'enargeia:inline-css',
    async closeBundle() {
      const assets = await readdir('dist/assets');
      const sheet = assets.find((name) => name.endsWith('.css'));
      if (!sheet) return;
      const css = await readFile(`dist/assets/${sheet}`, 'utf8');
      const html = await readFile('dist/index.html', 'utf8');
      await writeFile(
        'dist/index.html',
        html.replace(
          new RegExp(`<link rel="stylesheet"[^>]*href="/assets/${sheet}"[^>]*>`),
          `<style>${css}</style>`,
        ),
      );
    },
  };
}

export default defineConfig({
  plugins: [excludeModels(), inlineStylesheet()],
});
