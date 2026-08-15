import { rm } from 'node:fs/promises';
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

export default defineConfig({
  plugins: [excludeModels()],
});
