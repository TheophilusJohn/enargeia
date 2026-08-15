/**
 * Which browser the checks drive.
 *
 * Playwright's *bundled* Chromium, headless, was falling back to SwiftShader on this machine —
 * decode measured 0.2 tok/s against 34–38. Nothing else was: real Chrome gets `metal-3` with
 * `shader-f16` headless or headed, and so does the bundled build when it is headed. So it was
 * never a machine-wide GPU failure and never something a visitor would hit; it was one binary
 * in one mode.
 *
 * | launcher                    | adapter      | f16 |
 * |-----------------------------|--------------|-----|
 * | bundled Chromium, headless  | swiftshader  | no  |
 * | bundled Chromium, headed    | metal-3      | yes |
 * | real Chrome, headless       | metal-3      | yes |
 * | real Chrome, headed         | metal-3      | yes |
 *
 * The checks therefore prefer installed Chrome, which is also the engine most visitors use.
 * `ENGINE=chromium` forces the bundled build, `ENGINE=webkit` selects Safari's engine.
 */

import { chromium, webkit } from 'playwright';

const CHROME_ARGS = ['--enable-unsafe-webgpu'];

/** Launch a browser for the behavioural checks, preferring hardware WebGPU. */
export async function launchChecked(engineName = process.env.ENGINE) {
  if (engineName === 'webkit') {
    return { browser: await webkit.launch(), label: 'webkit' };
  }
  if (engineName !== 'chromium') {
    // Installed Chrome first: it has hardware WebGPU here where the bundled headless build
    // does not, and it is what visitors run.
    try {
      const browser = await chromium.launch({ channel: 'chrome', args: CHROME_ARGS });
      return { browser, label: 'chrome' };
    } catch {
      // Not installed. Fall through rather than fail: the checks are still worth running.
    }
  }
  return { browser: await chromium.launch({ args: CHROME_ARGS }), label: 'chromium (bundled)' };
}

/** The adapter a launched browser actually got, for reporting alongside any result. */
export async function adapterOf(page) {
  return page.evaluate(async () => {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) return { architecture: '', vendor: '', f16: false };
    return {
      architecture: adapter.info?.architecture ?? '',
      vendor: adapter.info?.vendor ?? '',
      f16: adapter.features.has('shader-f16'),
    };
  });
}
