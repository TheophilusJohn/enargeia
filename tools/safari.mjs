/**
 * A minimal W3C WebDriver client for real Safari, via `safaridriver`.
 *
 * Playwright ships WebKit, which is Safari's engine and close enough for most things — but it
 * is not Safari, and the whole reason this file exists is that a bug was reported in Safari
 * after every visual check had been done in headless Chromium. Checking in the actual browser
 * removes one layer of "close enough" from the answer.
 *
 * Start the driver first:  safaridriver -p 4444 &
 *
 * Only the handful of endpoints this project needs are implemented.
 */

const BASE = process.env.SAFARIDRIVER ?? 'http://localhost:4444';

async function call(method, path, body) {
  const response = await fetch(`${BASE}${path}`, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const json = await response.json().catch(() => ({}));
  if (!response.ok || json?.value?.error) {
    throw new Error(`${method} ${path}: ${json?.value?.error ?? response.status} — ${json?.value?.message ?? ''}`);
  }
  return json.value;
}

export async function session() {
  const created = await call('POST', '/session', {
    capabilities: { alwaysMatch: { browserName: 'safari' } },
  });
  const id = created.sessionId;
  const at = (path) => `/session/${id}${path}`;

  return {
    id,
    capabilities: created.capabilities,
    async go(url) {
      await call('POST', at('/url'), { url });
    },
    /** Synchronous script. Returns whatever it returns, JSON-serialised. */
    async run(script, args = []) {
      return call('POST', at('/execute/sync'), { script: `return (${script}).apply(null, arguments)`, args });
    },
    async size(width, height) {
      await call('POST', at('/window/rect'), { width, height, x: 0, y: 0 });
    },
    /** Full-window screenshot, base64 PNG decoded to a Buffer. */
    async screenshot() {
      return Buffer.from(await call('GET', at('/screenshot')), 'base64');
    },
    async wait(ms) {
      await new Promise((resolve) => setTimeout(resolve, ms));
    },
    async end() {
      await call('DELETE', at('')).catch(() => {});
    },
  };
}
