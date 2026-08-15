// Final check against the deployed site: cold load, generation, inspector, mobile.
import { chromium, devices } from 'playwright';
const browser = await chromium.launch({ channel: 'chromium', args: ['--enable-unsafe-webgpu'] });

const context = await browser.newContext();
const page = await context.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text().slice(0, 160)); });

await page.goto('https://enargeia.dev/', { waitUntil: 'domcontentloaded' });
const t0 = Date.now();
await page.locator('#app button.primary').click();
await page.waitForSelector('.composer textarea', { timeout: 1_800_000 });
const ready = (Date.now() - t0) / 1000;
await page.evaluate(() => document.getElementById('demo').scrollIntoView());
await page.locator('.composer textarea').fill('Explain what a GPU workgroup is, briefly.');
const asked = Date.now();
await page.locator('.composer button.primary').click();
await page.waitForFunction(() => (document.querySelector('.turn.assistant .body')?.textContent?.length ?? 0) > 0, { timeout: 120_000 });
const ttft = (Date.now() - asked) / 1000;
await page.waitForTimeout(14_000);
const state = await page.evaluate(() => ({
  tps: document.querySelector('.readout .big')?.textContent,
  kernels: [...document.querySelectorAll('.kernel-row')].map((r) => r.textContent),
  ledger: document.querySelectorAll('.inspector .kv')[2]?.textContent,
  reply: document.querySelector('.turn.assistant .body')?.textContent?.slice(0, 130),
}));
console.log(`COLD LOAD  ${ready.toFixed(1)} s   TTFT ${ttft.toFixed(2)} s   decode ${state.tps} tok/s`);
console.log(`reply   ${JSON.stringify(state.reply)}`);
console.log(`kernels ${state.kernels.join(' | ')}`);
console.log(`ledger  ${state.ledger}`);
await page.screenshot({ path: '/private/tmp/claude-501/-Users-theojohn-dev-enargeia/b988e848-6b32-4a21-a668-f572ba22b961/scratchpad/live-desktop.png' });

const page2 = await context.newPage();
await page2.goto('https://enargeia.dev/', { waitUntil: 'domcontentloaded' });
const w0 = Date.now();
await page2.locator('#app button.primary').click();
await page2.waitForSelector('.composer textarea', { timeout: 600_000 });
console.log(`WARM LOAD  ${((Date.now() - w0) / 1000).toFixed(1)} s`);
await context.close();

const m = await browser.newContext({ ...devices['iPhone 14 Pro'] });
const mp = await m.newPage();
mp.on('pageerror', (e) => errors.push('mobile: ' + e.message));
await mp.goto('https://enargeia.dev/', { waitUntil: 'networkidle' });
console.log('mobile overflow:', await mp.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth));
await mp.evaluate(() => document.getElementById('demo').scrollIntoView());
await mp.screenshot({ path: '/private/tmp/claude-501/-Users-theojohn-dev-enargeia/b988e848-6b32-4a21-a668-f572ba22b961/scratchpad/live-mobile.png' });
console.log('errors:', errors.length ? errors.slice(0, 6) : '(none)');
await browser.close();
