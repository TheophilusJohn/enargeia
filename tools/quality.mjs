#!/usr/bin/env node
// `npm run quality [-- --q4]` — perplexity plus decode/prefill timing for one weight format.
// The two formats run in separate processes because both resident at once is 2.3 GB.
import { spawnSync } from 'node:child_process';
const args = process.argv.slice(2);
const env = { ...process.env, ENARGEIA_PARITY: '1' };
if (args.includes('--q4')) env.VITE_QUALITY_Q4 = '1';
const embed = args.indexOf('--embed');
if (embed !== -1) { env.VITE_QUALITY_Q4 = '1'; env.VITE_QUALITY_EMBED = args[embed + 1]; }
if (args.includes('--large')) { env.VITE_QUALITY_LARGE = '1'; env.VITE_QUALITY_CACHE = '1'; env.VITE_QUALITY_TOKENS = '1258'; }
if (args.includes('--cache')) env.VITE_QUALITY_CACHE = '1';
const tokens = args.indexOf('--tokens');
if (tokens !== -1) env.VITE_QUALITY_TOKENS = args[tokens + 1];
const r = spawnSync('npx', ['vitest', 'run', 'test/parity/quality.test.ts', '--reporter=verbose', '--testTimeout=1800000'], { stdio: 'inherit', env });
process.exit(r.status ?? 1);
