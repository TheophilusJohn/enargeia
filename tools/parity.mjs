#!/usr/bin/env node
// `npm run parity` wrapper: translates the CLI the parity skill documents into the env vars
// the browser-mode test reads, since vitest owns argv itself.
import { spawnSync } from 'node:child_process';

const args = process.argv.slice(2);
const env = { ...process.env, ENARGEIA_PARITY: '1' };
const layer = args.indexOf('--layer');
if (layer !== -1) env.VITE_PARITY_LAYER = args[layer + 1];
if (args.includes('--strict')) env.VITE_PARITY_STRICT = '1';
if (args.includes('--clamp')) env.VITE_PARITY_CLAMP = '1';
if (args.includes('--q4')) env.VITE_PARITY_Q4 = '1';
const tokens = args.indexOf('--tokens');
if (tokens !== -1) env.VITE_PARITY_TOKENS = args[tokens + 1];

const result = spawnSync(
  'npx',
  ['vitest', 'run', 'test/parity', '--reporter=verbose', '--testTimeout=900000'],
  { stdio: 'inherit', env },
);
process.exit(result.status ?? 1);
