#!/usr/bin/env node
// `npm run ablation [-- --off flag1,flag2]` — one row of the M6 table.
import { spawnSync } from 'node:child_process';
const args = process.argv.slice(2);
const env = { ...process.env, ENARGEIA_PARITY: '1' };
const off = args.indexOf('--off');
if (off !== -1) env.VITE_ABLATION_OFF = args[off + 1];
const r = spawnSync('npx', ['vitest', 'run', 'test/parity/ablation.test.ts', '--reporter=verbose', '--testTimeout=1800000'], { stdio: 'inherit', env });
process.exit(r.status ?? 1);
