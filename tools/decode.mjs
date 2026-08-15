#!/usr/bin/env node
// `npm run decode [-- --fp32]` — TTFT and inter-token latency across context lengths.
import { spawnSync } from 'node:child_process';
const env = { ...process.env, ENARGEIA_PARITY: '1' };
if (process.argv.includes('--fp32')) env.VITE_DECODE_FP32 = '1';
const r = spawnSync('npx', ['vitest', 'run', 'test/parity/decode.test.ts', '--reporter=verbose', '--testTimeout=1800000'], { stdio: 'inherit', env });
process.exit(r.status ?? 1);
