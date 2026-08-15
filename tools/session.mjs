#!/usr/bin/env node
// `npm run session` — KV cache correctness and GPU sampling, against the no-cache path.
import { spawnSync } from 'node:child_process';
const r = spawnSync('npx', ['vitest', 'run', 'test/parity/session.test.ts', '--reporter=verbose', '--testTimeout=1800000'],
  { stdio: 'inherit', env: { ...process.env, ENARGEIA_PARITY: '1' } });
process.exit(r.status ?? 1);
