#!/usr/bin/env node
// `npm run residency` — where resident memory goes, and whether chunking prefill would pay.
import { spawnSync } from 'node:child_process';
const r = spawnSync('npx', ['vitest', 'run', 'test/parity/residency.test.ts', '--reporter=verbose', '--testTimeout=1800000'],
  { stdio: 'inherit', env: { ...process.env, ENARGEIA_PARITY: '1' } });
process.exit(r.status ?? 1);
