/**
 * What the inspector reads.
 *
 * The UI never touches a GPUBuffer. The runtime publishes a snapshot; panels render it. That
 * boundary is what keeps `src/ui` from growing engine knowledge, and it is why the inspector
 * can be closed without changing what the engine does.
 */

import type { DeviceProfile } from '../gpu/device.ts';

/** Kernel identity -> colour, per the design system. The same hue means the same operation. */
export const KERNEL_COLOURS: Array<{ match: RegExp; group: string; colour: string }> = [
  { match: /embed_gather|lm_head/, group: 'embedding', colour: '#FF4757' },
  { match: /rmsnorm/, group: 'rmsnorm', colour: '#FF8B27' },
  { match: /matmul|cache_pack/, group: 'projection', colour: '#FFD52E' },
  { match: /rope/, group: 'rope', colour: '#3DDC6B' },
  { match: /attn|softmax/, group: 'attention', colour: '#1FD4E8' },
  { match: /silu|add/, group: 'mlp', colour: '#5B8CFF' },
  { match: /sample|argmax/, group: 'sample', colour: '#B45BFF' },
];

export function kernelGroup(name: string): { group: string; colour: string } {
  for (const entry of KERNEL_COLOURS) {
    if (entry.match.test(name)) return { group: entry.group, colour: entry.colour };
  }
  return { group: 'other', colour: '#63637C' };
}

export interface KernelTiming {
  group: string;
  colour: string;
  ms: number;
  share: number;
  dispatches: number;
}

export interface MemoryLedger {
  weightsBytes: number;
  kvCacheBytes: number;
  scratchBytes: number;
  totalBytes: number;
}

export interface AttentionSample {
  layer: number;
  heads: number;
  positions: number;
  /** [heads, positions], row-normalised already by the softmax. */
  weights: Float32Array;
}

export interface Telemetry {
  phase: 'idle' | 'prefill' | 'decoding';
  tokensPerSecond: number;
  interTokenMs: number;
  tokensGenerated: number;
  contextUsed: number;
  contextMax: number;
  memory: MemoryLedger;
  /** Empty until the first profiled step lands. */
  kernels: KernelTiming[];
  gpuMsPerToken: number | null;
  attention: AttentionSample | null;
  device: DeviceProfile;
  /** True while the inspector is paying for extra readbacks. */
  instrumented: boolean;
}

/**
 * A rolling mean over the last N samples.
 *
 * Rolling rather than cumulative because the interesting number is what the machine is doing
 * now — a cumulative average buries a slowdown under everything that came before it.
 */
export class Rolling {
  private readonly values: number[] = [];
  private readonly window: number;

  constructor(window = 16) {
    this.window = window;
  }

  push(value: number): void {
    this.values.push(value);
    if (this.values.length > this.window) this.values.shift();
  }

  get mean(): number {
    if (this.values.length === 0) return 0;
    return this.values.reduce((a, b) => a + b, 0) / this.values.length;
  }

  get last(): number {
    return this.values[this.values.length - 1] ?? 0;
  }

  clear(): void {
    this.values.length = 0;
  }
}
