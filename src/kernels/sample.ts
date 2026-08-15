import code from './sample.wgsl?raw';
import type { KernelSpec } from './kernel.ts';

/**
 * logits, history, out, dims. One workgroup for the whole vocabulary.
 *
 * Everything the decode loop needs to choose a token happens here: repetition penalty,
 * temperature, top-p and the draw. Only `out[0]` is ever read back, which is the four bytes
 * the project's one-readback-per-token budget allows.
 */
export const SAMPLE: KernelSpec = {
  name: 'sample',
  code,
  bindings: ['read', 'read', 'read_write', 'uniform'],
  workgroupSize: [256, 1, 1],
  uniformBytes: 32,
};

export interface SamplingParams {
  /** 0 selects greedy, which takes the argmax path and ignores topP. */
  temperature: number;
  /** Nucleus mass in (0, 1]. 1 keeps the whole distribution. */
  topP: number;
  /** Logits of tokens already in the history are divided by this. 1 disables it. */
  repetitionPenalty: number;
}

export const GREEDY: SamplingParams = { temperature: 0, topP: 1, repetitionPenalty: 1 };

export function sampleDims(
  vocab: number,
  historyLength: number,
  params: SamplingParams,
  random: number,
): ArrayBuffer {
  const buffer = new ArrayBuffer(32);
  new Uint32Array(buffer, 0, 2).set([vocab, historyLength]);
  new Float32Array(buffer, 8, 4).set([
    params.temperature,
    params.topP,
    params.repetitionPenalty,
    random,
  ]);
  return buffer;
}

export const SAMPLE_WORKGROUPS: [number, number, number] = [1, 1, 1];
