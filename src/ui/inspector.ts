/**
 * The inspector.
 *
 * Every panel here renders a published telemetry snapshot and nothing else. Nothing moves that
 * is not a measurement: the bars are kernel shares, the fill is cache occupancy, the heatmap is
 * attention weights read off the GPU. There is no idle state in which anything animates.
 *
 * Updates are throttled by the runtime to 30 Hz. This file does not schedule its own frames.
 */

import type { Telemetry } from '../runtime/telemetry.ts';
import type { DeviceProfile } from '../gpu/device.ts';
import { el, mib, ms, pct } from './format.ts';

function gauge(title: string, body: HTMLElement): HTMLElement {
  return el('section', { class: 'gauge' }, [
    el('h3', { text: title }),
    body,
  ]);
}

/** Tokens per second, plus the inter-token time it is the reciprocal of. */
class ThroughputPanel {
  readonly root: HTMLElement;
  private readonly value = el('span', { class: 'big', text: '—' });
  private readonly detail = el('dl', { class: 'kv' });

  constructor() {
    this.root = gauge('Throughput', el('div', {}, [
      el('div', { class: 'readout' }, [this.value, el('span', { class: 'unit', text: 'tok/s' })]),
      this.detail,
    ]));
  }

  update(t: Telemetry): void {
    this.value.textContent = t.tokensPerSecond > 0 ? t.tokensPerSecond.toFixed(1) : '—';
    this.detail.replaceChildren(
      el('dt', { text: 'inter-token' }),
      el('dd', { text: t.interTokenMs > 0 ? ms(t.interTokenMs) : '—' }),
      el('dt', { text: 'tokens generated' }),
      el('dd', { text: String(t.tokensGenerated) }),
      el('dt', { text: 'phase' }),
      el('dd', { class: t.phase === 'idle' ? '' : 'ok', text: t.phase }),
    );
  }
}

/** KV cache occupancy. The bar is the fraction of the preallocated context in use. */
class CachePanel {
  readonly root: HTMLElement;
  private readonly fill = el('i');
  private readonly text = el('dl', { class: 'kv' });

  constructor() {
    const bar = el('div', { class: 'bar' }, [this.fill]);
    this.root = gauge('KV cache', el('div', {}, [this.text, bar]));
  }

  update(t: Telemetry): void {
    const fraction = t.contextMax > 0 ? t.contextUsed / t.contextMax : 0;
    this.fill.style.width = `${(fraction * 100).toFixed(2)}%`;
    this.text.replaceChildren(
      el('dt', { text: 'positions' }),
      el('dd', { text: `${t.contextUsed} / ${t.contextMax}` }),
      el('dt', { text: 'occupancy' }),
      el('dd', { text: pct(fraction) }),
      el('dt', { text: 'allocated' }),
      el('dd', { text: mib(t.memory.kvCacheBytes) }),
    );
  }
}

/**
 * Per-kernel GPU time, grouped by operation.
 *
 * Only populated when timestamp-query is available and instrumentation is on. When it is not,
 * the panel says which of the two is missing rather than showing an empty chart.
 */
class KernelPanel {
  readonly root: HTMLElement;
  private readonly rows = el('div', { class: 'kernel-rows' });
  private readonly note = el('p', { class: 'note' });

  private readonly hasTimestamps: boolean;

  constructor(hasTimestamps: boolean) {
    this.hasTimestamps = hasTimestamps;
    this.root = gauge('GPU time per kernel', el('div', {}, [this.rows, this.note]));
  }

  update(t: Telemetry): void {
    if (!this.hasTimestamps) {
      this.rows.replaceChildren();
      this.note.textContent =
        'This device does not expose timestamp-query, so per-kernel GPU time cannot be read. ' +
        'Everything else on this panel is wall-clock and unaffected.';
      return;
    }
    if (t.kernels.length === 0) {
      this.rows.replaceChildren();
      this.note.textContent = t.instrumented
        ? 'Waiting for the first profiled step.'
        : 'Instrumentation off.';
      return;
    }

    this.rows.replaceChildren(
      ...t.kernels.map((kernel) =>
        el('div', { class: 'kernel-row' }, [
          el('span', { class: 'name', text: kernel.group }),
          el('span', { class: 'track' }, [
            (() => {
              const bar = el('i');
              bar.style.width = `${(kernel.share * 100).toFixed(2)}%`;
              bar.style.background = kernel.colour;
              return bar;
            })(),
          ]),
          // The percentage is text as well as width, so the chart survives being read aloud.
          el('span', { class: 'pct', text: pct(kernel.share) }),
        ]),
      ),
    );
    const total = t.gpuMsPerToken;
    this.note.textContent =
      `${ms(total ?? 0)} of GPU time per token across ${t.kernels.reduce((n, k) => n + k.dispatches, 0)} ` +
      `dispatches. Profiling adds a compute pass per dispatch and runs on one step in 16.`;
  }
}

/** Where the memory went. Weights, cache, and scratch, summed. */
class MemoryPanel {
  readonly root: HTMLElement;
  private readonly list = el('dl', { class: 'kv' });
  private readonly note = el('p', { class: 'note' });

  constructor() {
    this.note.textContent =
      'Scratch is dominated by the prefill graph, which is allocated once for the largest ' +
      'prompt it must accept: two 14 × 2048 × 2048 attention buffers are 470 MiB of it. ' +
      'Decoding needs almost none of that — it is the price of never allocating mid-generation.';
    this.root = gauge('Memory', el('div', {}, [this.list, this.note]));
  }

  update(t: Telemetry): void {
    const m = t.memory;
    this.list.replaceChildren(
      el('dt', { text: 'weights (int4, int8 embed)' }),
      el('dd', { text: mib(m.weightsBytes) }),
      el('dt', { text: 'KV cache' }),
      el('dd', { text: mib(m.kvCacheBytes) }),
      el('dt', { text: 'activation scratch' }),
      el('dd', { text: mib(m.scratchBytes) }),
      el('dt', { text: 'resident total' }),
      el('dd', { text: mib(m.totalBytes) }),
      el('dt', { text: 'weights in fp32, for scale' }),
      el('dd', { text: '1884.6 MiB' }),
    );
  }
}

/**
 * Attention weights for the last layer, heads down, positions across.
 *
 * This costs an extra readback, which the one-readback-per-token decode budget does not
 * include. It is therefore only sampled while this panel is open, and only on the profiling
 * duty cycle — one step in 16. The panel says so.
 */
class HeatmapPanel {
  readonly root: HTMLElement;
  private readonly canvas = el('canvas', { class: 'heat', 'aria-hidden': 'true' });
  private readonly note = el('p', { class: 'note' });
  private readonly summary = el('p', { class: 'note' });
  private readonly toggle = el('button', { type: 'button', text: 'Sample attention' });
  private enabled = false;
  private readonly onToggle: (enabled: boolean) => void;

  constructor(onToggle: (enabled: boolean) => void) {
    this.onToggle = onToggle;
    this.toggle.addEventListener('click', () => {
      this.enabled = !this.enabled;
      this.toggle.textContent = this.enabled ? 'Stop sampling' : 'Sample attention';
      this.onToggle(this.enabled);
    });
    this.root = gauge('Attention', el('div', {}, [
      this.canvas, this.summary, this.note, this.toggle,
    ]));
    this.note.textContent =
      'Off by default: reading attention weights back costs a second GPU→CPU round trip per ' +
      'sampled step, which the decode budget of one readback per token does not include.';
  }

  update(t: Telemetry): void {
    const sample = t.attention;
    if (!sample || !this.enabled) return;

    const { heads, positions, weights } = sample;
    const ctx = this.canvas.getContext('2d');
    if (!ctx) return;
    this.canvas.width = positions;
    this.canvas.height = heads;
    const image = ctx.createImageData(positions, heads);

    // Normalised per head against that head's own maximum: heads differ by an order of
    // magnitude in how peaked they are, and a global scale renders the flat ones as black.
    for (let h = 0; h < heads; h++) {
      let peak = 0;
      for (let p = 0; p < positions; p++) peak = Math.max(peak, weights[h * positions + p]);
      const scale = peak > 0 ? 1 / peak : 0;
      for (let p = 0; p < positions; p++) {
        const v = Math.sqrt(weights[h * positions + p] * scale);
        const i = (h * positions + p) * 4;
        // Ramped through --k4, the attention hue, because that is what this is.
        image.data[i] = Math.round(31 * v + 8 * (1 - v));
        image.data[i + 1] = Math.round(212 * v + 8 * (1 - v));
        image.data[i + 2] = Math.round(232 * v + 16 * (1 - v));
        image.data[i + 3] = 255;
      }
    }
    ctx.putImageData(image, 0, 0);

    // The text equivalent: where each head is actually looking.
    let peakHead = 0;
    let peakPos = 0;
    let peakValue = 0;
    for (let h = 0; h < heads; h++) {
      for (let p = 0; p < positions; p++) {
        if (weights[h * positions + p] > peakValue) {
          peakValue = weights[h * positions + p];
          peakHead = h;
          peakPos = p;
        }
      }
    }
    this.summary.textContent =
      `Layer ${sample.layer}, ${heads} heads over the ${positions} most recent positions. ` +
      `Strongest weight ${peakValue.toFixed(3)} at head ${peakHead}, position ${peakPos}.`;
  }
}

/** What was detected, and which path the engine took as a result. */
export function devicePanel(
  profile: DeviceProfile,
  embeddingParts: number,
  maxContext: number,
): HTMLElement {
  const list = el('dl', { class: 'kv' });
  const row = (label: string, value: string, tone: '' | 'ok' | 'warn' = '') => {
    list.append(el('dt', { text: label }), el('dd', { class: tone, text: value }));
  };

  row('adapter', profile.description || profile.device || 'unnamed');
  row('tier', profile.tier);
  row('shader-f16', profile.f16 ? 'present' : 'absent', profile.f16 ? 'ok' : 'warn');
  row('timestamp-query', profile.timestampQuery ? 'present' : 'absent',
    profile.timestampQuery ? 'ok' : 'warn');
  row('subgroups', profile.subgroups ? 'present' : 'absent');
  row('max storage binding', mib(profile.maxStorageBufferBindingSize));
  row('storage buffers / stage', String(profile.maxStorageBuffersPerShaderStage));
  row('max workgroups / dim', profile.maxComputeWorkgroupsPerDimension.toLocaleString());
  row('context', `${maxContext} tokens`, maxContext < 2048 ? 'warn' : '');
  row('embedding bindings', embeddingParts === 1 ? '1 (table fits whole)' : `${embeddingParts} (split)`,
    embeddingParts === 1 ? 'ok' : 'warn');

  const fallbacks: string[] = [];
  fallbacks.push(
    profile.f16
      ? 'KV cache stored as f16 — 24 MiB at a 2048 context instead of 48.'
      : 'No shader-f16, so the KV cache runs in fp32: 48 MiB at a 2048 context, same speed. ' +
        'The f16 cache is a memory optimization, not a speed one — measured +3.4% before the ' +
        'attention kernel stopped being iteration-bound, and nothing after.',
  );
  fallbacks.push(
    profile.timestampQuery
      ? 'GPU timing read from timestamp-query.'
      : 'No timestamp-query, so per-kernel GPU time is unavailable. Wall-clock throughput is not affected.',
  );
  fallbacks.push(
    embeddingParts === 1
      ? 'The 151,936 × 896 embedding table fits in one binding at int8 — 130 MiB against this ' +
        `device's ${mib(profile.maxStorageBufferBindingSize)} limit.`
      : `The embedding table exceeds this device's binding limit and is split across ` +
        `${embeddingParts} buffers, with the gather and the tied LM head reading whichever part ` +
        'holds the row.',
  );
  if (maxContext < 2048) {
    fallbacks.push(
      `Mobile-class adapter, so the context is ${maxContext} rather than 2048. Prefill ` +
      'activations grow with the square of the context — the two attention buffers are ' +
      '470 MiB at 2048 and 118 at 1024 — and that is the difference between running and ' +
      'failing to allocate.',
    );
  }
  if (profile.storageBindingClamped) {
    fallbacks.push(
      `Binding size is clamped to 128 MiB for testing; the adapter would allow ` +
      `${mib(profile.adapterStorageBufferBindingSize)}.`,
    );
  }

  return gauge('This device', el('div', {}, [
    list,
    ...fallbacks.map((text) => el('p', { class: 'note', text })),
  ]));
}

export class Inspector {
  readonly root = el('aside', { class: 'inspector', 'aria-label': 'Live inspector' });
  private readonly panels: Array<{ update(t: Telemetry): void }>;

  constructor(options: {
    profile: DeviceProfile;
    embeddingParts: number;
    maxContext: number;
    onAttentionToggle: (enabled: boolean) => void;
  }) {
    const throughput = new ThroughputPanel();
    const cache = new CachePanel();
    const kernels = new KernelPanel(options.profile.timestampQuery);
    const memory = new MemoryPanel();
    const heatmap = new HeatmapPanel(options.onAttentionToggle);
    this.panels = [throughput, cache, kernels, memory, heatmap];

    this.root.append(
      el('header', { class: 'panel-head' }, [
        el('span', { text: 'Inspector' }),
        el('span', { text: 'live' }),
      ]),
      throughput.root,
      cache.root,
      kernels.root,
      memory.root,
      heatmap.root,
      devicePanel(options.profile, options.embeddingParts, options.maxContext),
    );
  }

  update(telemetry: Telemetry): void {
    for (const panel of this.panels) panel.update(telemetry);
  }
}
