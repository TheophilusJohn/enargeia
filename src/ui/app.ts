/**
 * Mounts the demo into the page.
 *
 * The page itself is static HTML and paints before any of this runs; nothing here is on the
 * critical path to first paint, and nothing downloads until the visitor asks for it.
 */

import { Loader } from './loading.ts';
import { el } from './format.ts';
import type { Telemetry } from '../runtime/telemetry.ts';
import type { LoadState } from './engine.ts';

/**
 * What an unsupported browser sees.
 *
 * Not a blank page and not a shrug: which requirement is missing, what to do about it, and the
 * measured numbers so the visit is not wasted. Someone who cannot run the demo can still read
 * what it does.
 */
function unsupported(reason: string, detail: string): HTMLElement {
  const browser = navigator.userAgent;
  const isFirefox = /firefox/i.test(browser);
  const isSafari = /safari/i.test(browser) && !/chrome|chromium|edg/i.test(browser);

  let advice: string;
  if (isSafari) {
    advice =
      'Safari 26 and later support WebGPU. On an older version, enable it under ' +
      'Develop → Feature Flags → WebGPU, or open this page in Chrome or Edge.';
  } else if (isFirefox) {
    advice =
      'Firefox ships WebGPU on Windows from version 141 and is still rolling it out on macOS ' +
      'and Linux. Setting dom.webgpu.enabled in about:config may work; Chrome or Edge will.';
  } else {
    advice = 'Chrome or Edge 113 and later support WebGPU on desktop. This page needs it.';
  }

  return el('div', { class: 'unsupported' }, [
    el('h2', { text: 'This browser cannot run the demo' }),
    el('p', { class: 'prose', text: `${reason} ${advice}` }),
    el('p', { class: 'note', text: detail }),
    el('p', { class: 'prose' }, [
      'Everything below still applies: the weights are 334.9 MiB against 1884.6 MiB for the ' +
      'same model in fp32 — 458.0 MiB live, with the KV cache and activations — and it decodes ' +
      'at 43.1 tokens per second at a 2048-token context and prefills at 1105. Those numbers ' +
      'were measured on an M2, and the ',
      el('a', { href: 'https://github.com/TheophilusJohn/enargeia/blob/main/BENCH.md',
        text: 'measurement log' }),
      ' records how.',
    ]),
  ]);
}

export function mountApp(host: HTMLElement): void {
  // Checked here rather than after the import, so a browser that cannot run any of this never
  // downloads the engine at all.
  if (!('gpu' in navigator)) {
    host.replaceChildren(unsupported('WebGPU is not available here.', 'navigator.gpu is undefined.'));
    return;
  }

  let update: (telemetry: Telemetry) => void = () => {};
  let pending: Telemetry | null = null;

  const fail = (heading: string, message: string, detail?: string) => {
    host.replaceChildren(
      el('div', { class: 'unsupported' }, [
        el('h2', { text: heading }),
        el('p', { class: 'prose', text: message }),
        el('p', { class: 'note', text: detail ?? 'Reloading usually clears a partial cache write.' }),
      ]),
    );
  };

  const loader = new Loader(() => {
    // The engine, the kernels, the tokenizer and the panels are all behind this import. None of
    // it is on the path to first paint, and a visitor who reads the page without running the
    // demo never downloads any of it.
    void (async () => {
      const [{ Engine, UnsupportedError }, { Chat }, { Inspector }] = await Promise.all([
        import('./engine.ts'),
        import('./chat.ts'),
        import('./inspector.ts'),
      ]);

      const engine = new Engine({
        onLoad: (state: LoadState) => loader.update(state),
        onTelemetry: (telemetry) => {
          update(telemetry);
          pending = telemetry;
        },
      });

      try {
        await engine.load();
      } catch (error) {
        if (error instanceof UnsupportedError) {
          host.replaceChildren(unsupported(error.message, error.detail));
        } else {
          fail('Loading failed', error instanceof Error ? error.message : String(error));
        }
        return;
      }

      const chat = new Chat(engine);
      const inspector = new Inspector({
        profile: engine.profile!,
        embeddingParts: engine.embeddingParts,
        maxContext: engine.maxContext,
        // Attention sampling rides the profiling duty cycle, so opening the heatmap turns it on
        // and closing it turns it back off.
        onAttentionToggle: (enabled) => engine.setAttentionSampling(enabled),
      });
      update = (telemetry) => inspector.update(telemetry);
      engine.setInstrumented(true);

      // A handle for measuring the app itself in a dev build. Never present in production.
      if (import.meta.env.DEV) {
        (window as unknown as Record<string, unknown>).__enargeia = { engine, inspector };
      }

      host.replaceChildren(el('div', { class: 'app' }, [chat.root, inspector.root]));
      if (pending) inspector.update(pending);
    })();
  });

  host.replaceChildren(loader.root);
}
