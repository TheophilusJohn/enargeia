/**
 * First load, which is the highest-abandonment moment on the site.
 *
 * Two bars, not one. A download of 335 MB and a compilation of 17 pipelines are different
 * waits — one is the network and takes a minute on a slow connection, the other is the CPU and
 * takes ten milliseconds — and a single merged bar that sits at 99% while the second happens
 * reads as a hang. Both report the numbers behind them, so a slow bar can be told from a stuck
 * one without waiting to find out.
 */

import { el, mb, duration, rate } from './format.ts';
import type { LoadState } from './engine.ts';

const MODEL_BYTES = 351_187_968;

function phaseRow(label: string): {
  root: HTMLElement;
  detail: HTMLElement;
  fill: HTMLElement;
  set(fraction: number, detail: string): void;
} {
  const detail = el('b', { text: '—' });
  const fill = el('i');
  const root = el('div', { class: 'phase' }, [
    el('div', { class: 'top' }, [el('span', { text: label }), detail]),
    el('div', {
      class: 'bar',
      role: 'progressbar',
      'aria-label': label,
      'aria-valuemin': '0',
      'aria-valuemax': '100',
      'aria-valuenow': '0',
    }, [fill]),
  ]);
  return {
    root,
    detail,
    fill,
    set(fraction, text) {
      const percent = Math.max(0, Math.min(1, fraction)) * 100;
      fill.style.width = `${percent.toFixed(1)}%`;
      root.querySelector('.bar')!.setAttribute('aria-valuenow', percent.toFixed(0));
      detail.textContent = text;
    },
  };
}

export class Loader {
  readonly root = el('div', { class: 'loader' });
  private readonly headline = el('h2', { text: 'Run a language model on your own GPU' });
  private readonly status = el('p', { class: 'prose' });
  private readonly download = phaseRow('Download — weights');
  private readonly compile = phaseRow('Compile — compute pipelines');
  private readonly start = el('button', { class: 'primary', type: 'button' });
  private readonly detail = el('p', { class: 'note' });

  constructor(onStart: () => void) {
    this.start.textContent = `Load the model — ${mb(MODEL_BYTES)}`;
    this.start.addEventListener('click', () => {
      this.start.disabled = true;
      this.start.textContent = 'Loading…';
      onStart();
    });

    this.status.innerHTML =
      'Nothing has been downloaded yet. Loading fetches <strong>335 MiB</strong> of int4 weights ' +
      'to your graphics card, where they stay — no text you type leaves this tab, because there ' +
      'is no server to send it to.';
    this.detail.textContent =
      'The download is cached, so a second visit skips it. On a 50 Mbit/s connection the first ' +
      'load takes about a minute.';

    this.root.append(
      this.headline,
      this.status,
      this.start,
      el('div', { class: 'phases' }, [this.download.root, this.compile.root]),
      this.detail,
    );
  }

  update(state: LoadState): void {
    switch (state.phase) {
      case 'device':
      case 'tokenizer':
        this.detail.textContent = state.message;
        break;
      case 'download': {
        const total = state.total ?? MODEL_BYTES;
        this.download.set(
          state.loaded / total,
          `${mb(state.loaded)} / ${mb(total)}`,
        );
        this.detail.textContent = state.warm
          ? 'Served from the Cache API — no network transfer.'
          : `${rate(state.bytesPerSecond)} · ${duration(state.etaSeconds)} remaining`;
        break;
      }
      case 'compile': {
        this.download.set(1, `${mb(state.total ?? MODEL_BYTES)} · done`);
        this.compile.set(
          state.pipelines > 0 ? state.compiled / state.pipelines : 0,
          `${state.compiled} / ${state.pipelines}`,
        );
        this.detail.textContent =
          'Compiling now rather than on the first dispatch, so the wait does not land on the ' +
          'first word of the first answer.';
        break;
      }
      case 'error':
        this.status.textContent = state.error ?? 'Loading failed.';
        this.start.disabled = false;
        this.start.textContent = 'Try again';
        break;
      default:
        break;
    }
  }

  /** Replaced wholesale once the session exists; kept as a method so app.ts stays declarative. */
  remove(): void {
    this.root.remove();
  }
}
