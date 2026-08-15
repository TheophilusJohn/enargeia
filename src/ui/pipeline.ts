/**
 * The pinned pipeline section: one decode step, under the reader's control.
 *
 * Scroll position is the playhead. Scrolling forward walks the dispatch sequence a real decode
 * step issues, in the order `src/model/graph_decode.ts` builds it — this is not an illustration
 * of a transformer, it is the list of compute passes this engine encodes to produce one token.
 *
 * Under `prefers-reduced-motion` the scrub is disabled and every stage is shown at once as a
 * plain list, which is the same information without the scroll dependency.
 */

import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { el } from './format.ts';

gsap.registerPlugin(ScrollTrigger);

interface Stage {
  name: string;
  colour: string;
  /** Dispatches per token, across all 24 layers. */
  dispatches: number;
  what: string;
}

/**
 * The decode graph, grouped. Counts are per token: a stage inside the layer loop runs 24 times,
 * one that runs once per token counts once. The KV pack stages exist only when the cache is
 * f16, which is the shipping configuration wherever `shader-f16` is present.
 */
export const STAGES: Stage[] = [
  {
    name: 'embedding',
    colour: '#FF4757',
    dispatches: 1,
    what:
      'One row of a 151,936 × 896 table, dequantized from int8 as it is read. The table is ' +
      '130 MiB — the largest single tensor in the model, and the first thing to exceed a ' +
      'device binding limit.',
  },
  {
    name: 'rmsnorm',
    colour: '#FF8B27',
    dispatches: 24 * 2 + 1,
    what:
      'Root-mean-square normalisation, twice per layer plus once at the end. Mean of squares, ' +
      'not sum — a detail worth two hours of parity debugging to anyone who assumes otherwise.',
  },
  {
    name: 'projection',
    colour: '#FFD52E',
    dispatches: 24 * 7 + 1,
    what:
      'Q, K, V, the attention output, and the three MLP matrices, plus the tied LM head. Every ' +
      'one is int4: four bits per weight, unpacked in registers inside the accumulation loop ' +
      'and never written out as floats.',
  },
  {
    name: 'rope',
    colour: '#3DDC6B',
    dispatches: 24 * 2,
    what:
      'Rotary position embedding, applied to Q and K. K is cached after rotation, so a cached ' +
      'key never has to be re-rotated when the context grows.',
  },
  {
    name: 'attention',
    colour: '#1FD4E8',
    dispatches: 24 * 3,
    what:
      'Scores against every cached key, softmax, then a weighted sum of values. 14 query heads ' +
      'share 2 key/value heads. A workgroup splits the history and reduces in shared memory ' +
      'rather than walking it one position per thread — the change that made decode flat in ' +
      'context.',
  },
  {
    name: 'mlp',
    colour: '#5B8CFF',
    dispatches: 24 * 3,
    what:
      'SwiGLU: the gate through a sigmoid-weighted linear unit, multiplied by the up ' +
      'projection, then back down. Plus the two residual adds that make the stream a stream.',
  },
  {
    name: 'sample',
    colour: '#B45BFF',
    dispatches: 1,
    what:
      'Repetition penalty, temperature, and top-p over 151,936 logits — as a compute shader. ' +
      'Top-p by bisecting for a probability-mass threshold in 32 reductions rather than sorting ' +
      'the vocabulary. Only the chosen token id crosses back to JavaScript.',
  },
];

export const TOTAL_DISPATCHES = STAGES.reduce((n, s) => n + s.dispatches, 0);

export function mountPipeline(section: HTMLElement): void {
  // Pin-and-scrub is a desktop affordance. On a narrow screen the two columns stack, and
  // holding the viewport for seven screens of scroll on a phone is hostile rather than
  // controlled — so the same content is laid out plainly instead.
  const reduced =
    matchMedia('(prefers-reduced-motion: reduce)').matches || matchMedia('(max-width: 760px)').matches;

  const rows = STAGES.map((stage) => {
    const swatch = el('span', { class: 'stage-swatch' });
    swatch.style.background = stage.colour;
    const row = el('button', { class: 'stage-row', type: 'button' }, [
      swatch,
      el('span', { class: 'stage-name', text: stage.name }),
      el('span', { class: 'stage-count', text: `${stage.dispatches}×` }),
    ]);
    return row;
  });

  const detail = el('div', { class: 'stage-detail' });
  const title = el('h3');
  const body = el('p', { class: 'prose' });
  const count = el('p', { class: 'note' });
  detail.append(title, body, count);

  const list = el('div', { class: 'stage-list' }, rows);
  const grid = el('div', { class: 'stage-grid' }, [list, detail]);
  section.append(grid);

  const show = (index: number) => {
    const stage = STAGES[index];
    rows.forEach((row, i) => row.classList.toggle('active', i === index));
    title.textContent = stage.name;
    body.textContent = stage.what;
    const share = stage.dispatches / TOTAL_DISPATCHES;
    count.textContent =
      `${stage.dispatches} of ${TOTAL_DISPATCHES} dispatches per token — ` +
      `${(share * 100).toFixed(1)}% of the encoded work.`;
    title.style.color = stage.colour;
  };
  rows.forEach((row, i) => row.addEventListener('click', () => show(i)));
  show(0);

  if (reduced) {
    // No pin, no scrub: every stage's text laid out at once.
    detail.replaceChildren(
      ...STAGES.map((stage) =>
        el('div', {}, [
          el('h3', { text: `${stage.name} — ${stage.dispatches}×` }),
          el('p', { class: 'prose', text: stage.what }),
        ]),
      ),
    );
    return;
  }

  ScrollTrigger.create({
    trigger: section,
    start: 'top top',
    end: `+=${STAGES.length * 320}`,
    pin: true,
    scrub: true,
    onUpdate: (self) => {
      const index = Math.min(STAGES.length - 1, Math.floor(self.progress * STAGES.length));
      show(index);
    },
  });
}
