/**
 * The pinned pipeline section: one decode step, under the reader's control.
 *
 * Scroll position is the playhead. Scrolling forward walks the dispatch sequence a real decode
 * step issues, in the order `src/model/graph_decode.ts` builds it — this is not an illustration
 * of a transformer, it is the list of compute passes this engine encodes to produce one token.
 *
 * Under `prefers-reduced-motion`, and at narrow widths, the scrub is disabled and every stage is
 * shown at once as a plain list — the same information without the scroll dependency. Both of
 * those are separate code paths and both are checked by `npm run sweep`.
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

/** Height of the sticky masthead, which everything that pins or anchors has to clear. */
function mastheadHeight(): number {
  return document.querySelector('.masthead')?.getBoundingClientRect().height ?? 53;
}

export function mountPipeline(container: HTMLElement): void {
  // Pin the *section*, not this container.
  //
  // Pinning the inner div was the bug: the section's heading scrolled away while the stage grid
  // stuck to the top of the viewport underneath a sticky masthead that covered its first line,
  // and because the grid is only ~280px tall, the rest of the pinned viewport was empty. It
  // looked correct in exactly one place — the first frame of the pin — which is the frame every
  // screenshot had been taken at.
  const section = container.closest('section') ?? container;

  // Pin-and-scrub is a desktop affordance. On a narrow screen the two columns stack, and
  // holding the viewport for seven screens of scroll on a phone is hostile rather than
  // controlled — so the same content is laid out plainly instead.
  const reduced =
    matchMedia('(prefers-reduced-motion: reduce)').matches || matchMedia('(max-width: 760px)').matches;

  // The list selects which stage the panel describes, which is a tablist. Marking it as one
  // gives a screen reader the relationship between the two halves, and arrow keys move between
  // stages the way they do in every other tablist — neither of which a screenshot review would
  // ever have caught.
  const rows = STAGES.map((stage, index) => {
    const swatch = el('span', { class: 'stage-swatch' });
    swatch.style.background = stage.colour;
    const row = el('button', {
      class: 'stage-row',
      type: 'button',
      role: 'tab',
      id: `stage-tab-${index}`,
      'aria-controls': 'stage-detail',
      'aria-selected': 'false',
    }, [
      swatch,
      el('span', { class: 'stage-name', text: stage.name }),
      el('span', { class: 'stage-count', text: `${stage.dispatches}×` }),
    ]);
    return row;
  });

  const detail = el('div', {
    class: 'stage-detail',
    id: 'stage-detail',
    role: 'tabpanel',
    'aria-live': 'polite',
  });
  const title = el('h3');
  const body = el('p', { class: 'prose' });
  const count = el('p', { class: 'note' });
  detail.append(title, body, count);

  const list = el('div', {
    class: 'stage-list',
    role: 'tablist',
    'aria-label': 'Stages of one decode step',
  }, rows);
  const grid = el('div', { class: 'stage-grid' }, [list, detail]);
  container.append(grid);

  const show = (index: number, focus = false) => {
    const stage = STAGES[index];
    rows.forEach((row, i) => {
      row.classList.toggle('active', i === index);
      row.setAttribute('aria-selected', String(i === index));
      // Roving tabindex: one stop for the whole list, arrows move within it.
      row.tabIndex = i === index ? 0 : -1;
    });
    detail.setAttribute('aria-labelledby', `stage-tab-${index}`);
    if (focus) rows[index].focus();
    title.textContent = stage.name;
    body.textContent = stage.what;
    const share = stage.dispatches / TOTAL_DISPATCHES;
    count.textContent =
      `${stage.dispatches} of ${TOTAL_DISPATCHES} dispatches per token — ` +
      `${(share * 100).toFixed(1)}% of the encoded work.`;
    title.style.color = stage.colour;
  };
  rows.forEach((row, i) => row.addEventListener('click', () => show(i)));
  list.addEventListener('keydown', (event) => {
    const current = rows.findIndex((row) => row.classList.contains('active'));
    const step =
      event.key === 'ArrowDown' || event.key === 'ArrowRight' ? 1 :
      event.key === 'ArrowUp' || event.key === 'ArrowLeft' ? -1 :
      event.key === 'Home' ? -current :
      event.key === 'End' ? STAGES.length - 1 - current : 0;
    if (step === 0) return;
    event.preventDefault();
    show((current + step + STAGES.length) % STAGES.length, true);
  });
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

  // The pinned section fills the viewport below the masthead and centres its content, so the
  // playhead sits in a composed frame rather than at the top of a mostly-empty screen.
  section.classList.add('is-pinned');
  document.documentElement.style.setProperty('--masthead-h', `${mastheadHeight()}px`);

  ScrollTrigger.create({
    trigger: section,
    // Start where the section's top meets the *bottom of the masthead*, not the top of the
    // viewport. `start: 'top top'` slides the first line of content under a sticky header.
    start: () => `top ${mastheadHeight()}px`,
    end: () => `+=${STAGES.length * 300}`,
    pin: section,
    pinSpacing: true,
    // Both recomputed on resize: the masthead wraps to two lines on a narrow window, and a
    // start offset measured once at load would be wrong from then on.
    invalidateOnRefresh: true,
    anticipatePin: 1,
    onUpdate: (self) => {
      const index = Math.min(STAGES.length - 1, Math.floor(self.progress * STAGES.length));
      show(index);
    },
  });

  // Fonts change the section's height, and a stale measurement leaves the pin ending in the
  // wrong place.
  void document.fonts?.ready.then(() => ScrollTrigger.refresh());
  addEventListener('resize', () => {
    document.documentElement.style.setProperty('--masthead-h', `${mastheadHeight()}px`);
  });
}
