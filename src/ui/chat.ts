/**
 * The chat surface.
 *
 * Output streams token by token because that is how it is produced — there is no buffering step
 * that could make it arrive whole. The transcript is a live region so a screen reader follows
 * generation rather than being told about it once at the end.
 */

import { el } from './format.ts';
import type { ChatTurn, Engine } from './engine.ts';

const SUGGESTIONS = [
  'Explain what a GPU workgroup is.',
  'Write a haiku about matrix multiplication.',
  'Why do bridges have expansion joints?',
];

export class Chat {
  readonly root = el('section', { class: 'chat' });
  // `data-lenis-prevent` on every nested scroller: Lenis handles wheel events at the window and
  // would otherwise scroll the page instead of the container under the pointer.
  private readonly transcript = el('div', {
    class: 'transcript',
    role: 'log',
    'aria-live': 'polite',
    'aria-label': 'Conversation',
    'data-lenis-prevent': '',
  });
  private readonly input = el('textarea', {
    rows: '1',
    placeholder: 'Ask it something — everything runs on your GPU',
    'aria-label': 'Message',
    // Scrolls internally once a message passes its max height.
    'data-lenis-prevent': '',
  });
  private readonly send = el('button', { class: 'primary', type: 'button', text: 'Send' });
  private readonly stop = el('button', { type: 'button', text: 'Stop' });
  private readonly turns: ChatTurn[] = [];
  private readonly engine: Engine;
  private readonly replyCap: number;
  private busy = false;

  constructor(engine: Engine) {
    this.engine = engine;
    this.replyCap = engine.replyCap;

    const composer = el('form', { class: 'composer' }, [this.input, this.send, this.stop]);
    composer.addEventListener('submit', (event) => {
      event.preventDefault();
      void this.submit();
    });
    this.send.addEventListener('click', () => void this.submit());
    this.stop.addEventListener('click', () => engine.cancel());
    this.stop.disabled = true;

    // Enter sends; Shift+Enter is a newline. The textarea grows to fit what has been typed.
    this.input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        void this.submit();
      }
    });
    this.input.addEventListener('input', () => {
      this.input.style.height = 'auto';
      this.input.style.height = `${Math.min(this.input.scrollHeight, 132)}px`;
    });

    this.root.append(
      el('header', { class: 'panel-head' }, [
        el('span', { text: 'Qwen2.5-0.5B-Instruct · int4' }),
        el('span', { text: 'on your GPU' }),
      ]),
      this.transcript,
      composer,
    );
    this.showSuggestions();
  }

  private showSuggestions(): void {
    const list = el('div', { class: 'turn' }, [
      el('span', { class: 'who', text: 'try' }),
    ]);
    const row = el('div', { class: 'body' });
    for (const suggestion of SUGGESTIONS) {
      const button = el('button', { type: 'button', text: suggestion });
      button.style.marginRight = '8px';
      button.style.marginBottom = '8px';
      button.addEventListener('click', () => {
        this.input.value = suggestion;
        void this.submit();
      });
      row.append(button);
    }
    list.append(row);
    this.transcript.append(list);
  }

  /** A line of engine commentary in the transcript — not something the model said. */
  private note(text: string): void {
    this.transcript.append(el('div', { class: 'turn note-turn' }, [
      el('span', { class: 'who', text: 'engine' }),
      el('div', { class: 'body', text }),
    ]));
    this.transcript.scrollTop = this.transcript.scrollHeight;
  }

  private appendTurn(role: 'user' | 'assistant', text: string): HTMLElement {
    const body = el('div', { class: 'body', text });
    this.transcript.append(
      el('div', { class: `turn ${role}` }, [
        el('span', { class: 'who', text: role === 'user' ? 'you' : 'qwen2.5-0.5b' }),
        body,
      ]),
    );
    this.transcript.scrollTop = this.transcript.scrollHeight;
    return body;
  }

  private async submit(): Promise<void> {
    const text = this.input.value.trim();
    if (!text || this.busy || !this.engine.ready) return;

    this.input.value = '';
    this.input.style.height = 'auto';
    this.busy = true;
    this.send.disabled = true;
    this.stop.disabled = false;

    this.turns.push({ role: 'user', content: text });
    this.appendTurn('user', text);
    const body = this.appendTurn('assistant', '');
    const turn = body.parentElement!;
    turn.classList.add('streaming');

    let reply = '';
    await this.engine.chat(this.turns, {
      onToken: (chunk) => {
        reply += chunk;
        body.textContent = reply;
        const atBottom =
          this.transcript.scrollHeight - this.transcript.scrollTop - this.transcript.clientHeight
          < 80;
        if (atBottom) this.transcript.scrollTop = this.transcript.scrollHeight;
      },
      onTrimmed: (dropped) => {
        this.note(`Dropped the ${dropped === 1 ? 'oldest exchange' : `${dropped} oldest exchanges`} — ` +
          'the conversation no longer fits in a 2048-token context.');
      },
      onDone: (stopped, message) => {
        turn.classList.remove('streaming');
        if (stopped === 'error') {
          body.textContent = `${reply}\n\n[stopped: ${message ?? 'unknown error'}]`;
        } else if (stopped === 'cancelled') {
          body.textContent = `${reply}\n\n[stopped]`;
        } else if (stopped === 'limit') {
          // A reply cut off at the cap looked exactly like a finished one, which is how a
          // runaway reads as a complete answer that simply stops mid-sentence.
          this.note(`Cut off at ${this.replyCap} tokens — the model did not end its turn. Small models ` +
            'sometimes fail to emit a stop token; ask again, or start a new conversation.');
        } else if (reply.length === 0) {
          body.textContent = '[the model produced nothing]';
        }
        this.turns.push({ role: 'assistant', content: reply });
      },
    });

    this.busy = false;
    this.send.disabled = false;
    this.stop.disabled = true;
    this.input.focus();
  }
}
