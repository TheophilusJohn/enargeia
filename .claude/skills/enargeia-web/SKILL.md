---
name: enargeia-web
description: Design system, motion rules, copy voice, and performance budget for the Enargeia website and inspector UI — colour tokens, typography, scroll-driven animation, the 3D residual-stream hero, and accessibility floor. Use this for any work on the landing page, marketing copy, inspector panels, charts, chat interface, or Three.js scene, whenever adding animation or scroll effects, whenever writing user-facing text, and whenever choosing colours or type. Also use when the task mentions making the site look better, less generic, or less AI-generated.
---

# The site

The site is the product. The engine could be perfect and unread; what the visitor takes
away is fifteen seconds of watching a language model run on their own graphics card.

Design accordingly, and hold one line: **the site is allowed libraries, the engine is not.**
Lenis, GSAP, and Three.js live under `src/ui` and never import from `src/gpu`,
`src/kernels`, `src/model`, or `src/runtime`.

## The motion rule

**If it moves, it is measuring something.**

Every animated element on the page displays real state: layers firing during a forward
pass, kernel timing percentages, attention weights, KV occupancy, tokens per second, load
progress. Nothing animates for atmosphere.

This is not asceticism. The project's credibility rests on being a serious performance
artifact, and decorative motion signals the opposite to precisely the audience that
matters. It also produces a better page — there is more real data here than most sites
have, and using it is more interesting than inventing effects.

The one licensed exception is ambient drift in the 3D scene: slow lattice movement and
braid rotation so the hero is alive before anyone scrolls. Keep it below the threshold of
conscious notice.

## Colour

Hue encodes kernel identity. The same colour means the same operation in the hero, the
pipeline diagram, the timing bars, and the attention heatmap. A spectrum used as an
encoding reads as designed; the same spectrum used decoratively reads as generated.

```
--k0 #FF4757   embedding        --bg      #05050A
--k1 #FF8B27   rmsnorm          --surface #0E0E16
--k2 #FFD52E   qkv projection   --line    #1C1C29
--k3 #3DDC6B   rope             --ink     #F5F5FA
--k4 #1FD4E8   attention        --dim     #9A9AB4
--k5 #5B8CFF   mlp              --muted   #63637C
--k6 #B45BFF   sample
```

Never introduce a colour outside these tokens. Never assign a kernel colour to something
that is not that kernel.

## Type

Space Grotesk for display, JetBrains Mono for data. Display is tight — letter-spacing
around `-.03em` at large sizes, line-height under 1.05 for headlines. Mono carries every
number, label, and identifier.

Do not reach for Inter. It is the default that makes a page look like every other page.

## The two-register rule

Every claim is stated twice: once in plain language at display size, once as hard numbers
in mono beside it, smaller and dimmer.

```
It reads all 494 million numbers to write one word.
1.98 GB/token · 24 layers · 170 dispatches
```

A visitor with no background gets a complete story from the display type alone. An
engineer's eye catches on the mono and finds it checks out. Neither has to read the other
register.

This is also the strongest available signal that a human wrote the page. Generated copy
picks one voice and stays in it, and it avoids specific numbers because specific numbers
can be wrong in public.

## Copy voice

Plain sentences, concrete numbers, active voice. Concede the real limitations — the FAQ
says outright that the model is far worse than a frontier model and that WebLLM is the
better choice for production use. Conceding is what makes the rest believable, and no
marketing generator ever does it.

Avoid: "revolutionary", "seamless", "powerful", "unlock", "cutting-edge", "leverage",
"blazing fast". Avoid three-item feature triplets. Avoid any sentence that would survive
unchanged on a different product's page.

## Scroll and 3D

Scroll position drives timelines rather than triggering them. The pinned pipeline section
maps scroll progress to which kernel is active — the reader controls the playhead. Use
GSAP ScrollTrigger for pinning and scrub; it handles resize, refresh, and back-navigation
correctly, which hand-rolled scroll math does not.

Lenis for scroll smoothing. Around 3 KB, and it is what makes scrub-driven sections feel
deliberate rather than twitchy.

The 3D hero is the residual stream: a braid of strands falling through 24 layer lattices,
with attention arcs reaching backward to earlier positions because a causal mask forbids
looking forward. Every element corresponds to something real in the architecture. Keep the
camera off-axis — a dead-centre view down the stack is the generic composition.

Constraints for the scene: pixel ratio capped at 2, additive blending with `depthWrite`
off, pause rendering when off-screen via IntersectionObserver, and a static fallback under
`prefers-reduced-motion`. If WebGL is unavailable, show a short explanatory message rather
than a blank canvas.

## Performance budget

The model is 384 MB, so JavaScript weight is not the constraint it usually is — GSAP,
Lenis, and Three together are about 0.05% of the download. Spend it.

What actually matters:

- Inspector panels update at most 30 times a second, regardless of decode rate. Faster is
  invisible and steals GPU time from the thing being measured.
- Panels read published snapshots, never GPU buffers directly.
- The 3D scene stops rendering when scrolled past.
- First paint does not wait on the model. The page is interactive and explains itself
  before a single byte of weights arrives.

## Accessibility floor

Not negotiable, and not announced on the page:

- Visible keyboard focus everywhere — `:focus-visible` with a `--k4` outline
- `prefers-reduced-motion` respected: no autoplay, no scroll-scrub, static hero
- Chat output in a live region so screen readers follow generation
- Every animated visual has a text equivalent; the timing bars carry their percentages as
  text, not only as bar width
- Contrast at least 4.5:1 for body copy. `--muted` on `--bg` fails this — it is for
  decorative labels only, never for content a visitor needs.
