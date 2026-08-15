/**
 * The residual stream, in three dimensions.
 *
 * Everything in the scene corresponds to something in the architecture. Twenty-four lattices
 * are the twenty-four layers. The strands falling through them are the residual stream, one per
 * sampled token position, carrying 896 numbers each. The arcs reach backwards only, because a
 * causal mask forbids a position from attending to one that comes after it — an arc pointing
 * forward would be a bug in the model, so there are none.
 *
 * This is the one licensed exception to the motion rule: the lattice drift and braid rotation
 * are ambient rather than measured, kept slow enough to sit below conscious notice so the hero
 * is alive before anyone scrolls.
 */

import {
  AdditiveBlending,
  BufferAttribute,
  BufferGeometry,
  Color,
  Group,
  Line,
  LineBasicMaterial,
  LineSegments,
  PerspectiveCamera,
  Scene,
  WebGLRenderer,
} from 'three';

const LAYERS = 24;
const STRANDS = 7;
const LAYER_GAP = 0.62;
const RADIUS = 1.5;

/** The kernel palette, in architecture order. Same hues as the timing bars. */
const HUES = ['#FF4757', '#FF8B27', '#FFD52E', '#3DDC6B', '#1FD4E8', '#5B8CFF', '#B45BFF'];

function lattice(y: number, colour: Color): LineSegments {
  // A hexagonal ring rather than a square grid: the square grid read as a generic "tech" motif
  // in every arrangement tried, and the ring is what a layer actually is — a boundary the whole
  // stream passes through at once.
  const points: number[] = [];
  const sides = 6;
  for (let i = 0; i < sides; i++) {
    const a = (i / sides) * Math.PI * 2;
    const b = ((i + 1) / sides) * Math.PI * 2;
    points.push(
      Math.cos(a) * RADIUS * 1.6, y, Math.sin(a) * RADIUS * 1.6,
      Math.cos(b) * RADIUS * 1.6, y, Math.sin(b) * RADIUS * 1.6,
    );
    points.push(0, y, 0, Math.cos(a) * RADIUS * 1.6, y, Math.sin(a) * RADIUS * 1.6);
  }
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(new Float32Array(points), 3));
  return new LineSegments(
    geometry,
    new LineBasicMaterial({
      color: colour, transparent: true, opacity: 0.14,
      blending: AdditiveBlending, depthWrite: false,
    }),
  );
}

function strand(index: number, colour: Color): Line {
  const points: number[] = [];
  const steps = LAYERS * 6;
  const phase = (index / STRANDS) * Math.PI * 2;
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const y = (t - 0.5) * LAYERS * LAYER_GAP;
    // A braid: each strand winds around the axis, and they do not intersect.
    const angle = phase + t * Math.PI * 2.4;
    const r = RADIUS * (0.55 + 0.35 * Math.sin(t * Math.PI * 3 + phase));
    points.push(Math.cos(angle) * r, y, Math.sin(angle) * r);
  }
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(new Float32Array(points), 3));
  return new Line(
    geometry,
    new LineBasicMaterial({
      color: colour, transparent: true, opacity: 0.8,
      blending: AdditiveBlending, depthWrite: false,
    }),
  );
}

/** Attention arcs: from a position down to an earlier one. Never upward. */
function arcs(colour: Color): LineSegments {
  const points: number[] = [];
  const random = seeded(0x1fd4e8);
  for (let i = 0; i < 44; i++) {
    const from = random();
    const to = from * random(); // strictly earlier — the causal mask, drawn
    const yFrom = (from - 0.5) * LAYERS * LAYER_GAP;
    const yTo = (to - 0.5) * LAYERS * LAYER_GAP;
    const aFrom = random() * Math.PI * 2;
    const aTo = random() * Math.PI * 2;
    const segments = 12;
    for (let s = 0; s < segments; s++) {
      const t0 = s / segments;
      const t1 = (s + 1) / segments;
      for (const t of [t0, t1]) {
        const y = yFrom + (yTo - yFrom) * t;
        const angle = aFrom + (aTo - aFrom) * t;
        // Bowed outward so the arc is visibly a path rather than a chord.
        const r = RADIUS * (0.6 + 1.5 * Math.sin(t * Math.PI));
        points.push(Math.cos(angle) * r, y, Math.sin(angle) * r);
      }
    }
  }
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(new Float32Array(points), 3));
  return new LineSegments(
    geometry,
    new LineBasicMaterial({
      color: colour, transparent: true, opacity: 0.22,
      blending: AdditiveBlending, depthWrite: false,
    }),
  );
}

function seeded(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

export function mountHero(canvas: HTMLCanvasElement): () => void {
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

  let renderer: WebGLRenderer;
  try {
    renderer = new WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'low-power' });
  } catch {
    canvas.replaceWith(fallbackNote());
    return () => {};
  }
  if (!renderer.getContext()) {
    canvas.replaceWith(fallbackNote());
    return () => {};
  }

  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor(0x000000, 0);

  const scene = new Scene();
  const camera = new PerspectiveCamera(42, 1, 0.1, 200);
  // Off-axis deliberately: looking straight down the stack is the generic composition.
  camera.position.set(6.4, 2.6, 11.0);
  camera.lookAt(0, 0, 0);

  const stack = new Group();
  for (let i = 0; i < LAYERS; i++) {
    const colour = new Color(HUES[i % HUES.length]);
    stack.add(lattice((i - LAYERS / 2) * LAYER_GAP, colour));
  }
  for (let i = 0; i < STRANDS; i++) {
    stack.add(strand(i, new Color(HUES[i % HUES.length])));
  }
  stack.add(arcs(new Color('#1FD4E8')));
  scene.add(stack);

  const resize = () => {
    const { clientWidth, clientHeight } = canvas;
    if (clientWidth === 0 || clientHeight === 0) return;
    renderer.setSize(clientWidth, clientHeight, false);
    camera.aspect = clientWidth / clientHeight;
    camera.updateProjectionMatrix();
  };
  resize();
  addEventListener('resize', resize);

  let visible = true;
  const observer = new IntersectionObserver(
    ([entry]) => {
      visible = entry.isIntersecting;
      if (visible && !reduced) frame = requestAnimationFrame(tick);
    },
    { threshold: 0 },
  );
  observer.observe(canvas);

  let frame = 0;
  let last = performance.now();
  const tick = (now: number) => {
    const dt = Math.min((now - last) / 1000, 0.05);
    last = now;
    // Slow enough to read as depth rather than as animation.
    stack.rotation.y += dt * 0.045;
    renderer.render(scene, camera);
    if (visible && !reduced) frame = requestAnimationFrame(tick);
  };

  if (reduced) {
    // A single frame: composed, still, and identical every time.
    stack.rotation.y = 0.4;
    renderer.render(scene, camera);
  } else {
    frame = requestAnimationFrame(tick);
  }

  return () => {
    cancelAnimationFrame(frame);
    observer.disconnect();
    removeEventListener('resize', resize);
    renderer.dispose();
  };
}

function fallbackNote(): HTMLElement {
  const note = document.createElement('p');
  note.className = 'note';
  note.textContent =
    'The 3D view needs WebGL, which this browser has not made available. Nothing else on the ' +
    'page depends on it.';
  return note;
}
