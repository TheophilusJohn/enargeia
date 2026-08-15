/**
 * Entry point.
 *
 * The page is static HTML and has already painted by the time this runs. Nothing here blocks
 * first paint, and nothing downloads weights until the visitor asks for them.
 */

import './style.css';
import { mountApp } from './ui/app.ts';

const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

const app = document.getElementById('app');
if (app) mountApp(app);

// Three.js is the largest thing on the page after the model, and the hero is decorative in the
// strict sense that nothing depends on it. It arrives after first paint.
const heroCanvas = document.getElementById('hero-canvas');
if (heroCanvas instanceof HTMLCanvasElement && !reduced) {
  void import('./ui/hero.ts').then(({ mountHero }) => mountHero(heroCanvas));
} else if (heroCanvas instanceof HTMLCanvasElement) {
  // Under reduced motion the scene still renders — one static frame — but it can wait for idle.
  requestIdleCallback(() => void import('./ui/hero.ts').then(({ mountHero }) => mountHero(heroCanvas)));
}

const pipeline = document.getElementById('pipeline-stages');
if (pipeline) void import('./ui/pipeline.ts').then(({ mountPipeline }) => mountPipeline(pipeline));

// Lenis smooths the scrub-driven section into something deliberate rather than twitchy. It is
// skipped entirely under reduced motion, where hijacking scroll is exactly the wrong thing.
if (!reduced) {
  void import('lenis').then(async ({ default: Lenis }) => {
    const { default: gsap } = await import('gsap');
    const { ScrollTrigger } = await import('gsap/ScrollTrigger');
    const lenis = new Lenis({ duration: 0.9, smoothWheel: true });
    lenis.on('scroll', ScrollTrigger.update);
    gsap.ticker.add((time) => lenis.raf(time * 1000));
    gsap.ticker.lagSmoothing(0);
  });
}
