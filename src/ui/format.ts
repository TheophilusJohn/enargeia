/** Number formatting for the panels. Every reading is a measurement, so none of them round away
 * a digit the reader would want. */

export function mib(bytes: number): string {
  return `${(bytes / 1048576).toFixed(1)} MiB`;
}

export function mb(bytes: number): string {
  return `${(bytes / 1e6).toFixed(0)} MB`;
}

export function ms(value: number): string {
  return value >= 100 ? `${value.toFixed(0)} ms` : `${value.toFixed(1)} ms`;
}

export function pct(fraction: number): string {
  return `${(fraction * 100).toFixed(1)}%`;
}

export function rate(bytesPerSecond: number | null): string {
  if (bytesPerSecond === null || !Number.isFinite(bytesPerSecond)) return '—';
  return `${(bytesPerSecond / 1e6).toFixed(1)} MB/s`;
}

export function duration(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds)) return '—';
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  return `${Math.floor(seconds / 60)}m ${String(Math.round(seconds % 60)).padStart(2, '0')}s`;
}

/** Build an element in one call. Not a framework; just less noise than six assignments. */
export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string> = {},
  children: Array<Node | string> = [],
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === 'class') node.className = value;
    else if (key === 'text') node.textContent = value;
    else node.setAttribute(key, value);
  }
  for (const child of children) {
    node.append(typeof child === 'string' ? document.createTextNode(child) : child);
  }
  return node;
}
