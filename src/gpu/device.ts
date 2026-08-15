/**
 * Device acquisition and capability detection.
 *
 * Probe once at startup, store the answers in a DeviceProfile, branch on the profile
 * everywhere else. Nothing outside this file calls `adapter.features.has` or reads
 * `device.limits` — a capability check at a call site is a capability check that will
 * eventually disagree with another one.
 */

/** WebGPU's spec default for `maxStorageBufferBindingSize`. The dev clamp targets this. */
export const DEFAULT_STORAGE_BINDING_SIZE = 128 * 1024 * 1024;

export type DeviceTier = 'discrete' | 'integrated' | 'mobile' | 'unknown';

export interface DeviceProfile {
  /** `shader-f16` is enabled on the device. Roughly a third of devices lack it. */
  f16: boolean;
  /** `timestamp-query` is enabled. Without it, timing falls back to wall clock. */
  timestampQuery: boolean;
  /** `subgroups` is enabled. */
  subgroups: boolean;

  maxBufferSize: number;
  maxStorageBufferBindingSize: number;
  maxUniformBufferBindingSize: number;
  maxComputeWorkgroupStorageSize: number;
  maxComputeInvocationsPerWorkgroup: number;
  maxComputeWorkgroupsPerDimension: number;
  maxStorageBuffersPerShaderStage: number;

  tier: DeviceTier;
  vendor: string;
  architecture: string;
  device: string;
  description: string;

  /** True when the dev clamp forced `maxStorageBufferBindingSize` down to the default. */
  storageBindingClamped: boolean;
  /** What the hardware would have allowed, whether or not the clamp engaged. */
  adapterStorageBufferBindingSize: number;
}

export interface GPUContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  queue: GPUQueue;
  profile: DeviceProfile;
}

export interface InitOptions {
  powerPreference?: GPUPowerPreference;
  /**
   * Clamp `maxStorageBufferBindingSize` to 128 MiB so the split-binding path for the
   * embedding table is exercised on hardware that would otherwise hide it. Leave
   * undefined to resolve from `VITE_GPU_CLAMP_STORAGE` or a `?clampStorage` query param.
   */
  clampStorageBindingSize?: boolean;
  /** Request the `subgroups` feature when the adapter offers it. Off by default. */
  requestSubgroups?: boolean;
  onDeviceLost?: (info: GPUDeviceLostInfo) => void;
  onUncapturedError?: (error: GPUError) => void;
  label?: string;
}

export class GPUUnavailableError extends Error {
  readonly reason: 'no-navigator-gpu' | 'no-adapter';
  constructor(reason: 'no-navigator-gpu' | 'no-adapter', message: string) {
    super(message);
    this.name = 'GPUUnavailableError';
    this.reason = reason;
  }
}

/**
 * Limits worth asking for. WebGPU hands out conservative defaults unless you ask —
 * `maxStorageBufferBindingSize` defaults to 128 MiB on hardware that allows 2 GiB, and
 * silently accepting that fails on the embedding table rather than at init.
 */
const REQUESTED_LIMITS = [
  'maxBufferSize',
  'maxStorageBufferBindingSize',
  'maxUniformBufferBindingSize',
  'maxComputeWorkgroupStorageSize',
  'maxComputeInvocationsPerWorkgroup',
  'maxComputeWorkgroupSizeX',
  'maxComputeWorkgroupSizeY',
  'maxComputeWorkgroupSizeZ',
  'maxComputeWorkgroupsPerDimension',
  'maxStorageBuffersPerShaderStage',
  'maxBindGroups',
] as const;

export async function initGPU(options: InitOptions = {}): Promise<GPUContext> {
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    throw new GPUUnavailableError(
      'no-navigator-gpu',
      'WebGPU is unavailable. Chrome 113+, Edge 113+, or Safari 18+ is required.',
    );
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: options.powerPreference ?? 'high-performance',
  });
  if (!adapter) {
    throw new GPUUnavailableError(
      'no-adapter',
      'navigator.gpu exists but returned no adapter. The GPU is likely blocklisted — check chrome://gpu.',
    );
  }

  const clamp = options.clampStorageBindingSize ?? clampRequestedByEnvironment();
  const limits = requiredLimits(adapter.limits, clamp);

  // Optional features, most valuable first. Requesting a feature the driver advertises
  // can still fail, so each is dropped in turn rather than failing device creation.
  const optional: GPUFeatureName[] = ['shader-f16', 'timestamp-query'];
  if (options.requestSubgroups) optional.push('subgroups');
  const wanted = optional.filter((f) => adapter.features.has(f));

  const device = await requestDeviceDroppingFeatures(adapter, wanted, limits, options.label);

  device.lost.then((info) => {
    // A lost device is terminal: every buffer and pipeline built from it is invalid.
    console.error(`[gpu] device lost (${info.reason}): ${info.message}`);
    options.onDeviceLost?.(info);
  });

  if (options.onUncapturedError) {
    device.addEventListener('uncapturederror', (event) => {
      options.onUncapturedError?.((event as GPUUncapturedErrorEvent).error);
    });
  }

  const info: Partial<GPUAdapterInfo> = adapter.info ?? {};
  const l = device.limits;
  const profile: DeviceProfile = {
    f16: device.features.has('shader-f16'),
    timestampQuery: device.features.has('timestamp-query'),
    subgroups: device.features.has('subgroups'),

    maxBufferSize: l.maxBufferSize,
    maxStorageBufferBindingSize: l.maxStorageBufferBindingSize,
    maxUniformBufferBindingSize: l.maxUniformBufferBindingSize,
    maxComputeWorkgroupStorageSize: l.maxComputeWorkgroupStorageSize,
    maxComputeInvocationsPerWorkgroup: l.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupsPerDimension: l.maxComputeWorkgroupsPerDimension,
    maxStorageBuffersPerShaderStage: l.maxStorageBuffersPerShaderStage,

    tier: classifyTier(info),
    vendor: info.vendor ?? '',
    architecture: info.architecture ?? '',
    device: info.device ?? '',
    description: info.description ?? '',

    storageBindingClamped:
      l.maxStorageBufferBindingSize < adapter.limits.maxStorageBufferBindingSize,
    adapterStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
  };

  return { adapter, device, queue: device.queue, profile };
}

/**
 * Ask for the adapter's maximum on everything the engine cares about, except the one
 * limit the dev clamp holds down. Requesting a limit worse than the default is legal and
 * is the only way to reproduce a constrained device on unconstrained hardware.
 */
function requiredLimits(
  adapterLimits: GPUSupportedLimits,
  clampStorageBinding: boolean,
): Record<string, number> {
  const limits: Record<string, number> = {};
  for (const key of REQUESTED_LIMITS) {
    const value = adapterLimits[key];
    if (typeof value === 'number') limits[key] = value;
  }
  if (clampStorageBinding) {
    limits.maxStorageBufferBindingSize = Math.min(
      limits.maxStorageBufferBindingSize ?? DEFAULT_STORAGE_BINDING_SIZE,
      DEFAULT_STORAGE_BINDING_SIZE,
    );
  }
  return limits;
}

async function requestDeviceDroppingFeatures(
  adapter: GPUAdapter,
  wanted: GPUFeatureName[],
  requiredLimits: Record<string, number>,
  label?: string,
): Promise<GPUDevice> {
  const attempts: GPUFeatureName[][] = [wanted];
  for (let i = wanted.length - 1; i >= 0; i--) attempts.push(wanted.slice(0, i));

  let lastError: unknown;
  for (const requiredFeatures of attempts) {
    try {
      const device = await adapter.requestDevice({ requiredFeatures, requiredLimits, label });
      if (requiredFeatures.length < wanted.length) {
        const dropped = wanted.filter((f) => !requiredFeatures.includes(f));
        console.warn(`[gpu] device created without ${dropped.join(', ')} — fallback path engaged`);
      }
      return device;
    } catch (error) {
      lastError = error;
    }
  }
  throw new Error(`requestDevice failed for every feature combination: ${String(lastError)}`);
}

/** `VITE_GPU_CLAMP_STORAGE=1`, or `?clampStorage` on the URL for a one-off reload. */
function clampRequestedByEnvironment(): boolean {
  const env = import.meta.env?.VITE_GPU_CLAMP_STORAGE;
  if (env === '1' || env === 'true') return true;
  if (typeof location !== 'undefined' && location.search) {
    return new URLSearchParams(location.search).has('clampStorage');
  }
  return false;
}

/**
 * Best-effort classification from adapter strings, which browsers redact to varying
 * degrees. Used for reporting and for choosing defaults, never for correctness — a wrong
 * guess here must never be the difference between working and broken.
 */
function classifyTier(info: Partial<GPUAdapterInfo>): DeviceTier {
  const hay = `${info.vendor ?? ''} ${info.architecture ?? ''} ${info.device ?? ''} ${info.description ?? ''}`.toLowerCase();

  if (/adreno|mali|powervr|immortalis|xclipse|apple-a\d/.test(hay)) return 'mobile';
  if (typeof navigator !== 'undefined' && /android|iphone|ipad|ipod/i.test(navigator.userAgent)) {
    return 'mobile';
  }
  if (/nvidia|geforce|rtx |gtx |quadro|radeon rx|navi|rdna/.test(hay)) return 'discrete';
  // Apple silicon is unified-memory; it behaves like an integrated part for bandwidth
  // planning even though its throughput is not integrated-class.
  if (/intel|iris|uhd graphics|apple|metal|vega|swiftshader|llvmpipe|warp/.test(hay)) {
    return 'integrated';
  }
  return 'unknown';
}
