/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Absolute or same-origin URL of the `.enargeia` weight file. See `.env`. */
  readonly VITE_MODEL_URL?: string;
  /** Clamp `maxStorageBufferBindingSize` to 128 MiB to exercise the split-binding path. */
  readonly VITE_GPU_CLAMP_STORAGE?: string;
  /** Comma-separated optimization keys to disable, for the ablation harness. */
  readonly VITE_ABLATION_OFF?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
