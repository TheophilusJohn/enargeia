/**
 * TypeScript 6's `lib.dom` ships every WebGPU interface but omits the four flag namespace
 * objects, so `GPUBufferUsage.STORAGE` does not typecheck out of the box. These are the
 * spec's values, declared here rather than pulling in `@webgpu/types` — that package
 * redeclares the whole API and collides with `lib.dom`.
 *
 * Compute-only: texture and colour-write flags are omitted because the engine has no
 * render path. Add them here if that ever changes.
 */

declare var GPUBufferUsage: {
  readonly MAP_READ: 0x0001;
  readonly MAP_WRITE: 0x0002;
  readonly COPY_SRC: 0x0004;
  readonly COPY_DST: 0x0008;
  readonly INDEX: 0x0010;
  readonly VERTEX: 0x0020;
  readonly UNIFORM: 0x0040;
  readonly STORAGE: 0x0080;
  readonly INDIRECT: 0x0100;
  readonly QUERY_RESOLVE: 0x0200;
};

declare var GPUShaderStage: {
  readonly VERTEX: 0x1;
  readonly FRAGMENT: 0x2;
  readonly COMPUTE: 0x4;
};

declare var GPUMapMode: {
  readonly READ: 0x1;
  readonly WRITE: 0x2;
};
