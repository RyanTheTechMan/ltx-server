# Third-party notices

Phase 4 contains no vendored LTX-2 or LTX-Desktop source, patches, model weights,
frontend assets or license-restricted artifacts. The implementation is original;
public interface facts and dimension constraints are documented as references.
The optional Linux inference extra installs official LTX packages from the exact
commit below. They remain separately licensed dependencies. The instance-local
retention/cancellation adapters are original code; no Desktop patch was copied.

Inspected sources:

- Lightricks/LTX-Desktop, commit `68cd86c15e5fd25f56229ea63c0dbcb0338f7812`.
  Its [LICENSE.txt](https://github.com/Lightricks/LTX-Desktop/blob/68cd86c15e5fd25f56229ea63c0dbcb0338f7812/LICENSE.txt)
  contains the Apache License 2.0 and `Copyright 2024 Lightricks`.
  Its separate `NOTICES.md` covers dependencies. If source is adapted in a later
  phase, preserve relevant notices, include the license and mark modifications.
- Lightricks/LTX-2, commit `a95ab856bf29407b6b066ede0abe1846050db56c`.
  The root [LICENSE](https://github.com/Lightricks/LTX-2/blob/a95ab856bf29407b6b066ede0abe1846050db56c/LICENSE)
  routes versions to `LICENSE-2` and `LICENSE-2_x` community license agreements.
  Do not assume Desktop's Apache license covers LTX-2 packages or model weights.
  Those license files were inspected before adding the optional dependency. Model
  downloads remain explicit and subject to the upstream repository's access and
  license terms. Packed Gemma assets also retain their upstream terms.

Python dependencies are installed separately under their own licenses. The
`uv.lock` file records exact versions and distributions; installed distribution
metadata contains their applicable license files. FFmpeg/ffprobe are externally
installed system tools and are not bundled or redistributed here.

SageAttention is an optional, separately built dependency under its upstream terms.
The original adapter uses its public API; no kernel source is vendored.
