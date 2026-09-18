# dev-tools Workflow Reference

For getting-started documentation, see [README.md](README.md).

This document provides a compact option reference for each command and documents
the auto-detection behaviour of the deps and runtime base image resolution.

---

## Command Reference

### `build deps`

Builds a local deps container image (~2 hours).

| Option | Default | Description |
|---|---|---|
| `--distro=ubuntu22.04\|rockylinux8` | `ubuntu22.04` | Target distro |
| `--cuda-version=<ver>` | `12.9.2` | CUDA version |
| `--lib-type=static\|shared` | `static` | Library type |
| `--tag=<tag>` | `YYYYMMDD` | Image tag |
| `--nproc=<n>` | `min(nproc,24)` | Parallel jobs |

**Preconditions:** `nvcr.io/nvidia/cuda:<ver>-devel-<distro>` available locally.

**Output:** `ghcr.io/heavyai/heavydb-internal/core-build-<distro>-<lib>-cuda<ver>-<arch>:<tag>`

---

### `build immerse / webserver / heavyiq / geos-dsos`

Builds a single component artifact.

| Option | Default | Description |
|---|---|---|
| `--deps-image=<image>` | auto-detected | Deps container image |
| `--distro=<distro>` | `ubuntu22.04` | Filter deps image detection |
| `--cuda-version=<ver>` | — | Filter deps image detection to a CUDA version |
| `--ref=<branch\|sha>` | existing checkout | Component repo ref |
| `--repos-dir=<path>` | parent of this repo | Parent dir for clones |
| `--output-dir=<path>` | `build/components/` | Artifact destination |
| `--pyheavydb-source=<src>` | `project` | `heavyiq` only: `project\|pypi\|testpypi\|build-local-wheel` |
| `--pyheavydb-ref=<ref>` | — | `heavyiq` only: implies `build-local-wheel` |

**Preconditions:** Deps image; SSH access to component repos; network for tool downloads (Node.js / Go / GEOS).

---

### `build heavydb`

Runs cmake + make + cpack inside the deps container.

| Option | Default | Description |
|---|---|---|
| `--deps-image=<image>` | auto-detected | Deps container image |
| `--distro=<distro>` | parsed from deps image | `ubuntu22.04\|rockylinux8` |
| `--cuda-version=<ver>` | — | Filter deps image detection |
| `--compiler=gcc\|clang` | `gcc` | Compiler |
| `--norendering` | off | Disable rendering |
| `--cuda` / `--cpu` | cuda | CUDA or CPU-only build |
| `--static` / `--shared` | static | Library linking |
| `--build-type=Release\|Debug\|RelWithDebInfo` | `Release` | CMake build type |
| `--heavyiq / --immerse / --webserver` | off | Bundle component from `--component-dir` |
| `--heavyiq-file=<path>` | — | Explicit component path (implies `--heavyiq`) |
| `--immerse-file=<path>` | — | Explicit component path (implies `--immerse`) |
| `--webserver-file=<path>` | — | Explicit component path (implies `--webserver`) |
| `--component-dir=<path>` | `build/components/` | Component artifact dir |
| `--output-dir=<path>` | `build/` | Build output dir |
| `--nproc=<n>` | `min(nproc,24)` | Parallel make jobs |
| `--no-package` | off | Skip cpack (build only) |
| `--clean` | off | Wipe output dir before building |
| `--docker` | off | Also build a product Docker image |
| `--product-base-image=<image>` | auto-detected from deps image | Override Docker base image |

**Output:** `build/heavyai-<ver>-<date>-<sha>-<distro>-<arch>[-render].tar.gz`
**With `--docker`:** also `heavyai-<distro>-cuda-<render|norender>-<arch>:<ver>-<date>-<sha>` + `:latest`

---

### `build image`

Builds a product Docker image from a pre-existing tarball without recompiling.

| Option | Default | Description |
|---|---|---|
| `--tar=<path>` | **required** | Path to `.tar.gz`; distro and render parsed from filename |
| `--product-base-image=<image>` | auto-detected | Explicit base image; distro derived from its name |
| `--tag=<tag>` | parsed from tarball name | Override image tag |

**Preconditions:** Tarball file must exist; a local `nvcr.io/nvidia/cuda:*-runtime-<distro>` image must be available (unless `--product-base-image` is given).

**Output:** `heavyai-<distro>-cuda-<render|norender>-<arch>:<tag>` + `:latest`

---

### `build / build all`

Runs component builds + `build heavydb` [+ docs] in one step.
`build` excludes HeavyIQ; `build all` includes it.

Accepts all options from `build heavydb` plus:

| Option | Default | Description |
|---|---|---|
| `--ref=<branch\|sha>` | — | Shared ref for all component checkouts |
| `--heavyiq-ref / --immerse-ref / --webserver-ref` | falls back to `--ref` | Per-component ref override |
| `--repos-dir=<path>` | parent of this repo | Parent dir for component clones |
| `--run-tests=none\|sanity\|all` | `none` | Run test suite after build |

---

### `build docs`

Builds Sphinx HTML docs; optionally runs Doxygen if available on the host.

| Option | Default | Description |
|---|---|---|
| `--output-dir=<path>` | `build/` | Build dir (also the docs destination) |

**Output:** `build/docs/html/`

---

### `build ci <config>`

Runs a raw CI-style build (intended to be run inside the deps container).

**Preconditions:** Must be inside the deps container (or have the mapd-deps env available at `/usr/local/mapd-deps/mapd-deps.sh`).

---

### `test [sanity | all]`

Runs the sanity or full test suite against an existing build.

| Option | Default | Description |
|---|---|---|
| `--deps-image=<image>` | auto-detected from `build/CMakeCache.txt` | Deps container image |
| `--build-dir=<path>` | `build/` | Build dir to test |
| `--all` | off | Run full test suite instead of sanity |
| `--nproc=<n>` | `min(nproc,24)` | Parallel jobs for test compilation |

**Preconditions:** A completed `build heavydb` in `--build-dir` (CMakeCache.txt must exist).

---

### `test <variant>`

Runs a CI test variant (e.g. `columnar`, `extended-geo`, `arrow-ipc`) against an existing local build. From the host, dev.sh auto-wraps the run in the deps container; from inside a container or CI image it runs directly.

**Host-side flow:**
1. Read `build/CMakeCache.txt` to recover distro, CUDA version, and lib type
2. Auto-select the matching local deps image
3. Re-invoke inside the container: reconfigure with `-DENABLE_TESTS=ON`, compile test binary, run

| Option | Default | Description |
|---|---|---|
| `--deps-image=<image>` | auto-detected from `build/CMakeCache.txt` | Deps container image |
| `--build-dir=<path>` | `build/` | Existing build tree to test against |
| `--nproc=<n>` | `min(nproc,24)` | Parallel jobs for test binary compilation |

**Preconditions:** A completed `build heavydb` (CMakeCache.txt must exist in `--build-dir`).

---

### `test immerse / webserver / heavyiq / pyheavydb`

Runs component-level test suites.

| Option | Default | Description |
|---|---|---|
| `--deps-image=<image>` | auto-detected | Deps container image |
| `--distro=<distro>` | `ubuntu22.04` | Distro filter for deps image auto-detection |
| `--repos-dir=<path>` | parent of this repo | Parent dir for clones |
| `--ref=<branch\|sha>` | existing checkout | Component ref |
| `--unit` | — | `pyheavydb` / `heavyiq` only: run unit tests only (no server) |
| `--integration` | — | `pyheavydb` only: run integration tests (needs product image) |
| `--heavydb-host=<host>` | managed mode | `pyheavydb` only: test against an external server |
| `--heavydb-image=<img>` | auto-detected locally built product image | `pyheavydb` only: server image for managed mode |

---

### `test in-image <config> <variant>`

Pulls a CI-built `pr-build/<config>` image from GHCR and runs a test variant
inside it. Use this to reproduce exact CI behaviour without a local build.

| Option | Default | Description |
|---|---|---|
| `--image=<local-image>` | pull from GHCR | Use a local image instead of pulling |
| `--build-dir=<path>` | — | Mount a local `build/` tree at `/workspace/build` inside the image (overrides the baked-in build) |
| `--tag=<tag>` | latest for the current HEAD | Specific image tag to pull |

**When to use `test in-image` vs `test <variant>`:**

| | `test <variant>` | `test in-image` |
|---|---|---|
| Source of binaries | Compiled from your local `build/` | Baked into the CI image |
| Deps container | Your local deps image | The CI pr-build image |
| Use case | Fast iteration on your own build | Reproduce a CI failure exactly |

---

### `shell`

| Invocation | What it does |
|---|---|
| `shell` | Interactive shell inside the local deps container |
| `shell <config>` | Pull + drop into a CI image shell |
| `shell pull <config>` | Pull a CI image and print its full ref |

---

## Auto-Detection Behaviour

### Deps image (`_resolve_deps_image`)

Called by: `build heavydb`, `build [all]`, `build immerse/webserver/heavyiq/geos-dsos`, `test`, `shell`.

Scans local Docker images matching `ghcr.io/heavyai/heavydb-internal/core-build-*-<lib_type>-*-<arch>`.
Filters by `--distro` and `--cuda-version` when supplied. Deduplicates by image ID (multiple tags of the same image count as one; when an image has multiple tags, the one listed first by `docker images` is used). Errors if:
- **0 matches** → `ERROR: no local deps image found. Run: build deps`
- **2+ distinct images remain** → `ERROR: multiple deps images found. Use --deps-image or narrow with --distro / --cuda-version`

### Product base image (`_resolve_product_base_image`)

Called by: `build image` (when `--product-base-image` is not given).

Scans local Docker images matching `nvcr.io/nvidia/cuda:*-runtime-<distro>`. Errors if:
- **0 matches** → `ERROR: no CUDA runtime base image found for '<distro>'. Pull one or run build deps.`
- **2+ matches** → `ERROR: multiple CUDA runtime base images. Use --product-base-image.`
