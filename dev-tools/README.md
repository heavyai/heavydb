# Local Development with dev.sh

For a complete workflow graph including all commands, options, and blocked paths, see [WORKFLOWS.md](WORKFLOWS.md).

`dev-tools/dev.sh` lets you build, test, and interact with the project locally
using the same toolchain as CI. Most commands run inside the local deps
container; the docs target uses the Sphinx docs Docker image instead, with
optional Doxygen on the host.

## Prerequisites

- Docker installed and running
- For GPU access inside containers: `nvidia-container-toolkit` (auto-detected
  via `nvidia-smi`; containers run CPU-only if absent)

## Quick start

```bash
# Create a working area
mkdir $WORKSPACE/heavyai
cd $WORKSPACE/heavyai

# Clone the repo
git clone https://github.com/heavyai/heavydb.git
cd heavydb

# Build a local build/dependencies container
./dev-tools/dev.sh build deps

# If doing a full build (including front-end) and you have
# these, set them in the env before the build
export MAPBOX_TOKEN=pk.xxx
export GOOGLE_API_KEY=AIzaXXX

# Default full build:
#   Back-end (HeavyDB server and renderer)
#   Front-end (Immerse and Webserver)
#   GEOS bundle
#   Sphinx docs
# This will create a tarball package
# To create a Docker container, add the --docker option
./dev-tools/dev.sh build [--docker]

# Alternative:
#   Just build back-end (HeavyDB server and renderer only) as a tarball
./dev-tools/dev.sh build heavydb
```

---

## `build` — build things

```
dev-tools/dev.sh build [target] [options]
```

| Target | What it does |
|---|---|
| *(none)* | Build immerse + webserver + geos-dsos + heavydb + docs (common dev case) |
| `all` | Build ALL components (including HeavyIQ) + heavydb + docs |
| `immerse` | Build just the Immerse frontend artifact |
| `webserver` | Build just the web server artifact |
| `heavyiq` | Build just the HeavyIQ artifact |
| `geos-dsos` | Build just the GEOS DSO artifact |
| `docs` | Build Sphinx developer docs into `build/docs/html/` |
| `heavydb` | Build heavydb only (using existing component artifacts) |
| `image` | Build a product Docker image from a pre-existing tarball (no recompile) |
| `deps` | Build the deps container image (multi-hour) |
| `ci <config>` | Run a CI-style build via `scripts/ci/build.sh` |

**Common options** (for `build` and `build all`):

```
--deps-image=<image>          Override auto-detected deps container image
--distro=ubuntu22.04|rockylinux8   (default: ubuntu22.04)
--cuda-version=<ver>          Filter deps image auto-detection to a CUDA version (e.g. 12.9.2)
--compiler=gcc|clang          (default: gcc)
--cuda / --cpu                (default: cuda)
--static / --shared           (default: static)
--build-type=Release|Debug|RelWithDebInfo   (default: Release)
--nproc=<n>                   Parallel make jobs (default: min(nproc, 24))
--no-package                  Skip cpack
--docker                      Also build a product Docker image after cpack
--clean                       Wipe build directories and repo artifacts before building
--ref=<branch|sha>            Ref to check out for all component repos
--repos-dir=<path>            Parent dir for component repo clones (default: ../)
```

**Environment variables for Immerse:**

```bash
PRIVATE_PACKAGES_TOKEN=$(gh auth token) \  # required for private npm packages
MAPBOX_TOKEN=pk.xxx \                       # for Mapbox maps to work
GOOGLE_API_KEY=AIzaXXX \                   # for Google Maps / Street View to work
  dev-tools/dev.sh build immerse
```

**Per-component options** (when building a single component):

```
--ref=<branch|sha>     Ref to check out
--repos-dir=<path>     Parent dir for repo clone
--output-dir=<path>    Where to write the artifact (default: build/components/)
--pyheavydb-source=<source>   HeavyIQ only: project, pypi, testpypi, or build-local-wheel
--pyheavydb-ref=<ref>         pyheavydb ref when building a local wheel
```

**HeavyIQ Python dependencies**

HeavyIQ's `requirements.txt` is authoritative. Its build artifact includes that
manifest and the application source. By default, `project` mode packages no
wheels and leaves dependency resolution entirely to HeavyIQ.

Because pyheavydb is managed with HeavyDB, the explicit modes obtain the version
declared by HeavyIQ and pass the selected wheel through HeavyIQ's existing
`--pyheavydb-wheel` hook. That pyheavydb wheel is the sole packaged
exception:

```bash
# Fetch the declared version specifically from TestPyPI
dev-tools/dev.sh build heavyiq --pyheavydb-source=testpypi

# Fetch it specifically from production PyPI
dev-tools/dev.sh build heavyiq --pyheavydb-source=pypi

# Build the declared version from a pyheavydb source ref
dev-tools/dev.sh build heavyiq \
  --pyheavydb-source=build-local-wheel \
  --pyheavydb-ref=<branch-or-sha>
```

Package-index modes download the selected wheel below
`build/components/.python-deps/`; `build-local-wheel` uses the
pyheavydb checkout's `dist/` directory. dev-tools checks that the wheel
satisfies HeavyIQ's declaration and passes it to HeavyIQ, which stores only it
under `packages/`. It neither changes `requirements.txt` nor creates
`requirements.packages.txt`. `build all` carries the same artifact into the
product without adding Python dependencies.

When the product starts HeavyIQ, `scripts/start_heavyiq.sh` creates a virtual
environment, installs the optional pyheavydb wheel with `--no-deps`, then
installs `requirements.txt`. The deployment environment supplies package-index
access and pip configuration. Air-gapped HeavyIQ installation is not supported.

**GEOS DSOs** — the artifact contains flat `libgeos*` shared objects for
`--libgeos-path /var/lib/heavyai/libgeos`. Install manually:

```bash
mkdir -p /var/lib/heavyai/libgeos
tar xJf heavydb-libgeos-<os>-<arch>.tar.xz -C /var/lib/heavyai/libgeos
```

**Docs** — Sphinx HTML is written to `build/docs/html/`. Docs build
outside the heavydb deps container (Sphinx runs in the dedicated docs Docker
image). The default `build` / `build all` targets run docs after heavydb.
Doxygen runs only when a configured heavydb build (`Doxyfile`) exists and
`doxygen` is on the host PATH; otherwise Sphinx builds with placeholder C++
API pages.

```bash
dev-tools/dev.sh build docs
```

Set `HEAVYDB_SPHINX_IMAGE` to override the Sphinx image name (default
`heavydb-sphinx-doc`). After editing `docs/Dockerfile` or `docs/requirements.txt`,
rebuild explicitly: `docker build -t heavydb-sphinx-doc docs/`.

---

## `test` — run tests

```
dev-tools/dev.sh test [target] [options]
```

| Target | What it does |
|---|---|
| `sanity` *(default)* | Sanity tests against an existing heavydb build |
| `all` | Full test suite |
| `pyheavydb` | pyheavydb pytest suite (`--unit` / `--integration`) |
| `immerse` | Immerse npm tests (ESLint + Jest unit + Jest component) |
| `webserver` | WebServer verify (goimports + golint) |
| `heavyiq` | HeavyIQ pytest suite (`--unit` to skip server tests) |
| `integration-encrypted-jdbc` | TLS + JDBC Docker integration test (requires `--build-dir`) |
| `integration-kafka-import` | KafkaImporter Docker integration test (requires `--build-dir`) |
| `in-image <config> <variant>` | Run a test variant inside a CI-built image |
| `<variant>` | Run a CI test variant against `./build` |

```bash
dev-tools/dev.sh test sanity
dev-tools/dev.sh test pyheavydb --unit
dev-tools/dev.sh test immerse
dev-tools/dev.sh test heavyiq --unit
dev-tools/dev.sh test integration-encrypted-jdbc --build-dir build
PRIVATE_PACKAGES_TOKEN=$(gh auth token) dev-tools/dev.sh test immerse
```

See `dev-tools/integration-tests/README.md` for integration test build dependencies and options.

---

## `shell` — open a shell

```
dev-tools/dev.sh shell [target]
```

| Target | What it does |
|---|---|
| *(none)* | Enter the deps container (repo mounted at `/workspace`) |
| `<config>` | Pull + drop into a bash shell in a CI-built image |
| `pull <config>` | Just pull a CI-built image from GHCR |

```bash
dev-tools/dev.sh shell            # enter deps container
dev-tools/dev.sh shell gcc        # pull + enter CI image
dev-tools/dev.sh shell pull gcc   # just pull
```

---

## `list` — list configs and test variants

```bash
dev-tools/dev.sh list
```

---

## Deps image auto-detection

All build/test subcommands auto-detect a local deps image matching
`ghcr.io/heavyai/heavydb/core-build-*-<lib-type>-*-<arch>`. Candidates
are deduplicated by image ID; the `:latest` tag is preferred when multiple tags
point to the same image. Override with `--deps-image=<image>` or narrow with
`--distro=<distro>` and/or `--cuda-version=<ver>`. Ambiguous results (multiple
distinct images after filtering) are an error — add dimensions or use `--deps-image`.

## Log files

Build output is redirected to `build/logs/` to keep the terminal clean.
One log file per step:

| Step | File |
|---|---|
| Immerse | `immerse.log` |
| WebServer | `webserver.log` |
| HeavyIQ | `heavyiq.log` |
| pyheavydb artifact | `pyheavydb.log` or `pyheavydb-<source>.log` |
| GEOS DSOs | `geos-dsos.log` |
| heavydb cmake+make | `heavydb.log` |
| Docs (Doxygen, optional) | `docs-doxygen.log` |
| Docs (Sphinx) | `docs.log` |
| Docker image build | `docker-build.log` |

On failure the last 50 lines of the relevant log are printed automatically.
