# HeavyDB Developer Documentation

User-facing product documentation is published at:

https://docs.nvidia.com/heavyai

## Fern docs

The developer documentation site is built with [Fern](https://buildwithfern.com)
from `fern/` at the repository root:

* `fern/docs.yml` — site configuration and navigation.
* `docs/pages/` — narrative Markdown/MDX pages, plus `docs/pages/api/cpp/`,
  the C++ API reference generated from source via Fern's library docs
  generator (replaces the old Doxygen/breathe integration).
* `docs/images/` — images referenced by the pages above.

See [AGENTS.md](AGENTS.md) for the full picture: page/nav conventions, how
the C++ API reference is generated and kept in sync, known gaps (a handful
of diagrams still aren't rendered), and how the CI (preview links, publish
on merge) is wired up.

### Validating

From the repository root:

```
dev-tools/dev.sh build docs
```

This runs `fern check` against the site. Pass `--regenerate-api` to
regenerate the C++ API reference pages first (`fern docs md generate
--local`, requires Docker). The default `dev-tools/dev.sh build` / `build
all` targets also validate docs after building heavydb.

Equivalently, from `fern/`:

```
fern check
```

Requires the Fern CLI on PATH (`npm install -g fern-api`).

### Previewing

```
cd fern && fern docs dev
```

### Publishing

```
cd fern && fern generate --docs
```

## Other contents of this directory

* `docs/source/` — the original Sphinx/reStructuredText developer docs,
  retained for reference. Superseded by the Fern site above; not built or
  published anymore.
* `docs/internal/` — internal-only guides (e.g. the release process).
