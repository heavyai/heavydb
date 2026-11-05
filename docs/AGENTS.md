# AGENTS.md — docs/

Instructions for AI agents (and a useful reference for humans) working in
this directory. Read this before editing anything under `docs/` or `fern/`.

## What lives where

```
fern/                    Fern project config (NOT under docs/)
  fern.config.json       Org name + pinned Fern CLI version
  docs.yml               Site config: navigation, libraries, instances, theme

docs/
  pages/                 Narrative Markdown/MDX pages (Fern site content)
    api/cpp/              ← generated, see "C++ API reference" below
    <section>/index.mdx   ← one dir per nav section, matches docs.yml nesting
  images/                Images referenced by pages/ (same relative layout)
  source/                LEGACY Sphinx/reStructuredText docs. Retained for
                          history only — not built, not published, not
                          linked from the Fern site. Do not edit to "keep it
                          in sync"; it is frozen.
  internal/               Internal-only guides (e.g. mapd-release-guide.md).
                          Unrelated to the Fern site; leave alone unless
                          asked.
  README.md               Human-facing quick reference (short).
  AGENTS.md               This file.
```

`docs.yml` lives in `fern/`, but the paths it points at start with
`../docs/pages/...` — the actual page/image content was intentionally moved
out of `fern/` into `docs/pages/` and `docs/images/` so it sits alongside
`docs/source/` and `docs/internal/`. `fern/` itself should only ever contain
`fern.config.json`, `docs.yml`, and `.gitignore`.

## Site structure

- Single Fern instance: `heavyai-heavydb.docs.buildwithfern.com/heavyai/heavydb`,
  custom domain `docs.nvidia.com/heavyai/heavydb`, theme `nvidia`.
- `docs.yml`'s `navigation:` mirrors the old Sphinx `docs/source/index.rst`
  toctree: Introduction → System Architecture (Overview, Quickstart, Catalog,
  Data Model, Data Flow, Calcite Parser, Query Execution, Foreign Storage,
  Table Functions, ML Models, Row-Level Security, Runtime, Configuration) →
  C++ API Reference (one `folder:` entry per module, see below) →
  Additional Resources (Glossary only) → Detailed Class Information
  (Logger, QueryState).
- Every narrative page has `---\ntitle: <Title>\n---` frontmatter. Keep this
  convention for new pages — link labels elsewhere are written from these
  titles (see "Cross-links" below).

## C++ API reference (`docs/pages/api/cpp/`)

This directory is **entirely generated** — never hand-edit files under it.
It replaces the project's old Doxygen/breathe Sphinx integration with Fern's
own (beta) library docs generator, which parses C++ source directly into
MDX.

`fern/docs.yml`'s `libraries:` block has one entry per top-level C++ module
(matching the old `Doxyfile.in` scope, minus vendored/test/tooling dirs):
analyzer, archive, calcite, catalog, cudamgr, datamgr, fragmenter,
geospatial, importexport, lockmgr, logger, migrationmgr, parser,
queryengine, queryrunner, shared, sqliteconnector, stringdictionary,
stringops, tablearchiver, thrifthandler, utils. To add another module,
add a `heavydb-<name>: {lang: cpp, input: {path: ../<Dir>}, output: {path:
../docs/pages/api/cpp/<name>}}` entry and a matching `folder:` item under
the "C++ API Reference" navigation section.

Regenerate after any C++ source change under those modules:

```
dev-tools/dev.sh build docs --regenerate-api
# or, equivalently, from fern/:
fern docs md generate --local     # requires Docker
```

CI enforces this doesn't drift: `docs-tests.yml`'s `api-generate` job
regenerates and fails if it produces a diff (see "CI" below).

One known generator quirk: anonymous C++ enums produce a page literally
named `.mdx` with blank frontmatter, which fails `fern check`. If
regeneration produces one, delete it rather than trying to give it a name
(this happened once already, for an anonymous enum in `QueryEngine`).

## Editing narrative pages

- Filenames and directories are kebab-case (`data-model/physical-layout.mdx`,
  not `data_model/physical_layout.mdx`), independent of the old RST source
  naming.
- **Cross-links**: use the target page's title as link text, not its path —
  e.g. `[Runtime](../runtime/index.mdx)`, never
  `[../runtime/index.mdx](../runtime/index.mdx)`. This was a real cleanup
  pass (see git history) after a first migration draft left raw paths as
  link text; don't reintroduce it.
- Anchors (`#some-heading`) must match an actual heading's generated slug.
  Don't invent figure/section anchors that don't correspond to a real
  heading — link to prose ("see diagram below") instead if there's nothing
  to anchor to.
- `.. note::` / `.. warning::` (old RST) map to `<Note>...</Note>` /
  `<Warning>...</Warning>` MDX components.
- Images: copy into `docs/images/` (preserving a sensible subpath), then
  reference with a normal relative markdown image link
  (`![alt](../../images/foo/bar.png)`). Compute the relative path from the
  page's actual location.
- **PlantUML / Graphviz diagrams are not yet rendered.** Six pages currently
  carry raw diagram source in a fenced ```plantuml or ```dot code block,
  immediately followed by `<Note>TODO: render this diagram to an image and
  replace this code block.</Note>`:
  `components/logger.mdx`, `execution/kernels.mdx`, `execution/overview.mdx`,
  `execution/scheduler.mdx`, `flow/data.mdx`, `overview/index.mdx`. If you're
  touching one of these pages, consider rendering the diagram (e.g. via
  `plantuml`/`graphviz` locally) to SVG/PNG under `docs/images/` and
  replacing the placeholder — but this wasn't in scope for the original
  migration, so don't be surprised it's still pending.
- No math/LaTeX rendering has been verified — inline `:math:` roles from the
  old RST were converted to `$...$` spans on a best-effort basis
  (`execution/results.mdx`) but nobody has confirmed Fern renders them.

## Validating, previewing, publishing

Always run `fern check` (or `dev-tools/dev.sh build docs`) after editing
anything under `fern/` or `docs/pages/` — broken nav references or malformed
frontmatter fail loudly and cheaply here, don't leave it for CI.

```bash
# Validate
dev-tools/dev.sh build docs                  # fern check
dev-tools/dev.sh build docs --regenerate-api # + regenerate C++ API pages first

# Equivalent, from fern/
fern check
fern docs md generate --local   # requires Docker

# Preview locally
cd fern && fern docs dev

# Publish to production (also happens automatically on merge to master — see CI)
cd fern && fern generate --docs
```

Requires the Fern CLI on `PATH`: `npm install -g fern-api`.

## CI

Four workflows under `.github/workflows/`, following this repo's existing
pattern (manual `workflow_dispatch`, triggered by `/test <subcommand>` PR
comments and auto-pended by `pr-gatekeeper.yml` based on changed files —
*not* automatic `pull_request` triggers, and no third-party marketplace
actions, only `actions/*` + `actions/github-script`):

- **`docs-tests.yml`** (`/test docs`) — `fern-check` (validate),
  `api-generate` (C++ reference drift check), `preview` (when dispatched
  with a PR number: builds a stable Fern preview link and posts/updates a
  PR comment).
- **`cleanup-docs-preview.yml`** — deletes a PR's preview on merge.
  Automatic, no `/test` gating needed (push-only trust boundary).
- **`publish-docs.yml`** — publishes to production on push to `master`
  touching `fern/`, `docs/pages/`, or `docs/images/`. Automatic. Regenerates
  the C++ API reference pages from current source before publishing
  (ephemeral — not committed back), so production always reflects the
  latest source rather than whatever was last manually regenerated and
  committed.
- **`pr-gatekeeper.yml`** — has a `docs` category (context `docs tests`)
  that pends whenever a PR touches `fern/`, `docs/pages/`, or
  `docs/images/`.

**Requires a `FERN_TOKEN` repository secret** (Settings → Secrets and
variables → Actions; generate via the Fern Dashboard or `fern token`) for
`preview`, `cleanup-docs-preview`, and `publish-docs` to work.

**`docs tests` is not yet a required branch-protection status check** —
that's a GitHub admin setting outside this repo's files, still outstanding.

## Retired: the old Sphinx build

`docs/Dockerfile`, `docs/Makefile`, `docs/requirements.txt`, and the CMake
`sphinx` target are gone — replaced by the Fern flow above. `docs/source/`
(the RST content they built) is kept only as a historical reference; it is
frozen and disconnected from the live site.
