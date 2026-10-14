# AGENTS.md

Guidance for contributors (human and AI) working on HeavyDB. This file currently
covers C++ only (`.h`, `.cpp`, and `.cu`/`.cuh` where applicable). Code under
`ThirdParty/` is vendored and must not be reformatted or restyled.

The golden rule: **match the surrounding code.** HeavyDB is a large, long-lived
codebase with some legacy inconsistencies. When the guidance below conflicts with the
established style of the file or module you are editing, follow the file, and don't
mix style-only cleanups into functional changes.

## Language and build

- C++17 (`CMAKE_CXX_STANDARD 17`). Do not use C++20 features.
- Builds use `-Wall -Werror -Wvla`. New code must compile warning-free; no
  variable-length arrays.
- Use C++ headers (`<cstdint>`, `<cstring>`), not C headers (`<stdint.h>`).

## Formatting (automated, do not hand-format)

- Formatting is defined by `.clang-format` (Chromium base, 90-column limit, 2-space
  indent) and enforced with **clang-format 14.0.6**. Other versions produce different
  output, so use the pinned one:
  - `pip install pre-commit && pre-commit install` (runs on commit), or
  - `pre-commit run clang-format --files <changed files>`
- Only format the files you touch. Do not reformat unrelated code.
- Notable consequences of the config:
  - When a call or declaration doesn't fit on one line, put each argument or
    parameter on its own line (no bin-packing).
  - Constructor initializer lists break before the comma.
  - `public:`/`private:` are indented by one space.
- Use `// clang-format off` / `// clang-format on` sparingly, only for tables or
  layouts that are clearer when hand-aligned.

## Static analysis (`.clang-tidy`)

The enabled checks are mandatory for new code:
- Always use braces around `if`/`else`/`for`/`while` bodies, even single statements.
- `using Alias = Type;` instead of `typedef`.
- `override` (or `final`) on every overriding virtual function; don't repeat `virtual`.
- Prefer `emplace_back`/`emplace` over constructing a temporary and pushing it.

## File layout

- Every source file starts with the NVIDIA SPDX header (pre-commit's
  `verify-copyright` hook inserts it and updates the year automatically):

      /*
       * SPDX-FileCopyrightText: Copyright (c) <year>, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
       * SPDX-License-Identifier: Apache-2.0
       */

  When you touch a file with a legacy HEAVY.AI/OmniSci/MapD header, the hook replaces
  it; let it.
- New headers use `#pragma once`. Don't convert existing `#ifndef` guards unless
  you're already rewriting the file.
- File names are PascalCase matching the main class (`RelScanTree.h`/`.cpp`). Use
  `.cpp` and `.h` (not `.cc`/`.hpp`).
- Includes:
  - In a `.cpp`, include its own header first.
  - Project headers use quotes and paths relative to the repo root, e.g.
    `#include "QueryEngine/RelAlgDag.h"`. Don't write `"RelAlgDag.h"` or use `../`.
  - Separate groups (project / third-party such as boost / standard library) with
    blank lines. clang-format sorts within each group but won't reorder groups
    (`IncludeBlocks: Preserve`).
  - Include what you use. Prefer forward declarations in headers when possible.

## Naming

| Entity | Convention | Example |
|---|---|---|
| Classes, structs, enums, type aliases | PascalCase | `QueryState`, `RelScanTree` |
| Member functions | camelCase | `getQueryStr()`, `createTimer()` |
| Free functions (incl. anonymous-namespace helpers) | snake_case | `find_top_level_project_node()` |
| Local variables, parameters, public struct fields | snake_case | `current_index`, `user_id` |
| Private/protected data members | snake_case + trailing `_` | `query_str_`, `events_mutex_` |
| Global variables | `g_` prefix, snake_case | `g_enable_watchdog` |
| Constants | `kPascalCase` or `ALL_CAPS`; match the module | `kMaxRetries`, `MAX_SIZE` |
| Macros | `ALL_CAPS` | `CHECK_EQ`, `DEBUG_TIMER` |

Namespaces: use the existing namespace of the module you're in (e.g.
`Catalog_Namespace`, `foreign_storage`, `QueryRenderer`). New top-level namespaces
should be snake_case. Put file-local helpers in an anonymous namespace, not `static`
free functions in headers. Never write `using namespace std;`.

## Idioms and preferred utilities

- **Invariants:** use `CHECK(cond)`, `CHECK_EQ/NE/LT/LE/GT/GE(a, b)` from
  `Logger/Logger.h`, not `assert`. They are active in release builds and accept
  streamed context: `CHECK_EQ(node->inputCount(), size_t(1)) << node->toString(...)`.
  Use `UNREACHABLE()` for impossible branches (e.g. the default of an exhaustive
  `switch`).
- **User-facing errors:** `throw std::runtime_error(...)` (or an existing
  domain-specific exception type) with an actionable message. `CHECK` failures are
  fatal and are only for programmer errors, never for bad input.
- **Logging:** `LOG(INFO|WARNING|ERROR|FATAL)` and `VLOG(n)` for verbose/debug output.
  Use `DEBUG_TIMER(__func__)` for timing. Don't use `std::cout`/`std::cerr`/`printf`
  outside tests and command-line tools.
- **Ownership:** `std::unique_ptr` by default, `std::shared_ptr` only when ownership
  is genuinely shared. Create them with `std::make_unique`/`std::make_shared`, not
  `new`. Pass non-owning references as `const T&` or `T*` (when nullable).
- **Null:** `nullptr`, never `NULL` or `0`.
- **Optional values:** prefer `std::optional` for new code (`boost::optional` is legacy).
- **Strings:** use `std::string_view` for non-owning read-only string parameters where
  practical. See `Shared/StringTransform.h` for common helpers before writing new ones.
- **Filesystem:** much existing code uses `boost::filesystem`; follow the module you're
  in. Prefer `std::filesystem` for new standalone code.
- **Locking:** use the aliases in `Shared/heavyai_shared_mutex.h`
  (`heavyai::shared_mutex`, `heavyai::unique_lock`, `heavyai::shared_lock`,
  `heavyai::lock_guard`) for consistency with existing lock code; plain
  `std::lock_guard<std::mutex>` is fine for simple private mutexes. Mark mutexes that
  guard data read in `const` methods `mutable`.
- **Cleanup on scope exit:** `ScopeGuard` from `Shared/scope.h`.
- **const:** be const-correct (methods, parameters, locals). Both `const auto` and
  `auto const` appear; match the file.
- **Enums:** prefer `enum class` for new enumerations.
- **Casts:** C++ casts (`static_cast`, `dynamic_cast`); no C-style casts.
- **Literal arguments:** label non-obvious literals with a comment:
  `buildDag(plan, /*optimize_dag=*/false)`.

## Comments and documentation

- Use Doxygen-style `/** ... */` with `@brief`/`@param` for public APIs and
  non-obvious functions; `//` for inline comments.
- Comments explain *why* (constraints, invariants, non-obvious behavior), not *what*
  the next line does.
- Don't commit commented-out code. `TODO`s should say what is missing and ideally who
  or what ticket owns it (`// TODO(name): ...`).

## Tests

- Tests use GoogleTest and live in `Tests/` as `<Feature>Test.cpp`. Register a new test
  binary in `Tests/CMakeLists.txt` with both `add_executable(...)` and
  `add_test(<Name> <Name> ${TEST_ARGS})`.
- Follow the standard `main` pattern used by existing tests:
  `TestHelpers::init_logger_stderr_only(argc, argv)`, `testing::InitGoogleTest(...)`,
  `QR::init(BASE_PATH)` when a QueryRunner is needed, and wrap `RUN_ALL_TESTS()` in a
  try/catch that `LOG(ERROR)`s the exception.
- Reuse fixtures and helpers (`TestHelpers.h`, `DBHandlerTestHelpers.h`,
  `DataMgrTestHelpers.h`) instead of duplicating setup.
- Bug fixes should come with a regression test.

## Commits

- All commits must be signed off (`git commit -s`); see `CONTRIBUTING.md` for the DCO.
- Keep formatting-only or renaming-only changes in separate commits/PRs from functional
  changes.