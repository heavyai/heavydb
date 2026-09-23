# HeavyDB (formerly MapD Core or OmniSciDB)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/heavyai/heavydb/blob/master/LICENSE.txt)
[![Security](https://img.shields.io/badge/Security-Report%20a%20Vulnerability-red.svg)](https://github.com/heavyai/heavydb/blob/master/SECURITY.md)
[![GitHub Discussions](https://img.shields.io/badge/GitHub-Discussions-blue?logo=github)](https://github.com/orgs/heavyai/discussions)


HeavyDB is an in-memory, column store, SQL relational database designed from the ground up to run on GPUs.

The repository includes a number of third party packages provided under separate licenses. Details about these packages and their respective licenses is at [ThirdParty/licenses/index.md](ThirdParty/licenses/index.md).

***

# Developing HeavyDB: Table of Contents

- [Links](#links)
- [License](#license)
- [Contributing](#contributing)
- [Building](#building)
- [Code Style](#code-style)
- [Dependencies](#dependencies)
- [Security](#security)
- [Support](#support)

# Links

- [Documentation](https://docs.nvidia.com/heavyai/)
- [Release Notes](https://docs.nvidia.com/heavyai/release-notes)


# License

This project is licensed under the [Apache License, Version 2.0](https://github.com/heavyai/heavydb/blob/master/LICENSE.txt).

The repository includes a number of third party packages provided under separate licenses. Details about these packages and their respective licenses is at [ThirdParty/licenses/index.md](ThirdParty/licenses/index.md).

# Contributing

Follow the instructions noted in [CONTRIBUTIONING.md](https://github.com/heavyai/heavydb/blob/master/CONTRIBUTING.md)

# Building

Follow the instructions noted in [Build From Source](https://docs.nvidia.com/heavyai/installation-and-configuration/installation/build-from-source). This will build the entire HeavyAI platform which includes Immerse, WebServer, GEOS DSOs, and HeavyDB into a usable docker image.

# Code Style

Contributed code should compile without generating warnings by recent compilers on most Linux distributions. Changes to the code should follow the [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines).

## clang-format

A [`.clang-format`](https://clang.llvm.org/docs/ClangFormat.html) style configuration, based on the Chromium style guide, is provided at the top level of the repository. Please format your code using a recent version (9.0+ preferred) of ClangFormat before submitting.

To use:

    clang-format -i File.cpp

## clang-tidy

A [`.clang-tidy`](https://clang.llvm.org/extra/clang-tidy/) configuration is provided at the top level of the repository. Please lint your code using a recent version (6.0+ preferred) of clang-tidy before submitting.

`clang-tidy` requires all generated files to exist before running. The easiest way to accomplish this is to simply run a full build before running `clang-tidy`. A build target which runs `clang-tidy` is provided. To use:

    make run-clang-tidy

Note: `clang-tidy` may make invalid or overly verbose changes to the source code. It is recommended to first commit your changes, then run `clang-tidy` and review its recommended changes before amending them to your commit.

Note: the `clang-tidy` target uses the `run-clang-tidy.py` script provided with LLVM, which may depend on `PyYAML`. The target also depends on `jq`, which is used to filter portions of the `compile_commands.json` file.

## License headers

Source files must carry the NVIDIA SPDX license header. For C/C++/CUDA/Java this is a leading comment block:

    /*
     * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
     * SPDX-License-Identifier: Apache-2.0
     */

(`#`-style comments are used for Python, shell, and CMake.) The header — including its copyright year range — is added and kept up to date automatically by a [pre-commit](https://pre-commit.com/) hook. Install it once:

    pip install pre-commit
    pre-commit install

After that, every commit checks the files you changed; any missing or out-of-date header is fixed in place (re-stage and commit). Legacy `HEAVY.AI`/`OmniSci`/`MapD` headers are replaced with the NVIDIA header when a file is touched. Enforcement is **going forward** — only files you add or modify are checked, so the existing tree is not rewritten en masse. The same check runs in CI (the `license-headers` job in `.github/workflows/pr-required-checks.yml`).

Vendored third-party code keeps its upstream license and is exempt (`ThirdParty/`, `Archive/`, `ODBC/`, and the individual third-party files listed in [`.pre-commit-config.yaml`](.pre-commit-config.yaml)).

To run the check manually against the files changed on your branch:

    scripts/check_license_headers.sh            # compares against origin/master

# Dependencies

The maintained dependency scripts support Ubuntu 22.04 and Rocky
Linux 8.x. They install under `/usr/local/mapd-deps` and generate
`/usr/local/mapd-deps/mapd-deps.sh`, which must be sourced before configuring
HeavyDB:

    source /usr/local/mapd-deps/mapd-deps.sh

To install a prepared dependency archive supplied with the developer or CI
build context, use:

    sudo scripts/mapd-deps-prebuilt.sh --static

Ubuntu also supports `--shared`; Rocky Linux supports static dependencies.
Pass `--enable` to install a profile entry that loads the generated
environment in future login shells.

To build dependency archives from source:

    cd scripts
    ./mapd-deps-ubuntu.sh --static --compress

or, on Rocky Linux:

    cd scripts
    ./mapd-deps-rockylinux.sh --compress

GPU builds also require a compatible NVIDIA driver and CUDA toolkit. Follow
the [NVIDIA CUDA installation instructions](https://developer.nvidia.com/cuda-downloads)
for the target distribution. For complete dependency-script details, see the
[dependency quickstart](docs/source/quickstart/deps.rst).


# Security
> [!WARNING]
> **Do not report security vulnerabilities through public GitHub issues!**

NVIDIA takes security seriously. If you discover a vulnerability in heavydb, **DO NOT open a public issue**. Use one of the private reporting channels described in [SECURITY.md](https://github.com/heavyai/heavydb/blob/master/SECURITY.md).

# Support
Join the [HeavyAI GitHub Discussions](https://github.com/orgs/heavyai/discussions) to ask questions, share feedback, and report issues. HeavyAI maintainers review issues, discussions, and pull requests on a best effort basis without guaranteed response timelines.
  

