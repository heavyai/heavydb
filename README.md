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
- [Testing](#testing)
- [Using](#using)
- [Code Style](#code-style)
- [Dependencies](#dependencies)
- [Roadmap](ROADMAP.md)

# Links

- [Documentation](https://docs.nvidia.com/heavyai/)
- [Release Notes](https://docs.nvidia.com/heavyai/release-notes)


# License

This project is licensed under the [Apache License, Version 2.0](https://github.com/heavyai/heavydb/blob/master/LICENSE.txt).

The repository includes a number of third party packages provided under separate licenses. Details about these packages and their respective licenses is at [ThirdParty/licenses/index.md](ThirdParty/licenses/index.md).

# Contributing

Follow the instructions noted in [CONTRIBUTIONING.md](https://github.com/heavyai/heavydb/blob/master/CONTRIBUTING.md)

# Building

If this is your first time building HeavyDB, install the dependencies mentioned in the [Dependencies](#dependencies) section below.

HeavyDB uses CMake for its build system.

    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Debug ..
    make -j 4

HeavyDB requires CMake 3.25 or newer. CUDA and tests are enabled by default;
backend rendering is disabled by default. `PREFER_STATIC_LIBS` defaults to
`on` on RedHat-derived systems and `off` elsewhere. Optional subsystems can be
disabled automatically if their dependencies are unavailable.

See the [build quickstart](docs/source/quickstart/build.rst) for the maintained
list of common CMake options and defaults.

# Testing

HeavyDB uses [Google Test](https://github.com/google/googletest) as its main testing framework. Tests reside under the [Tests](Tests) directory.

The `sanity_tests` target runs the most common tests. If using Makefiles to build, the tests may be run using:

    make sanity_tests

## AddressSanitizer

[AddressSanitizer](https://github.com/google/sanitizers/wiki/AddressSanitizer) can be activated by setting the `ENABLE_ASAN` CMake flag in a fresh build directory. At this time CUDA must also be disabled. In an empty build directory run CMake and compile:

    mkdir build && cd build
    cmake -DENABLE_ASAN=on -DENABLE_CUDA=off ..
    make -j 4

Finally run the tests:

    export ASAN_OPTIONS=alloc_dealloc_mismatch=0:handle_segv=0:protect_shadow_gap=0
    make sanity_tests

## ThreadSanitizer

[ThreadSanitizer](https://github.com/google/sanitizers/wiki/ThreadSanitizerCppManual) can be activated by setting the `ENABLE_TSAN` CMake flag in a fresh build directory. At this time CUDA must also be disabled. In an empty build directory run CMake and compile:

    mkdir build && cd build
    cmake -DENABLE_TSAN=on -DENABLE_CUDA=off ..
    make -j 4

We use a TSAN suppressions file to ignore warnings in third party libraries. Source the suppressions file by adding it to your `TSAN_OPTIONS` env:

    export TSAN_OPTIONS="suppressions=/path/to/mapd/config/tsan.suppressions"

Finally run the tests:

    make sanity_tests

# Generating Packages

HeavyDB uses [CPack](https://cmake.org/cmake/help/latest/manual/cpack.1.html) to generate packages for distribution. Packages generated on CentOS with static linking enabled can be used on most other recent Linux distributions.

To generate packages on CentOS (assuming starting from top level of the HeavyDB repository):

    mkdir build-package && cd build-package
    cmake -DPREFER_STATIC_LIBS=on -DCMAKE_BUILD_TYPE=release ..
    make -j 4
    cpack -G TGZ

The first command creates a fresh build directory, to ensure there is nothing left over from a previous build.

The second command configures the build to prefer linking to the dependencies' static libraries instead of the (default) shared libraries, and to build using CMake's `release` configuration (enables compiler optimizations). Linking to the static versions of the libraries libraries reduces the number of dependencies that must be installed on target systems.

The last command generates a `.tar.gz` package. The `TGZ` can be replaced with, for example, `RPM` or `DEB` to generate a `.rpm` or `.deb`, respectively.

# Using

The [`startheavy`](startheavy) wrapper script may be used to start HeavyDB in a testing environment. This script performs the following tasks:

- initializes the `storage` directory via `initheavy`, if required
- starts the main HeavyDB server, `heavydb`
- starts `heavy_web_server` when its binary and a `frontend` directory are present
- starts HeavyIQ when its directory is present
- attempts to open the bundled frontend when one is available

Assuming you are in the `build` directory, and it is a subdirectory of the `HeavyDB` repository, `startheavy` may be run by:

    ../startheavy

## Starting Manually

It is assumed that the following commands are run from inside the `build` directory.

Initialize the `storage` directory. This command only needs to be run once.

    mkdir -p storage
    ./bin/initheavy -f --data storage

Start the HeavyDB server:

    ./bin/heavydb storage

If the optional web server was built, it can be started in another terminal:

    ./bin/heavy_web_server

You can now start using the database. The `heavysql` utility may be used to interact with the database from the command line:

    ./bin/heavysql -p HyperInteractive

where `HyperInteractive` is the default password. The default user `admin` is assumed if not provided.

When the bundled frontend is available, visit the web server's default port:

[http://localhost:6273](http://localhost:6273)

Note: the bundled web frontend, when present, is subject to separate license
terms from HeavyDB itself.

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
  
# License
Apache 2.0. See [LICENSE](https://github.com/heavyai/heavydb/blob/master/LICENSE.txt).

