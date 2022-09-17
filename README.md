OmniSciDB
==============================

OmniSciDB is an open source SQL-based, relational, columnar database engine that leverages the full performance and parallelism of modern hardware (both CPUs and GPUs) to enable querying of multi-billion row datasets in milliseconds, without the need for indexing, pre-aggregation, or downsampling.  OmniSciDB can be run on hybrid CPU/GPU systems (Nvidia GPUs are currently supported), as well as on CPU-only systems featuring X86, Power, and ARM (experimental support) architectures. To achieve maximum performance, OmniSciDB features multi-tiered caching of data between storage, CPU memory, and GPU memory, and an innovative Just-In-Time (JIT) query compilation framework.

# Downloads and Installation Instructions

OmniSci can be installed as a part of [HDK](https://github.com/conda-forge/hdk-feedstock).

# Developing OmniSciDB: Table of Contents

- [License](#license)
- [Building](#building)
- [Code Style](#code-style)

# License

This project is licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).

The repository includes a number of third party packages provided under separate licenses. Details about these packages and their respective licenses is at [ThirdParty/licenses/index.md](ThirdParty/licenses/index.md).

# Building

If this is your first time building OmniSciDB, install the dependencies mentioned in the [Dependencies](#dependencies) section below.

OmniSciDB uses CMake for its build system.

    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=debug ..
    make -j 4

The following `cmake`/`ccmake` options can enable/disable different features:

- `-DCMAKE_BUILD_TYPE=release` - Build type and compiler options to use.
                                 Options are `Debug`, `Release`, `RelWithDebInfo`, `MinSizeRel`, and unset.
- `-DENABLE_ASAN=off` - Enable address sanitizer. Default is `off`.
- `-DENABLE_AWS_S3=on` - Enable AWS S3 support, if available. Default is `on`.
- `-DENABLE_CUDA=off` - Disable CUDA. Default is `on`.
- `-DENABLE_CUDA_KERNEL_DEBUG=off` - Enable debugging symbols for CUDA kernels. Will dramatically reduce kernel performance. Default is `off`.
- `-DENABLE_DECODERS_BOUNDS_CHECKING=off` - Enable bounds checking for column decoding. Default is `off`.
- `-DENABLE_FOLLY=on` - Use Folly. Default is `on`.
- `-DENABLE_IWYU=off` - Enable include-what-you-use. Default is `off`.
- `-DENABLE_JIT_DEBUG=off` - Enable debugging symbols for the JIT. Default is `off`.
- `-DENABLE_PROFILER=off` - Enable google perftools. Default is `off`.
- `-DENABLE_STANDALONE_CALCITE=off` - Require standalone Calcite server. Default is `off`.
- `-DENABLE_TESTS=on` - Build unit tests. Default is `on`.
- `-DENABLE_TSAN=off` - Enable thread sanitizer. Default is `off`.
- `-DENABLE_CODE_COVERAGE=off` - Enable code coverage symbols (clang only). Default is `off`.
- `-DMAPD_DOCS_DOWNLOAD=on` - Download the latest master build of the documentation / `docs.mapd.com`. Default is `off`.
                              **Note:** this is a >50MB download.
- `-DPREFER_STATIC_LIBS=off` - Static link dependencies, if available. Default is `off`.

# Building in Conda environment

This is default and recommended way to build the project

    conda env create -f scripts/mapd-deps-conda-dev-env.yml --force
    conda activate omnisci-dev
    bash scripts/conda/build-install-all.sh

By default, tests are not included in the build. To build (only) tests use:

    RUN_TESTS=1 bash scripts/conda/build-install-all.sh

To build & run tests in a conda environment launch:

    RUN_TESTS=2 bash scripts/conda/build-install-all.sh

For debug build use (default is Release):

    CMAKE_BUILD_TYPE=Debug bash scripts/conda/build-install-all.sh

# Testing

OmniSciDB uses [Google Test](https://github.com/google/googletest) as its main testing framework. Tests reside under the [Tests](Tests) directory.

The `sanity_tests` target runs the most common tests. If using Makefiles to build, the tests may be run using:

    make sanity_tests

Unit tests can be run with (requires tests to be enabled):

`cd build && make all_tests`

## AddressSanitizer

[AddressSanitizer](https://github.com/google/sanitizers/wiki/AddressSanitizer) can be activated by setting the `ENABLE_ASAN` CMake flag in a fresh build directory. At this time CUDA must also be disabled. In an empty build directory run CMake and compile:

    mkdir build && cd build
    cmake -DENABLE_ASAN=on -DENABLE_CUDA=off ..
    make -j 4

Finally run the tests:

    export ASAN_OPTIONS=alloc_dealloc_mismatch=0:handle_segv=0
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


## Docker

To build a docker image use `docker/dev/Dockerfile`. For CUDA build use the following file

    cat docker/dev/Dockerfile docker/dev/Dockerfile.cuda >Dockerfile

Use the following command to build the image `omni-build`:

    docker build -t omni-build . # replace . with the path to the Dockerfile

Assuming the sources are in the `$(pwd)` directory run a container using the following command:

    docker run -id --name omni-build --network host -v $(pwd):/_work omni-build:latest

Add `--device /dev/nvidia-caps/nvidia-cap2:/dev/nvidia-caps/nvidia-cap2 --device /dev/nvidia-caps/nvidia-cap1:/dev/nvidia-caps/nvidia-cap1 --device /dev/nvidia-modeset:/dev/nvidia-modeset --device /dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools --device /dev/nvidia-uvm:/dev/nvidia-uvm --device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl` options to run CUDA tests.

To build executables use the following commands in the container:

    cmake -B/_work/build -S/_work -DENABLE_TESTS=on # add more options if needed
    cmake --build /_work/build/ --parallel 2

To install use the following command in the container:

    cmake --install /_work/build/ --parallel 2


## Ubuntu

Most build dependencies required by OmniSciDB are available via APT. Certain dependencies such as Folly must be built as they either do not exist in the default repositories or have outdated versions. A prebuilt package containing all these dependencies is provided for Ubuntu 18.04 (x86_64). The dependencies will be installed to `/usr/local/mapd-deps/` by default; see the Environment Variables section below for how to add these dependencies to your environment.


### Environment Variables

The CUDA and mapd-deps `lib` directories need to be added to `LD_LIBRARY_PATH`; the CUDA and mapd-deps `bin` directories need to be added to `PATH`. The `mapd-deps-ubuntu.sh` and `mapd-deps-prebuilt.sh` scripts will generate a script named `mapd-deps.sh` containing the environment variables which need to be set. Simply source this file in your current session (or symlink it to `/etc/profile.d/mapd-deps.sh`) in order to activate it:

    source /usr/local/mapd-deps/mapd-deps.sh

### CUDA

Recent versions of Ubuntu provide the NVIDIA CUDA Toolkit and drivers in the standard repositories. To install:

    sudo apt install -y \
        nvidia-cuda-toolkit

Be sure to reboot after installing in order to activate the NVIDIA drivers.

### Building Dependencies

The [scripts/mapd-deps-ubuntu.sh](scripts/mapd-deps-ubuntu.sh) and [scripts/mapd-deps-ubuntu1604.sh](scripts/mapd-deps-ubuntu1604.sh) scripts are used to build the dependencies for Ubuntu 18.04 and 16.04, respectively. The scripts will install all required dependencies (except CUDA) and build the dependencies which require it. Modify this script and run if you would like to change dependency versions or to build on alternative CPU architectures.

    cd scripts
    ./mapd-deps-ubuntu.sh --compress


# Code Style

Contributed code should compile without generating warnings by recent compilers on most Linux distributions. Changes to the code should follow the [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines).

## clang-format

A [`.clang-format`](https://clang.llvm.org/docs/ClangFormat.html) style configuration, based on the Chromium style guide, is provided at the top level of the repository. Please format your code using a recent version (8.0+ preferred) of ClangFormat before submitting.

To use:

    clang-format -i File.cpp

## clang-tidy

A [`.clang-tidy`](https://clang.llvm.org/extra/clang-tidy/) configuration is provided at the top level of the repository. Please lint your code using a recent version (6.0+ preferred) of clang-tidy before submitting.

`clang-tidy` requires all generated files to exist before running. The easiest way to accomplish this is to simply run a full build before running `clang-tidy`. A build target which runs `clang-tidy` is provided. To use:

    make run-clang-tidy

Note: `clang-tidy` may make invalid or overly verbose changes to the source code. It is recommended to first commit your changes, then run `clang-tidy` and review its recommended changes before amending them to your commit.

Note: the `clang-tidy` target uses the `run-clang-tidy.py` script provided with LLVM, which may depend on `PyYAML`. The target also depends on `jq`, which is used to filter portions of the `compile_commands.json` file.


