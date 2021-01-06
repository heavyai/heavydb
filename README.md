OmniSciDB (formerly MapD Core)
==============================

OmniSciDB is an open source SQL-based, relational, columnar database engine that leverages the full performance and parallelism of modern hardware (both CPUs and GPUs) to enable querying of multi-billion row datasets in milliseconds, without the need for indexing, pre-aggregation, or downsampling.  OmniSciDB can be run on hybrid CPU/GPU systems (Nvidia GPUs are currently supported), as well as on CPU-only systems featuring X86, Power, and ARM (experimental support) architectures. To achieve maximum performance, OmniSciDB features multi-tiered caching of data between storage, CPU memory, and GPU memory, and an innovative Just-In-Time (JIT) query compilation framework.

For usage info, see the [product documentation](https://docs.omnisci.com/latest/), and for more details about the system's internal architecture, check out the [developer documentation](https://omnisci.github.io/omniscidb/). Further technical discussion can be found on the [OmniSci Community Forum](https://community.omnisci.com).

The repository includes a number of third party packages provided under separate licenses. Details about these packages and their respective licenses is at [ThirdParty/licenses/index.md](ThirdParty/licenses/index.md).

# Downloads and Installation Instructions

OmniSci provides pre-built binaries for Linux for stable releases of the project:

| Distro | Package type | CPU/GPU | Repository | Docs |
| --- | --- | --- | --- | --- |
| CentOS | RPM | CPU | https://releases.omnisci.com/os/yum/stable/cpu | https://docs.omnisci.com/latest/4_centos7-yum-cpu-os-recipe.html |
| CentOS | RPM | GPU | https://releases.omnisci.com/os/yum/stable/cuda | https://docs.omnisci.com/latest/4_centos7-yum-gpu-os-recipe.html |
| Ubuntu | DEB | CPU | deb https://releases.omnisci.com/os/apt/ stable cpu | https://docs.omnisci.com/latest/4_ubuntu-apt-cpu-os-recipe.html |
| Ubuntu | DEB | GPU | deb https://releases.omnisci.com/os/apt/ stable cuda | https://docs.omnisci.com/latest/4_ubuntu-apt-gpu-os-recipe.html |
| * | tarball | CPU | https://releases.omnisci.com/os/tar/omnisci-os-latest-Linux-x86_64-cpu.tar.gz |  |
| * | tarball | GPU | https://releases.omnisci.com/os/tar/omnisci-os-latest-Linux-x86_64-cuda.tar.gz |  |

***

# Developing OmniSciDB: Table of Contents

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

- [Developer Documentation](https://omnisci.github.io/omniscidb/)
- [Doxygen-generated Documentation](http://doxygen.omnisci.com/)
- [Product Documentation](https://docs.omnisci.com/latest/)
- [Release Notes](https://docs.omnisci.com/latest/7_0_release.html)
- [Community Forum](https://community.omnisci.com)
- [OmniSci Homepage](https://www.omnisci.com)
- [OmniSci Blog](https://www.omnisci.com/blog/)
- [OmniSci Downloads](https://www.omnisci.com/platform/downloads/)

# License

This project is licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).

The repository includes a number of third party packages provided under separate licenses. Details about these packages and their respective licenses is at [ThirdParty/licenses/index.md](ThirdParty/licenses/index.md).

# Contributing

In order to clarify the intellectual property license granted with Contributions from any person or entity, OmniSci must have a Contributor License Agreement ("CLA") on file that has been signed by each Contributor, indicating agreement to the [Contributor License Agreement](CLA.txt). After making a pull request, a bot will notify you if a signed CLA is required and provide instructions for how to sign it. Please read the agreement carefully before signing and keep a copy for your records.

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

# Testing

OmniSciDB uses [Google Test](https://github.com/google/googletest) as its main testing framework. Tests reside under the [Tests](Tests) directory.

The `sanity_tests` target runs the most common tests. If using Makefiles to build, the tests may be run using:

    make sanity_tests

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

# Generating Packages

OmniSciDB uses [CPack](https://cmake.org/cmake/help/latest/manual/cpack.1.html) to generate packages for distribution. Packages generated on CentOS with static linking enabled can be used on most other recent Linux distributions.

To generate packages on CentOS (assuming starting from top level of the omniscidb repository):

    mkdir build-package && cd build-package
    cmake -DPREFER_STATIC_LIBS=on -DCMAKE_BUILD_TYPE=release ..
    make -j 4
    cpack -G TGZ

The first command creates a fresh build directory, to ensure there is nothing left over from a previous build.

The second command configures the build to prefer linking to the dependencies' static libraries instead of the (default) shared libraries, and to build using CMake's `release` configuration (enables compiler optimizations). Linking to the static versions of the libraries libraries reduces the number of dependencies that must be installed on target systems.

The last command generates a `.tar.gz` package. The `TGZ` can be replaced with, for example, `RPM` or `DEB` to generate a `.rpm` or `.deb`, respectively.

# Using

The [`startomnisci`](startomnisci) wrapper script may be used to start OmniSciDB in a testing environment. This script performs the following tasks:

- initializes the `data` storage directory via `initdb`, if required
- starts the main OmniSciDB server, `omnisci_server`
- offers to download and import a sample dataset, using the `insert_sample_data` script

Assuming you are in the `build` directory, and it is a subdirectory of the `omniscidb` repository, `startomnisci` may be run by:

    ../startomnisci

## Starting Manually

It is assumed that the following commands are run from inside the `build` directory.

Initialize the `data` storage directory. This command only needs to be run once.

    mkdir data && ./bin/initdb data

Start the OmniSciDB server:

    ./bin/omnisci_server

If desired, insert a sample dataset by running the `insert_sample_data` script in a new terminal:

    ../insert_sample_data

You can now start using the database. The `omnisql` utility may be used to interact with the database from the command line:

    ./bin/omnisql -p HyperInteractive

where `HyperInteractive` is the default password. The default user `admin` is assumed if not provided.

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

# Dependencies

OmniSciDB has the following dependencies:

| Package | Min Version | Required |
| ------- | ----------- | -------- |
| [CMake](https://cmake.org/) | 3.16 | yes |
| [LLVM](http://llvm.org/) | 9.0 | yes |
| [GCC](http://gcc.gnu.org/) | 8.4.0 | no, if building with clang |
| [Go](https://golang.org/) | 1.12 | yes |
| [Boost](http://www.boost.org/) | 1.72.0 | yes |
| [OpenJDK](http://openjdk.java.net/) | 1.7 | yes |
| [CUDA](http://nvidia.com/cuda) | 11.0 | yes, if compiling with GPU support |
| [gperftools](https://github.com/gperftools/gperftools) | | yes |
| [gdal](http://gdal.org/) | 2.4.2 | yes |
| [Arrow](https://arrow.apache.org/) | 1.0.0 | yes |

## CentOS 7

OmniSciDB requires a number of dependencies which are not provided in the common CentOS/RHEL package repositories. A prebuilt package containing all these dependencies is provided for CentOS 7 (x86_64).

Use the [scripts/mapd-deps-prebuilt.sh](scripts/mapd-deps-prebuilt.sh) build script to install prebuilt dependencies.

These dependencies will be installed to a directory under `/usr/local/mapd-deps`. The `mapd-deps-prebuilt.sh` script also installs [Environment Modules](http://modules.sf.net) in order to simplify managing the required environment variables. Log out and log back in after running the `mapd-deps-prebuilt.sh` script in order to active Environment Modules command, `module`.

The `mapd-deps` environment module is disabled by default. To activate for your current session, run:

    module load mapd-deps

To disable the `mapd-deps` module:

    module unload mapd-deps

WARNING: The `mapd-deps` package contains newer versions of packages such as GCC and ncurses which might not be compatible with the rest of your environment. Make sure to disable the `mapd-deps` module before compiling other packages.

Instructions for installing CUDA are below.

### CUDA

It is preferred, but not necessary, to install CUDA and the NVIDIA drivers using the .rpm using the [instructions provided by NVIDIA](https://developer.nvidia.com/cuda-downloads). The `rpm (network)` method (preferred) will ensure you always have the latest stable drivers, while the `rpm (local)` method allows you to install does not require Internet access.

The .rpm method requires DKMS to be installed, which is available from the [Extra Packages for Enterprise Linux](https://fedoraproject.org/wiki/EPEL) repository:

    sudo yum install epel-release

Be sure to reboot after installing in order to activate the NVIDIA drivers.

### Environment Variables

The `mapd-deps-prebuilt.sh` script includes two files with the appropriate environment variables: `mapd-deps-<date>.sh` (for sourcing from your shell config) and `mapd-deps-<date>.modulefile` (for use with [Environment Modules](http://modules.sf.net), yum package `environment-modules`). These files are placed in mapd-deps install directory, usually `/usr/local/mapd-deps/<date>`. Either of these may be used to configure your environment: the `.sh` may be sourced in your shell config; the `.modulefile` needs to be moved to the modulespath.

### Building Dependencies

The [scripts/mapd-deps-centos.sh](scripts/mapd-deps-centos.sh) script is used to build the dependencies. Modify this script and run if you would like to change dependency versions or to build on alternative CPU architectures.

    cd scripts
    module unload mapd-deps
    ./mapd-deps-centos.sh --compress

## macOS

[scripts/mapd-deps-osx.sh](scripts/mapd-deps-osx.sh) is provided that will automatically install and/or update [Homebrew](http://brew.sh/) and use that to install all dependencies. Please make sure macOS is completely up to date and Xcode is installed before running. Xcode can be installed from the App Store.

### CUDA

`mapd-deps-osx.sh` will automatically install CUDA via Homebrew and add the correct environment variables to `~/.bash_profile`.

### Java

`mapd-deps-osx.sh` will automatically install Java and Maven via Homebrew and add the correct environment variables to `~/.bash_profile`.

## Ubuntu

Most build dependencies required by OmniSciDB are available via APT. Certain dependencies such as Thrift, Blosc, and Folly must be built as they either do not exist in the default repositories or have outdated versions. A prebuilt package containing all these dependencies is provided for Ubuntu 18.04 (x86_64). The dependencies will be installed to `/usr/local/mapd-deps/` by default; see the Environment Variables section below for how to add these dependencies to your environment.

### Ubuntu 16.04

OmniSciDB requires a newer version of Boost than the version which is provided by Ubuntu 16.04. The [scripts/mapd-deps-ubuntu1604.sh](scripts/mapd-deps-ubuntu1604.sh) build script will compile and install a newer version of Boost into the `/usr/local/mapd-deps/` directory.

### Ubuntu 18.04

Use the [scripts/mapd-deps-prebuilt.sh](scripts/mapd-deps-prebuilt.sh) build script to install prebuilt dependencies.

These dependencies will be installed to a directory under `/usr/local/mapd-deps`. The `mapd-deps-prebuilt.sh` script above will generate a script named `mapd-deps.sh` containing the environment variables which need to be set. Simply source this file in your current session (or symlink it to `/etc/profile.d/mapd-deps.sh`) in order to activate it:

    source /usr/local/mapd-deps/mapd-deps.sh

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

## Arch

[scripts/mapd-deps-arch.sh](scripts/mapd-deps-arch.sh) is provided that will use [yay](https://aur.archlinux.org/packages/yay/) to install packages from the [Arch User Repository](https://wiki.archlinux.org/index.php/Arch_User_Repository) and a custom PKGBUILD script for Apache Arrow. If you don't have yay yet, install it first: https://github.com/Jguer/yay#installation

Note: Apache Arrow, while available in the AUR, requires a few custom build flags in order to be used with Core. A custom PKGBUILD for it is included.

### CUDA

CUDA and the NVIDIA drivers may be installed using the following.

    yay -S \
        linux-headers \
        cuda \
        nvidia

Be sure to reboot after installing in order to activate the NVIDIA drivers.

### Environment Variables

The `cuda` package should set up the environment variables required to use CUDA. If you receive errors saying `nvcc` is not found, then CUDA `bin` directories need to be added to `PATH`: the easiest way to do so is by creating a new file named `/etc/profile.d/mapd-deps.sh` containing the following:

    PATH=/opt/cuda/bin:$PATH
    export PATH
