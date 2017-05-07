MapD Core
=========

MapD Core is an in-memory, column store, SQL relational database that was designed from the ground up to run on GPUs.

# Table of Contents

- [Links](#links)
- [License](#license)
- [Contributing](#contributing)
- [Building](#building)
- [Testing](#testing)
- [Using](#using)
- [Code Style](#code-style)
- [Dependencies](#dependencies)

# Links

- [Documentation](https://www.mapd.com/docs/)
- [Release Notes](https://www.mapd.com/docs/latest/release-notes/platform/)
- [Community Forum](https://community.mapd.com)
- [MapD Homepage](https://www.mapd.com)
- [MapD Blog](https://www.mapd.com/blog/)

# License

This project is licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).

The repository includes a number of third party packages provided under separate licenses. Details about these packages and their respective licenses is at [ThirdParty/licenses/index.md](ThirdParty/licenses/index.md).

The standard build process for this project downloads the Community Edition of the MapD Immerse visual analytics client. This version of MapD Immerse is governed by a separate license agreement, included in the file `EULA-CE.txt`, and may only be used for non-commercial purposes.

# Contributing

In order to clarify the intellectual property license granted with Contributions from any person or entity, MapD must have a Contributor License Agreement ("CLA") on file that has been signed by each Contributor, indicating agreement to the [Contributor License Agreement](CLA.txt). If you have not already done so, please complete and sign, then scan and email a pdf file of this Agreement to [contributors@mapd.com](mailto:contributors@mapd.com). Please read the agreement carefully before signing and keep a copy for your records.

# Building

If this is your first time building MapD Core, install the dependencies mentioned in the [Dependencies](#dependencies) section below.

MapD uses CMake for its build system.

    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=debug ..
    make -j 4

The following `cmake`/`ccmake` options can enable/disable different features:

- `-DCMAKE_BUILD_TYPE=release` build type and compiler options to use. Options: `Debug`, `Release`, `RelWithDebInfo`, `MinSizeRel`, and unset.
- `-DENABLE_CUDA=off` disable CUDA. Default `on`.
- `-DMAPD_IMMERSE_DOWNLOAD=on` download the latest master build of Immerse / `mapd2-frontend`. Default `on`.
- `-DMAPD_DOCS_DOWNLOAD=on` download the latest master build of the documentation / `docs.mapd.com`. Default `off`. Note: this is a >50MB download.
- `-DPREFER_STATIC_LIBS=on` static link dependencies, if available. Default `off`.

# Testing

MapD Core uses [Google Test](https://github.com/google/googletest) as its main testing framework. Tests reside under the [Tests](Tests) directory.

The `sanity_tests` target runs the most common tests. If using Makefiles to build, the tests may be run using:

    make sanity_tests

## AddressSanitizer

[AddressSanitizer](https://github.com/google/sanitizers/wiki/AddressSanitizer) can be activated by setting the `ENABLE_ASAN` CMake flag in a fresh build directory. At this time CUDA must also be disabled, and Calcite must be run in standalone/server mode. In an empty build directory run CMake and compile:

    mkdir build && cd build
    cmake -DENABLE_ASAN=on -DENABLE_CUDA=off ..
    make -j 4

In a separate terminal start Calcite in standalone mode from the build directory:

    java -jar bin/mapd-1.0-SNAPSHOT-jar-with-dependencies.jar --data=Tests/tmp

Finally run the tests:

    export ASAN_OPTIONS=alloc_dealloc_mismatch=0:handle_segv=0
    make sanity_tests

## ThreadSanitizer

[ThreadSanitizer](https://github.com/google/sanitizers/wiki/ThreadSanitizerCppManual) can be activated by setting the `ENABLE_TSAN` CMake flag in a fresh build directory. At this time CUDA must also be disabled, and Calcite must be run in standalone/server mode. In an empty build directory run CMake and compile:

    mkdir build && cd build
    cmake -DENABLE_TSAN=on -DENABLE_CUDA=off ..
    make -j 4

In a separate terminal start Calcite in standalone mode from the build directory:

    java -jar bin/mapd-1.0-SNAPSHOT-jar-with-dependencies.jar --data=Tests/tmp

Finally run the tests:

    make sanity_tests

# Using

The [`startmapd`](startmapd) wrapper script may be used to start MapD Core in a testing environment. This script performs the following tasks:

- initializes the `data` storage directory via `initdb`, if required
- starts the main MapD Core server, `mapd_server`
- starts the MapD Core web server, `mapd_web_server`, for serving MapD Immerse
- offers to download and import a sample dataset, using the `insert_sample_data` script
- attempts to open MapD Immerse in your web browser

Assuming you are in the `build` directory, and it is a subdirectory of the `mapd-core` repository, `startmapd` may be run by:

    ../startmapd

## Starting Manually

It is assumed that the following commands are run from inside the `build` directory.

Initialize the `data` storage directory. This command only needs to be run once.

    mkdir data && ./bin/initdb data

Start the MapD Core server:

    ./bin/mapd_server

In a new terminal, start the MapD Core web server:

    ./bin/mapd_web_server

If desired, insert a sample dataset by running the `insert_sample_data` script in a new terminal:

    ../insert_sample_data

You can now start using the database. The `mapdql` utility may be used to interact with the database from the command line:

    ./bin/mapdql -p HyperInteractive

where `HyperInteractive` is the default password. The default user `mapd` is assumed if not provided.

You can also interact with the database using the web-based MapD Immerse frontend by visiting the web server's default port of `9092`:

    http://localhost:9092

Note: usage of MapD Immerse is governed by a separate license agreement, provided under `EULA-CE.txt`. The version bundled with this project may only be used for non-commercial purposes.

# Code Style

A [`.clang-format`](http://clang.llvm.org/docs/ClangFormat.html) style configuration, based on the Chromium style guide, is provided at the top level of the repository. Please format your code using a recent version (3.8+) of ClangFormat before submitting.

To use:

    clang-format -i File.cpp

Contributed code should compile without generating warnings by recent compilers (gcc 4.9, gcc 5.3, clang 3.8) on most Linux distributions. Changes to the code should follow the [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines).

# Dependencies

MapD has the following dependencies:

- [CMake 3.3+](https://cmake.org/)
- [LLVM 3.8](http://llvm.org/)
- [GCC 4.9+](http://gcc.gnu.org/): not required if building with Clang
- [Boost 1.5.7+](http://www.boost.org/)
- [Thrift 0.9.2+](https://thrift.apache.org/)
- [bison++](https://code.google.com/p/flexpp-bisonpp/)
- [Google glog](https://github.com/google/glog)
- [Go 1.5+](https://golang.org/)
- [OpenJDK](http://openjdk.java.net/)
- [CUDA 7.0+](http://nvidia.com/cuda)
- [gperftools](https://github.com/gperftools/gperftools)
- [gdal](http://gdal.org/)

Dependencies for `mapd_web_server` and other Go utils are in [`ThirdParty/go`](ThirdParty/go). See [`ThirdParty/go/src/mapd/vendor/README.md`](ThirdParty/go/src/mapd/vendor/README.md) for instructions on how to add new deps.

## CentOS 7

MapD Core requires a number of dependencies which are not provided in the common CentOS/RHEL package repositories. The script [scripts/mapd-deps-linux.sh](scripts/mapd-deps-linux.sh) is provided to automatically build and install these dependencies. A prebuilt package containing these dependencies is also provided for CentOS 7 (x86_64).

First install the basic build tools:

    sudo yum groupinstall -y "Development Tools"
    sudo yum install -y \
        zlib-devel \
        epel-release \
        libssh \
        openssl-devel \
        ncurses-devel \
        git \
        maven \
        java-1.8.0-openjdk-devel \
        java-1.8.0-openjdk-headless \
        gperftools \
        gperftools-devel \
        gperftools-libs \
        environment-modules

Next download and install the prebuilt dependencies:

    curl -OJ https://internal-dependencies.mapd.com/mapd-deps/deploy.sh
    sudo bash deploy.sh

These dependencies will be installed to a directory under `/usr/local/mapd-deps`. The `deploy.sh` script also installs [Environment Modules](http://modules.sf.net) in order to simplify managing the required environment variables. Log out and log back in after running the `deploy.sh` script in order to active Environment Modules command, `module`.

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

[scripts/mapd-deps-linux.sh](scripts/mapd-deps-linux.sh) generates two files with the appropriate environment variables: `mapd-deps-<date>.sh` (for sourcing from your shell config) and `mapd-deps-<date>.modulefile` (for use with [Environment Modules](http://modules.sf.net), yum package `environment-modules`). These files are placed in mapd-deps install directory, usually `/usr/local/mapd-deps/<date>`. Either of these may be used to configure your environment: the `.sh` may be sourced in your shell config; the `.modulefile` needs to be moved to the modulespath.

The Java server lib directory containing `libjvm.so` must also be added to your `LD_LIBRARY_PATH`. Add one of the following to your shell config:

    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/jvm/jre/lib/amd64/server
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/jvm/java-1.8.0/jre/lib/amd64/server

## macOS

[scripts/mapd-deps-osx.sh](scripts/mapd-deps-osx.sh) is provided that will automatically install and/or update [Homebrew](http://brew.sh/) and use that to install all dependencies. Please make sure macOS is completely update to date and Xcode is installed before running. Xcode can be installed from the App Store.

### CUDA

`mapd-deps-osx.sh` will automatically install CUDA via Homebrew and add the correct environment variables to `~/.bash_profile`.

### Java

`mapd-deps-osx.sh` will automatically install Java and Maven via Homebrew and add the correct environment variables to `~/.bash_profile`.

## Ubuntu 16.04, 16.10

Most build dependencies required by MapD Core are available via APT. Thrift, Blosc, and Folly must be built manually. The following will install all required dependencies and build the ones not available in the APT repositories.

    sudo apt update
    sudo apt install -y \
        build-essential \
        cmake \
        cmake-curses-gui \
        git \
        clang \
        clang-format \
        llvm \
        llvm-dev \
        libboost-all-dev \
        libgoogle-glog-dev \
        golang \
        libssl-dev \
        libevent-dev \
        default-jre \
        default-jre-headless \
        default-jdk \
        default-jdk-headless \
        maven \
        libncurses5-dev \
        binutils-dev \
        google-perftools \
        libdouble-conversion-dev \
        libevent-dev \
        libgdal-dev \
        libgflags-dev \
        libgoogle-perftools-dev \
        libiberty-dev \
        libjemalloc-dev \
        liblz4-dev \
        liblzma-dev \
        libsnappy-dev \
        zlib1g-dev \
        autoconf \
        autoconf-archive

    sudo apt build-dep -y thrift-compiler
    VERS=0.10.0
    wget http://apache.claz.org/thrift/$VERS/thrift-$VERS.tar.gz
    tar xvf thrift-$VERS.tar.gz
    pushd thrift-$VERS
    ./configure \
        --with-lua=no \
        --with-python=no \
        --with-php=no \
        --with-ruby=no \
        --prefix=/usr/local/mapd-deps
    make -j $(nproc)
    sudo make install
    popd

    VERS=1.11.3
    wget --continue https://github.com/Blosc/c-blosc/archive/v$VERS.tar.gz
    tar xvf v$VERS.tar.gz
    BDIR="c-blosc-$VERS/build"
    rm -rf "$BDIR"
    mkdir -p "$BDIR"
    pushd "$BDIR"
    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local/mapd-deps \
        -DBUILD_BENCHMARKS=off \
        -DBUILD_TESTS=off \
        -DPREFER_EXTERNAL_SNAPPY=off \
        -DPREFER_EXTERNAL_ZLIB=off \
        -DPREFER_EXTERNAL_ZSTD=off \
        ..
    make -j $(nproc)
    sudo make install
    popd

    VERS=2017.04.10.00
    wget --continue https://github.com/facebook/folly/archive/v$VERS.tar.gz
    tar xvf v$VERS.tar.gz
    pushd folly-$VERS/folly
    /usr/bin/autoreconf -ivf
    ./configure --prefix=/usr/local/mapd-deps
    make -j $(nproc)
    sudo make install
    popd

    VERS=1.21-45
    wget --continue https://github.com/jarro2783/bisonpp/archive/$VERS.tar.gz
    tar xvf $VERS.tar.gz
    pushd bisonpp-$VERS
    ./configure
    make -j $(nproc)
    sudo make install
    popd

### CUDA

It is preferred, but not necessary, to install CUDA and the NVIDIA drivers using the .deb using the [instructions provided by NVIDIA](https://developer.nvidia.com/cuda-downloads). The `deb (network)` method (preferred) will ensure you always have the latest stable drivers, while the `deb (local)` method allows you to install does not require Internet access.

Be sure to reboot after installing in order to activate the NVIDIA drivers.

### Environment Variables

The CUDA, Java, and mapd-deps `lib` directories need to be added to `LD_LIBRARY_PATH`; the CUDA and mapd-deps `bin` directories need to be added to `PATH`. The easiest way to do so is by creating a new file named `/etc/profile.d/mapd-deps.sh` containing the following:

    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    LD_LIBRARY_PATH=/usr/lib/jvm/default-java/jre/lib/amd64/server:$LD_LIBRARY_PATH
    LD_LIBRARY_PATH=/usr/local/mapd-deps/lib:$LD_LIBRARY_PATH
    LD_LIBRARY_PATH=/usr/local/mapd-deps/lib64:$LD_LIBRARY_PATH

    PATH=/usr/local/cuda/bin:$PATH
    PATH=/usr/local/mapd-deps/bin:$PATH

    export LD_LIBRARY_PATH PATH

## Arch

The following uses [yaourt](https://wiki.archlinux.org/index.php/Yaourt) to install packages from the [Arch User Repository](https://wiki.archlinux.org/index.php/Arch_User_Repository).

    yaourt -S \
        git \
        cmake \
        boost \
        google-glog \
        extra/jdk8-openjdk \
        clang \
        llvm \
        thrift \
        go \

    VERS=1.21-45
    wget --continue https://github.com/jarro2783/bisonpp/archive/$VERS.tar.gz
    tar xvf $VERS.tar.gz
    pushd bisonpp-$VERS
    ./configure
    make -j $(nproc)
    sudo make install
    popd

### CUDA

CUDA and the NVIDIA drivers may be installed using the following.

    yaourt -S \
        linux-headers \
        cuda \
        nvidia

Be sure to reboot after installing in order to activate the NVIDIA drivers.
