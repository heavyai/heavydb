mapd2
=====

Central repo for the buildout of MapD V2

# Building MapD

MapD uses CMake for its build system. Only the `Unix Makefiles` and `Ninja` generators are regularly used. Others, such as `Xcode` might need some work.

    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=debug ..
    make -j

The following `cmake`/`ccmake` options can enable/disable different features:

- `-DENABLE_CALCITE=on` enable Calcite SQL parser. Default `off`.
- `-DENABLE_RENDERING=on` enable backend rendering. Default `off`.
- `-DENABLE_CUDA=off` disable CUDA (CUDA still required for build). Default `on`.
- `-DMAPD2_FRONTEND_DOWNLOAD=on` download the latest pre-built `mapd2-frontend`. Default `off`.
- `-DPREFER_STATIC_LIBS=on` static link dependencies, if available. Default `off`.

# Dependencies

MapD has the following dependencies:

- [CMake 3.3+](https://cmake.org/)
- [LLVM 3.5](http://llvm.org/)
- [GCC 4.9+](http://gcc.gnu.org/): not required if building with Clang
- [Boost 1.5.7+](http://www.boost.org/)
- [Thrift 0.9.2+](https://thrift.apache.org/)
- [bison++](https://code.google.com/p/flexpp-bisonpp/)
- [Google glog](https://github.com/google/glog)
- [Go 1.5+](https://golang.org/)
- [OpenJDK](http://openjdk.java.net/)
- [CUDA 7.0+](http://nvidia.com/cuda)
- [GLEW](http://glew.sourceforge.net/)
- [GLFW 3.1.2+](http://www.glfw.org/)
- [libpng](http://libpng.org/pub/png/libpng.html)
- [libcurl](https://curl.haxx.se/)
- [crypto++](https://www.cryptopp.com/)

//TODO(@vastcharade): add backend rendering deps
//TODO(@dwayneberry: add Calcite deps

Generating PDFs of the documentation requires `pandoc` and `texlive` (specifically `pdflatex`).

Dependencies for `mapd_web_server` and other Go utils are in [`ThirdParty/go`](ThirdParty/go). See [`ThirdParty/go/src/mapd/vendor/README.md`](ThirdParty/go/src/mapd/vendor/README.md) for instructions on how to add new deps.

## CentOS 6/7

[scripts/mapd-deps-linux.sh](scripts/mapd-deps-linux.sh) is provided that will automatically download, build, and install most dependencies. Before running this script, make sure you have the basic build tools installed:

    yum groupinstall "Development Tools"
    yum install git

Instructions for installing CUDA are below.

### CUDA

[scripts/cuda-autoinstall.sh](scripts/cuda_autoinstall.sh) will install CUDA and the drives via the runfile method. For the RPM method, first enable EPEL via `yum install epel-release` and then follow the instructions provided by Nvidia.

## Mac OS X 10.11

[scripts/mapd-deps-osx.sh](scripts/mapd-deps-osx.sh) is provided that will automatically install and/or update [Homebrew](http://brew.sh/) and use that to install all dependencies. Please make sure OSX is completely update to date and Xcode is installed before running.

Note: installing LLVM 3.5 view Homebrew requires some slight modifications to the build config arguments. [scripts/mapd-deps-osx.sh](scripts/mapd-deps-osx.sh) will run `brew edit llvm35`, which opens up the build config in your editor. Jump to the configure args (should be around lines 196-203) and add this line, if running on OS X 10.11:

    "--with-c-include-dirs=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.11.sdk/usr/include",

### CUDA

`mapd-deps-osx.sh` will automatically install CUDA via Homebrew and add the correct environment variables to `~/.bash_profile`.

## Ubuntu 15.10

Most build dependencies are available via APT. Thrift is the one exception and must be built by hand (Thrift 0.9.1 is available in APT, but that version is not supported by MapD).

    apt-get update
    apt-get install build-essential \
                    cmake \
                    cmake-curses-gui \
                    clang-3.5 \
                    clang-format-3.5 \
                    llvm-3.5 \
                    libboost-all-dev \
                    bison++ \
                    libgoogle-glog-dev \
                    golang \
                    libssl-dev \
                    libevent-dev \
                    libglew-dev \
                    libglfw3-dev \
                    libpng12-dev \
                    libcurl4-openssl-dev \
                    libcrypto++-dev

    apt-get build-dep thrift-compiler
    wget http://apache.claz.org/thrift/0.9.2/thrift-0.9.3.tar.gz
    tar xvf thrift-0.9.3.tar.gz
    cd thrift-0.9.3
    patch -p1 < /path/to/mapd2/scripts/mapd-deps-thrift-refill-buffer.patch
    ./configure --with-lua=no --with-python=no --with-php=no --with-ruby=no --prefix=/usr/local/mapd-deps
    make -j $(nproc)
    make install

### CUDA

CUDA should be installed via the .deb method, following the instructions provided by Nvidia.

## Arch

Assuming you already have [yaourt](https://wiki.archlinux.org/index.php/Yaourt) or some other manager that supports the AUR installed:

    yaourt -S git cmake boost google-glog extra/jdk8-openjdk clang llvm thrift go cuda nvidia glew glfw libpng
    wget https://flexpp-bisonpp.googlecode.com/files/bisonpp-1.21-45.tar.gz
    tar xvf bisonpp-1.21-45.tar.gz
    cd bison++-1.21
    ./configure && make && make install

### CUDA

CUDA is installed to `/opt/cuda` instead of the default `/usr/local/cuda`. You may have to add the following to `CMakeLists.txt` to support this:

    include_directories("/opt/cuda/include")

# Environment variables

If using `mapd-deps-linux.sh` or Ubuntu 15.10, you will need to add following environment variables to your `~/.bashrc` or a file such as `/etc/profile.d/mapd-deps.sh` (or use [Environment Modules](http://modules.sourceforge.net/)):

    MAPD_PATH=/usr/local/mapd-deps
    PATH=$MAPD_PATH/bin:$PATH
    LD_LIBRARY_PATH=$MAPD_PATH/lib:$LD_LIBRARY_PATH
    LD_LIBRARY_PATH=$MAPD_PATH/lib64:$LD_LIBRARY_PATH
    export PATH LD_LIBRARY_PATH

CUDA requires the following environment variables, assuming CUDA was installed to `/usr/local/cuda`:

    CUDA_PATH=/usr/local/cuda
    PATH=$CUDA_PATH/bin:$PATH
    LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
    export PATH LD_LIBRARY_PATH

CUDA on OS X is usually installed under `/Developer/NVIDIA/CUDA-7.5`:

    CUDA_PATH=/Developer/NVIDIA/CUDA-7.5
    PATH=$CUDA_PATH/bin:$PATH
    DYLD_LIBRARY_PATH=$CUDA_PATH/lib64:$DYLD_LIBRARY_PATH
    export PATH DYLD_LIBRARY_PATH
