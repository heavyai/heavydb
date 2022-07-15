FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Tools
RUN apt-get update && apt-get install -y \
    autoconf \
    automake \
    build-essential \
    ccache \
    cmake \
    gdb \
    git \
    libtbb-dev \
    make \
    python \
    python-dev \
    rsync \
    tar \
    vim \
    wget \
    --

# LLVM
RUN apt update && apt install -y llvm-12 llvm-12-dev clang-12

# Toolchains
RUN apt install -y \
    gcc-9 \
    g++-9 \
    openjdk-11-jdk \
    --

RUN update-alternatives \
    --install /usr/bin/gcc gcc /usr/bin/gcc-9 800 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-9

# Dependencies
RUN apt install -y \
    binutils-dev \
    bzip2 \
    libboost-all-dev \
    libbz2-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libncurses-dev \
    libsnappy-dev \
    libssl-dev \
    libtbb-dev \
    libunwind-dev \
    libz-dev \
    --

RUN apt-get install -y maven

# Arrow
# https://arrow.apache.org/install/
RUN apt install -y -V ca-certificates lsb-release wget
RUN wget https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
RUN apt install -y -V ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
RUN apt update
RUN apt install -y -V libarrow-dev libparquet-dev


# Intel GPU stack
# https://dgpu-docs.intel.com/installation-guides/ubuntu/ubuntu-focal.html
RUN apt install -y gpg-agent software-properties-common
RUN wget -qO - https://repositories.intel.com/graphics/intel-graphics.key | apt-key add -
RUN apt-add-repository \
    'deb [arch=amd64] https://repositories.intel.com/graphics/ubuntu focal main'
RUN apt-get update
RUN apt-get install -y \
    intel-igc-cm \
    intel-level-zero-gpu \
    intel-media-va-driver-non-free \
    intel-opencl-icd \
    level-zero \
    level-zero-dev \
    libigc-dev \
    libigdfcl-dev \
    libigfxcmrt-dev \
    libmfx1 \
    --

# SPIRV-Translator
RUN apt install -y pkg-config unzip
RUN mkdir /translator && \
    cd /translator && \
    wget https://github.com/KhronosGroup/SPIRV-LLVM-Translator/releases/download/v12.0.0/SPIRV-LLVM-Translator-v12.0.0-linux-Release.zip && \
    unzip SPIRV-LLVM-Translator-v12.0.0-linux-Release.zip && \
    rm SPIRV-LLVM-Translator-v12.0.0-linux-Release.zip \
    --

RUN sed -i "1s/.*/\/translator/" /translator/lib/pkgconfig/LLVMSPIRVLib.pc

RUN apt-get upgrade -y
RUN apt-get clean
