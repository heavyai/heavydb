# Build libglvnd for libGL, libEGL, libOpenGL
# Not currently pulled in by nvidia-docker2
FROM nvidia/cudagl:10.0-devel-ubuntu18.04

ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

# Add entrypoint script to run ldconfig
RUN echo $'#!/bin/bash\n\
      ldconfig\n\
      exec "$@"'\
    >> /docker-entrypoint.sh && \
    chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]

RUN apt-get update 
RUN apt-get install -y software-properties-common \
      build-essential \
      ccache \
      cmake \
      cmake-curses-gui \
      git \
      wget \
      curl \
      gcc \
      g++ \
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
      libldap2-dev \
      binutils-dev \
      google-perftools \
      libdouble-conversion-dev \
      libevent-dev \
      libgflags-dev \
      libgoogle-perftools-dev \
      libiberty-dev \
      libjemalloc-dev \
      libglu1-mesa-dev \
      liblz4-dev \
      liblzma-dev \
      libbz2-dev \
      libarchive-dev \
      libcurl4-openssl-dev \
      libedit-dev \
      uuid-dev \
      libsnappy-dev \
      zlib1g-dev \
      autoconf \
      autoconf-archive \
      automake \
      bison \
      flex-old \
      libpng-dev \
      rsync \
      unzip \
      jq \
      python-dev \
      python-yaml \
      swig \
      libxerces-c-dev \
      libxmlsec1-dev \
      sudo

RUN mkdir -p /etc/vulkan/icd.d && \
    echo '{ "file_format_version" : "1.0.0", "ICD" : { "library_path" : "libGLX_nvidia.so.0", "api_version" : "1.1.99" } }' > /etc/vulkan/icd.d/nvidia_icd.json

RUN echo > /etc/ld.so.preload

RUN curl -OJ https://internal-dependencies.mapd.com/mapd-deps/mapd-deps-prebuilt.sh \
    && USER=root sudo bash ./mapd-deps-prebuilt.sh \
    && rm mapd-deps-prebuilt.sh
