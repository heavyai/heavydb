# Build libglvnd for libGL, libEGL, libOpenGL
# Not currently pulled in by nvidia-docker2
FROM nvidia/cuda:9.0-devel-centos7

ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

# Add entrypoint script to run ldconfig
RUN echo $'#!/bin/bash\n\
      ldconfig\n\
      exec "$@"'\
    >> /docker-entrypoint.sh && \
    chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]

RUN yum groupinstall -y "Development Tools"
RUN yum install -y \
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
        python-devel \
        wget \
        curl \
        sudo \
        openldap-devel \
        libX11-devel \
        mesa-libGL-devel \
        environment-modules \
        PyYAML \
        valgrind && \
    rm -rf /var/cache/yum/*
RUN yum install -y \
        cloc \
        jq && \
    rm -rf /var/cache/yum/*

WORKDIR /libglvnd/
ADD https://github.com/NVIDIA/libglvnd/archive/v1.0.0.tar.gz .
RUN tar xvf v1.0.0.tar.gz --strip-components=1 \
    && ./autogen.sh \
    && ./configure --disable-glx \
    && make -j 4 \
    && make install \
    && rm -rf /libglvnd
WORKDIR /

RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    echo '{ "file_format_version" : "1.0.0", "ICD" : { "library_path" : "libEGL_nvidia.so.0" } }' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
RUN mkdir -p /usr/local/share/glvnd/egl_vendor.d && \
    echo '{ "file_format_version" : "1.0.0", "ICD" : { "library_path" : "libEGL_nvidia.so.0" } }' > /usr/local/share/glvnd/egl_vendor.d/10_nvidia.json

RUN curl -OJ https://internal-dependencies.mapd.com/mapd-deps/deploy.sh \
    && USER=root sudo bash ./deploy.sh --testing \
    && rm deploy.sh
