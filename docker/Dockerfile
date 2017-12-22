# Build libglvnd for libGL, libEGL, libOpenGL
# Not currently pulled in by nvidia-docker2
FROM nvidia/cuda:8.0-devel-ubuntu16.04 AS glvnd

RUN apt-get update && apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        libtool \
        pkg-config \
        python \
        libxext-dev \
        libx11-dev \
        x11proto-gl-dev && \
    apt-get remove --purge -y && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /libglvnd/
ADD https://github.com/NVIDIA/libglvnd/archive/v1.0.0.tar.gz .
RUN tar xvf v1.0.0.tar.gz --strip-components=1
RUN ./autogen.sh
RUN ./configure --disable-glx && \
    make -j 4 && \
    make install

# Copy and extract MapD tarball. In own stage so that the temporary tarball
# isn't included in a layer.
FROM ubuntu:16.04 AS extract

WORKDIR /mapd/
COPY mapd-latest-Linux-x86_64.tar.gz /mapd/
RUN tar xvf mapd-latest-Linux-x86_64.tar.gz --strip-components=1 && \
    rm -rf mapd-latest-Linux-x86_64.tar.gz

# Build final stage
FROM nvidia/cuda:8.0-runtime-ubuntu16.04
LABEL maintainer "Andrew Seidl <andrew@mapd.com>"

ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

RUN apt-get update && apt-get install -y --no-install-recommends \
        libldap-2.4-2 \
        bsdmainutils \
        wget \
        curl \
        default-jre-headless && \
    apt-get remove --purge -y && \
    rm -rf /var/lib/apt/lists/*

COPY --from=extract /mapd /mapd
COPY --from=glvnd /usr/local /usr/local/

RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    echo '{ "file_format_version" : "1.0.0", "ICD" : { "library_path" : "libEGL_nvidia.so.0" } }' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
RUN mkdir -p /usr/local/share/glvnd/egl_vendor.d && \
    echo '{ "file_format_version" : "1.0.0", "ICD" : { "library_path" : "libEGL_nvidia.so.0" } }' > /usr/local/share/glvnd/egl_vendor.d/10_nvidia.json

WORKDIR /mapd

EXPOSE 9091 9092

CMD /mapd/startmapd --non-interactive --data /mapd-storage/data --config /mapd-storage/mapd.conf
