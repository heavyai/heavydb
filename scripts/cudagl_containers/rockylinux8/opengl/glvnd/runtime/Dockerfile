ARG from

# Build libglvnd
FROM rockylinux:8 as glvnd

RUN dnf install -y \
        git \
        make \
        libtool \
        gcc \
        pkgconfig \
        python2 \
        libXext-devel \
        libX11-devel \
        xorg-x11-proto-devel && \
    rm -rf /var/cache/dnf/*

ARG LIBGLVND_VERSION

WORKDIR /opt/libglvnd
RUN git clone --branch="${LIBGLVND_VERSION}" https://github.com/NVIDIA/libglvnd.git . && \
    ./autogen.sh && \
    ./configure --prefix=/usr/local --libdir=/usr/local/lib64 && \
    make -j"$(nproc)" install-strip && \
    find /usr/local/lib64 -type f -name 'lib*.la' -delete

FROM ${from}
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

COPY --from=glvnd /usr/local/lib64 /usr/local/lib64

COPY 10_nvidia.json /usr/local/share/glvnd/egl_vendor.d/10_nvidia.json

RUN echo '/usr/local/lib64' >> /etc/ld.so.conf.d/glvnd.conf && \
    ldconfig && \
    echo '/usr/local/$LIB/libGL.so.1' >> /etc/ld.so.preload && \
    echo '/usr/local/$LIB/libEGL.so.1' >> /etc/ld.so.preload
