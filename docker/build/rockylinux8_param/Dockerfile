# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build libglvnd for libGL, libEGL, libOpenGL
# Not currently pulled in by nvidia-docker2
ARG BASE_CUDA_IMAGE=nvcr.io/nvidia/cuda:12.9.2-devel-rockylinux8
FROM ${BASE_CUDA_IMAGE}

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# Add entrypoint script to run ldconfig
RUN echo $'#!/bin/bash\n\
      ldconfig\n\
      exec "$@"'\
    >> /docker-entrypoint.sh && \
    chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]

RUN dnf groupinstall -y "Development Tools"

# Enable PowerTools (CRB) — it provides perl module deps (perl-Role::Tiny,
# Class::Method::Modifiers, Import::Into, Devel::GlobalDestruction) that
# perl-Moo (pulled in by cloc) needs. Disabled by default on the nvcr base.
RUN dnf install -y dnf-plugins-core && \
    dnf config-manager --set-enabled powertools

# Install the Docker CLI (client only) so builds inside this image can run
# `docker` against a mounted host daemon socket. The nvcr/cuda base has no
# docker client.
RUN dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo && \
    dnf install -y docker-ce-cli && \
    rm -rf /var/cache/dnf/*

RUN dnf install -y \
        zlib-devel \
        epel-release \
        libssh \
        git \
        maven \
        java-1.8.0-openjdk-devel \
        java-1.8.0-openjdk-headless \
        wget \
        curl \
        sudo \
        libX11-devel \
        mesa-libGL-devel \
        which \
        valgrind \
        zip unzip && \
    rm -rf /var/cache/dnf/*
RUN dnf install -y \
        cloc \
        jq && \
    rm -rf /var/cache/dnf/*

# Vulkan ICD is injected at runtime by the NVIDIA Container Toolkit (graphics
# capability), so it is not baked into the image.

RUN echo > /etc/ld.so.preload

# copy the script into the docker context
# the script needs to already be in '.'
COPY mapd-deps-prebuilt.sh .
COPY common-functions.sh .
COPY nvidia-graphics-env.sh .
#
# example cmd use --build-arg DEPS_TARBALL="mapd-deps-ubuntu22.04-static-x86_64-20251207-faaa7c06f.tar.xz"
#
ARG DEPS_TARBALL
COPY ${DEPS_TARBALL} .

ARG PREBUILT_CUSTOM_ARG
## Use the default location for --deps-tar-path in mapd-deps-prebuilt.sh
## The default is . for the docker build
RUN USER=root sudo bash ./mapd-deps-prebuilt.sh ${PREBUILT_CUSTOM_ARG}
