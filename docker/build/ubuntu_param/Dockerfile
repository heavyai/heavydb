# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# cudagl required to expose cuda and graphics (vulkan)

ARG BASE_CUDA_IMAGE=placeholder
FROM ${BASE_CUDA_IMAGE}

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# Add entrypoint script to run ldconfig
RUN echo '#!/bin/bash\n\
      ldconfig\n\
      exec "$@"'\
    >> /docker-entrypoint.sh && \
    chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]

#
# add --build-arg TZDATA="" to cmd for ubuntu 1804 build
ARG TZDATA_INSTALL="DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y tzdata" 

RUN apt-get update && \
    apt-get install -y sudo curl zip unzip ca-certificates
# Install the Docker CLI (client only) so builds inside this image can run
# `docker` against a mounted host daemon socket. The nvcr/cuda base has no
# docker client (the old cudagl-based core-build images shipped one).
RUN install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc && \
    chmod a+r /etc/apt/keyrings/docker.asc && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo \"$VERSION_CODENAME\") stable" > /etc/apt/sources.list.d/docker.list && \
    apt-get update && \
    apt-get install -y docker-ce-cli
RUN /bin/bash -c "${TZDATA_INSTALL}"
RUN rm -rf /var/lib/apt/lists/*

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

#
# example cmd use --build-arg PREBUILT_CUSTOM_ARG="--custom=tsan-20230706" 
#
ARG PREBUILT_CUSTOM_ARG
## Use the default location for --deps-tar-path in mapd-deps-prebuilt.sh
## The default is . for the docker build
RUN USER=root sudo bash ./mapd-deps-prebuilt.sh ${PREBUILT_CUSTOM_ARG}
