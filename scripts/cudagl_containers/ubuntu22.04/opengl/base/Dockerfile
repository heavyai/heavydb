ARG from

FROM ${from}

RUN apt-get update && apt-get install -y --no-install-recommends \
        libxau6 \
        libxdmcp6 \
        libxcb1 \
        libxext6 \
        libx11-6 && \
    rm -rf /var/lib/apt/lists/*

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
        ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
        ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,compat32,utility

RUN echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

COPY NGC-DL-CONTAINER-LICENSE /

# Required for non-glvnd setups.
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
