# cudagl required to expose cuda and graphics (vulkan)
FROM docker-internal.mapd.com/cudagl:11.8.0-devel-ubuntu20.04
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

# Add entrypoint script to run ldconfig
RUN echo '#!/bin/bash\n\
      ldconfig\n\
      exec "$@"'\
    >> /docker-entrypoint.sh && \
    chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]

RUN apt-get update && \
    apt-get install -y \
      sudo \
      curl && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y tzdata && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/vulkan/icd.d && \
    echo '{ "file_format_version" : "1.0.0", "ICD" : { "library_path" : "libGLX_nvidia.so.0", "api_version" : "1.1.99" } }' > /etc/vulkan/icd.d/nvidia_icd.json

RUN echo > /etc/ld.so.preload

RUN curl -OJ https://dependencies.mapd.com/mapd-deps/mapd-deps-prebuilt.sh \
    && USER=root sudo bash ./mapd-deps-prebuilt.sh --enable \
    && rm mapd-deps-prebuilt.sh
