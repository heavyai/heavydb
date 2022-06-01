FROM nvidia/cudagl:11.0-devel-ubuntu18.04

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
    rm -rf /var/lib/apt/lists/*

RUN echo > /etc/ld.so.preload

RUN curl -OJ https://dependencies.mapd.com/mapd-deps/mapd-deps-prebuilt.sh \
    && USER=root sudo bash ./mapd-deps-prebuilt.sh --enable \
    && rm mapd-deps-prebuilt.sh
