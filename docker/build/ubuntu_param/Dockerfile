# cudagl required to expose cuda and graphics (vulkan)

ARG BASE_CUDA_IMAGE
FROM ${BASE_CUDA_IMAGE}

ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

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
    apt-get install -y  sudo curl
RUN /bin/bash -c "${TZDATA_INSTALL}"
RUN rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/vulkan/icd.d && \
    echo '{ "file_format_version" : "1.0.0", "ICD" : { "library_path" : "libGLX_nvidia.so.0", "api_version" : "1.1.99" } }' > /etc/vulkan/icd.d/nvidia_icd.json

RUN echo > /etc/ld.so.preload

# copy the script into the docker context
# the script needs to already be in '.'
COPY mapd-deps-prebuilt.sh .

#
# example cmd use --build-arg PREBUILT_CUSTOM_ARG="--custom=tsan-20230706" 
#
ARG PREBUILT_CUSTOM_ARG
RUN USER=root sudo bash ./mapd-deps-prebuilt.sh ${PREBUILT_CUSTOM_ARG}
