# cudagl required to expose cuda and graphics (vulkan)
FROM docker-internal.mapd.com/cudagl:11.8.0-devel-centos7

ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

# Add entrypoint script to run ldconfig
RUN echo $'#!/bin/bash\n\
      ldconfig\n\
      exec "$@"'\
    >> /docker-entrypoint.sh && \
    chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]

# patch repo files now that CentOS 7 is EOL
RUN sed -i 's/mirror.centos.org/vault.centos.org/' /etc/yum.repos.d/*.repo
RUN sed -i 's/^#.*baseurl=http:/baseurl=https:/' /etc/yum.repos.d/*.repo
RUN sed -i 's/^mirrorlist=http:/#mirrorlist=https:/' /etc/yum.repos.d/*.repo
RUN echo 'sslverify=false' | tee -a /etc/yum.conf

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
        which \
        PyYAML \
        valgrind && \
    rm -rf /var/cache/yum/*
RUN yum install -y \
        cloc \
        jq && \
    rm -rf /var/cache/yum/*

RUN mkdir -p /etc/vulkan/icd.d && \
    echo '{ "file_format_version" : "1.0.0", "ICD" : { "library_path" : "libGLX_nvidia.so.0", "api_version" : "1.1.99" } }' > /etc/vulkan/icd.d/nvidia_icd.json

RUN echo > /etc/ld.so.preload

RUN curl -OJ https://internal-dependencies.mapd.com/mapd-deps/mapd-deps-prebuilt.sh \
    && USER=root sudo bash ./mapd-deps-prebuilt.sh \
    && rm mapd-deps-prebuilt.sh
