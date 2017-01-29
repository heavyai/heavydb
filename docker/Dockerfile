FROM nvidia/cuda:8.0-runtime-ubuntu16.04
MAINTAINER Andrew Seidl <andrew@mapd.com>

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglu1-mesa-dev \
        libldap-2.4-2 \
        bsdmainutils \
        wget \
        curl \
        default-jre-headless && \
    apt-get remove --purge -y && \
    rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH /usr/lib/jvm/default-java/jre/lib/amd64/server:/mapd/bin:${LD_LIBRARY_PATH}

ADD mapd2-latest-Linux-x86_64.tar.gz /installs
RUN rm -f /mapd && ln -s $(ls -dtr /installs/mapd* | head -n 1) /mapd

WORKDIR /mapd

EXPOSE 9092

CMD /mapd/startmapd --non-interactive --data /mapd-storage/data --config /mapd-storage/mapd.conf
