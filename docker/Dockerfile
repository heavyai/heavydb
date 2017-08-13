FROM nvidia/cuda:8.0-runtime-ubuntu16.04
MAINTAINER Andrew Seidl <andrew@mapd.com>

RUN apt-get update && apt-get install -y --no-install-recommends \
        libldap-2.4-2 \
        bsdmainutils \
        wget \
        curl \
        default-jre-headless && \
    apt-get remove --purge -y && \
    rm -rf /var/lib/apt/lists/*

ADD mapd-latest-Linux-x86_64.tar.gz /installs
RUN rm -f /mapd && ln -s $(ls -dtr /installs/mapd* | head -n 1) /mapd

WORKDIR /mapd

EXPOSE 9092

CMD /mapd/startmapd --non-interactive --data /mapd-storage/data --config /mapd-storage/mapd.conf
