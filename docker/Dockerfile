FROM ubuntu:16.04 AS extract

WORKDIR /mapd/
COPY mapd-latest-Linux-x86_64.tar.gz /mapd/
RUN tar xvf mapd-latest-Linux-x86_64.tar.gz --strip-components=1 && \
    rm -rf mapd-latest-Linux-x86_64.tar.gz

FROM nvidia/cuda:8.0-runtime-ubuntu16.04
LABEL maintainer "Andrew Seidl <andrew@mapd.com>"

RUN apt-get update && apt-get install -y --no-install-recommends \
        libldap-2.4-2 \
        bsdmainutils \
        wget \
        curl \
        default-jre-headless && \
    apt-get remove --purge -y && \
    rm -rf /var/lib/apt/lists/*

COPY --from=extract /mapd /mapd

WORKDIR /mapd

EXPOSE 9092

CMD /mapd/startmapd --non-interactive --data /mapd-storage/data --config /mapd-storage/mapd.conf
