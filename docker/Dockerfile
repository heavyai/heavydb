FROM nvidia/cuda:8.0-ubuntu16.04
MAINTAINER Andrew Seidl <andrew@mapd.com>

ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}:/usr/local/cuda-8.0/targets/x86_64-linux/lib/stubs
RUN ln -sf /usr/local/cuda-8.0/targets/x86_64-linux/lib/stubs/libcuda.so /usr/local/cuda-8.0/targets/x86_64-linux/lib/stubs/libcuda.so.1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglu1-mesa-dev \
        default-jre-headless && \
    apt-get remove --purge -y && \
    rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH /usr/lib/jvm/default-java/jre/lib/amd64/server:/mapd/bin:${LD_LIBRARY_PATH}

ADD mapd2-latest-Linux-x86_64.tar.gz /installs
RUN rm -f /mapd && ln -s $(ls -dtr /installs/mapd* | head -n 1) /mapd

RUN mkdir -p /mapd-storage/data \
  && /mapd/bin/initdb /mapd-storage/data \
  && touch /mapd-storage/mapd.conf

WORKDIR /mapd

EXPOSE 9092

CMD /mapd/startmapd --data /mapd-storage/data --config /mapd-storage/mapd.conf
