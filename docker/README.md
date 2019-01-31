## Docker and nvidia-docker Installation

Install Docker and nvidia-docker:
- https://docs.docker.com/engine/installation/
  - Be sure to use Docker's repositories for the latest version of Docker.
- https://github.com/NVIDIA/nvidia-docker/blob/master/README.md

To use Docker as a normal user, add the user to both the `docker` and `nvidia-docker` groups:

    sudo usermod -aG docker username
    sudo usermod -aG nvidia-docker username

## Building OmniSci container

The `Dockerfile` assumes a copy of the OmniSci tarball is in the same directory as the Dockerfile and is named `omnisci-latest-Linux-x86_64.tar.gz`.

To build the container, run:

    mv ../../omnisci-4.5.0-*-render.tar.gz omnisci-latest-Linux-x86_64.tar.gz
    nvidia-docker build .

where `../../omnisci-4.5.0-*-render.tar.gz` is the path to the OmniSci tarball.

The container image id will be output on the last line of the `build` step. To assign a custom name and tag:

    nvidia-docker build -t omnisci/omnisci:v4.5.0 .

which will assign the name `omnisci/omnisci` and the tag `v4.5.0` to the image.

### Image layout

The data directory is at `/omnisci-storage/data`.

The config file lives at `/omnisci-storage/omnisci.conf`.

## Running OmniSci inside a container

    nvidia-docker run -d \
      -p 6273:6273 \
      --name omnisci \
      -v /path/to/omnisci-storage:/omnisci-storage \
      omnisci/omnisci:v4.5.0

This starts the OmniSci Core Database inside a container named `omnisci`, and exposes the Immerse visualization client on port 6273..

Data will be persisted to the host directory `/path/to/omnisci-storage`.

## Compatibility with existing installations

The provided `Dockerfile.mapd` provides compatibility with pre-v4.5.0 installations which used `/mapd-storage` for the storage directory and `/mapd-storage/mapd.conf` for the config file, as well as ports 9091 (Thrift TCP binary protocol) and 9092 (Thrift HTTP/JSON protocol, Immerse).
