## Docker and NVIDIA Container Toolkit Installation

The NVIDIA Container Toolkit is required in order to use GPUs inside Docker containers.

Install Docker and NVIDIA Container Toolkit:
- https://docs.docker.com/engine/installation/
  - Be sure to use Docker's repositories for the latest version of Docker.
- https://github.com/NVIDIA/nvidia-docker/blob/master/README.md

## Building OmniSci container

The `Dockerfile` assumes a copy of the OmniSci tarball is in the same directory as the Dockerfile and is named `omnisci-latest-Linux-x86_64.tar.gz`.

To build the container, run:

    mv omnisci-5.2.1-*-render.tar.gz omnisci-latest-Linux-x86_64.tar.gz
    tar -xvf omnisci-latest-Linux-x86_64.tar.gz --strip-components=2 --no-anchored "docker/Dockerfile"
    docker build .

where `omnisci-5.2.1-*-render.tar.gz` is the path to the OmniSci tarball.

The container image id will be output on the last line of the `build` step. To assign a custom name and tag:

    docker build -t omnisci/omnisci:v5.2.1 .

which will assign the name `omnisci/omnisci` and the tag `v5.2.1` to the image.

### Image layout

The data directory is at `/omnisci-storage/data`.

The config file lives at `/omnisci-storage/omnisci.conf`.

## Running OmniSci inside a container

    docker run -d \
      --gpus all \
      -p 6273:6273 \
      --name omnisci \
      -v /path/to/omnisci-storage:/omnisci-storage \
      omnisci/omnisci:v5.2.1

This starts OmniSciDB inside a container named `omnisci`, and exposes the Immerse visualization client on port 6273..

Data will be persisted to the host directory `/path/to/omnisci-storage`.
