## Docker and NVIDIA Container Toolkit Installation

The NVIDIA Container Toolkit is required in order to use GPUs inside Docker containers.

Install Docker and NVIDIA Container Toolkit:
- https://docs.docker.com/engine/installation/
  - Be sure to use Docker's repositories for the latest version of Docker.
- https://github.com/NVIDIA/nvidia-docker/blob/master/README.md

## Building HEAVY.AI container

The `Dockerfile` assumes a copy of the HEAVY.AI tarball is in the same directory as the Dockerfile and is named `heavyai-latest-Linux-x86_64.tar.gz`.

To build the container, run:

    mv heavyai-6.0.0-*-render.tar.gz heavyai-latest-Linux-x86_64.tar.gz
    tar -xvf heavyai-latest-Linux-x86_64.tar.gz --strip-components=2 --no-anchored "docker/Dockerfile"
    docker build .

where `heavyai-6.0.0-*-render.tar.gz` is the path to the HEAVY.AI tarball.

The container image id will be output on the last line of the `build` step. To assign a custom name and tag:

    docker build -t heavyai/heavyai:v6.0.0 .

which will assign the name `heavyai/heavyai` and the tag `v6.0.0` to the image.

### Image layout

The data directory is at `/var/lib/heavyai/data`.

The config file lives at `/var/lib/heavyai/heavy.conf`.

## Running HEAVY.AI inside a container

    docker run -d \
      --gpus all \
      -p 6273:6273 \
      --name heavyai \
      -v /path/to/storage:/var/lib/heavyai \
      heavyai/heavyai:v6.0.0

This starts the HEAVY.AI platform inside a container named `heavyai`, and exposes the Immerse visualization client on port 6273..

Data will be persisted to the host directory `/path/to/storage`.
