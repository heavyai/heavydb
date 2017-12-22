## Docker and nvidia-docker Installation

Install Docker and nvidia-docker:
- https://docs.docker.com/engine/installation/
  - Be sure to use Docker's repositories for the latest version of Docker.
- https://github.com/NVIDIA/nvidia-docker/blob/master/README.md

To use Docker as a normal user, add the user to both the `docker` and `nvidia-docker` groups:

    sudo usermod -aG docker username
    sudo usermod -aG nvidia-docker username

## Building MapD container

The `Dockerfile` assumes a copy of the MapD tarball is in the same directory as the Dockerfile and is named `mapd-latest-Linux-x86_64.tar.gz`.

To build the container, run:

    mv ../../mapd-3.0.0-*-render.tar.gz mapd-latest-Linux-x86_64.tar.gz
    nvidia-docker build .

where `../../mapd-3.0.0-*-render.tar.gz` is the path to the MapD tarball.

The container image id will be output on the last line of the `build` step. To assign a custom name and tag:

    nvidia-docker build -t mapd/mapd:v3.0.0 .

which will assign the name `mapd/mapd` and the tag `v3.0.0` to the image.

### Image layout

The tarball is extracted to `/installs`. The extracted tarball is also symlinked to `/mapd`.

The data directory is at `/mapd-storage/data`.

The config file lives at `/mapd-storage/mapd.conf`.

## Running MapD inside a container

    nvidia-docker run -d \
      -p 9092:9092 \
      --name mapd \
      -v /path/to/mapd-storage:/mapd-storage \
      mapd/mapd:v3.0.0

This starts the MapD Core Database inside a container named `mapd`, and exposes the Immerse visualization client on port 9092..

Data will be persisted to the host directory `/path/to/mapd-storage`.
