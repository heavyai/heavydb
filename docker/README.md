## Docker Install

Install Docker and nvidia-docker:
- https://docs.docker.com/engine/installation/
  - Install from Docker's repos, not your distro's. Normal distros are often many versions behind.
- https://github.com/NVIDIA/nvidia-docker/blob/master/README.md

If you want to run as a normal user, add the user to both the `docker` and `nvidia-docker` groups:

    sudo usermod -aG docker username
    sudo usermod -aG nvidia-docker username

## Building MapD container

Provided `Dockerfile` is for CUDA 8, which means your host system must have up-to-date drivers installed (367 or later preferred).

The `Dockerfile` assumes that there is a MapD tarball sitting in the same directory named `mapd2-latest-Linux-x86_64.tar.gz`.

To build the container, run:

    wget --ask-password https://user@builds.mapd.com/mapd2-latest-Linux-x86_64.tar.gz # use your own user/pass
    nvidia-docker build .

The image id will be output on the last line of the `build` step. To assign a custom name do something like:

    nvidia-docker build -t mapd/mapd-norender:v1.2.10

which will assign the name `mapd/mapd-norender` and the tag `v1.2.10` to the image.

### Image layout

The tarball is extracted to `/installs`. The extracted tarball also gets symlinked to `/mapd`.

Data directory lives at `/mapd-storage/data`.

Config file lives at `/mapd-storage/mapd.conf`.

## Running MapD inside a container

    nvidia-docker run -p 19092:9092 mapd/mapd-norender:v1.2.10

will expose the web server on port `19092`.

Saved data inside containers is ephemeral. To preserve your data you probably want to use a data container or at least bind mount in a host directory.

    nvidia-docker run -v /home/mapd/prod/mapd-storage:/mapd-storage -p 19092:9092 mapd/mapd-norender:v1.2.10

will mount the host directory `/home/mapd/prod/mapd-storage` to `/mapd-storage` in the container.

See the Docker docs for more info on how to run as a daemon, how to spawn a shell inside the container, how to autostart on reboot, etc.

Note: the `Dockerfile` currently uses `startmapd` to start both `mapd_web_server` and `mapd_server`. It will automatically run `initdb` to create a data directory if one does not exist.

## Rendering Support

An EGL-enabled build must be used for backend rendering due to issues related to running X inside a container. To build a container, rename the provided EGL-enabled tarball to `mapd2-latest-Linux-x86_64.tar.gz` and then build the container with:

    nvidia-docker build .
