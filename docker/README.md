## Docker Install

Install Docker and nvidia-docker:
- https://docs.docker.com/engine/installation/
  - Install from Docker's repos, not your distro's. Normal distros are often many versions behind.
- https://github.com/NVIDIA/nvidia-docker/blob/master/README.md

If you want to run as a normal user, add the user to both the `docker` and `nvidia-docker` groups:

    sudo usermod -aG docker username
    sudo usermod -aG nvidia-docker username

## Building MapD container

Provided `Dockerfile` is for CUDA 8, which means your host system must have up-to-date drivers installed. This was chosen as it's a requirement to run on the DGX-1/anything Pascal.

The `Dockerfile` assumes that there is a MapD tarball sitting in the same directory named `mapd2-latest-Linux-x86_64.tar.gz` (standard tarball for our CUDA-enabled, rendering-disabled builds).

To build the container do:

    wget --ask-password https://mapd@builds.mapd.com/mapd2-latest-Linux-x86_64.tar.gz # use your own user/pass
    nvidia-docker build .

The image id will be output on the last line of the `build` step. To assign a custom name do something like:

    nvidia-docker build -t mapd/mapd-norender:v1.1.9

which will assign the name `mapd/mapd-norender` and the tag `v1.1.9` to the image.

If you need to run with older drivers, replace the tag in the `FROM` line with `7.5-ubuntu14.04` and replace all references to `8.0` with `7.5`.

### Image layout

The tarball gets extracted to `/installs`. When building the `Dockerfile`, the extracted tarball then gets symlinked to `/mapd`.

Data directory lives at `/mapd-storage/data`.

Config file lives at `/mapd-storage/mapd.conf`.

## Running MapD inside a container

    nvidia-docker run -p 19092:9092 mapd/mapd-norender:v1.1.9

That will expose the webserver on port `19092`.

Saved data inside containers is ephemeral. To preserve your data you probably want to use a data container or at least bind mount in a host directory.

    nvidia-docker run -v /home/mapd/prod/mapd-storage:/mapd-storage -p 19092:9092 mapd/mapd-norender:v1.1.9

will mount the host directory `/home/mapd/prod/mapd-storage` to `/mapd-storage` in the container.

See the Docker docs for more info on how to run as a daemon, how to spawn a shell inside the container, how to autostart on reboot, etc.

Note: the `Dockerfile` currently uses `startmapd` to start both `mapd_web_server` and `mapd_server`. Make sure you have run `initdb` on your data folder before running.

## CPU

- change `FROM` to your favorite recent distro. Busybox and Alpine probably won't work right now due to libldap.
- remove any lines specific to CUDA
- grab `nocuda` tarball, update Dockerfile
- remote `nvidia-` from all commands above

## EGL-based rendering

Due to issues related to running X inside a container, you must use an EGL-enabled build in order to use backend rendering. Grab an EGL-enabled build from Jenkins (select `EGL` for `renderer_context_type` in `mapd2-multi`) and then rename the tarball to `mapd2-latest-Linux-x86_64.tar.gz`. You can then build the container via the usual:

    nvidia-docker build .

There is currently an issue where EGL+CUDA code will segfault if `libEGL.so` does not exist in your `LD_LIBRARY_PATH`. While this symlink is created by the NVIDIA driver install, it is not currently created by the drivers that are brought in by `nvidia-docker`. A patch has been submitted to NVIDIA to resolve this (see: https://github.com/NVIDIA/nvidia-docker/pull/146), but until that is merged, you can work around the issue by bind mounting the library in from the host:

    nvidia-docker run -v /usr/lib/nvidia-367/libEGL.so.1:/usr/lib/libEGL.so ...

This assumes that you are running NVIDIA driver 367.xx and the file `libEGL.so.1` exists under `/usr/lib/nvidia-367`. Adjust if necessary for your driver version and distro.

## TODO
- bump file descriptors count / ulimit
- keys / licensing / limiting
