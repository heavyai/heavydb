These scripts generate the base Docker images used for `heavydb` build containers and final `heavydb` product containers. 

While Nvidia publishes images that can be used for `cuda`, they do not publish derived `cudagl` images that add necessary components for graphics APIs (OpenGL / Vulkan). Generating these derived containers requires running the Nvidia build scripts that are hosted in gitlab: https://gitlab.com/nvidia/container-images/cuda/

Two Docker image are produced:
- `devel` provides the necessary libraries for building Cuda applications. Used as the base for build containers
- `runtime` is a smaller image used for wrapping the final Cuda application. Used as the base for `heavydb` containers

Example final image names with date tags:
- `docker-internal.mapd.com/cudagl/ubuntu22.04-cuda12.2.2-x86_64-devel:20241113`
- `docker-internal.mapd.com/cudagl/ubuntu22.04-cuda12.2.2-x86_64-runtime:20241113`

Another shortcoming of the Nvidia images is a lack of prompt security updates. Nvidia does periodically refresh the `cuda` images to pick up important updates in the base OS packages, the cadence is infrequent, and the updates incomplete. These scripts ensure the latest package updates are applied to the two Docker images

The `scripts/cudagl_containers` directory contains HEAVY.AI scripts to build the `cudagl` images. These scripts will either clone or update an existing local copy of the Nvidia gitlab repo, patch the Nvidia scripts as needed for `aarch64` support, then invoke the Nvidia `build.sh` script to build the base images. Once the standard Nvidia containers are built, the scripts apply the latest base package updates, then change the image naming and tagging scheme to conform to our internal naming requirements (including date tagging). Finally, the images can be pushed to `docker-internal.mapd.com` or kept locally (or discarded, useful for testing)

## Usage
Run `build_cudagl_containers.sh` from any directory. By default it will create a temp folder in `scripts/cudagl_containers`. The temp folder is deleted on script exit

The defaults do not keep or push images, essentially a "dry run". You need to specify `--keep-images` and/or `--push` or the images will be discarded when the script completes

Rerunning the script can be very quick as much of the work will be cached by docker. A basic `docker system prune` (no `-f`) is recommended once final images are built to clean up any cached containers and images

There is also a Jenkins project https://jenkins.mapd.com/job/dependency_scripts/job/cudagl-container-build/

## Files
### `build_cudagl_containers.sh`

This is the primary script to run. By default the script will clone the Nvidia repository to a temp directory, but a different location can be specified with `--repo-path=<path>`. If an existing repository is found in that location the script will perform a `fetch` && `pull`, otherwise it will perform a fresh clone. By default the repository is deleted, but passing `--keep-repo` will leave the Nvidia repository in place (this must be used with `--repo-path` as the temp directory is always deleted)

During the build process, the Nvidia scripts clone a sub-repository `opengl`, in order to add the necessary components to support graphics APIs, including Vulkan. Unfortunately the `opengl` scripts do not cleanly support building for `aarch64` (ARM) targets. The problem lies in the Dockerfiles pulling i386 (32-bit) packages alongside the `aarch64` packages, which fails on Arm systems (and also bloats `x86_64` containers). To resolve this, 2 patch files are used. The first patches the `cuda/build.sh` script so it will apply the `opengl` patch after it clones that repository. The `opengl` patch removes the `i386` package installation

By default the final `cudagl` images ARE removed and NOT pushed, meaning they will be lost. Pass `--push` to push the images to `docker-internal`, or use `--keep-images` to keep the local copies for testing.

#### Parameters:
```
--cuda-version=xx.x.x      default = 12.2.2
--os=value                 default = ubuntu
--os-version=value         default = 22.04
--arch=[x86_64 | aarch64]  default = x86_64
--repo-path=path           REQUIRED (no default)
--keep-repo
--push
--keep-images
```

### `gen_cudagl_package_updater.sh`

Captures upgradable packages via `update list --upgradable`, and generates `cudagl_package_updater.sh`, the bash script that will `apt-get install` the specific versions that would be upgraded. Held packages are automatically removed. This generated script is necessary as the base `cuda` container where the script needs to run would require additional packages to be installed to do an `apt-get upgrade`, however `apt-get install` works fine.

This script can also be run manually to generate an upgrade "patch" script that can be run in final build and heavydb containers to upgrade the base packages for `heavydb` point releases. This patching has been integrated into existing `heavydb-internal` Dockerfiles, scripts, and Jenkins jobs

### `cudagl_package_updater.sh`

This is a do nothing stub. It can be replaced by using `gen_cudagl_package_updater.sh` or via manual editing to allow patching the cudagl containers during several stages of the process:
  - Rerunning the `heavyai-deps-pipeline` to rebuild deps tarballs and images will run this script prior to building the tarballs, and when creating the final build containers
  - When building `heavydb` containers for patching without building dependencies. This is useful for patching security issues in the final `heavydb` images without having to rebuild deps. Any security fixes to static libraries we link with will not be updated of course, that requires patching the build containers, or building fresh `cudagl` images with the latest updates

The stub needs to exist so impacted `Dockerfiles` can do the necessary `COPY` commmand to copy the file into the Docker build context, which cannot be done conditionally based on file existence

### `Dockerfile`

A very simple Dockerfile that is used to apply the generated `cudagl_package_updater` script to an existing docker container

### Patch files

- `apply_cudagl_remove_i386_patch.patch` - changes `cuda/build.sh` so it will apply the `opengl` patch after cloning that sub-repository
- `cudagl_remove_i386.patch` - changes `cudagl` Dockerfiles to remove installation of `i386` packages