# MapD Install Guide (Release 0.1 BETA)

## Dependencies
MapD is distributed as a group of mostly statically-linked executables, which minimizes the number of dependencies required. The following are the minimum requirements for running MapD.

Operating Systems

* CentOS/RHEL 7.0 or later. CentOS/RHEL 6.x builds may be provided upon request, but are not supported.
* Ubuntu 15.04 or later.
* Mac OS X 10.9 or later, on 2013 or later hardware. Builds which support 2012 hardware may be provided upon request, but are not supported.

Libraries

* CUDA 7.0 or later. Basic installation instructions are provided below.

## Terminology

Environment variables:

* `$MAPD_PATH`: MapD install directory, e.g. `/opt/mapd/mapd2`
* `$MAPD_DATA`: MapD data directory, e.g. `/var/lib/mapd/data`

Programs and scripts:

* `mapd_server`: MapD database server. Located at `$MAPD_PATH/bin/mapd_server`.
* `mapd_web_server`: Web server which hosts the web-based frontend and provides database access over HTTP(S). Located at `$MAPD_PATH/bin/mapd_web_server`.
* `initdb`: Initializes the MapD data directory. Located at `$MAPD_PATH/bin/initdb`.
* `mapdql`: Command line-based program that gives direct access to the database. Located at `$MAPD_PATH/bin/mapdql`.
* `startmapd`: All-in-one script that will initialize a MapD data directory at `$MAPD_PATH/data`, offer to load a sample dataset, and then start the MapD server and web server. Located at `$MAPD_PATH/startmapd`.

Other

* `systemd`: init system used by most major Linux distributions. Sample `systemd` target files for starting MapD are provided in `$MAPD_PATH/systemd`.

## Installation

### CUDA Installation
CUDA-enabled installations of MapD depend on `libcuda` and `libnvvm` which are provided by the NVIDIA GPU drivers and NVIDIA CUDA Toolkit, respectively. As of January 2016, both CUDA 7.0 and CUDA 7.5 are supported by MapD.

The NVIDIA CUDA Toolkit, which includes the NVIDIA GPU drivers, is available at: https://developer.nvidia.com/cuda-downloads . 

The installation notes below are just a summary of what is required to install the CUDA Toolkit. Please see the [CUDA Quick Start Guide](http://developer.download.nvidia.com/compute/cuda/7.5/Prod/docs/sidebar/CUDA_Quick_Start_Guide.pdf) for full instructions.

Before proceeding, please make sure your system is completely up-to-date and you have restarted to activate the latest kernel, etc.

#### CentOS / Red Hat Enterprise Linux (RHEL)
Please download the RPM package provided by NVIDIA from https://developer.nvidia.com/cuda-downloads .

RHEL-based distributions require Dynamic Kernel Module Support (DKMS) in order to build the GPU driver kernel modules, which is provided by the Extra Packages for Enterprise Linux (EPEL) repository. See the [EPEL website](https://fedoraproject.org/wiki/EPEL) for complete instructions for enabling this repository.

1. Enable EPEL
```
sudo yum install epel-release
```

2. Install GCC and Linux headers
```
sudo yum groupinstall "Development Tools"
sudo yum install linux-headers
```

3. Install the CUDA repository, update local repository cache, and then install the CUDA Toolkit and GPU drivers
```
sudo rpm --install cuda-repo-<distro>-<version>.<architecture>.rpm
sudo yum clean expire-cache
sudo yum install cuda
```

#### Ubuntu / Debian
Please download the DEB package provided by NVIDIA from https://developer.nvidia.com/cuda-downloads .

Install the CUDA repository, update local repository cache, and then install the CUDA Toolkit and GPU drivers
```
sudo dpkg --install cuda-repo-<distro>-<version>.<architecture>.deb
sudo apt-get update
sudo apt-get install cuda
```

#### Mac OS X
Please download the DMG package provided by NVIDIA from https://developer.nvidia.com/cuda-downloads .

The DMG package will walk you through all required steps to install CUDA.

#### Environment Variables
MapD depends on `libcuda` and `libnvvm`, both of which must be available in your environment in order to run MapD. The NVIDIA GPU drivers usually make `libcuda` available by default by installing it to a system-wide `lib` directory such as `/usr/lib64` (on CentOS/RHEL) or `/usr/lib/x86_64-linux-gnu` (on Ubuntu). However, `libnvvm` is typically installed to a CUDA-specific directory which is not added to your environment by default (via `$LD_LIBRARY_PATH`). To make `libnvvm` available, add the following to `/etc/profile.d/cuda.sh` (on Linux) or your user's `$HOME/.bashrc`:
```
LD_LIBRARY_PATH=/usr/local/cuda-7.5/nvvm/lib64:$LD_LIBRARY_PATH
```
for Linux or
```
DYLD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-7.5/nvvm/lib:$DYLD_LIBRARY_PATH
```
for Mac OS X, where `/usr/local/cuda-7.5` and `/Developer/NVIDIA/CUDA-7.5` are the default CUDA Toolkit install directories.

#### Verifying Installation
After installing CUDA and setting up the environment variables, please restart your machine to activate the GPU drivers.

On Linux, you can verify installation of the GPU drivers by running `nvidia-smi`.

### MapD Installation
MapD is distributed as a self-extracting archive, that is, a shell script which contains the complete contents of a .tar.gz file. Other package types are available upon request.

To install, move the archive to the desired installation directory (`$MAPD_PATH`) and run:
```
sh mapd2-<date>-<hash>-<platform>-<architecture>.sh
```
replacing `mapd2-<date>-<hash>-<platform>-<architecture>.sh` with the name of the archive provided to you.

The installer will then present the EULA and, if accepted, ask for the installation path. 

#### Systemd
For Linux, the MapD archive includes `systemd` target files which allows `systemd` to manage MapD as a service on your server. The provided `install_mapd_systemd.sh` script will ask a few questions about your environment and then install the target files into the correct location.

```
cd $MAPD_PATH/systemd
./install_mapd_systemd.sh
```

## Configuration
Before starting MapD, the `data` directory must be initialized. To do so, create an empty directory at the desired path (`/var/lib/mapd/data`) and run `$MAPD_PATH/bin/initdb` with that path as the argument. For example:

```
sudo mkdir -p /var/lib/mapd/data
sudo $MAPD_PATH/bin/initdb /var/lib/mapd/data
```

Finally, make sure this directory is owned by the user that will be running MapD (i.e. `mapd`):
```
sudo chown -R mapd /var/lib/mapd
```

You can now test your installation of MapD with the `startmapd` script:
```
$MAPD_PATH/startmapd --data $MAPD_DATA
```

### Configuration file
MapD also supports storing options in a configuration file. This is useful if, for example, you need to run the MapD database and/or web servers on different ports than the default. An example configuration file is provided under `$MAPD_PATH/mapd.conf.sample`.

To use options provided in this file, provide the path the the config file to the `--config` flag of `startmapd` or `mapd_server` and `mapd_web_server`. For example:
```
$MAPD_PATH/startmapd --config $MAPD_DATA/mapd.conf
```

## Starting and Stopping MapD Services
MapD consists of two system services: `mapd_server` and `mapd_web_server`. These services may be started individually or run via the interactive script `startmapd`. For permanent installations, it is recommended that you use `systemd` to manage the MapD services.

### MapD Via `startmapd`
MapD may be run via the `startmapd` script provided in `$MAPD_PATH/startmapd`. This script handles creating the `data` directory if it does not exist, inserting a sample dataset if desired, and starting both `mapd_server` and `mapd_web_server`.

#### Starting MapD Via `startmapd`
To use `startmapd` to start MapD, run:
```
$MAPD_PATH/startmapd --config /path/to/mapd.conf
```
if using a configuration file, or
```
$MAPD_PATH/startmapd --data $MAPD_DATA
```
to explicitly specify the `$MAPD_DATA` directory.

#### Stopping MapD Via `startmapd`
To stop an instance of MapD that was started with the `startmapd` script, simply kill the `startmapd` process via `CTRL-C` or `pkill startmapd`. You can also use `pkill mapd` to ensure all processes have been killed.

### MapD Via `systemd`
For permenant installations of MapD, it is recommended that you use `systemd` to manage the MapD services. `systemd` automatically handles tasks such as log management, starting the services on restart, and restarting the services in case they die. It is assumed that you have followed the instructions above for installing the `systemd` service unit files for MapD.

#### Starting MapD Via `systemd`
To manually start MapD via `systemd`, run:
```
systemctl start mapd_server
systemctl start mapd_web_server
```

#### Stopping MapD Via `systemd`
To manually stop MapD via `systemd`, run:
```
systemctl stop mapd_server
systemctl stop mapd_web_server
```

#### Enabling MapD on Startup
To enable the MapD services to be started on restart, run:
```
systemctl enable mapd_server
systemctl enable mapd_web_server
```
