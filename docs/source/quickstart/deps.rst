.. OmniSciDB Quickstart

####################
Server Dependencies
####################

Installation scripts are provided for building and installing OmniSciDB dependencies for most major linux operating systems and MacOS in the `scripts` directory. Prebuilt dependencies for CentOS 7 and Ubuntu 16.04, 18.04, 19.04, and  can also be downloaded from OmniSci using the included `mapd-deps-prebuilt.sh` script (see `Prebuilt Dependencies`, below).

Dependencies for ``omnisci_web_server`` and other Go utils are in `ThirdParty/go`. See `ThirdParty/go/src/mapd/vendor/README.md` for instructions on how to add new deps.

Prebuilt Dependencies
=====================

OmniSciDB requires a number of dependencies which are not provided in the common linux distribution packages. While the developer is welcome to install these dependencies themselves using the included dependency script for their respective OS in the `scripts` directory, OmniSci provides versioned, prebuilt dependencies which can be downloaded using the `mapd-deps-prebuilt.sh` script in the `scripts` directory. The prebuilt dependencies are only available for CentOS 7 and Ubuntu 16.04, 18.04, 19.04.

First, use the `scripts/mapd-deps-prebuilt.sh` build script to download the prebuilt dependencies from OmniSci and install them. By default, the dependencies will be installed to the `/usr/local/mapd-deps` directory. The `mapd-deps-prebuilt.sh` script also sets up an `environment module <http://modules.sf.net>`_ in order to simplify managing the required environment variables. Log out and log back in after running the `mapd-deps-prebuilt.sh` script in order to activate the environment modules command, ``module``.

The ``mapd-deps`` environment module is disabled by default. To activate for your current session, run:

.. code-block::

    module load mapd-deps

To disable the `mapd-deps` module:

.. code-block::

    module unload mapd-deps

.. warning::

    The `mapd-deps` package contains newer versions of packages such as GCC and ncurses which might not be compatible with the rest of your environment. Make sure to disable the `mapd-deps` module before compiling other packages.

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

The `mapd-deps-prebuilt.sh` script includes two files with the appropriate environment variables: `mapd-deps-<date>.sh` (for sourcing from your shell config) and `mapd-deps-<date>.modulefile` (for use with environment modules). These files are placed in mapd-deps install directory, usually `/usr/local/mapd-deps/<date>`. Either of these may be used to configure your environment: the `.sh` may be sourced in your shell config; the `.modulefile` needs to be moved to the modulespath.

Building on Ubuntu
==================

Most build dependencies required by OmniSciDB are available via APT. Certain dependencies such as Thrift, Blosc, and Folly must be built as they either do not exist in the default repositories or have outdated versions. A prebuilt package containing all these dependencies is provided for Ubuntu 18.04 (x86_64). The dependencies will be installed to `/usr/local/mapd-deps/` by default; see the Environment Variables section below for how to add these dependencies to your environment.

These dependencies will be installed to a directory under `/usr/local/mapd-deps`. The `mapd-deps-prebuilt.sh` script above will generate a script named `mapd-deps.sh` containing the environment variables which need to be set. Simply source this file in your current session (or symlink it to `/etc/profile.d/mapd-deps.sh`) in order to activate it:

.. code-block:: shell

    source /usr/local/mapd-deps/mapd-deps.sh

To install: 

.. code-block:: shell

    cd scripts
    ./mapd-deps-ubuntu.sh --compress

Ubuntu 16.04
~~~~~~~~~~~~

OmniSciDB requires a newer version of Boost than the version which is provided by Ubuntu 16.04. The `scripts/mapd-deps-ubuntu1604.sh` build script will compile and install a newer version of Boost into the `/usr/local/mapd-deps/` directory.

Ubuntu 18.04
~~~~~~~~~~~~

Some installs of Ubuntu 18.04 may fail while building with a message similar to:

..code-block::

    java.security.InvalidAlgorithmParameterException: the trustAnchors parameter must be non-empty

This is a known issue in 18.04 which will be resolved in `Ubuntu 18.04.1 <https://bugs.launchpad.net/ubuntu/+source/ca-certificates-java/+bug/1739631>`_. To resolve on 18.04:

.. code-block:: shell

    sudo rm /etc/ssl/certs/java/cacerts
    sudo update-ca-certificates -f

Recent versions of Ubuntu provide the NVIDIA CUDA Toolkit and drivers in the standard repositories. To install:

.. code-block:: shell

    sudo apt install -y \
        nvidia-cuda-toolkit

Be sure to reboot after installing in order to activate the NVIDIA drivers.

Building on CentOS 
==================

The `scripts/mapd-deps-centos.sh` script is used to build the dependencies for CentOS. Modify this script and run if you would like to change dependency versions or to build on alternative CPU architectures.

.. code-block:: shell 

    cd scripts
    module unload mapd-deps
    ./mapd-deps-centos.sh --compress

Building on macOS
=================

The `scripts/mapd-deps-osx.sh` is used to build the dependencies for macOS. Note that the macOS deps script will automatically install and/or update `Homebrew <http://brew.sh/>`_ and use that to install all dependencies. Please make sure macOS is completely up to date and `Xcode` is installed before running. `Xcode` can be installed from the App Store.

Notes:
* `mapd-deps-osx.sh` will automatically install CUDA via Homebrew and add the correct environment variables to `~/.bash_profile`.
* `mapd-deps-osx.sh` will automatically install Java and Maven via Homebrew and add the correct environment variables to `~/.bash_profile`.

Building on Arch Linux
======================

`scripts/mapd-deps-arch.sh` is provided that will use `yay <https://aur.archlinux.org/packages/yay/>`_ to install packages from the `Arch User Repository <https://wiki.archlinux.org/index.php/Arch_User_Repository>`_. Note that some dependencies may be installed using custom `PKGBUILD` scripts. See the `scripts/arch` directory for an up to date list of packages requiring a custom `PKGBUILD`. If you don't have yay yet, install it first: https://github.com/Jguer/yay#installation

..note::
    Apache Arrow, while available in the AUR, requires a few custom build flags in order to be used with Core. A custom PKGBUILD for it is included.

.. note::
    packages aws-sdk-cpp and folly, while available in the AUR, are not supported while for building Core on Arch. If these packages are installed, support for them should be disabled when building Core. To do so, use the following options when running CMake:

    ``cmake -DENABLE_FOLLY=off -DENABLE_AWS_S3=off ..``

CUDA
----

CUDA and the NVIDIA drivers may be installed using the following.

..code-block:: shell

    yay -S \
        linux-headers \
        cuda \
        nvidia

Be sure to reboot after installing in order to activate the NVIDIA drivers.
