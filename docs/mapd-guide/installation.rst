.. _installation:

Installation
============

Common Dependencies / CPU-only Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All installations of MapD Core depend on Java 1.6+/\ ``libjvm``, provided by
a Java Runtime Environment (JRE), and ``libldap``, provided by OpenLDAP
and similar packages. ``libldap`` is available by default on most common
Linux distributions and therefore does not require any special
installation. The following describes how to install a JRE and configure
the appropriate environment variables.

CentOS / Red Hat Enterprise Linux (RHEL)
----------------------------------------

``libjvm`` is provided by the packages ``java-1.7.0-openjdk-headless``
or ``java-1.8.0-openjdk-headless``. ``1.8.0`` is preferred, but not
currently required. To install run:

::

    sudo yum install java-1.8.0-openjdk-headless

By default a symlink pointing to the newly installed JRE will be placed
at ``/usr/lib/jvm/jre-1.8.0-openjdk``, with the ``libjvm`` library
residing in the subdirectory ``lib/amd64/server``. This subdirectory
must be added to your ``LD_LIBRARY_PATH`` environment variable in order
to start the MapD Core Server:

::

    export LD_LIBRARY_PATH=/usr/lib/jvm/jre-1.8.0-openjdk/lib/amd64/server:$LD_LIBRARY_PATH

This command may be added to any file managing your environment such as
``$HOME/.bash_profile``, ``/etc/profile``, or
``/etc/profile.d/java.sh``.

Ubuntu / Debian
---------------

``libjvm`` is provided by the package ``default-jre-headless``. To
install run:

::

    sudo apt install default-jre-headless

By default a symlink pointing to the newly installed JRE will be placed
at ``/usr/lib/jvm/default-java``, with the ``libjvm`` library residing
in the subdirectory ``jre/lib/amd64/server``. This subdirectory must be
added to your ``LD_LIBRARY_PATH`` environment variable in order to start
the MapD Core Server:

::

    export LD_LIBRARY_PATH=/usr/lib/jvm/default-java/jre/lib/amd64/server:$LD_LIBRARY_PATH

This command may be added to any file managing your environment such as
``$HOME/.bash_profile``, ``/etc/profile``, or
``/etc/profile.d/java.sh``.

Xorg and NVIDIA GPU Driver Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CUDA-enabled installations of MapD Core depend on ``libcuda``, which is
provided by the NVIDIA GPU Drivers and NVIDIA CUDA Toolkit. The backend
rendering feature of MapD Core additionally requires a working installation
of Xorg.

The NVIDIA CUDA Toolkit, which includes the NVIDIA GPU drivers, is
available from the `NVIDIA CUDA
Zone <https://developer.nvidia.com/cuda-downloads>`__.

The installation notes below are just a summary of what is required to
install the CUDA Toolkit. Please see the `CUDA Quick Start
Guide <http://developer.download.nvidia.com/compute/cuda/7.5/Prod/docs/sidebar/CUDA_Quick_Start_Guide.pdf>`__
for full instructions.

Before proceeding, please make sure your system is completely up-to-date
and you have restarted to activate the latest kernel, etc.

CentOS / Red Hat Enterprise Linux (RHEL)
----------------------------------------

Please download the network install RPM package provided by NVIDIA from
the `NVIDIA CUDA Zone <https://developer.nvidia.com/cuda-downloads>`__.

RHEL-based distributions require Dynamic Kernel Module Support (DKMS) in
order to build the GPU driver kernel modules, which is provided by the
Extra Packages for Enterprise Linux (EPEL) repository. See the `EPEL
website <https://fedoraproject.org/wiki/EPEL>`__ for complete
instructions for enabling this repository.

1. Enable EPEL

   ::

       sudo yum install epel-release

2. Install Xorg and required libraries. This is only required to take
   advantage of the backend rendering features of MapD Core.

   ::

       sudo yum install xorg-x11-server-Xorg mesa-libGLU libXv

3. Update and reboot The GPU drivers and their dependencies,
   specifically ``kernel-devel`` and ``kernel-headers``, require that
   the latest available kernel is installed and active. To ensure this,
   update the entire system and reboot to activate the latest kernel:

   ::

       sudo yum update
       sudo reboot

4. Install the CUDA repository, update local repository cache, and then
   install the GPU drivers. ``yum`` will automatically install some
   additional dependencies for the ``cuda-drivers`` package, including:
   ``dkms``, ``gcc``, ``kernel-devel``, and ``kernel-headers``. The CUDA
   Toolkit (package ``cuda``) is *not* required to run MapD Core, but the GPU
   drivers (package ``cuda-drivers``, which include libcuda) are.

   ::

       sudo rpm --install cuda-repo-rhel7-8.0.44-1.x86_64.rpm
       sudo yum clean expire-cache
       sudo yum install cuda-drivers

   Where ``cuda-repo-rhel7-8.0.44-1.x86_64.rpm`` is the name of the RPM
   package provided by NVIDIA.

5. Reboot and continue to section `Environment
   Variables <#environment-variables>`__ below.

Ubuntu / Debian
---------------

Please download the DEB package provided by NVIDIA from the `NVIDIA CUDA
Zone <https://developer.nvidia.com/cuda-downloads>`__.

1. Install Xorg and required libraries, and disable the automatically
   enabled ``graphical`` target. This is only required to take advantage
   of the backend rendering features of MapD Core.

   ::

       sudo apt install xserver-xorg libglu1-mesa
       sudo systemctl set-default multi-user

2. Install the CUDA repository, update local repository cache, and then
   install the CUDA Toolkit and GPU drivers

   ::

       sudo dpkg --install cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
       sudo apt update
       sudo apt install cuda-drivers linux-image-extra-virtual

   Where ``cuda-repo-ubuntu1604_8.0.44-1_amd64.deb`` is the name of the
   package provided by NVIDIA.

3. Reboot and continue to section `Environment
   Variables <#environment-variables>`__ below.

Mac OS X
--------

Please download the DMG package provided by NVIDIA from the `NVIDIA CUDA
Zone <https://developer.nvidia.com/cuda-downloads>`__.

The DMG package will walk you through all required steps to install
CUDA.

Environment Variables
---------------------

For CPU-only installations of MapD Core, skip to section `MapD
Installation <#mapd-installation>`__ below.

MapD Core Server depends on ``libcuda``, which must be available in your environment
in order to run MapD Core. The NVIDIA GPU drivers usually make ``libcuda``
available by default by installing it to a system-wide ``lib`` directory
such as ``/usr/lib64`` (on CentOS/RHEL) or ``/usr/lib/x86_64-linux-gnu``
(on Ubuntu).

Verifying Installation
----------------------

After installing CUDA and setting up the environment variables, please
restart your machine to activate the GPU drivers.

On Linux, you can verify installation of the GPU drivers by running
``nvidia-smi``.

Xorg Configuration
~~~~~~~~~~~~~~~~~~

The ``nvidia-xconfig`` tool provided by the GPU drivers may be used to
generate a valid ``/etc/X11/xorg.conf``. To use, run:

::

    sudo nvidia-xconfig --use-display-device=none --enable-all-gpus --preserve-busid

Run the following to verify configuration:

::

    sudo X :1

If ``X`` starts without issues, kill it via ``<ctrl-c>`` (or
``sudo pkill X`` in a different session) and then proceed to `MapD
Installation <#mapd-installation>`__.

Troubleshooting
---------------

``no screens defined``, NVIDIA Tesla K20 GPUs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The NVIDIA Tesla K20 GPU requires graphics support to be explicitly
enabled in order to use Xorg. This mode may be enabled by running:

::

    sudo nvidia-smi --gom=0

``no screens defined``
^^^^^^^^^^^^^^^^^^^^^^

In rare circumstances ``nvidia-xconfig`` generates an ``xorg.conf`` that
does not include the PCIe BusID for each GPU. When this happens,
``X :1`` will fail with the error message ``no screens defined``. To
resolve this issue, verify that the BusIDs are not listed by opening
``/etc/X11/xorg.conf`` and look for the ``BusID`` option under each
``Section "Device"``. For example, you should see something similar to:

::

    Section "Device"
        Identifier     "Device0"
        Driver         "nvidia"
        VendorName     "NVIDIA Corporation"
        BoardName      "Tesla K80"
        BusID          "PCI:131:0:0"
    EndSection

    Section "Device"
        Identifier     "Device1"
        Driver         "nvidia"
        VendorName     "NVIDIA Corporation"
        BoardName      "Tesla K80"
        BusID          "PCI:132:0:0"
    EndSection

If the ``BusID`` is not listed, they may be determined by running the
command ``nvidia-smi``:

::

    $ nvidia-smi
    +-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  Tesla K80           On   | 0000:83:00.0     Off |                    0 |
    | N/A   29C    P8    26W / 149W |     74MiB / 11519MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   1  Tesla K80           On   | 0000:84:00.0     Off |                    0 |
    | N/A   25C    P8    29W / 149W |     74MiB / 11519MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+

In this case, the BusIDs are ``83:00.0`` and ``84:00.0``. Note: this
values are in hexadecimal and must be converted to decimal to use in
``xorg.conf``. One way to do this is by running ``echo $((16#xx))``,
replacing ``xx`` with the values from ``nvidia-smi``:

::

    $ echo $((16#83))
    131
    $ echo $((16#84))
    132

This means that the BusIDs to use would be ``PCI:131:0:0`` and
``PCI:132:0:0``. ``nvidia-smi`` can then be used to regenerate
``xorg.conf`` with these values:

::

    sudo nvidia-xconfig --use-display-device=none --busid=PCI:131:0:0 --busid=PCI:132:0:0

Note: On some systems, such as those provided by Amazon Web Services,
``nvidia-smi`` will report the BusID as, for example, ``00:03.0``. In
these cases the Xorg BusIDs would be of the form ``PCI:0:3:0``.

MapD Installation
~~~~~~~~~~~~~~~~~

MapD Core is distributed as a .tar.gz archive. Other package types are
available upon request.

To install, move the archive to the desired installation directory
(``$MAPD_PATH``) and run:

::

    tar -xvf mapd2-<date>-<hash>-<platform>-<architecture>.tar.gz

replacing ``mapd2-<date>-<hash>-<platform>-<architecture>.tar.gz`` with
the name of the archive provided to you. For example, a release for
x86-64 Linux built on 15 April 2016 will have the file name
``mapd2-20160415-86fec7b-Linux-x86_64.tar.gz``.
