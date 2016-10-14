Dependencies
============

MapD is distributed as a group of mostly statically-linked executables,
which minimizes the number of dependencies required. The following are
the minimum requirements for running MapD.

Basic installation instructions for all dependencies are provided in the
`Installation <#installation>`__ section below.

Operating Systems, Officially Supported
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  CentOS/RHEL 7.0 or later.
-  Ubuntu 15.04 or later.

Operating Systems, Not Officially Supported
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Available on a case-by-case basis.

-  CentOS/RHEL 6.x.
-  Ubuntu 14.x.
-  Debian 7.x.
-  Debian 8.x. Lacks official driver support from NVIDIA.
-  Mac OS X 10.9 or later, 2013 or later hardware. Mac OS X builds do
   not support CUDA or backend rendering.

Libraries and Drivers
~~~~~~~~~~~~~~~~~~~~~

-  libjvm, provided by Java 1.6 or later.
-  libldap.
-  NVIDIA GPU Drivers. Not required for CPU-only installations.
-  Xorg. Only required to utilize MapD's backend rendering feature.
