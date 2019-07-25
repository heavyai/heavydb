OmniSciDB (formerly MapD Core)
==============================

OmniSciDB is an open source SQL-based, relational, columnar database engine. This project is specifically developed to harness the parallel processing power of graphics processing units (GPUs). OmniSciDB can query up to billions of rows in milliseconds, and benefits from the advantages that GPUs provide, such as parallelism or the ability to process in parallel, which can boost performance. OmniSciDB also uses multi-tiered memory caching, a Just-In-Time (JIT) query compilation framework and in-situ graphics rendering.

To find out more, please check out the [OmniSci Website](https://www.omnisci.com) and the [OmniSciDB wiki](https://github.com/snowcrash007/omniscidb/wiki/).

# Quickstart

- Install the dependencies mentioned in the [Dependencies](https://github.com/snowcrash007/omniscidb/wiki/OmniSciDB-Dependencies) page.
- [Download and Install](#DownloadsandInstallationInstructions) the package.
- Read up about [Usage and Configuration](#building)
- Initialize servers [using a wrapper](https://github.com/snowcrash007/omniscidb/wiki/OmniSciDB-Documentation/_edit#starting-using-the-startomnisci-wrapper) or [manually](https://github.com/snowcrash007/omniscidb/wiki/OmniSciDB-Documentation/_edit#starting-manually)
- [Load sample data.](#working-with-data) Sample sets can be found in the [Data Set Library](https://community.omnisci.com/browse/new-item2).
- Perform a Sample Query

## Diving In

- How-tos, code snippets and more on the [OmniSci Blog](https://www.omnisci.com/blog/)
- Tutorials
- Need a data set to practice with? Search the [Data Set Library](https://community.omnisci.com/browse/new-item2)
- Video overview of the Architecture

## Learn more
| [OmniSciDB](https://github.com/snowcrash007/omniscidb/wiki/OmniSciDB-Overview) | [Documentation](https://github.com/snowcrash007/omniscidb/wiki/OmniSciDB-Documentation) | [Setup](https://github.com/snowcrash007/omniscidb/wiki/Setup) | [Community & Resources](https://github.com/snowcrash007/omniscidb/wiki/Community-&-Resources) |
| -- | -- | --|--|
| Overview of OmniSciDB| Developer Friendly Technical Documentation | Step-by-step getting started documentation | Important links, community resources and updates |

# Downloads and Installation Instructions

OmniSci provides pre-built binaries for Linux for stable releases of the project:

| Distro | Package type | CPU/GPU | Download Link | Installation Guide |
| --- | --- | --- | --- | --- |
| CentOS | RPM | CPU | https://releases.omnisci.com/os/yum/stable/cpu | https://www.omnisci.com/docs/latest/4_centos7-yum-cpu-os-recipe.html |
| CentOS | RPM | GPU | https://releases.omnisci.com/os/yum/stable/cuda | https://www.omnisci.com/docs/latest/4_centos7-yum-gpu-os-recipe.html |
| Ubuntu | DEB | CPU | deb https://releases.omnisci.com/os/apt/ stable cpu | https://www.omnisci.com/docs/latest/4_ubuntu-apt-cpu-os-recipe.html |
| Ubuntu | DEB | GPU | deb https://releases.omnisci.com/os/apt/ stable cuda | https://www.omnisci.com/docs/latest/4_ubuntu-apt-gpu-os-recipe.html |
| * | tarball | CPU | https://releases.omnisci.com/os/tar/omnisci-os-latest-Linux-x86_64-cpu.tar.gz |  |
| * | tarball | GPU | https://releases.omnisci.com/os/tar/omnisci-os-latest-Linux-x86_64-cuda.tar.gz |  |

***

Get more detailed download instructions, videos, resources and tutorials by visiting our [Downloads](https://github.com/snowcrash007/omniscidb/wiki/OmniSciDB-Downloads) page and [Documentation](https://github.com/snowcrash007/omniscidb/wiki/OmniSciDB-Documentation).

# Contributing
Developers are encouraged to contribute to this Open Source project to expand and enhance OmniSciDB capabilities. Check out our Contributing page on the wiki! If you have questions and would like to connect with the maintainers of this open source project, please visit the official [online forum and community.](https://community.omnisci.com/home)

## Making your first contribution? 
Check out [contribution wishlist](https://github.com/omnisci/omniscidb/contribute) with highlighted issues that would make great first contributions to this project. We also recommend visiting the [Roadmap](ROADMAP.md) to learn more about upcoming features or checking out the [Issues](https://github.com/omnisci/omniscidb/issues) to see the latest updates or proposed content.

## Contributor License Agreement (CLA)
In order to clarify the intellectual property license granted with Contributions from any person or entity, OmniSci must have a Contributor License Agreement ("CLA") on file that has been signed by each Contributor, indicating agreement to the [Contributor License Agreement](CLA.txt). After making a pull request, a bot will notify you if a signed CLA is required and provide instructions for how to sign it. Please read the agreement carefully before signing and retain a copy for your records.

## Need Help?
Have questions? Post questions and get answers on the official [online forum and community](https://community.omnisci.com/home) or create an issue in the repo.

# Copyright & License
This project is licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).

The repository includes a number of third party packages provided under separate licenses. Details about these packages and their respective licenses is at [ThirdParty/licenses/index.md](ThirdParty/licenses/index.md).
