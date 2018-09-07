MapD Core
=========

MapD Core is an in-memory, column store, SQL relational database that was designed from the ground up to run on GPUs.

Selected details about this project are listed below. For the full details about building from source, installing MapD Core and contributing to the project, please see the [wiki](https://github.com/mapd/mapd-core/wiki).

# License

This project is licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).

The repository includes a number of third party packages provided under separate licenses. Details about these packages and their respective licenses is at [ThirdParty/licenses/index.md](ThirdParty/licenses/index.md).

The standard build process for this project downloads the Community Edition of the MapD Immerse visual analytics client. This version of MapD Immerse is governed by a separate license agreement, included in the file `EULA-CE.txt`, and may only be used for non-commercial purposes.

# Contributing

In order to clarify the intellectual property license granted with Contributions from any person or entity, MapD must have a Contributor License Agreement ("CLA") on file that has been signed by each Contributor, indicating agreement to the [Contributor License Agreement](CLA.txt). After making a pull request, a bot will notify you if a signed CLA is required and provide instructions for how to sign it. Please read the agreement carefully before signing and keep a copy for your records.

# Dependencies

MapD has the following dependencies:

| Package | Min Version | Required |
| ------- | ----------- | -------- |
| [CMake](https://cmake.org/) | 3.3 | yes |
| [LLVM](http://llvm.org/) | 3.8-4.0, 6.0 | yes |
| [GCC](http://gcc.gnu.org/) | 5.1 | no, if building with clang |
| [Go](https://golang.org/) | 1.6 | yes |
| [Boost](http://www.boost.org/) | 1.65.0 | yes |
| [OpenJDK](http://openjdk.java.net/) | 1.7 | yes |
| [CUDA](http://nvidia.com/cuda) | 8.0 | yes, if compiling with GPU support |
| [gperftools](https://github.com/gperftools/gperftools) | | yes |
| [gdal](http://gdal.org/) | | yes |
| [Arrow](https://arrow.apache.org/) | 0.7.0 | yes |

Dependencies for `mapd_web_server` and other Go utils are in [`ThirdParty/go`](ThirdParty/go). See [`ThirdParty/go/src/mapd/vendor/README.md`](ThirdParty/go/src/mapd/vendor/README.md) for instructions on how to add new deps.
