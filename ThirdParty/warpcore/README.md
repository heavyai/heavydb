# WARPCORE

**Hashing at the speed of light on modern CUDA-accelerators**

## Introduction
`warpcore` is a framework for creating high-throughput, purpose-built hashing data structures on CUDA-accelerators.

This library provides the following data structures:
- [`SingleValueHashTable`](include/warpcore/single_value_hash_table.cuh): stores a set of key-value pairs
- [`HashSet`](include/warpcore/hash_set.cuh): stores a set of keys
- [`CountingHashTable`](include/warpcore/counting_hash_table.cuh): keeps track of the number of occurrences of each inserted key
- [`BloomFilter`](include/warpcore/bloom_filter.cuh): pattern-blocked bloom filter for approximate membership queries
- [`MultiValueHashTable`](include/warpcore/multi_value_hash_table.cuh): stores a multi-set of key-value pairs
- [`BucketListHashTable`](include/warpcore/bucket_list_hash_table.cuh): alternative variant of `MultiValueHashTable`
- [`MultiBucketHashTable`](include/warpcore/multi_bucket_hash_table.cuh): alternative variant of `MultiValueHashTable`

Implementations support key types `std::uint32_t` and `std::uint64_t` together with any trivially copyable value type. In order to be adaptable to a wide range of possible usecases, we provide a multitude of combinable modules such as [hash functions](include/warpcore/hashers.cuh), [probing schemes](include/warpcore/probing_schemes.cuh), and [data layouts](include/warpcore/storage.cuh) (visit the [documentation](https://sleeepyjack.github.io/warpcore/) for further information).

`warpcore` has won the best paper award at the [IEEE HiPC 2020 conference](https://hipc.org/) ([link to manuscript](https://ieeexplore.ieee.org/document/9406635))([link to preprint](https://arxiv.org/abs/2009.07914)) and is based on our previous work on massively parallel GPU hash tables `warpdrive` which has been published in the prestigious [IEEE IPDPS conference](https://www.ipdps.org/) ([link to manuscript](https://ieeexplore.ieee.org/document/8425198)).

## Development Status

This library is still under heavy development. Users should expect breaking changes and refactoring to be common.
Developement mainly takes place on our in-house GitLab instance. However, we plan to migrate to GitHub in the near future.

## Requirements
- [CUDA-capable device](https://developer.nvidia.com/cuda-gpus) with device architecture 6.0 or higher (Pascal+; see [link](https://en.wikipedia.org/wiki/CUDA#GPUs_supported))
- [NVIDIA CUDA toolkit/compiler](https://developer.nvidia.com/cuda-toolkit) version $\ge$ v11.2
- C++14 or higher

## Dependencies
- [hpc_helpers](https://gitlab.rlp.net/pararch/hpc_helpers) - utils, timers, etc.
- [kiss_rng](https://github.com/sleeepyjack/kiss_rng) - a fast and lightweight GPU PRNG
- [CUB](https://nvlabs.github.io/cub/) - high-throughput primitives for GPUs (already included in newer versions of the CUDA toolkit, i.e., $\ge$ v10.2)

**Note:** Dependencies are automatically managed via [CMake](https://cmake.org/).

## Getting `warpcore`

`warpcore` is header only and can be incorporated manually into your project by downloading the headers and placing them into your source tree.

### Adding `warpcore` to a CMake Project

`warpcore` is designed to make it easy to include within another CMake project.
 The `CMakeLists.txt` exports a `warpcore` target that can be linked<sup>[1](#link-footnote)</sup> into a target to setup include directories, dependencies, and compile flags necessary to use `warpcore` in your project.


We recommend using [CMake Package Manager (CPM)](https://github.com/TheLartians/CPM.cmake) to fetch `warpcore` into your project.
With CPM, getting `warpcore` is easy:

```
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

include(path/to/CPM.cmake)

CPMAddPackage(
  NAME warpcore
  GITHUB_REPOSITORY sleeepyjack/warpcore
  GIT_TAG/VERSION XXXXX
)

target_link_libraries(my_target warpcore)
```

This will take care of downloading `warpcore` from GitHub and making the headers available in a location that can be found by CMake. Linking against the `warpcore` target will provide everything needed for `warpcore` to be used by `my_target`.

<a name="link-footnote">1</a>: `warpcore` is header-only and therefore there is no binary component to "link" against. The linking terminology comes from CMake's `target_link_libraries` which is still used even for header-only library targets.

## Building `warpcore`

Since `warpcore` is header-only, there is nothing to build to use it.

To build the tests, benchmarks, and examples:

```
cd $WARPCORE_ROOT
mkdir -p build
cd build
cmake .. -DWARPCORE_BUILD_TESTS=ON -DDWARPCORE_BUILD_BENCHMARKS=ON -DDWARPCORE_BUILD_EXAMPLES=ON
make
```
Binaries will be built into:
- `build/tests/`
- `build/benchmarks/`
- `build/examples/`


## [Documentation](docs/index.html)

## Where to go from here?
Take a look at the [examples](examples/README.md), test your own system performance using the [benchmark suite](benchmarks/README.md) and be sure everything works as expected by running the [test suite](tests/README.md).

## How to cite `warpcore`?
BibTeX:
```console
@inproceedings{DBLP:conf/hipc/JungerKM0XLS20,
  author    = {Daniel J{\"{u}}nger and
               Robin Kobus and
               Andr{\'{e}} M{\"{u}}ller and
               Christian Hundt and
               Kai Xu and
               Weiguo Liu and
               Bertil Schmidt},
  title     = {WarpCore: {A} Library for fast Hash Tables on GPUs},
  booktitle = {27th {IEEE} International Conference on High Performance Computing,
               Data, and Analytics, HiPC 2020, Pune, India, December 16-19, 2020},
  pages     = {11--20},
  publisher = {{IEEE}},
  year      = {2020},
  url       = {https://doi.org/10.1109/HiPC50609.2020.00015},
  doi       = {10.1109/HiPC50609.2020.00015},
  timestamp = {Wed, 05 May 2021 09:45:30 +0200},
  biburl    = {https://dblp.org/rec/conf/hipc/JungerKM0XLS20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

@inproceedings{DBLP:conf/ipps/Junger0S18,
  author    = {Daniel J{\"{u}}nger and
               Christian Hundt and
               Bertil Schmidt},
  title     = {WarpDrive: Massively Parallel Hashing on Multi-GPU Nodes},
  booktitle = {2018 {IEEE} International Parallel and Distributed Processing Symposium,
               {IPDPS} 2018, Vancouver, BC, Canada, May 21-25, 2018},
  pages     = {441--450},
  publisher = {{IEEE} Computer Society},
  year      = {2018},
  url       = {https://doi.org/10.1109/IPDPS.2018.00054},
  doi       = {10.1109/IPDPS.2018.00054},
  timestamp = {Sat, 19 Oct 2019 20:31:38 +0200},
  biburl    = {https://dblp.org/rec/conf/ipps/Junger0S18.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
***
warpcore Copyright (C) 2018-2021 [Daniel Jünger](https://github.com/sleeepyjack)

This program comes with ABSOLUTELY NO WARRANTY.
This is free software, and you are welcome to redistribute it under certain
conditions. See the file [LICENSE](LICENSE) for details.

[repository]: https://github.com/sleeepyjack/warpcore


