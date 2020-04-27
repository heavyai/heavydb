The provided CMake files have been modified to silence various `message(STATUS)` calls. Grep for `# message` to find all of them.

Also disabled option `BENCHMARK_ENABLE_INSTALL` so that the libs are not included in our packages.
