Google Mock and Google Test
---------------------------

This is Google Test v1.8.1.

This directory contains [Google Test and Google
Mock](https://github.com/google/googletest), which have been fused together to
a single set of files.

Use of the source code in this directory is governed by the license provided
in this directory and at https://github.com/google/googletest/blob/master/LICENSE

### Instructions for regenerating fused gmock and gtest files

Remove the current files:
```
rm -rf gtest gmock gmock-gtest-all.cc
```

Grab the latest Google Test source from GitHub:
```
git clone https://github.com/google/googletest
```

Run the included script to fuse the files together:
```
python2 googletest/googlemock/scripts/fuse_gmock_files.py .
```

Remove source directory:
```
rm -rf googletest
```

Note: as of 2019-05-28 the fuse scripts are written for Python 2; running with
Python 3 will result in a `SyntaxError` due to a `print`.

### Usage

The provided CMakeLists.txt will generate a target named `gtest`.

The following assumes these files are in a directory named
`ThirdParty/googletest`.

In the top-level CMakeLists.txt add:
```
include_directories(ThirdParty/googletest)
add_subdirectory(ThirdParty/googletest)
```

To link against the Google Test library:
```
target_link_libraries(mylib gtest)
```
