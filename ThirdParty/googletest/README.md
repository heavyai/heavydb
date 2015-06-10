Google Mock and Google Test
---------------------------

This directory contains [Google Test](https://code.google.com/p/googletest/)
and [Google Mock](https://code.google.com/p/googlemock/), which have been fused
together for distribution.

Use of the source code in this directory is governed by the license provided
in this directory and at https://googletest.googlecode.com/svn/trunk/LICENSE

### Instructions for regenerating gmock and gtest files

Note: Google Mock includes all files from Google Test.

Remove the current files:
```
rm -rf gtest gmock gmock-gtest-all.cc
```

Grab the latest Google Mock source from SVN:
```
svn checkout http://googlemock.googlecode.com/svn/trunk/ googlemock-read-only
```

Run the included script to fuse the files together:
```
python googlemock-read-only/scripts/fuse_gmock_files.py .
```

Remove source directory:
```
rm -rf googlemock-read-only
```

The fuse scripts are written for Python 2; running with Python 3 will result in
a `SyntaxError`.

### Usage

The provided CMakeLists.txt will generate a target named `gtest`.

The following assumes these files are in a directory named
`ThirdParty/googletest`.

In the top-level CMakeLists.txt, add:
```
include_directories(ThirdParty/googletest)
add_subdirectory(ThirdParty/googletest)
```

To link against the Google Test library, do:
```
target_link_libraries(mylib gtest)
```
