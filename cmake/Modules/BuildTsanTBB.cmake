# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# TSAN must observe synchronization inside the scheduler, not just its callers.
# Keep this version in sync with install_tbb() in scripts/common-functions.sh.
set(_tsan_tbb_version "2021.9.0")
set(_tsan_tbb_version_header "${TBB_INCLUDE_DIR}/oneapi/tbb/version.h")
if(NOT EXISTS "${_tsan_tbb_version_header}")
  message(FATAL_ERROR "TSAN requires oneTBB ${_tsan_tbb_version} headers in the dependency prefix")
endif()
foreach(_part MAJOR MINOR PATCH)
  file(STRINGS "${_tsan_tbb_version_header}" _version_line
    REGEX "^#define TBB_VERSION_${_part} [0-9]+")
  string(REGEX REPLACE "^#define TBB_VERSION_${_part} +([0-9]+).*" "\\1"
    _tsan_tbb_${_part} "${_version_line}")
endforeach()
set(_installed_tbb_version "${_tsan_tbb_MAJOR}.${_tsan_tbb_MINOR}.${_tsan_tbb_PATCH}")
if(NOT _installed_tbb_version STREQUAL _tsan_tbb_version)
  message(FATAL_ERROR
    "TSAN TBB source/header mismatch: building ${_tsan_tbb_version}, but found "
    "${_installed_tbb_version} in ${TBB_INCLUDE_DIR}. Update the pinned TSAN "
    "source together with the dependency recipe.")
endif()

include(ExternalProject)
find_package(Threads REQUIRED)
set(_tsan_tbb_prefix "${CMAKE_BINARY_DIR}/tsan-tbb")
set(_tsan_tbb_install "${_tsan_tbb_prefix}/install")
set(HEAVYDB_TSAN_TBB_SOURCE_URL
  "https://github.com/oneapi-src/oneTBB/archive/v${_tsan_tbb_version}.tar.gz"
  CACHE STRING "Pinned oneTBB source archive or local archive path for TSAN builds")
mark_as_advanced(HEAVYDB_TSAN_TBB_SOURCE_URL)

# Static linkage preserves the existing CI configuration and makes the packed
# test binaries independent of both build-tree paths and the base image's TBB.
ExternalProject_Add(heavydb_tsan_tbb_build
  PREFIX "${_tsan_tbb_prefix}"
  URL "${HEAVYDB_TSAN_TBB_SOURCE_URL}"
  URL_HASH SHA256=1ce48f34dada7837f510735ff1172f6e2c261b09460e3bf773b49791d247d24e
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  CMAKE_ARGS
    "-DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}"
    "-DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}"
    "-DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo"
    "-DCMAKE_POLICY_VERSION_MINIMUM:STRING=3.5"
    "-DCMAKE_INSTALL_PREFIX:PATH=${_tsan_tbb_install}"
    "-DCMAKE_INSTALL_LIBDIR:STRING=lib"
    "-DCMAKE_CXX_FLAGS:STRING=-fno-omit-frame-pointer"
    "-DBUILD_SHARED_LIBS:BOOL=OFF"
    "-DTBB_TEST:BOOL=OFF"
    "-DTBB_STRICT:BOOL=OFF"
    "-DTBB_ENABLE_IPO:BOOL=OFF"
    "-DTBBMALLOC_PROXY_BUILD:BOOL=OFF"
    "-DTBB_SANITIZE:STRING=thread"
  BUILD_BYPRODUCTS
    "${_tsan_tbb_install}/lib/libtbb.a"
    "${_tsan_tbb_install}/lib/libtbbmalloc.a"
)

# Repoint the discovered targets without changing FindTBB's cache entries, so
# turning ENABLE_TSAN off cannot leave instrumented libraries selected.
foreach(_component tbb tbbmalloc)
  set(_library "${_tsan_tbb_install}/lib/lib${_component}.a")
  set_target_properties(TBB::${_component} PROPERTIES
    IMPORTED_LOCATION "${_library}"
    IMPORTED_LOCATION_DEBUG "${_library}"
    IMPORTED_LOCATION_RELEASE "${_library}")
  set_property(TARGET TBB::${_component} APPEND PROPERTY
    INTERFACE_LINK_LIBRARIES "${CMAKE_DL_LIBS};Threads::Threads")
  add_dependencies(TBB::${_component} heavydb_tsan_tbb_build)
endforeach()
set(TBB_LIBRARY TBB::tbb)
set(TBB_LIBRARIES TBB::tbb)
set(TBB_MALLOC_LIBRARY TBB::tbbmalloc)
set(TBB_MALLOC_LIBRARIES TBB::tbbmalloc)
message(STATUS "TSAN: building instrumented oneTBB ${_tsan_tbb_version} with ${CMAKE_CXX_COMPILER}")
