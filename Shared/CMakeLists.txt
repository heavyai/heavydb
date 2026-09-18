# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2014-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

set(shared_source_files
    Datum.cpp
    StringTransform.cpp
    Compressor.cpp
    DateTimeParser.cpp
    File.cpp
    StackTrace.cpp
    base64.cpp
    misc.cpp
    timedate.cpp
    thread_count.cpp
    threading.cpp
    MathUtils.cpp
    file_path_util.cpp
    file_type.cpp
    TargetInfo.cpp
    JsonUtils.cpp
    DbObjectKeys.cpp
    QuotedIdentifierUtil.cpp
    FullyQualifiedTableName.cpp
    LonLatBoundingBox.cpp
    heavyai_fs.cpp
    heavyai_path.cpp
    heavyai_env.cpp
    heavyai_glob.cpp
    heavyai_hostname.cpp)

include_directories(${CMAKE_SOURCE_DIR})
list(APPEND shared_source_files Encryption.cpp heavyai_locks.cpp)

if(ENABLE_NVTX)
  list(APPEND shared_source_files nvtx_helpers.cpp)
endif()

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/funcannotations.h DESTINATION ${CMAKE_BINARY_DIR}/Shared/)

add_library(Shared ${shared_source_files} "boost_stacktrace.hpp")
target_link_libraries(Shared Logger ${BLOSC_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} ${OPENSSL_LIBRARIES})

# Required by ThriftClient.cpp
add_definitions("-DTHRIFT_PACKAGE_VERSION=\"${Thrift_VERSION}\"")
add_library(ThriftClient ThriftClient.cpp "boost_stacktrace.hpp")
target_link_libraries(ThriftClient  ${Thrift_LIBRARIES} ${Boost_LIBRARIES})
