# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2014-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 3.25)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Required for LLVM / JIT, alternative to setting CMP0065 to OLD
set(CMAKE_ENABLE_EXPORTS True)

# Get architecture
EXECUTE_PROCESS(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCH)
message(STATUS "Architecture is ${ARCH}")

# Check for Redhat based OS
find_file(IS_REDHAT redhat-release PATHS /etc)
if(IS_REDHAT)
  message(STATUS "OS is Redhat or derivative")
endif()

# force `Release` build type if left unspecified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
string(TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_LOWER)

if("${CMAKE_VERSION}" VERSION_GREATER 3.11.999)
  cmake_policy(SET CMP0074 NEW)
endif()

find_program(CCACHE_EXE ccache)
if(CCACHE_EXE)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_EXE}")
endif()

project(heavyai)

# ENABLE_IWYU: Generates .iwyu files suitable for input into include-what-you-use's fix_includes.py tool.
option(ENABLE_IWYU "Enable include-what-you-use advice (.iwyu files for fix_includes.py)" OFF)
if(ENABLE_IWYU)
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    find_program(IWYU_EXE include-what-you-use)
    if(IWYU_EXE)
      set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE "${CMAKE_SOURCE_DIR}/scripts/do-iwyu;${IWYU_EXE};-Xiwyu --cxx${CMAKE_CXX_STANDARD}ns")
    else()
      message(FATAL_ERROR "ENABLE_IWYU failed to find the include-what-you-use binary")
    endif()
  else()
    message(FATAL_ERROR "ENABLE_IWYU may only be used with the clang compiler (need to set CMAKE_CXX_COMPILER)")
  endif()
endif()

# ENABLE_TIME_TRACE: Generates .json files with build timings viewable in a chrome://tracing/ flame graph.
option(ENABLE_TIME_TRACE "Enable build time tracing (.json files for chrome://tracing/)" OFF)
if(ENABLE_TIME_TRACE)
  if(("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9.0))
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-ftime-trace>")
    # Adding -ftime-trace to clang++ for linking seems to have no effect.
    # Leaving the commented-out setting here for further investigation.
    #add_link_options("$<$<LINK_LANGUAGE:CXX>:-ftime-trace>")  # For after we upgrade to: cmake 3.18+
    #add_link_options("-ftime-trace")
  else()
    message(FATAL_ERROR "ENABLE_TIME_TRACE may only be used with the clang compiler 9.0+ (need to set CMAKE_CXX_COMPILER)")
  endif()
endif()

# HeavyDB version number
set(MAPD_VERSION_MAJOR "10")
set(MAPD_VERSION_MINOR "0")
set(MAPD_VERSION_PATCH "0")
set(MAPD_VERSION_EXTRA "dev")
set(MAPD_VERSION_RAW "${MAPD_VERSION_MAJOR}.${MAPD_VERSION_MINOR}.${MAPD_VERSION_PATCH}${MAPD_VERSION_EXTRA}")

set(MAPD_BUILDS_REPO_URL "http://builds.mapd.com")
set(MAPD_BUILDS_REPO_PATH "/theHoard/export/home/www/builds.mapd.com")

set(MAPD_IMMERSE_BUILD_ID "immerse-v2-latest-master-prod" CACHE STRING "Immerse Build ID")
set(MAPD_IMMERSE_URL "${MAPD_BUILDS_REPO_URL}/frontend/${MAPD_IMMERSE_BUILD_ID}.zip")

set(MAPD_WEBSERVER_BUILD_ID "webserver/master" CACHE STRING "Web Server Build ID")
set(MAPD_WEBSERVER_BASE_URL "${MAPD_BUILDS_REPO_URL}/frontend/${MAPD_WEBSERVER_BUILD_ID}/")

string(TIMESTAMP MAPD_BUILD_DATE "%Y%m%d")

if($ENV{BUILD_NUMBER})
  set(MAPD_BUILD_NUMBER "$ENV{BUILD_NUMBER}")
else()
  set(MAPD_BUILD_NUMBER "dev")
endif()
set(MAPD_VERSION "${MAPD_VERSION_RAW}-${MAPD_BUILD_NUMBER}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

add_custom_target(clean-all
  COMMAND ${CMAKE_BUILD_TOOL} clean
 )

# Find dependency path (required for boost and geo files)
find_file(MapdDepsVersion_FILE mapd_deps_version.txt CMAKE_PREFIX_PATH NO_CACHE)
if(NOT MapdDepsVersion_FILE)
  message(WARNING "Build could NOT find deps version file mapd_deps_version.txt in \"${CMAKE_PREFIX_PATH}\"")
  message(WARNING "Unable to locate dependencies path")
else()
  get_filename_component(MAPD_DEPS_PATH "${MapdDepsVersion_FILE}" DIRECTORY)
  message(STATUS "Found dependencies path: ${MAPD_DEPS_PATH}")
  list(APPEND CMAKE_LIBRARY_PATH
    ${MAPD_DEPS_PATH}
    ${MAPD_DEPS_PATH}/lib
    ${MAPD_DEPS_PATH}/lib64)
endif()

macro(set_alternate_linker linker)
  find_program(LINKER_EXECUTABLE ld.${USE_ALTERNATE_LINKER} ${USE_ALTERNATE_LINKER})
  if(LINKER_EXECUTABLE)
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" AND "${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS 12.0.0)
      add_link_options("-ld-path=${USE_ALTERNATE_LINKER}")
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND "${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS 12.1.0 AND "${USE_ALTERNATE_LINKER}" STREQUAL "mold")
      # LINKER_EXECUTABLE will be a full path to ld.mold, so we replace the end of the path, resulting in the relative
      # libexec/mold dir, and tell GCC to look there first for an override version of executables, in this case, ld
      string(REPLACE "bin/ld.mold" "libexec/mold" PATH_TO_LIBEXEC_MOLD ${LINKER_EXECUTABLE})
      add_link_options("-B${PATH_TO_LIBEXEC_MOLD}")
    else()
      add_link_options("-fuse-ld=${USE_ALTERNATE_LINKER}")
    endif()
  else()
    set(USE_ALTERNATE_LINKER "" CACHE STRING "Use alternate linker" FORCE)
  endif()
endmacro()

macro(InstallVersionFile)
  if(MapdDepsVersion_FILE)
    message(STATUS "Found deps version file ${MapdDepsVersion_FILE}")
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/version)
    set(HEAVY_DEPS_CUSTOM ${CMAKE_BINARY_DIR}/version/heavyai_deps_version.txt)
    # cp mapd_version data to heavyai deps, removing un-tagged lines.
    add_custom_command(OUTPUT ${HEAVY_DEPS_CUSTOM}
      DEPENDS ${MapdDepsVersion_FILE}
      COMMAND ${CMAKE_COMMAND} -E copy ${MapdDepsVersion_FILE} ${HEAVY_DEPS_CUSTOM}
      # In the copied deps files leave the first line with the deps generated info
      # and any other line starting with 'Public Release:', though remove the public release tag
      VERBATIM
      COMMAND "sed" "-i" "-n" "/^Public Release:/s/^Public Release://p" ${HEAVY_DEPS_CUSTOM})

    add_custom_target(HeavyDepsVersionTarget DEPENDS ${HEAVY_DEPS_CUSTOM})
    add_dependencies(heavydb HeavyDepsVersionTarget)
    install(FILES ${HEAVY_DEPS_CUSTOM} DESTINATION "."  COMPONENT "doc")
  endif()
endmacro()

set(USE_ALTERNATE_LINKER "" CACHE STRING "Use alternate linker. Leave empty for system default; alternatives are 'gold', 'lld', 'bfd', 'mold'")
if("${ARCH}" STREQUAL "aarch64" AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  message(STATUS "Detected GCC build on ARM, forcing use of mold linker")
  set(USE_ALTERNATE_LINKER "mold" CACHE STRING "Use alternate linker" FORCE)
endif()
if(NOT "${USE_ALTERNATE_LINKER}" STREQUAL "")
  set_alternate_linker(${USE_ALTERNATE_LINKER})
endif()

if (IS_REDHAT)
  # RedHat currently can't link without static libraries, so default to ON.
  option(PREFER_STATIC_LIBS "Prefer linking against static libraries" ON)
  message("Redhat system detected, setting PREFER_STATIC_LIBS=ON by default.")
else()
  option(PREFER_STATIC_LIBS "Prefer linking against static libraries" OFF)
endif (IS_REDHAT)

if(PREFER_STATIC_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
  set(BUILD_SHARED_LIBS OFF)
  set(Arrow_USE_STATIC_LIBS ON)
  set(Boost_USE_STATIC_LIBS ON)
  set(EXPAT_USE_STATIC_LIBS ON)
  set(LibOpenSaml_USE_STATIC_LIBS ON)
  set(OPENSSL_USE_STATIC_LIBS ON)
  set(RdKafka_USE_STATIC_LIBS ON)
  set(Thrift_USE_STATIC_LIBS ON)
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++")
  endif()

  set(CUDA_USE_STATIC_CUDA_RUNTIME ON CACHE STRING "Use static CUDA runtime")

  # On ppc, build failures occur for targets that depend on locale related functions due to unresolved symbols that are
  # present in the stdc++ library. Add the library flag to these targets to be used in resolving these symbols.
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "ppc64le")
    set(LOCALE_LINK_FLAG "-lstdc++")
  endif()
else()
  add_definitions("-DBOOST_LOG_DYN_LINK")
endif()

# Required with Boost 1.71.0+
# See https://gitlab.kitware.com/cmake/cmake/issues/19714
set(Boost_NO_BOOST_CMAKE 1)

option(ENABLE_JAVA_REMOTE_DEBUG "Enable Java Remote Debug" OFF )
if(ENABLE_JAVA_REMOTE_DEBUG)
  add_definitions("-DENABLE_JAVA_REMOTE_DEBUG")
endif()

# Disable by default until a more satisfactory resolution to QE-215 is found.
option(ENABLE_UTM_TRANSFORM "Enable ST_TRANSFORM() with UTM Support" OFF )
if(ENABLE_UTM_TRANSFORM )
  add_definitions("-DENABLE_UTM_TRANSFORM")
endif()

option(ENABLE_CUDA "Enable CUDA support" ON)
if(ENABLE_CUDA)
  enable_language(CUDA)
  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  include_directories(ThirdParty/warpcore/include)
  include_directories(ThirdParty/hpc_helpers/include)
  list(APPEND CUDA_LIBRARIES cuda)
  list(APPEND CMAKE_CUDA_FLAGS --extended-lambda)  # required by warpcore
  add_definitions("-DHAVE_CUDA")
else()
  set(CUDA_LIBRARIES "")
  set(MAPD_PACKAGE_FLAGS "${MAPD_PACKAGE_FLAGS}-cpu")
endif()

# CUDA architecture flags
if("${CMAKE_BUILD_TYPE_LOWER}" STREQUAL "debug")
  option(ENABLE_ONLY_ONE_ARCH "Enable quicker building for only one GPU arch" ON)
else()
  option(ENABLE_ONLY_ONE_ARCH "Enable quicker building for only one GPU arch" OFF)
endif()
if(ENABLE_CUDA)
  set(MAPD_CUDA_OPTIONS)
  # Set Thrust debug mode for CUDA compilation project-wide
  string(TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_UPPERCASE)
  if(CMAKE_BUILD_TYPE_UPPERCASE MATCHES DEBUG)
    list(APPEND MAPD_CUDA_OPTIONS -DTHRUST_DEBUG --debug)
  else()
    list(APPEND MAPD_CUDA_OPTIONS -O3)
  endif()

  list(APPEND MAPD_CUDA_OPTIONS -Xcompiler -fPIC -D_FORCE_INLINES -std=c++${CMAKE_CXX_STANDARD})

  if(ENABLE_ONLY_ONE_ARCH)
    execute_process(
      COMMAND cmake -S ${CMAKE_SOURCE_DIR}/NvidiaComputeCapability -B NvidiaComputeCapability
      OUTPUT_QUIET
      ERROR_QUIET
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    execute_process(
      COMMAND cmake --build NvidiaComputeCapability
      OUTPUT_FILE ${CMAKE_BINARY_DIR}/NvidiaComputeCapability/build.out.txt
      ERROR_FILE ${CMAKE_BINARY_DIR}/NvidiaComputeCapability/build.err.txt
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    set(NVIDIA_COMPUTE_CAPABILITY "")
    if (EXISTS ${CMAKE_BINARY_DIR}/NvidiaComputeCapability.txt)
      file(STRINGS ${CMAKE_BINARY_DIR}/NvidiaComputeCapability.txt NVIDIA_COMPUTE_CAPABILITY)
    endif()
  endif()
  if (ENABLE_ONLY_ONE_ARCH AND NOT "${NVIDIA_COMPUTE_CAPABILITY}" STREQUAL "")
    # set one arch for regular CUDA compiles
    set(CMAKE_CUDA_ARCHITECTURES ${NVIDIA_COMPUTE_CAPABILITY}-virtual)
    list(APPEND MAPD_CUDA_OPTIONS -Wno-deprecated-gpu-targets)
    # set one arch for custom_command CUDA compiles
    set (CUSTOM_COMMAND_CUDA_ARCHITECTURES
      -gencode=arch=compute_${NVIDIA_COMPUTE_CAPABILITY},code=compute_${NVIDIA_COMPUTE_CAPABILITY}
      -Wno-deprecated-gpu-targets
    )
    # clean up compute capability build
    add_custom_target(clean_nvidia_compute_capability
      COMMAND ${CMAKE_BUILD_TOOL} clean
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/NvidiaComputeCapability
    )
    add_dependencies(clean-all clean_nvidia_compute_capability)
  else()
    # set all supported archs for regular CUDA compiles
    set(CMAKE_CUDA_ARCHITECTURES
        60-virtual
        70-virtual
        75-virtual
        80-virtual
        86-virtual
        89-virtual
        90-virtual)
    list(APPEND MAPD_CUDA_OPTIONS -Wno-deprecated-gpu-targets)
    # set all supported archs for custom_command CUDA compiles
    set(CUSTOM_COMMAND_CUDA_ARCHITECTURES
      -gencode=arch=compute_60,code=compute_60;
      -gencode=arch=compute_70,code=compute_70;
      -gencode=arch=compute_75,code=compute_75;
      -gencode=arch=compute_80,code=compute_80;
      -gencode=arch=compute_86,code=compute_86;
      -gencode=arch=compute_89,code=compute_89;
      -gencode=arch=compute_90,code=compute_90;
      -Wno-deprecated-gpu-targets)
    if(ENABLE_ONLY_ONE_ARCH)
      message(STATUS "ENABLE_ONLY_ONE_ARCH ignored because NvidiaComputeCapability.txt not found or not readable")
    endif()
  endif()
  # report architectures to use
  message(STATUS "Building for CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
  # use multi-threaded nvcc?
  if("${CMAKE_CUDA_COMPILER_ID}" STREQUAL "NVIDIA")
    include(ProcessorCount)
    ProcessorCount(N)
    if(CMAKE_CUDA_COMPILER_VERSION GREATER_EQUAL 11.3 AND NOT N EQUAL 0)
      message(STATUS "Enabling NVCC multi-threaded compilation with ${N} threads.")
      # Adding to MAPD_CUDA_OPTIONS ensures that code that is compiled via generator expression (eg thrust) uses threads
      # This breaks the VSCode language server with CMake versions > 16.5, so disable this for now
      # list(APPEND MAPD_CUDA_OPTIONS --threads ${N})
      set(NVCC_THREADS --threads ${N}) # used in explicit custom build steps which is most kernel compilation
    endif()
  endif()

  add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:${MAPD_CUDA_OPTIONS}>")
endif()

option(ENABLE_NVTX "Enable NVidia Tools Extension library" OFF)
if(ENABLE_NVTX)
  if(NOT ENABLE_CUDA)
    set(ENABLE_NVTX OFF CACHE BOOL "Enable NVidia Tools Extension library" FORCE)
    message(STATUS "Cuda must be enabled to use NVTX, disabling NVTX support.")
  else()
    find_package(NVTX)
    if (NVTX_FOUND)
      message(STATUS "Using NVTX profiling markers")
      add_definitions("-DHAVE_NVTX")
    else()
      set(ENABLE_NVTX OFF CACHE BOOL "Enable NVidia Tools Extension library" FORCE)
      message(STATUS "NVTX not found, disabling NVTX support.")
    endif()
  endif()
endif()

option(SUPPRESS_NULL_LOGGER_DEPRECATION_WARNINGS "Suppress NullLogger deprecated warnings.")
if (SUPPRESS_NULL_LOGGER_DEPRECATION_WARNINGS)
  add_definitions("-DSUPPRESS_NULL_LOGGER_DEPRECATION_WARNINGS")
endif()

option(ENABLE_CUDA_KERNEL_DEBUG "Enable debugging symbols for CUDA device Kernels" OFF)

option(ENABLE_JIT_DEBUG "Enable debugging symbols for the JIT" OFF)
if (ENABLE_JIT_DEBUG)
  add_definitions("-DWITH_JIT_DEBUG")
endif()

option(ENABLE_GEOS "Enable GEOS Support" ON)
if (ENABLE_GEOS)
  find_package(GEOS)
  if(GEOS_NOTFOUND)
    set(ENABLE_GEOS OFF CACHE BOOL "Enable GEOS support" FORCE)
    message(STATUS "GEOS not found, disabling support.")
  else()
    set(GEOS_LIBRARY_FILENAME '"${GEOS_LIBRARY}"')
    add_definitions("-DENABLE_GEOS -DGEOS_LIBRARY_FILENAME=${GEOS_LIBRARY_FILENAME}")
    set(GEOS_RT_DEFINITIONS "-DENABLE_GEOS")
  endif()
endif()

find_package(OpenSSL REQUIRED)
add_definitions("-DOPENSSL_SUPPRESS_DEPRECATED")
include_directories(${OPENSSL_INCLUDE_DIR})

find_package(Thrift REQUIRED)
include_directories(${Thrift_INCLUDE_DIRS})
if("${Thrift_VERSION}" VERSION_LESS "0.13.0")
  add_definitions("-DHAVE_THRIFT_PLATFORMTHREADFACTORY")
else()
  add_definitions("-DHAVE_THRIFT_THREADFACTORY")
  if("${Thrift_VERSION}" VERSION_GREATER_EQUAL "0.14.0")
    add_definitions("-DHAVE_THRIFT_MESSAGE_LIMIT")
  endif()
endif()

if(IS_REDHAT)
  list(APPEND CMAKE_MODULE_PATH "${MAPD_DEPS_PATH}/lib64/cmake/gdal/packages")
else()
  list(APPEND CMAKE_MODULE_PATH "${MAPD_DEPS_PATH}/lib/cmake/gdal/packages")
endif()

find_package(Git)
find_package(PNG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(GDAL REQUIRED)
find_package(GDALExtra REQUIRED)
find_package(BLOSC REQUIRED)
find_package(EXPAT REQUIRED)
if(PREFER_STATIC_LIBS)
  find_package(LibKML REQUIRED)
  list(APPEND GDAL_LIBRARIES ${LIBKML_LIBRARIES})
endif()
if (NOT IS_REDHAT)
  find_package(PROJ REQUIRED)
  find_package(TIFF REQUIRED)
endif()
find_library(XML2_LIBRARY xml2 REQUIRED)
list(APPEND GDAL_LIBRARIES ${PNG_LIBRARIES} ${GDALExtra_LIBRARIES} ${EXPAT_LIBRARIES})
if (NOT IS_REDHAT)
  list (APPEND GDAL_LIBRARIES ${PROJ_LIBRARIES} ${TIFF_LIBRARIES}) 
endif()
list (APPEND GDAL_LIBRARIES ${XML2_LIBRARY})
include_directories(${GDAL_INCLUDE_DIRS})

option(ENABLE_SYSTEM_TFS "Enable system table functions" ON)
option(ENABLE_ML_ONEDAL_TFS "Enable Intel oneDal ML system table functions" ON)
option(ENABLE_ML_MLPACK_TFS "Enable MLPack ML system table functions" OFF)
option(ENABLE_RUNTIME_LIBS "Enable runtime library support" OFF)
option(ENABLE_TORCH_TFS "Enable Torch system table functions" OFF)

option(ENABLE_PDAL "Enable PDAL support" ON)
option(ENABLE_POINT_CLOUD_TFS "Enable point cloud table functions" ON)
option(ENABLE_GFX_TFS "Enable graphics table functions" ON)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # Clang will not complile these tests quickly enough to avoid timeout
  option(ENABLE_RF_PROP_TFS "Enable RF Propagation module" OFF)
else()
  option(ENABLE_RF_PROP_TFS "Enable RF Propagation module" ON)
endif()

if (ENABLE_PDAL)
  find_package(PDAL)
  if(PDAL_FOUND)
    include_directories(${PDAL_INCLUDE_DIRS})
    add_definitions( ${PDAL_DEFINITIONS} )
    set(ENABLE_PDAL ON)
    add_definitions("-DHAVE_PDAL")
  else()
    set(ENABLE_PDAL OFF CACHE BOOL "Enable PDAL support" FORCE)
    set(ENABLE_POINT_CLOUD_TFS OFF CACHE BOOL "Enable point cloud table functions" FORCE)
  endif()
else()
  set(ENABLE_POINT_CLOUD_TFS OFF CACHE BOOL "Enable point cloud table functions" FORCE)
endif()

if(ENABLE_SYSTEM_TFS)
  add_definitions ("-DHAVE_SYSTEM_TFS")
  if(ENABLE_RF_PROP_TFS)
    add_definitions ("-DHAVE_RF_PROP_TFS")
  endif()
  if (ENABLE_POINT_CLOUD_TFS)
    add_definitions("-DHAVE_POINT_CLOUD_TFS")
  endif()
else()
  set(ENABLE_POINT_CLOUD_TFS OFF CACHE BOOL "Enable point cloud table functions" FORCE)
  set(ENABLE_RF_PROP_TFS OFF CACHE BOOL "Enable RF Propagation module" FORCE)
endif()

if(ENABLE_ML_ONEDAL_TFS)
  if(NOT ENABLE_SYSTEM_TFS)
    set(ENABLE_ML_ONEDAL_TFS OFF CACHE BOOL "Enable Intel oneDal ML system table functions" FORCE)
    message(STATUS "System table functions must be enabled to use oneDal ML functions, disabling oneDal support.")
  else()
    set(ONEDAL_USE_DPCPP no)
    set(ONEDAL_INTERFACE yes)
    set(ONEDAL_LINK static)
    find_package(oneDAL)
    if (oneDAL_FOUND)
      message(STATUS "Found the following oneDAL libraries:")
      foreach(DAL_LIBRARY IN LISTS DAL_LIBRARIES)
        string(REPLACE "oneDAL::" "" DAL_LIBRARY "${DAL_LIBRARY}")
        message(STATUS "  ${DAL_LIBRARY}")
      endforeach()
      include_directories(${oneDAL_INCLUDE_DIRS})
      add_definitions("-DHAVE_ONEDAL")
    else()
      set(ENABLE_ML_ONEDAL_TFS OFF CACHE BOOL "Enable Intel oneDal ML system table functions" FORCE)
      message(STATUS "oneDal library not found, disabling oneDal support.")
    endif()
  endif()
endif()

if(ENABLE_ML_MLPACK_TFS)
  if(NOT ENABLE_SYSTEM_TFS)
    set(ENABLE_ML_MLPACK_TFS OFF CACHE BOOL "Enable MLPACK ML system table functions" FORCE)
    message(STATUS "System table functions must be enabled to use MLPACK ML functions, disabling MLPACK support.")
  else()
    find_package(OpenMP REQUIRED)
    find_package(Armadillo REQUIRED)
    find_package(Boost COMPONENTS serialization REQUIRED)
    find_package(MLPACK REQUIRED)
    include_directories(${MLPACK_INCLUDE_DIRS})
    add_definitions("-DHAVE_MLPACK")
  endif()
endif()

find_package(Curses)
include_directories(${CURSES_INCLUDE_DIRS})
if (CURSES_HAVE_NCURSES_CURSES_H AND NOT CURSES_HAVE_CURSES_H)
  include_directories(${CURSES_INCLUDE_DIRS}/ncurses/)
endif()

function(install_thirdparty_files src_hdr_list lib_base_path_list)
  foreach(hdr ${src_hdr_list})
    install(FILES ${hdr} DESTINATION ThirdParty/include)
  endforeach()
  include_directories(ThirdParty/include)

  foreach(lib_base ${lib_base_path_list})
    file(GLOB lib_versions_list ${lib_base}*)
    foreach(lib ${lib_versions_list})
      install(FILES ${lib} DESTINATION ThirdParty/lib)
    endforeach()
  endforeach()
endfunction()

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

option(ENABLE_WARNINGS_AS_ERRORS "Enable treat warnings as errors" ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-local-typedefs -Wvla -fdiagnostics-color=auto -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS")
if (ENABLE_WARNINGS_AS_ERRORS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
  if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    # Allow maybe-uninitialized warnings with GCC as it is prone to false positives
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=maybe-uninitialized")
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Disable warnings as errors for misleading-indentation due to a boost header issue
    # This can be removed once the boost fix is in our deps https://github.com/boostorg/serialization/pull/262/files
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=misleading-indentation")
  endif()
endif()
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-register")
endif()
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()
if (${ARCH} STREQUAL "aarch64")
  # Disable floating point expression contraction which allows FMA in baseline aarch64
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffp-contract=off")
  if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    # Suppress "parameter passing for argument of type 'std::pair<X,X>' notes" on ARM
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-psabi")
  endif()
endif()

if (ENABLE_ML_MLPACK_TFS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# address and thread sanitizer
option(ENABLE_STANDALONE_CALCITE "Require standalone Calcite server" OFF)
option(ENABLE_ASAN "Enable address sanitizer" OFF)
option(ENABLE_TSAN "Enable thread sanitizer" OFF)
option(ENABLE_UBSAN "Enable undefined behavior sanitizer" OFF)
if(ENABLE_ASAN)
  set(SAN_FLAGS "-fsanitize=address -O1 -fno-omit-frame-pointer")
  add_definitions("-DHAVE_ASAN")
  add_definitions("-DWITH_DECODERS_BOUNDS_CHECKING")
elseif(ENABLE_TSAN)
  add_definitions("-DHAVE_TSAN")
  add_definitions("-DTBB_PREVIEW_WAITING_FOR_WORKERS") # might not be required in later versions
  # Copy the config directory to the build dir for TSAN suppressions
  file(COPY config DESTINATION ${CMAKE_BINARY_DIR})

  set(SAN_FLAGS "-fsanitize=thread -fPIC -O1 -fno-omit-frame-pointer")
  # required for older GCC, see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=64354
  add_definitions("-D__SANITIZE_THREAD__")
elseif(ENABLE_UBSAN)
  set(SAN_FLAGS "-fsanitize=undefined -fPIC -O1 -fno-omit-frame-pointer")
endif()
if(ENABLE_ASAN OR ENABLE_TSAN OR ENABLE_UBSAN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SAN_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SAN_FLAGS}")
  set(ENABLE_STANDALONE_CALCITE ON)
endif()

# Code coverage
option(ENABLE_CODE_COVERAGE "Enable compile time code coverage" OFF)
if(ENABLE_CODE_COVERAGE)
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(COVERAGE_FLAGS "-fprofile-instr-generate -fcoverage-mapping")
  else()
    message(FATAL_ERROR "Code coverage currently only supported with Clang compiler")
  endif()
  set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE ON)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COVERAGE_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COVERAGE_FLAGS}")
endif()

option(ENABLE_DECODERS_BOUNDS_CHECKING "Enable bounds checking for column decoding" OFF)

if(ENABLE_STANDALONE_CALCITE)
  add_definitions("-DSTANDALONE_CALCITE")
endif()

include_directories(${CMAKE_SOURCE_DIR}
                    ${CMAKE_SOURCE_DIR}/Parser
                    ${CMAKE_CURRENT_BINARY_DIR})

## Dependencies

# LLVM
find_package(LLVM CONFIG REQUIRED)
if (TARGET LLVM)
  get_target_property(LLVM_LIB LLVM IMPORTED_LOCATION_RELEASE)
endif()
if (NOT LLVM_LIB)
  find_library(LLVM_LIB LLVM)
endif()
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using LLVM_LIB: ${LLVM_LIB}")

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

find_package(Clang CONFIG)
if (TARGET clang-cpp)
  get_target_property(CLANG_LIB clang-cpp IMPORTED_LOCATION_RELEASE)
endif()
if (NOT CLANG_LIB)
  find_library(CLANG_LIB clang-cpp)
endif()
message(STATUS "Using CLANG_LIB: ${CLANG_LIB}")

# Deps builds use separate libs for each clang component, while some distros now bundle into a single lib
if (NOT LLVM_LIB)
  set(LLVM_COMPONENTS support mcjit core irreader option linker)
  option(ENABLE_INTEL_JIT_LISTENER "Enable Intel Vtune JIT Listener" OFF)
  if(ENABLE_INTEL_JIT_LISTENER)
    list(APPEND LLVM_COMPONENTS inteljitevents)
  endif()

  llvm_map_components_to_libnames(llvm_libs ${LLVM_TARGETS_TO_BUILD} ${LLVM_COMPONENTS})
  set(clang_libs
      clangFrontend
      clangSerialization
      clangDriver
      clangTooling
      clangParse
      clangSema
      clangAnalysis
      clangEdit
      clangAST
      clangLex
      clangBasic
      clangRewrite
      clangRewriteFrontend)

  # LLVMSupport explicitly lists tinfo in its INTERFACE_LINK_LIBRARIES, even
  # though we provide it in our build of ncurses. Since LLVMSupport is listed
  # as a requirement for other llvm libs, we need to walk through the entire
  # list in order to remove all instances of tinfo.
  foreach(lib ${llvm_libs})
    get_target_property(interface_libs ${lib} INTERFACE_LINK_LIBRARIES)
    list(REMOVE_ITEM interface_libs tinfo z rt pthread -lpthread m dl)
    set_target_properties(${lib} PROPERTIES INTERFACE_LINK_LIBRARIES "${interface_libs}")
  endforeach()

  list(APPEND llvm_libs ${CURSES_NCURSES_LIBRARY})
else()
  if(NOT CLANG_LIB)
    message(FATAL_ERROR "Could not find CLANG library.")
  endif()

  set(clang_libs ${CLANG_LIB})
  set(llvm_libs ${LLVM_LIB})
endif()

# International Components for Unicode (ICU) required by boost_locale
find_library(ICUUC_LIBRARY icuuc REQUIRED)
find_library(ICUDATA_LIBRARY icudata REQUIRED)
find_library(ICUI18N_LIBRARY icui18n REQUIRED)
set(ICU_LIBRARIES ${ICUUC_LIBRARY} ${ICUDATA_LIBRARY} ${ICUI18N_LIBRARY})

# Boost
find_package(Boost COMPONENTS log log_setup filesystem program_options regex system thread timer locale iostreams serialization REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
list(APPEND Boost_LIBRARIES ${ICU_LIBRARIES})
if(Boost_VERSION GREATER_EQUAL 108600)
  add_definitions("-DHAVE_BOOST_1_86")
endif()

# EGL
include_directories(ThirdParty/egl)

# Google Test and Google Mock
add_definitions("-DGTEST_USE_OWN_TR1_TUPLE=0")
include_directories(ThirdParty/googletest)
add_subdirectory(ThirdParty/googletest)

# Google Benchmark
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Suppress benchmark install rules for packaging" FORCE)
add_subdirectory(ThirdParty/googlebenchmark)

# aws-sdk
find_package(CURL REQUIRED QUIET)
# static libcurl may require OpenSSL/zlib (and optional zstd below) at link time
list(APPEND CURL_LIBRARIES ${OPENSSL_LIBRARIES} ${ZLIB_LIBRARIES})
find_library(ZSTD_LIBRARY zstd)
if(ZSTD_LIBRARY)
  list(APPEND CURL_LIBRARIES ${ZSTD_LIBRARY})
endif()
option(ENABLE_AWS_S3 "Enable AWS S3 support" ON)
if(ENABLE_AWS_S3)
  find_package(LibAwsS3)
  if(NOT LibAwsS3_FOUND)
    # Note for a build with static arrow libs (RedHat)
    # aws-sdk-cpp must be included.
    set(ENABLE_AWS_S3 OFF CACHE BOOL "Enable AWS S3 support" FORCE)
  else()
    add_definitions("-DHAVE_AWS_S3")
    set(AwsS3_CURL_SUPPORT ${CURL_LIBRARIES})
  endif()
endif()
# Arrow
find_package(Arrow REQUIRED)
add_definitions("-DARROW_NO_DEPRECATED_API")
add_definitions("-DENABLE_ARROW_4")
include_directories(${Arrow_INCLUDE_DIRS})

option(ENABLE_IMPORT_PARQUET "Enable Parquet Importer support" ON)
if(ENABLE_IMPORT_PARQUET)
  find_package(Parquet)
  if(NOT Parquet_FOUND)
    set(ENABLE_IMPORT_PARQUET OFF CACHE BOOL "Enable Parquet Importer support" FORCE)
    message(STATUS "Parquet not found. Disabling Parquet Importer support.")
  else()
    add_definitions("-DENABLE_IMPORT_PARQUET")
    # when we found libparquet it means we're using arrow 11+
    # and deps scripts must have built parquet as well as snappy
    find_package(Snappy REQUIRED)
  endif()
endif()

find_library(lz4_LIBRARY lz4 REQUIRED)
find_library(bz2_LIBRARY bz2 REQUIRED)

list(APPEND Arrow_LIBRARIES ${lz4_LIBRARY} ${Snappy_LIBRARIES} ${bz2_LIBRARY}) # zstd linked elsewhere for GDAL
if(ENABLE_AWS_S3)
  list(INSERT Arrow_LIBRARIES 0 ${LibAwsS3_LIBRARIES})
endif()
if (ENABLE_CUDA)
  list(INSERT Arrow_LIBRARIES 0 ${Arrow_GPU_CUDA_LIBRARIES})
endif()

# RapidJSON
include_directories(ThirdParty/rapidjson)
add_definitions(-DRAPIDJSON_HAS_STDSTRING)
add_definitions(-DRAPIDJSON_NOMEMBERITERATORCLASS)


# Linenoise
add_subdirectory(ThirdParty/linenoise)

# SQLite3
find_package(SQLite3 3.53 REQUIRED)
if(PREFER_STATIC_LIBS)
  # Static GDAL/PROJ (via GDALExtra) reference sqlite3 symbols; not every
  # target that links GDAL also links SqliteConnector (e.g. GeoTypesTest).
  list(APPEND GDAL_LIBRARIES SQLite::SQLite3)
endif()

# rdkafka
find_package(RdKafka REQUIRED)
include_directories(${RdKafka_INCLUDE_DIRS})

# Message Digest functions used by libarchive
find_library(MD_LIBRARY md REQUIRED)

# libarchive
find_package(LibArchive REQUIRED)
include_directories(${LibArchive_INCLUDE_DIRS})
list(APPEND LibArchive_LIBRARIES ${MD_LIBRARY})

# bcrypt
include_directories(ThirdParty/bcrypt)
add_subdirectory(ThirdParty/bcrypt)

# PicoSHA2
include_directories(ThirdParty/PicoSHA2)

# ldap
option(ENABLE_LDAP "Enable LDAP support" ON)
if(ENABLE_LDAP)
  add_definitions("-DHAVE_LDAP")
endif()

# opensaml
option(ENABLE_SAML "Enable SAML support" ON)
if(ENABLE_SAML)
  find_package(OpenSaml)
  if(NOT OpenSaml_FOUND)
    set(ENABLE_SAML OFF CACHE BOOL "Enable SAML support" FORCE)
  else()
    add_definitions("-DHAVE_SAML")
  endif()
endif()

# TBB

option(ENABLE_TBB "Enable OneTBB for threading (if found)" ON)
set(TBB_LIBS "")
find_package(TBB)
if(TBB_FOUND)
  message(STATUS "TBB library is found with ${TBB_DIR}")
  add_definitions("-DHAVE_TBB")
  add_definitions("-DTBB_PREVIEW_TASK_GROUP_EXTENSIONS")
  list(APPEND TBB_LIBS ${TBB_LIBRARIES})
  if(ENABLE_TBB)
    add_definitions("-DENABLE_TBB")
  else()
    message(STATUS "Using TBB for threading is DISABLED")
  endif()
else()
  set(ENABLE_TBB OFF)
endif()

option(DISABLE_CONCURRENCY "Disable parallellism at the threading layer" OFF)
if(DISABLE_CONCURRENCY)
  add_definitions("-DDISABLE_CONCURRENCY")
endif()

set(gen_cpp_files
    ${CMAKE_BINARY_DIR}/gen-cpp/Heavy.cpp
    ${CMAKE_BINARY_DIR}/gen-cpp/Heavy.h
    ${CMAKE_BINARY_DIR}/gen-cpp/heavy_types.cpp
    ${CMAKE_BINARY_DIR}/gen-cpp/common_types.cpp
    ${CMAKE_BINARY_DIR}/gen-cpp/completion_hints_types.cpp
    ${CMAKE_BINARY_DIR}/gen-cpp/serialized_result_set_types.cpp
    ${CMAKE_BINARY_DIR}/gen-cpp/extension_functions_types.cpp
    ${CMAKE_BINARY_DIR}/gen-cpp/extension_functions_types.h
)
add_custom_command(
  DEPENDS
    ${CMAKE_SOURCE_DIR}/heavy.thrift
    ${CMAKE_SOURCE_DIR}/common.thrift
    ${CMAKE_SOURCE_DIR}/completion_hints.thrift
    ${CMAKE_SOURCE_DIR}/QueryEngine/serialized_result_set.thrift
    ${CMAKE_SOURCE_DIR}/QueryEngine/extension_functions.thrift
  OUTPUT ${gen_cpp_files}
  COMMAND ${Thrift_EXECUTABLE}
  ARGS -gen cpp -r -o ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/heavy.thrift)
list(APPEND ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_BINARY_DIR}/gen-cpp/)

add_custom_target(thrift_gen DEPENDS ${gen_cpp_files})

add_library(mapd_thrift ${gen_cpp_files})

target_compile_options(mapd_thrift PRIVATE -fPIC)

target_link_libraries(mapd_thrift ${Thrift_LIBRARIES})

option(ENABLE_OMNIVERSE_CONNECTOR "Enable Omniverse Connector" ON)

option(ENABLE_PROFILER "Enable google perftools" OFF)
if(ENABLE_PROFILER)
  find_package(Gperftools REQUIRED COMPONENTS TCMALLOC PROFILER)
  set(_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
  set(CMAKE_FIND_LIBRARY_SUFFIXES .so)
  find_library(UNWIND_LIB unwind REQUIRED) # The static lib results in relocation linker errors
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_CMAKE_FIND_LIBRARY_SUFFIXES})
  set(PROFILER_LIBS ${Gperftools_TCMALLOC} ${Gperftools_PROFILER} ${UNWIND_LIB})
  add_definitions("-DHAVE_PROFILER")
else()
  set(PROFILER_LIBS "")
endif()

option(ENABLE_FSI_ODBC "Enable ODBC data wrapper" ON)
if(ENABLE_FSI_ODBC)
  add_definitions("-DEE_FSI_ODBC")
endif()

# Set RPATH for all executables
if(IS_REDHAT)
  set(CMAKE_INSTALL_RPATH "$ORIGIN/../ThirdParty/lib64")
else()
  set(CMAKE_INSTALL_RPATH "$ORIGIN/../ThirdParty/lib")
endif()

# Some subdirectories will add optional dependencies to initheavy so we want to declare
# it before subdirs since it is otherwise independent.
add_executable(initheavy initdb.cpp)

add_subdirectory(SqliteConnector)

add_subdirectory(StringDictionary)
add_subdirectory(Calcite)
get_target_property(CalciteThrift_BINARY_DIR calciteserver_thrift BINARY_DIR)
include_directories(${CalciteThrift_BINARY_DIR})

include_directories(ThirdParty/muparserx)
add_subdirectory(ThirdParty/muparserx)

set(MAPD_RENDERING_LIBRARIES "")
option(ENABLE_RENDERING "Build backend renderer" OFF)
if(ENABLE_RENDERING)
  option(ENABLE_RENDER_TESTS "Build backend renderer tests" ON)
  option(ENABLE_RENDER_LOGGER "Enable detailed render logging in debug builds" OFF)
  option(ENABLE_SHADER_DEBUG_PRINTF "Enable Vulkan Shader Debug printf extension" OFF)

  option(HOTLOAD_SHADER_TEMPLATE_FILES "Enable dynamic reloading of glsl shader templates" OFF)

  add_definitions("-DHAVE_RENDERING")
  add_subdirectory(GfxDriver)
  add_subdirectory(GfxInterop)
  add_subdirectory(QueryRenderer)
  add_subdirectory(ExecuteRenderInterface)

  include_directories(${GFXDRIVER_INCLUDE_DIRS})
  set(MAPD_RENDERING_LIBRARIES ExecuteRenderInterface QueryRenderer GfxInterop GfxDriver)

  set(MAPD_PACKAGE_FLAGS "${MAPD_PACKAGE_FLAGS}-render")
endif()

find_package(CPR REQUIRED)

if(ENABLE_RUNTIME_LIBS)
  if(ENABLE_TORCH_TFS)
    if(NOT ENABLE_SYSTEM_TFS)
      set(ENABLE_TORCH_TFS OFF CACHE BOOL "Enable Torch system table functions" FORCE)
      message(STATUS "System table functions must be enabled (-DENABLE_SYSTEM_TFS=ON) to use LibTorch functions, disabling LibTorch support.")
    else()
      find_package(Torch REQUIRED)
      if (DEFINED TORCH_CUDA_LIBRARIES)
        # Torch removes the CUDA Driver target from CUDA_LIBRARIES when doing its own CUDA CMake setup, so we have to re-add it
        list(APPEND CUDA_LIBRARIES cuda)
        add_compile_definitions("HAVE_CUDA_TORCH")
      endif()
      add_compile_definitions("HAVE_TORCH_TFS")
    endif()
  endif()
  add_subdirectory(RuntimeLibManager)
  add_compile_definitions("HAVE_RUNTIME_LIBS")
endif()

add_subdirectory(Catalog)
add_subdirectory(StringOps)
add_subdirectory(Parser)
add_subdirectory(Analyzer)
add_subdirectory(ImportExport)
add_subdirectory(QueryEngine)
add_subdirectory(DataMgr)
add_subdirectory(CudaMgr)
add_subdirectory(LockMgr)
add_subdirectory(Logger)
add_subdirectory(MigrationMgr)
add_subdirectory(Fragmenter)
add_subdirectory(Shared)
add_subdirectory(Utils)
add_subdirectory(QueryRunner)
add_subdirectory(SQLFrontend)
add_subdirectory(TableArchiver)
add_subdirectory(Geospatial)
add_subdirectory(UdfCompiler)
add_subdirectory(ThriftHandler)

set(MAPD_LIBRARIES Shared Catalog SqliteConnector MigrationMgr TableArchiver Parser Analyzer StringOps ImportExport QueryRunner QueryEngine QueryState LockMgr DataMgr Fragmenter Logger Geospatial)

list(APPEND MAPD_LIBRARIES Calcite)

if(ENABLE_RENDERING)
  add_dependencies(QueryRenderer Parser)
  list(APPEND MAPD_LIBRARIES ${MAPD_RENDERING_LIBRARIES})
endif()

list(APPEND MAPD_LIBRARIES ${Arrow_LIBRARIES})

if(ENABLE_PDAL)
  list(APPEND MAPD_LIBRARIES ${PDAL_LIBRARIES} ${GDAL_LIBRARIES} ${GDALExtra_LIBRARIES})
endif()

if (ENABLE_ML_ONEDAL_TFS)
    list(APPEND MAPD_LIBRARIES ${DAL_LIBRARIES})
    list(APPEND TBB_LIBS ${TBB_MALLOC_LIBRARY})
endif()

if(ENABLE_ML_MLPACK_TFS)
  list(APPEND MAPD_LIBRARIES ${MLPACK_LIBRARIES} ${ARMADILLO_LIBRARIES})
endif()

list(APPEND MAPD_LIBRARIES ${TBB_LIBS})

if(CPR_FOUND)
  list(APPEND MAPD_LIBRARIES ${CPR_LIBRARIES})
endif()

option(ENABLE_TESTS "Build unit tests" ON)
if (ENABLE_TESTS)
  enable_testing()
  option(ENABLE_INIT_DIR "Add test dependency on db init" ON)
  add_subdirectory(Tests)
  add_subdirectory(SampleCode)
  if (ENABLE_RUNTIME_LIBS)
    add_subdirectory(QueryEngine/TableFunctions/RuntimeLibTestFunctions)
  endif()
endif()

if(ENABLE_RENDERING AND ENABLE_RENDER_TESTS)
  enable_testing()
  add_subdirectory(Tests/RenderTests)
endif()

add_executable(heavydb HeavyDB.cpp ${CMAKE_BINARY_DIR}/MapDRelease.h)
set_target_properties(heavydb PROPERTIES COMPILE_DEFINITIONS "${TIME_LIMITED_DEFINITIONS}")
InstallVersionFile()

add_custom_command(
    DEPENDS ${CMAKE_SOURCE_DIR}/heavy.thrift
    OUTPUT
        ${CMAKE_SOURCE_DIR}/java/thrift/src/gen/ai/heavy/thrift/server/Heavy.java
        ${CMAKE_SOURCE_DIR}/java/thrift/src/gen/ai/heavy/thrift/server/TRow.java
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/java/thrift/src/gen
    COMMAND ${Thrift_EXECUTABLE}
    ARGS -gen java:generated_annotations=suppress -r -out ${CMAKE_SOURCE_DIR}/java/thrift/src/gen/ ${CMAKE_SOURCE_DIR}/heavy.thrift)

add_custom_command(
    DEPENDS ${CMAKE_SOURCE_DIR}/common.thrift
    OUTPUT
        ${CMAKE_SOURCE_DIR}/java/thrift/src/gen/ai/heavy/thrift/server/common.java
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/java/thrift/src/gen
    COMMAND ${Thrift_EXECUTABLE}
    ARGS -gen java:generated_annotations=suppress -r -out ${CMAKE_SOURCE_DIR}/java/thrift/src/gen/ ${CMAKE_SOURCE_DIR}/common.thrift)

add_custom_command(
    DEPENDS ${CMAKE_SOURCE_DIR}/QueryEngine/serialized_result_set.thrift
    OUTPUT
        ${CMAKE_SOURCE_DIR}/java/thrift/src/gen/ai/heavy/thrift/server/serialized_result_set.java
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/java/thrift/src/gen
    COMMAND ${Thrift_EXECUTABLE}
    ARGS -gen java:generated_annotations=suppress -r -out ${CMAKE_SOURCE_DIR}/java/thrift/src/gen/ ${CMAKE_SOURCE_DIR}/QueryEngine/serialized_result_set.thrift)

add_custom_command(
    DEPENDS ${CMAKE_SOURCE_DIR}/java/thrift/calciteserver.thrift ${CMAKE_SOURCE_DIR}/completion_hints.thrift ${CMAKE_SOURCE_DIR}/QueryEngine/extension_functions.thrift
    OUTPUT ${CMAKE_SOURCE_DIR}/java/thrift/src/gen/ai/heavy/thrift/calciteserver/CalciteServer.java
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/java/thrift/src/gen
    COMMAND ${Thrift_EXECUTABLE}
    ARGS -gen java:generated_annotations=suppress -r -I ${CMAKE_SOURCE_DIR} -out ${CMAKE_SOURCE_DIR}/java/thrift/src/gen/ ${CMAKE_SOURCE_DIR}/java/thrift/calciteserver.thrift)

list(APPEND ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_SOURCE_DIR}/java/thrift/src/gen/)

# Check if the git hash file exists and set the CPack version if it does
# If not, get the hash from git and create the file
execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMAND ${GIT_EXECUTABLE} rev-parse --short=10 HEAD
  OUTPUT_VARIABLE MAPD_GIT_HASH
)

# Ensure the git hash file exists
file(WRITE ${CMAKE_BINARY_DIR}/heavyai_git_hash.txt "${MAPD_GIT_HASH}")

# Make the CMake configuration dependent on the git hash file
# This will trigger a re-run of CMake if it changes, updating the CPack and version files
set_property(
  DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_BINARY_DIR}/heavyai_git_hash.txt
)

# Ensure MAPD_GIT_HASH is set and embed in CPack
file(STRINGS ${CMAKE_BINARY_DIR}/heavyai_git_hash.txt MAPD_GIT_HASH)
set(CPACK_PACKAGE_VERSION "${MAPD_VERSION_RAW}-${MAPD_BUILD_DATE}-${MAPD_GIT_HASH}")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/Shared/release.h"
  "${CMAKE_BINARY_DIR}/MapDRelease.h"
  @ONLY
  )

# On every build, check if the git hash has changed, and if so update cache file
# triggering a reconfigure
add_custom_target(get_git_hash ALL
  COMMENT "Checking git hash"
  BYPRODUCTS ${CMAKE_BINARY_DIR}/heavyai_git_hash_new.txt
  OUTPUT ${CMAKE_BINARY_DIR}/heavyai_git_hash.txt
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMAND ${GIT_EXECUTABLE} rev-parse --short=10 HEAD > ${CMAKE_BINARY_DIR}/heavyai_git_hash_new.txt
  COMMAND bash -c "${CMAKE_SOURCE_DIR}/scripts/update_git_hash.sh ${CMAKE_BINARY_DIR}"
)

list(APPEND ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_BINARY_DIR}/heavyai_git_hash.txt)
list(APPEND ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_BINARY_DIR}/MapDRelease.h)

# note if aws isn't enabled Aws_CURL_SUPPORT is empty
target_link_libraries(heavydb mapd_thrift thrift_handler ${MAPD_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS}
    ${CUDA_LIBRARIES} ${PROFILER_LIBS} ${ZLIB_LIBRARIES} ${BLOSC_LIBRARIES}
    ${LOCALE_LINK_FLAG} ${AwsS3_CURL_SUPPORT})

target_link_libraries(initheavy mapd_thrift thrift_handler ${MAPD_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS}
    ${CUDA_LIBRARIES} ${PROFILER_LIBS} ${ZLIB_LIBRARIES} ${BLOSC_LIBRARIES}
    ${LOCALE_LINK_FLAG} ${AwsS3_CURL_SUPPORT})

macro(set_dpkg_arch arch_in arch_out)
  if("${arch_in}" STREQUAL "x86_64")
    set(${arch_out} "amd64")
  elseif("${arch_in}" STREQUAL "aarch64")
    set(${arch_out} "arm64")
  elseif("${arch_in}" STREQUAL "ppc64le")
    set(${arch_out} "ppc64el")
  else()
    set(${arch_out} "${arch_in}")
  endif()
endmacro()

# clang-tidy
find_program(JQ_EXECUTABLE NAMES jq)
if (NOT ${JQ_EXECUTABLE} STREQUAL "JQ_EXECUTABLE-NOTFOUND")
  file(WRITE ${CMAKE_BINARY_DIR}/jq.filter "map(select(.file | test(\".*/(build|ThirdParty)/.*\") | not))")
  add_custom_target(run-clang-tidy
# For now add these as a dependency, do it better later
    DEPENDS ${CMAKE_BINARY_DIR}/gen-cpp/Heavy.h ${CMAKE_BINARY_DIR}/Parser/parser.h
    COMMAND mkdir -p clang-tidy
    COMMAND ${JQ_EXECUTABLE} -f jq.filter ${CMAKE_BINARY_DIR}/compile_commands.json > clang-tidy/compile_commands.json
    COMMAND cd clang-tidy && ${CMAKE_SOURCE_DIR}/ThirdParty/clang/run-clang-tidy.py -quiet -format -fix -header-filter="${CMAKE_SOURCE_DIR}/.*" 2> /dev/null
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
else()
  message(STATUS "jq not found, disabling run-clang-tidy target")
endif()

# doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
  configure_file(
    ${CMAKE_SOURCE_DIR}/Doxyfile.in ${CMAKE_BINARY_DIR}/Doxyfile @ONLY)
  add_custom_target(doxygen ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Generate API documentation with Doxygen"
    VERBATIM
    USES_TERMINAL
  )
  add_custom_target(doxygen_zip zip -r doxygen-${MAPD_GIT_HASH}.zip doxygen
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Generating .zip with API documentation from Doxygen"
    VERBATIM
    USES_TERMINAL
  )
  add_dependencies(doxygen_zip doxygen)
endif(DOXYGEN_FOUND)

# Packaging

if(NOT "${CMAKE_BUILD_TYPE_LOWER}" STREQUAL "debug" AND NOT "${CMAKE_BUILD_TYPE_LOWER}" STREQUAL "relwithdebinfo")
  set(CPACK_STRIP_FILES ON)
else()
  set(MAPD_PACKAGE_FLAGS "${MAPD_PACKAGE_FLAGS}-debug")
endif()
set(CPACK_PACKAGE_VENDOR "NVIDIA Corporation")
#set(CPACK_PACKAGE_CONTACT "TBD")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "HeavyDB Database")
set(CPACK_PROJECT_CONFIG_FILE ${CMAKE_SOURCE_DIR}/CMakePackaging.txt)
set(CPACK_DEBIAN_PACKAGE_DEPENDS "default-jre-headless | openjdk-8-jre-headless | java8-runtime-headless, bsdmainutils, curl | wget")
set(CPACK_RPM_PACKAGE_REQUIRES "java-headless, util-linux, curl")
set(CPACK_RPM_PACKAGE_AUTOREQ OFF)
set(CPACK_RPM_SPEC_MORE_DEFINE "%define __jar_repack %{nil}")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "${CPACK_DEBIAN_PACKAGE_DEPENDS}, libldap-2.4-2")
if(ENABLE_RENDERING)
  get_filename_component(VULKAN_LIBRARY_DIR ${VULKAN_LIBRARY} DIRECTORY)
  file(GLOB VULKAN_LOADER_LIBS "${VULKAN_LIBRARY_DIR}/libvulkan.so*")
  if(IS_REDHAT)
    install(FILES ${VULKAN_LOADER_LIBS} DESTINATION "ThirdParty/lib64")
  else()
    install(FILES ${VULKAN_LOADER_LIBS} DESTINATION "ThirdParty/lib")
  endif()
endif()

set_dpkg_arch(${CMAKE_SYSTEM_PROCESSOR} CPACK_DEBIAN_PACKAGE_ARCHITECTURE)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/licenses" DESTINATION "ThirdParty" COMPONENT "doc")

# heavydbTypes.h local includes (for UDF/UDTF)
install(FILES Shared/funcannotations.h DESTINATION "Shared/" COMPONENT "include")
install(FILES Shared/InlineNullValues.h DESTINATION "Shared/" COMPONENT "include")
install(FILES Logger/Logger.h DESTINATION "Logger/" COMPONENT "include")

# Frontend
message(STATUS "Webserver URL: \"${MAPD_WEBSERVER_BASE_URL}\"")

option(MAPD_IMMERSE_DOWNLOAD "Download HEAVY.AI Immerse for packaging" ON)
set(MAPD_IMMERSE_URL ${MAPD_IMMERSE_URL} CACHE STRING "URL to bundled frontend")
set(MAPD_IMMERSE_FILE "${MAPD_BUILDS_REPO_PATH}/frontend/${MAPD_IMMERSE_BUILD_ID}.zip" CACHE STRING "Local path to Immerse frontend zip")
if(MAPD_IMMERSE_DOWNLOAD)
  include(ExternalProject)
  include(heavyai_functions)
  externalproject_add_url_or_file(frontend ${MAPD_IMMERSE_URL} ${MAPD_IMMERSE_FILE})
  externalproject_get_property(frontend source_dir)

  install(DIRECTORY ${source_dir}/ DESTINATION "frontend/" PATTERN .git EXCLUDE PATTERN node_modules EXCLUDE)
  add_custom_command(TARGET frontend COMMAND ${CMAKE_COMMAND} -E copy_directory ${source_dir} frontend)
  list(APPEND ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_BINARY_DIR}/frontend)

  ## Go web server
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Immerse")
    add_subdirectory(Immerse)
  endif()
endif()

# HeavyIQ
option(HEAVYIQ_DOWNLOAD "Download HeavyIQ for packaging" ON)
if(HEAVYIQ_DOWNLOAD)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/HeavyIQ")
    add_subdirectory(HeavyIQ)
  endif()
endif()

# systemd
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  install(FILES systemd/heavydb.service.in systemd/heavydb@.service.in DESTINATION systemd)
  install(FILES systemd/heavy_web_server.service.in systemd/heavy_web_server@.service.in DESTINATION systemd)
  install(FILES systemd/heavyiq.service.in DESTINATION systemd)
  install(FILES systemd/heavy.conf.in DESTINATION systemd)
  install(PROGRAMS systemd/install_heavy_systemd.sh DESTINATION systemd)
endif()

## mvn process for java code
find_program(MVN_EXECUTABLE NAMES mvn)
if(NOT MVN_EXECUTABLE)
  message(FATAL_ERROR "mvn not found. Install Apache Maven.")
endif()
file(GLOB_RECURSE JAVA_POM RELATIVE ${CMAKE_SOURCE_DIR} java/**/pom.xml)
file(GLOB_RECURSE JAVA_FTL RELATIVE ${CMAKE_SOURCE_DIR} java/calcite/src/main/codegen/includes/*.ftl)
file(GLOB_RECURSE JAVA_SOURCES RELATIVE ${CMAKE_SOURCE_DIR} java/**/*.java)
list(FILTER JAVA_SOURCES EXCLUDE REGEX ".*/gen/.*")
list(FILTER JAVA_SOURCES EXCLUDE REGEX ".*/generated-sources/.*")
set(JAVA_ASSEMBLY_DESCRIPTORS
  ${CMAKE_SOURCE_DIR}/java/heavyaijdbc/src/assembly/runtime.xml
  ${CMAKE_SOURCE_DIR}/java/heavydb/src/assembly/production-utilities.xml)

set(OMNISCI_JAR_RELEASE_VERSION "${MAPD_VERSION_MAJOR}.${MAPD_VERSION_MINOR}.${MAPD_VERSION_PATCH}")
if("${MAPD_VERSION_EXTRA}" STREQUAL "dev")
  set (OMNISCI_JAR_RELEASE_VERSION "${OMNISCI_JAR_RELEASE_VERSION}-SNAPSHOT")
endif()

set (JDBC_JAR "heavyai-jdbc-${OMNISCI_JAR_RELEASE_VERSION}.jar")
set (UTILITY_JAR "heavyai-utility-${OMNISCI_JAR_RELEASE_VERSION}.jar")

set(MVN_PATH_COMMAND "MVNPATH=${CMAKE_SOURCE_DIR}/java")

set(THRIFT_MVN_VERSION "0.24.0") # independent of Thrift C++ version
add_custom_command(
  OUTPUT
    ${CMAKE_BINARY_DIR}/bin/${UTILITY_JAR}
    ${CMAKE_BINARY_DIR}/bin/${JDBC_JAR}
    ${CMAKE_BINARY_DIR}/bin/calcite-1.41.0-SNAPSHOT-jar-with-dependencies.jar
    COMMAND ${MVN_PATH_COMMAND} ${MVN_EXECUTABLE} -T 4 -l ${CMAKE_BINARY_DIR}/mvn_build.log -e clean install -Dthrift.version="${THRIFT_MVN_VERSION}" -Dmaven.compiler.showDeprecation=true -Dmaven.compiler.showWarnings=true -Domnisci.release.version="${OMNISCI_JAR_RELEASE_VERSION}" -Djava.net.preferIPv4Stack=true -Dmaven.wagon.http.retryHandler.count=3 -DLOG_DIR="${CMAKE_BINARY_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/java/heavydb/target/${UTILITY_JAR} ${CMAKE_BINARY_DIR}/bin
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/java/heavyaijdbc/target/${JDBC_JAR} ${CMAKE_BINARY_DIR}/bin
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/java/calcite/target/calcite-1.41.0-SNAPSHOT-jar-with-dependencies.jar ${CMAKE_BINARY_DIR}/bin
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/java
  DEPENDS
    ${CMAKE_SOURCE_DIR}/java/thrift/src/gen/ai/heavy/thrift/server/Heavy.java
    ${CMAKE_SOURCE_DIR}/java/thrift/src/gen/ai/heavy/thrift/calciteserver/CalciteServer.java
    ${CMAKE_SOURCE_DIR}/java/calcite/src/main/codegen/config.fmpp
    ${CMAKE_SOURCE_DIR}/java/pom.xml
    ${CMAKE_SOURCE_DIR}/heavy.thrift
    ${JAVA_ASSEMBLY_DESCRIPTORS}
    ${JAVA_POM}
    ${JAVA_SOURCES}
    ${JAVA_FTL}
  )
add_custom_target(mapd_java_components ALL DEPENDS
  ${CMAKE_BINARY_DIR}/bin/${UTILITY_JAR}
  ${CMAKE_BINARY_DIR}/bin/${JDBC_JAR}
  ${CMAKE_BINARY_DIR}/bin/calcite-1.41.0-SNAPSHOT-jar-with-dependencies.jar)
add_custom_target(mapd_java_clean
  COMMAND ${MVN_PATH_COMMAND} ${MVN_EXECUTABLE} -q clean -Domnisci.release.version="${OMNISCI_JAR_RELEASE_VERSION}"
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/java
  )
add_dependencies(clean-all mapd_java_clean)
install(FILES ${CMAKE_SOURCE_DIR}/java/heavydb/target/${UTILITY_JAR} DESTINATION bin COMPONENT "jar")
install(FILES ${CMAKE_SOURCE_DIR}/java/heavyaijdbc/target/${JDBC_JAR} DESTINATION bin COMPONENT "jar")
install(FILES ${CMAKE_SOURCE_DIR}/java/calcite/target/calcite-1.41.0-SNAPSHOT-jar-with-dependencies.jar DESTINATION bin COMPONENT "jar")

add_custom_target(maven_populate_cache
  COMMAND ${MVN_PATH_COMMAND} ${MVN_EXECUTABLE} -q verify -Domnisci.release.version="${OMNISCI_JAR_RELEASE_VERSION}"
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/java
  )

set_directory_properties(PROPERTIES ADDITIONAL_CLEAN_FILES "${ADDITIONAL_MAKE_CLEAN_FILES}")
set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "${ADDITIONAL_MAKE_CLEAN_FILES}")

install(TARGETS initheavy heavydb DESTINATION bin COMPONENT "exe")
install(FILES ${CMAKE_BINARY_DIR}/heavyai_git_hash.txt DESTINATION "." COMPONENT "doc")
if(ENABLE_CUDA)
  install(FILES ${CMAKE_BINARY_DIR}/QueryEngine/cuda_mapd_rt.fatbin DESTINATION QueryEngine COMPONENT "exe")
endif()
install(FILES completion_hints.thrift DESTINATION "." COMPONENT "thrift")
install(FILES heavy.thrift DESTINATION "." COMPONENT "thrift")
install(FILES common.thrift DESTINATION "." COMPONENT "thrift")
install(FILES QueryEngine/serialized_result_set.thrift DESTINATION "QueryEngine/" COMPONENT "thrift")
install(FILES QueryEngine/extension_functions.thrift DESTINATION "QueryEngine/" COMPONENT "thrift")

if(NOT PREFER_STATIC_LIBS)
  install(FILES ${Boost_LIBRARIES} DESTINATION ThirdParty/lib)
endif()

install(PROGRAMS startheavy DESTINATION ".")
install(PROGRAMS scripts/nvidia-graphics-env.sh DESTINATION "scripts")
install(PROGRAMS scripts/innerstartheavy DESTINATION "scripts")
install(PROGRAMS scripts/start_heavyiq.sh DESTINATION "scripts")

install(FILES docker/README.md DESTINATION "docker")
if(ENABLE_CUDA OR ENABLE_RENDERING)
  install(FILES docker/Dockerfile.cuda RENAME Dockerfile DESTINATION "docker")
else()
  install(FILES docker/Dockerfile.cpu RENAME Dockerfile DESTINATION "docker")
endif()


# provide backwards compatible symlinks to the old names through < v6.3.0
if("${MAPD_VERSION_RAW}" VERSION_LESS "6.3" AND ${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink heavydb omnisci_server
    COMMAND ${CMAKE_COMMAND} -E create_symlink heavy_web_server omnisci_web_server
    COMMAND ${CMAKE_COMMAND} -E create_symlink heavysql omnisql
    COMMAND ${CMAKE_COMMAND} -E create_symlink initheavy initdb
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  install(FILES
    ${CMAKE_BINARY_DIR}/bin/omnisci_server
    ${CMAKE_BINARY_DIR}/bin/omnisci_web_server
    ${CMAKE_BINARY_DIR}/bin/omnisql
    ${CMAKE_BINARY_DIR}/bin/initdb
    DESTINATION "bin"
  )
endif()

execute_process(COMMAND uname -m OUTPUT_VARIABLE MAPD_HOST_SYSTEM_ARCHITECTURE OUTPUT_STRIP_TRAILING_WHITESPACE)
# OS/distro segment of the package filename. Defaults to CMAKE_SYSTEM_NAME (e.g.
# "Linux"); pass -DMAPD_PACKAGE_DISTRO_NAME=ubuntu22.04 to embed the distro
# instead — "Linux" is implicit for these packages.
set(MAPD_PACKAGE_DISTRO_NAME "${CMAKE_SYSTEM_NAME}" CACHE STRING
  "OS/distro segment of the cpack package filename (e.g. ubuntu22.04, rockylinux8)")
set(CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${CPACK_PACKAGE_VERSION}-${MAPD_PACKAGE_DISTRO_NAME}-${MAPD_HOST_SYSTEM_ARCHITECTURE}${MAPD_PACKAGE_FLAGS}")

set(CPACK_GENERATOR "STGZ")

include(CPack)

if(DOXYGEN_FOUND)
  add_custom_target(sphinx
    COMMAND python3 -m venv sphinx-env
    COMMAND . sphinx-env/bin/activate && pip install -r requirements.txt
    COMMAND rm -rf build
    COMMAND . sphinx-env/bin/activate && make html SPHINXOPTS="-D version=${MAPD_VERSION_MAJOR}.${MAPD_VERSION_MINOR}.${MAPD_VERSION_PATCH}"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/docs
    )

  add_dependencies(sphinx doxygen)
endif(DOXYGEN_FOUND)

add_dependencies(heavydb get_git_hash)

# heavy: build our main executables (the entire bin/ directory)
list(APPEND HEAVY_TARGETS heavydb initheavy heavysql)
add_dependencies(initheavy mapd_java_components)
add_dependencies(heavydb mapd_java_components)
add_custom_target(heavy ALL DEPENDS ${HEAVY_TARGETS})
