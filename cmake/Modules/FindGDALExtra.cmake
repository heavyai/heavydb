# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

#.rst:
# FindGDALExtra.cmake
# -------------
#
# Find additional libraries required by GDAL.
# This is only relevant when static linking GDAL.
#
# find_package(GDALExtra ...)
#
# The following are set after the configuration is done:
#
# ::
#
#   GDALExtra_FOUND            - Set to TRUE if the libraries were found.
#   GDALExtra_INCLUDE_DIRS     - Include directories
#   GDALExtra_LIBRARIES        - Path to the libraries.
#   GDALExtra_LIBRARY_DIRS     - compile time link directories
#
#
# Sample usage:
#
# ::
#
#    find_package(GDALExtra)
#    if(GDALExtra_FOUND)
#      target_link_libraries(<YourTarget> ${GDALExtra_LIBRARIES})
#    endif()

if(PREFER_STATIC_LIBS)
  set(_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
  set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

set(GDALExtra_LIBRARIES "")
function(find_static_lib name)
  set(_mapd_deps_lib_paths "")
  if(MAPD_DEPS_PATH)
    list(APPEND _mapd_deps_lib_paths
      "${MAPD_DEPS_PATH}/lib"
      "${MAPD_DEPS_PATH}/lib64")
  endif()
  find_library(${name}_LIBRARY
    NAMES ${ARGN}
    HINTS ENV LD_LIBRARY_PATH
    HINTS ENV DYLD_LIBRARY_PATH
    PATHS
    ${_mapd_deps_lib_paths}
    /usr/lib
    /usr/local/lib
    /usr/local/homebrew/lib
    /opt/local/lib)
endfunction()

if(PREFER_STATIC_LIBS)
  find_static_lib(EXPAT expat)
  find_static_lib(KMLDOM kmldom)
  find_static_lib(MINIZIP minizip)
  find_static_lib(KMLENGINE kmlengine)
  find_static_lib(KMLBASE kmlbase)
  find_static_lib(URIPARSER uriparser)
  find_static_lib(PROJ proj)
  find_static_lib(BLOSC blosc)
  find_static_lib(HDF5 hdf5)
  find_static_lib(HDF5_HL hdf5_hl)
  find_static_lib(NETCDF netcdf)

  set(GDALExtra_LIBRARIES ${KMLDOM_LIBRARY} ${EXPAT_LIBRARY} ${KMLENGINE_LIBRARY} ${KMLBASE_LIBRARY} ${MINIZIP_LIBRARY} ${URIPARSER_LIBRARY} ${PROJ_LIBRARY} ${BLOSC_LIBRARY} ${NETCDF_LIBRARY} ${HDF5_LIBRARY} ${HDF5_HL_LIBRARY} ${HDF5_LIBRARY})

  if(GDAL_CONFIG)
    exec_program(${GDAL_CONFIG} ARGS --prefix OUTPUT_VARIABLE GDAL_CONFIG_PREFIX)
  else()
    message(FATAL_ERROR "Failed to find gdal-config executable in PATH")
  endif()

  execute_process(COMMAND ${GDAL_CONFIG} --version
    OUTPUT_VARIABLE GDAL_version_output
    ERROR_VARIABLE GDAL_version_output
    RESULT_VARIABLE GDAL_version_result)

  if("${GDAL_version_output}" VERSION_GREATER_EQUAL "3.7.3")
    # additional libs for 8.0 deps
    find_static_lib(OPENJPEG openjp2)
    find_static_lib(GEOTIFF geotiff)
    find_static_lib(TIFF tiff)
    find_static_lib(LZMA lzma)
    find_static_lib(LCMS2 lcms2)
    find_static_lib(WEBP webp)
    find_static_lib(SHARPYUV sharpyuv)
    list(APPEND GDALExtra_LIBRARIES ${OPENJPEG_LIBRARY} ${GEOTIFF_LIBRARY} ${TIFF_LIBRARY} ${LZMA_LIBRARY} ${LCMS2_LIBRARY} ${WEBP_LIBRARY} ${SHARPYUV_LIBRARY})
  endif()

  if(APPLE)
    find_static_lib(ICONV iconv)
    list(APPEND GDALExtra_LIBRARIES ${ICONV_LIBRARY})
  endif()
endif()

if(PREFER_STATIC_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

include(FindPackageHandleStandardArgs)
if(PREFER_STATIC_LIBS)
  find_package_handle_standard_args(GDALExtra REQUIRED_VARS EXPAT_LIBRARY KMLDOM_LIBRARY MINIZIP_LIBRARY KMLENGINE_LIBRARY KMLBASE_LIBRARY URIPARSER_LIBRARY PROJ_LIBRARY BLOSC_LIBRARY NETCDF_LIBRARY HDF5_LIBRARY HDF5_HL_LIBRARY)
else()
  # Nothing to find: when GDAL is linked dynamically its dependencies are
  # resolved through libgdal.so itself.
  set(GDALExtra_NO_EXTRA_LIBS TRUE)
  find_package_handle_standard_args(GDALExtra REQUIRED_VARS GDALExtra_NO_EXTRA_LIBS)
endif()
