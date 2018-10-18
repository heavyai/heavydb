# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

#.rst:
# FindGDAL
# --------
#
# Locate gdal
#
# This module defines the following CMake variables:
#
#     GDAL_FOUND - True if libgdal is found
#     GDAL_LIBRARY - A variable pointing to the GDAL library
#     GDAL_INCLUDE_DIR - Where to find the headers
#

find_program(GDAL_CONFIG gdal-config
    PATH_SUFFIXES bin
    PATHS
        /sw # Fink
        /opt/local # DarwinPorts
        /opt/csw # Blastwave
        /opt
)

if(GDAL_CONFIG)
  exec_program(${GDAL_CONFIG} ARGS --prefix OUTPUT_VARIABLE GDAL_CONFIG_PREFIX)
else()
  message(FATAL_ERROR "Failed to find gdal-config executable in PATH")
endif()

find_path(GDAL_INCLUDE_DIR gdal.h
  HINTS
    ${GDAL_CONFIG_PREFIX}
  PATH_SUFFIXES
     include/gdal
     include/GDAL
     include
  PATHS
      ~/Library/Frameworks/gdal.framework/Headers
      /Library/Frameworks/gdal.framework/Headers
      /sw # Fink
      /opt/local # DarwinPorts
      /opt/csw # Blastwave
      /opt
)

find_library(GDAL_LIBRARY
  NAMES gdal gdal_i gdal1.5.0 gdal1.4.0 gdal1.3.2 GDAL
  HINTS
    ${GDAL_CONFIG_PREFIX}
  PATH_SUFFIXES lib
  PATHS
    /sw
    /opt/local
    /opt/csw
    /opt
    /usr/freeware
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GDAL DEFAULT_MSG GDAL_LIBRARY GDAL_INCLUDE_DIR)

set(GDAL_LIBRARIES ${GDAL_LIBRARY})
set(GDAL_INCLUDE_DIRS ${GDAL_INCLUDE_DIR})
