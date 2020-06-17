# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

#.rst:
# FindGeos
# --------
#
# Locate geos
#
# This module defines the following CMake variables:
#
#     GEOS_NOTFOUND    - Set to TRUE if geos was not found
#     GEOS_LIBRARY     - A variable pointing to the Geos library
#     GEOS_INCLUDE_DIR - Where to find the headers
#

find_program(GEOS_CONFIG geos-config
    PATH_SUFFIXES bin
    PATHS
        /sw # Fink
        /opt/local # DarwinPorts
        /opt/csw # Blastwave
        /opt
)

if(GEOS_CONFIG)
	exec_program(${GEOS_CONFIG} ARGS --prefix OUTPUT_VARIABLE GEOS_CONFIG_PREFIX)
else()
  message(FATAL_ERROR "Failed to find geos-config executable in PATH")
endif()

find_path(GEOS_INCLUDE_DIR geos_c.h
  HINTS
     ${GEOS_CONFIG_PREFIX}
  PATH_SUFFIXES
     include
  PATHS
      ~/Library/Frameworks/gdal.framework/Headers
      /Library/Frameworks/gdal.framework/Headers
      /sw # Fink
      /opt/local # DarwinPorts
      /opt/csw # Blastwave
      /opt
)

find_library(GEOS_LIBRARY
  NAMES libgeos_c.so libgeos_c.dylib
  HINTS
    ${GEOS_CONFIG_PREFIX}
  PATH_SUFFIXES lib
  PATHS
    /sw
    /opt/local
    /opt/csw
    /opt
    /usr/freeware
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GEOS DEFAULT_MSG GEOS_LIBRARY GEOS_INCLUDE_DIR)

set(GEOS_LIBRARY ${GEOS_LIBRARY})
set(GEOS_INCLUDE_DIR ${GEOS_INCLUDE_DIR})
