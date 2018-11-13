#.rst:
# FindBlosc
# --------
#
# Find the blosc compression library
#
# IMPORTED Targets
# ^^^^^^^^^^^^^^^^
#
# This module defines the :prop_tgt:`IMPORTED` target ``BLOSC::BLOSC``,
# if BLOSC has been found.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ::
#
#   BLOSC_INCLUDE_DIRS - include directories for BLOSC
#   BLOSC_LIBRARIES - libraries to link against BLOSC
#   BLOSC_FOUND - true if BLOSC has been found and can be used

find_library(BLOSC_LIBRARY 
  NAMES blosc   
  HINTS 
  ENV LD_LIBRARY_PATH
  ENV DYLD_LIBRARY_PATH
  PATHS
  /usr/lib
  /usr/local/lib
  /usr/local/homebrew/lib
  /opt/local/lib
)

get_filename_component(BLOSC_LIBRARY_DIR ${BLOSC_LIBRARY} DIRECTORY)

find_path(BLOSC_INCLUDE_DIR 
  NAMES blosc.h
  HINTS
  ${BLOSC_LIBRARY_DIR}/../include
  PATHS
  /include
  /usr/include
  /usr/local/include
  /usr/local/homebrew/include
  /opt/local/include
)
find_path(BLOSC_INCLUDE_DIR blosc.h)

find_library(BLOSC_LIBRARY
  NAMES blosc
  HINTS
  ENV LD_LIBRARY_PATH
  ENV DYLD_LIBRARY_PATH
  PATHS
  /usr/lib
  /usr/local/lib
  /usr/local/mapd-deps/lib
  /usr/local/homebrew/lib
  /opt/local/lib)

get_filename_component(BLOSC_LIBRARY_DIR ${BLOSC_LIBRARY} DIRECTORY)

set(BLOSC_LIBRARIES ${BLOSC_LIBRARY})
set(BLOSC_INCLUDE_DIR ${BLOSC_LIBRARY_DIR}/../include)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(BLOSC REQUIRED_VARS BLOSC_INCLUDE_DIR BLOSC_LIBRARY)

if(BLOSC_FOUND AND NOT TARGET BLOSC::BLOSC)
  add_library(BLOSC::BLOSC UNKNOWN IMPORTED)
  set_target_properties(BLOSC::BLOSC PROPERTIES
    IMPORTED_LOCATION "${BLOSC_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${BLOSC_INCLUDE_DIRS}")
endif()

mark_as_advanced(BLOSC_INCLUDE_DIR BLOSC_LIBRARY)
