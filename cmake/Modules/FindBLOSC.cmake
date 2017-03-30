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

find_path(BLOSC_INCLUDE_DIR blosc.h)
find_library(BLOSC_LIBRARY NAMES blosc PATH_SUFFIXES lib64)

set(BLOSC_INCLUDE_DIRS ${BLOSC_INCLUDE_DIR})
set(BLOSC_LIBRARIES ${BLOSC_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(BLOSC REQUIRED_VARS BLOSC_INCLUDE_DIR BLOSC_LIBRARY)

if(BLOSC_FOUND AND NOT TARGET BLOSC::BLOSC)
  add_library(BLOSC::BLOSC UNKNOWN IMPORTED)
  set_target_properties(BLOSC::BLOSC PROPERTIES
    IMPORTED_LOCATION "${BLOSC_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${BLOSC_INCLUDE_DIRS}")
endif()

mark_as_advanced(BLOSC_INCLUDE_DIR BLOSC_LIBRARY)
