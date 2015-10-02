#.rst:
# FindGD
# --------
#
# Find the gd Library
#
# IMPORTED Targets
# ^^^^^^^^^^^^^^^^
#
# This module defines the :prop_tgt:`IMPORTED` target ``GD::GD``,
# if GD has been found.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ::
#
#   GD_INCLUDE_DIRS - include directories for GD
#   GD_LIBRARIES - libraries to link against GD
#   GD_FOUND - true if GD has been found and can be used

find_path(GD_INCLUDE_DIR gd.h)
find_library(GD_LIBRARY NAMES gd PATH_SUFFIXES lib64)

set(GD_INCLUDE_DIRS ${GD_INCLUDE_DIR})
set(GD_LIBRARIES ${GD_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GD REQUIRED_VARS GD_INCLUDE_DIR GD_LIBRARY)

if(GD_FOUND AND NOT TARGET GD::GD)
  add_library(GD::GD UNKNOWN IMPORTED)
  set_target_properties(GD::GD PROPERTIES
    IMPORTED_LOCATION "${GD_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${GD_INCLUDE_DIRS}")
endif()

mark_as_advanced(GD_INCLUDE_DIR GD_LIBRARY)
