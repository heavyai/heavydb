#.rst:
# FindGLFW3
# --------
#
# Find the GLFW3 Library
#
# IMPORTED Targets
# ^^^^^^^^^^^^^^^^
#
# This module defines the :prop_tgt:`IMPORTED` target ``GLFW3::GLFW3``,
# if GLFW3 has been found.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ::
#
#   GLFW3_INCLUDE_DIRS - include directories for GLFW3
#   GLFW3_LIBRARIES - libraries to link against GLFW3
#   GLFW3_FOUND - true if GLFW3 has been found and can be used

find_path(GLFW3_INCLUDE_DIR GLFW/glfw3.h)
find_library(GLFW3_LIBRARY NAMES glfw glfw3 PATH_SUFFIXES lib64)

set(GLFW3_INCLUDE_DIRS ${GLFW3_INCLUDE_DIR})
set(GLFW3_LIBRARIES ${GLFW3_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GLFW3 REQUIRED_VARS GLFW3_INCLUDE_DIR GLFW3_LIBRARY)

if(GLFW3_FOUND AND NOT TARGET GLFW3::GLFW3)
  add_library(GLFW3::GLFW3 UNKNOWN IMPORTED)
  set_target_properties(GLFW3::GLFW3 PROPERTIES
    IMPORTED_LOCATION "${GLFW3_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${GLFW3_INCLUDE_DIRS}")
endif()

mark_as_advanced(GLFW3_INCLUDE_DIR GLFW3_LIBRARY)
