#.rst:
# FindGLFW3
# --------
#
# Find the glfw3 Library
#
# IMPORTED Targets
# ^^^^^^^^^^^^^^^^
#
# This module defines the :prop_tgt:`IMPORTED` target ``GLFW3::GLFW3``,
# if glfw3 has been found.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ::
#
#   GLFW3_INCLUDE_DIRS - include directories for glfw3
#   GLFW3_LIBRARIES - libraries to link against glfw3
#   GLFW3_FOUND - true if glfw3 has been found and can be used

find_library(GLFW3_LIBRARY
  NAMES glfw glfw3
  HINTS
  ENV LD_LIBRARY_PATH
  ENV DYLD_LIBRARY_PATH
  PATHS
	/usr/lib
	/usr/local/lib
  /usr/local/homebrew/lib
  /opt/local/lib
)

get_filename_component(GLFW3_LIBRARY_DIR ${GLFW3_LIBRARY} DIRECTORY)

find_path(GLFW3_INCLUDE_DIR
  NAMES GLFW/glfw3.h
  HINTS
  ${GLFW3_LIBRARY_DIR}/../include
  PATHS
  /include
  /usr/include
  /usr/local/include
  /usr/local/homebrew/include
  /opt/local/include
)

# Set standard CMake FindPackage variables
set(GLFW3_LIBRARIES ${GLFW3_LIBRARY} ${CMAKE_DL_LIBS})
set(GLFW3_INCLUDE_DIRS ${GLFW3_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GLFW3 REQUIRED_VARS GLFW3_LIBRARY GLFW3_INCLUDE_DIR)

if (GLFW3_FOUND AND NOT TARGET GLFW3::GLFW3)
  add_library(GLFW3::GLFW3 UNKNOWN IMPORTED)
  set_target_properties(GLFW3::GLFW3 PROPERTIES
    IMPORTED_LOCATION "${GLFW3_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${GLFW3_INCLUDE_DIRS}")
endif()

mark_as_advanced(GLFW3_INCLUDE_DIR GLFW3_LIBRARY)
