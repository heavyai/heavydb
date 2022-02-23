#.rst:
# FindGLFW3
# --------
#
# Find the glfw3 Library
#
# IMPORTED Targets
# ^^^^^^^^^^^^^^^^
#
# This module defines the :prop_tgt:`IMPORTED` target ``GLFW::GLFW``,
# if glfw3 has been found.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ::
#
#   GLFW_INCLUDE_DIRS - include directories for glfw3
#   GLFW_LIBRARIES - libraries to link against glfw3
#   GLFW_FOUND - true if glfw3 has been found and can be used

find_library(GLFW_LIBRARY
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

get_filename_component(GLFW_LIBRARY_DIR ${GLFW_LIBRARY} DIRECTORY)

find_path(GLFW_INCLUDE_DIR
  NAMES GLFW/glfw3.h
  HINTS
  ${GLFW_LIBRARY_DIR}/../include
  PATHS
  /include
  /usr/include
  /usr/local/include
  /usr/local/homebrew/include
  /opt/local/include
)

# Set standard CMake FindPackage variables
set(GLFW_LIBRARIES ${GLFW_LIBRARY} ${CMAKE_DL_LIBS})
set(GLFW_INCLUDE_DIRS ${GLFW_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GLFW REQUIRED_VARS GLFW_LIBRARY GLFW_INCLUDE_DIR)

if (GLFW_FOUND AND NOT TARGET GLFW::GLFW)
  add_library(GLFW::GLFW UNKNOWN IMPORTED)
  set_target_properties(GLFW::GLFW PROPERTIES
    IMPORTED_LOCATION "${GLFW_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${GLFW_INCLUDE_DIRS}")
endif()

mark_as_advanced(GLFW_INCLUDE_DIR GLFW_LIBRARY)
