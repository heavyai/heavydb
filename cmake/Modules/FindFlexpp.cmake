#.rst:
# FindFlexpp.cmake
# -------------
#
# Find a Flex++ installation.
#
# This module finds if Flexpp is installed and selects a default
# configuration to use.
#
# find_package(Flexpp ...)
#
#
# The following are set after the configuration is done:
#
# ::
#
#   Flexpp_FOUND            - Set to TRUE if Flexpp was found.
#   Flexpp_EXECUTABLE       - Path to the Flexpp executable.

find_program(Flexpp_EXECUTABLE
  NAMES flex++
  PATHS
  /usr/bin
  /usr/local/bin
  /usr/local/homebrew/bin
  /opt/local/bin)

# Set standard CMake FindPackage variables if found.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Flexpp REQUIRED_VARS Flexpp_EXECUTABLE)
