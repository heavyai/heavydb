#.rst:
# FindBisonpp.cmake
# -------------
#
# Find a Bison++ installation.
#
# This module finds if Bisonpp is installed and selects a default
# configuration to use.
#
# find_package(Bisonpp ...)
#
#
# The following are set after the configuration is done:
#
# ::
#
#   Bisonpp_FOUND            - Set to TRUE if Bisonpp was found.
#   Bisonpp_EXECUTABLE       - Path to the Bisonpp executable.

find_program(Bisonpp_EXECUTABLE
  NAMES bison++
  PATHS
  /usr/bin
  /usr/local/bin
  /usr/local/homebrew/bin
  /opt/local/bin)

# Set standard CMake FindPackage variables if found.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Bisonpp REQUIRED_VARS Bisonpp_EXECUTABLE)
