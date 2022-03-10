#.rst:
# FindImGui
# --------
#
# Find the path to the ImGui source tree
#
# Minimal search, checks CMAKE_PREFIX_PATH and IMGUI_PATH env var
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ::
#
#   ImGui_PATH - Path to the ImGui source tree
#   ImGui_FOUND - true if ImGui has been found and can be used

find_path(ImGui_PATH
  NAMES imgui.h
  PATHS ENV CMAKE_PREFIX_PATH
  PATHS ENV IMGUI_PATH
  PATH_SUFFIXES "imgui"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ImGui REQUIRED_VARS ImGui_PATH)
