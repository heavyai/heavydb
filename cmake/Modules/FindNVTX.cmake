# FindNVTX
# --------
#
# Find NVTX3, the header only implementation of NVTX
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
#   NVTX_INCLUDE_DIRS - include directories for NVTX3
#   NVTX_FOUND - Set to TRUE if NVTX was found

if(DEFINED CUDA_CUDA_LIBRARY)
  get_filename_component(CUDA_CUDART_LIBRARY_DIR "${CUDA_CUDA_LIBRARY}" PATH CACHE)
  # Cuda is the reliable location, but standalone NSight Systems also redistributes NVTX3
  # locating NVTX within NSight is difficult since the folder naming is inconsistent
  # between versions. It's also common to have both Cuda and NSight installed, and
  # the NVTX versions may differ, so stick with includes shipped with Cuda
  find_path(NVTX_INCLUDE_DIRS
      NAMES
          nvToolsExt.h
      PATHS
        "${CUDA_CUDART_LIBRARY_DIR}"
        "${CUDA_TOOLKIT_ROOT_DIR}"
      PATH_SUFFIXES
          include/nvtx3
          nvtx3
      )

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(NVTX DEFAULT_MSG NVTX_INCLUDE_DIRS)
else()
  message(WARNING "CUDA_CUDA_LIBRARY not found, unable to local NVTX")
endif()
