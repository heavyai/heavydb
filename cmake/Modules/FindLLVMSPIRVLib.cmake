# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: MIT

include(FindPackageHandleStandardArgs)

find_path(LLVMSPIRVLib_INCLUDE_DIRS
  NAMES LLVMSPIRVLib/LLVMSPIRVLib.h LLVMSPIRVLib/LLVMSPIRVOpts.h
)
find_library(LLVMSPIRVLib_LIBRARIES
  NAMES LLVMSPIRVLib
)

find_package_handle_standard_args(LLVMSPIRVLib
  REQUIRED_VARS
    LLVMSPIRVLib_INCLUDE_DIRS
    LLVMSPIRVLib_LIBRARIES
  HANDLE_COMPONENTS
)

MESSAGE(STATUS "LLVMSPIRVLib_FOUND: " ${LLVMSPIRVLib_FOUND})
MESSAGE(STATUS "LLVMSPIRVLib_LIBRARIES: " ${LLVMSPIRVLib_LIBRARY})
MESSAGE(STATUS "LLVMSPIRVLib_INCLUDE_DIRS: " ${LLVMSPIRVLib_INCLUDE_DIRS})
