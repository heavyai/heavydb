# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2015-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

if(ENABLE_CUDA)
    add_library(CudaMgr CudaMgr.cpp DeviceMemoryAllocationMap.cpp JumpBufferTransferMgr.cpp)
else()
    add_library(CudaMgr CudaMgrNoCuda.cpp)
endif()

target_link_libraries(CudaMgr ${CUDA_LIBRARIES})
