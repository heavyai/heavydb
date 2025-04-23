if(ENABLE_CUDA)
    add_library(CudaMgr CudaMgr.cpp DeviceMemoryAllocationMap.cpp JumpBufferTransferMgr.cpp)
else()
    add_library(CudaMgr CudaMgrNoCuda.cpp)
endif()

target_link_libraries(CudaMgr ${CUDA_LIBRARIES})
