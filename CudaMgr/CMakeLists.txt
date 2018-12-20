if(ENABLE_CUDA)
    add_library(CudaMgr CudaMgr.cpp)
else()
    add_library(CudaMgr CudaMgrNoCuda.cpp)
endif()

target_link_libraries(CudaMgr ${CUDA_LIBRARIES})
