add_library(CudaMgr CudaMgr.cpp)

target_link_libraries(CudaMgr ${CUDA_LIBRARIES})
