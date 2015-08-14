cmake_minimum_required(VERSION 2.8)
SET (cudamgr_source_files
    CudaMgr.cpp
)

ADD_LIBRARY (CudaMgr ${cudamgr_source_files})

TARGET_LINK_LIBRARIES (CudaMgr -L/usr/local/cuda/lib ${CUDA_LIBRARIES})

