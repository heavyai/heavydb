cmake_minimum_required(VERSION 2.8)
find_package(CUDA)

#SET (CMAKE_CXX_FLAGS "-std=c++0x -O0 -g")

SET (datamgr_source_files
    DataMgr.cpp
    Encoder.cpp
    StringNoneEncoder.cpp
    FileMgr/FileMgr.cpp
    FileMgr/FileBuffer.cpp
    FileMgr/FileInfo.cpp
    FileMgr/File.cpp
    BufferMgr/GpuCudaBufferMgr/GpuCudaBufferMgr.cpp
    BufferMgr/GpuCudaBufferMgr/GpuCudaBuffer.cpp
    BufferMgr/CpuBufferMgr/CpuBufferMgr.cpp
    BufferMgr/CpuBufferMgr/CpuBuffer.cpp
    BufferMgr/BufferMgr.cpp
    BufferMgr/Buffer.cpp
)

ADD_LIBRARY(DataMgr ${datamgr_source_files})

TARGET_LINK_LIBRARIES (DataMgr CudaMgr ${Boost_THREAD_LIBRARY} "-lglog")

