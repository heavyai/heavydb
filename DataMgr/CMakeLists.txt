cmake_minimum_required(VERSION 2.8)
SET(CMAKE_CXX_FLAGS "-std=c++0x -g")

SET (datamgr_source_files
    DataMgr.cpp
    Encoder.cpp
    FileMgr/FileMgr.cpp
    FileMgr/FileBuffer.cpp
    FileMgr/FileInfo.cpp
    FileMgr/File.cpp
    BufferMgr/CpuBufferMgr/CpuBufferMgr.cpp
    BufferMgr/CpuBufferMgr/CpuBuffer.cpp
    BufferMgr/BufferMgr.cpp
    BufferMgr/Buffer.cpp
)

ADD_LIBRARY(DataMgr ${datamgr_source_files})

TARGET_LINK_LIBRARIES (DataMgr ${Boost_THREAD_LIBRARY})

