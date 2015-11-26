cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-result")

SET (chunk_source_files
    Chunk.cpp
)

ADD_LIBRARY(Chunk ${chunk_source_files})
TARGET_LINK_LIBRARIES(Chunk DataMgr)
