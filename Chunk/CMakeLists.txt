cmake_minimum_required(VERSION 2.8)
#SET(CMAKE_CXX_FLAGS "-std=c++0x -O0 -g")

SET (chunk_source_files
    Chunk.cpp
)

ADD_LIBRARY(Chunk ${chunk_source_files})
TARGET_LINK_LIBRARIES(Chunk DataMgr)
