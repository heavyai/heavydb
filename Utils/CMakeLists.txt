cmake_minimum_required(VERSION 2.8)
#SET(CMAKE_CXX_FLAGS "-std=c++0x -O3")
SET (utils_source_files
    StringLike.cpp
    ChunkIter.cpp
)

ADD_LIBRARY (Utils ${utils_source_files})
