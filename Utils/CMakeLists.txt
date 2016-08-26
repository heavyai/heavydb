cmake_minimum_required(VERSION 2.8)
#SET(CMAKE_CXX_FLAGS "-std=c++0x -O3")
SET (utils_source_files
    StringLike.cpp
    Regexp.cpp
    ChunkIter.cpp
)

ADD_LIBRARY (Utils ${utils_source_files})
target_link_libraries(Utils ${Boost_LIBRARIES})
