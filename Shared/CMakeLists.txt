cmake_minimum_required(VERSION 2.8)
SET(CMAKE_CXX_FLAGS "-std=c++0x -g")
SET (shared_source_files
    DatumString.cpp
)

ADD_LIBRARY (Shared ${shared_source_files})
