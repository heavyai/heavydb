cmake_minimum_required(VERSION 2.8)
#SET(CMAKE_CXX_FLAGS "-std=c++0x -O3")
SET (shared_source_files
    Datum.cpp
    timegm.cpp
    mapd_glob.cpp
)

ADD_LIBRARY (Shared ${shared_source_files})
