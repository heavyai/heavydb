cmake_minimum_required(VERSION 2.8)
#SET(CMAKE_CXX_FLAGS "-std=c++0x -O3 -g")
SET (planner_source_files
    Planner.cpp
    Planner.h
)

include_directories("/usr/local/include")

ADD_LIBRARY (Planner ${planner_source_files})

TARGET_LINK_LIBRARIES (Planner Parser Analyzer)
