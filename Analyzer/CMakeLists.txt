cmake_minimum_required(VERSION 2.8)
#SET(CMAKE_CXX_FLAGS "-std=c++0x -O3 -g")
SET (analyzer_source_files
    Analyzer.cpp
    Analyzer.h
)

include_directories("/usr/local/include")

ADD_LIBRARY (Analyzer ${analyzer_source_files})

TARGET_LINK_LIBRARIES (Analyzer Catalog)
