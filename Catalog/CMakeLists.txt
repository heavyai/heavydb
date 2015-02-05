cmake_minimum_required(VERSION 2.8)
SET(CMAKE_CXX_FLAGS "-std=c++0x -g")
SET (catalog_source_files
    Catalog.cpp
    Catalog.h
)

include_directories("/usr/local/include")

ADD_LIBRARY (Catalog ${catalog_source_files})

TARGET_LINK_LIBRARIES (Catalog SqliteConnector Fragmenter)
