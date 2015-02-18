cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wno-attributes -O3")
set(csv_import_source_files
    CsvImport.cpp
    csvparser.c
    csvparser.h)

add_library(CsvImport ${csv_import_source_files})
target_link_libraries(CsvImport Catalog Chunk DataMgr StringDictionary "-ldl -lglog")
