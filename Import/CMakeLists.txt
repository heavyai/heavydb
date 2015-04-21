cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wno-attributes -O3 -g")
set(csv_import_source_files
    Importer.cpp
    )

add_library(CsvImport ${csv_import_source_files})
target_link_libraries(CsvImport Shared Catalog Chunk DataMgr StringDictionary "-ldl -lglog")
