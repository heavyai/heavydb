cmake_minimum_required(VERSION 2.8)
set(csv_import_source_files
    Importer.cpp
    Importer.h
    ${CMAKE_BINARY_DIR}/gen-cpp/MapD.h
    )

set_source_files_properties(${mapd_thrift_gen_files} GENERATED)

add_library(CsvImport ${csv_import_source_files})
target_link_libraries(CsvImport Shared Catalog Chunk DataMgr StringDictionary ${Glog_LIBRARIES} "-ldl")
