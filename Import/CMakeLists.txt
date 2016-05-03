cmake_minimum_required(VERSION 2.8)
set(csv_import_source_files
    Importer.cpp
    Importer.h
    ${CMAKE_BINARY_DIR}/gen-cpp/MapD.h
    ${CMAKE_BINARY_DIR}/gen-cpp/mapd_types.cpp
    )

set_source_files_properties(${CMAKE_BINARY_DIR}/gen-cpp/mapd_types.cpp GENERATED)

add_library(CsvImport ${csv_import_source_files})
target_link_libraries(CsvImport Shared Catalog Chunk DataMgr StringDictionary poly2tri shapelib ${Glog_LIBRARIES} "-ldl")

add_custom_command(
    DEPENDS ${CMAKE_SOURCE_DIR}/mapd.thrift
    OUTPUT ${CMAKE_BINARY_DIR}/gen-cpp/MapD.h
    COMMAND ${Thrift_EXECUTABLE}
    ARGS -gen cpp -o ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/mapd.thrift)
