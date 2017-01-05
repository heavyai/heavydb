cmake_minimum_required(VERSION 2.8)

add_library(CsvImport Importer.cpp Importer.h)
target_link_libraries(CsvImport mapd_thrift Shared Catalog Chunk DataMgr StringDictionary poly2tri shapelib ${Glog_LIBRARIES} "-ldl")
