add_library(CsvImport Importer.cpp Importer.h)
target_link_libraries(CsvImport mapd_thrift Shared Catalog Chunk DataMgr StringDictionary poly2tri ${GDAL_LIBRARIES} ${Glog_LIBRARIES} ${CMAKE_DL_LIBS})
