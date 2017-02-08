add_library(CsvImport Importer.cpp Importer.h)
target_link_libraries(CsvImport mapd_thrift Shared Catalog Chunk DataMgr StringDictionary poly2tri ${GDAL_LIBRARIES} ${Glog_LIBRARIES} ${CMAKE_DL_LIBS})

install(DIRECTORY ${CMAKE_SOURCE_DIR}/ThirdParty/gdal-data DESTINATION "ThirdParty")
add_custom_target(gdal-data ALL COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/ThirdParty/gdal-data" "${CMAKE_BINARY_DIR}/ThirdParty/gdal-data")
