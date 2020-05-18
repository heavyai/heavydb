if(ENABLE_AWS_S3)
  include_directories(${LibAwsS3_INCLUDE_DIRS})
  list(APPEND IMPORT_LIBRARIES "${LibAwsS3_LIBRARIES}")
  list(APPEND S3Archive ../Archive/S3Archive.cpp)
endif()

if(ENABLE_IMPORT_PARQUET)
  list(APPEND IMPORT_LIBRARIES "${Parquet_LIBRARIES}")
endif()

add_library(CsvImport Importer.cpp Importer.h DelimitedParserUtils.cpp DelimitedParserUtils.h ${S3Archive})

target_link_libraries(CsvImport mapd_thrift Shared Catalog Chunk DataMgr StringDictionary ${GDAL_LIBRARIES} ${CMAKE_DL_LIBS}
 ${LibArchive_LIBRARIES} ${IMPORT_LIBRARIES} ${Arrow_LIBRARIES})

install(DIRECTORY ${CMAKE_SOURCE_DIR}/ThirdParty/gdal-data DESTINATION "ThirdParty")
add_custom_target(gdal-data ALL COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/ThirdParty/gdal-data" "${CMAKE_BINARY_DIR}/ThirdParty/gdal-data")

install(DIRECTORY ${CMAKE_SOURCE_DIR}/ThirdParty/geo_samples DESTINATION "ThirdParty")
add_custom_target(geo_samples ALL COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/ThirdParty/geo_samples" "${CMAKE_BINARY_DIR}/ThirdParty/geo_samples")

add_library(RowToColumn RowToColumnLoader.cpp RowToColumnLoader.h DelimitedParserUtils.cpp DelimitedParserUtils.h)
target_link_libraries(RowToColumn ThriftClient)

add_executable(StreamImporter StreamImporter.cpp)
target_link_libraries(StreamImporter RowToColumn mapd_thrift Shared ${CMAKE_DL_LIBS} ${Boost_LIBRARIES} ${PROFILER_LIBS})

add_executable(KafkaImporter KafkaImporter.cpp)
target_link_libraries(KafkaImporter RowToColumn mapd_thrift ${RdKafka_LIBRARIES} Shared ${CMAKE_DL_LIBS} ${Boost_LIBRARIES} ${PROFILER_LIBS})

install(TARGETS StreamImporter KafkaImporter DESTINATION bin)
