if(ENABLE_AWS_S3)
  include_directories(${LibAwsS3_INCLUDE_DIRS})
  list(APPEND IMPORT_EXPORT_LIBRARIES "${LibAwsS3_LIBRARIES}")
  list(APPEND S3Archive ../Archive/S3Archive.cpp)
endif()

if(ENABLE_IMPORT_PARQUET)
  list(APPEND IMPORT_EXPORT_LIBRARIES "${Parquet_LIBRARIES}")
endif()

set(IMPORT_SOURCES
  Importer.cpp
  DelimitedParserUtils.cpp)

set(EXPORT_SOURCES
  QueryExporter.cpp
  QueryExporterCSV.cpp)

add_library(RenderGroupAnalyzer RenderGroupAnalyzer.cpp)
add_library(ImportExport ${IMPORT_SOURCES} ${EXPORT_SOURCES} ${S3Archive})

target_link_libraries(ImportExport RenderGroupAnalyzer mapd_thrift Logger Shared Catalog DataMgr StringDictionary ${CMAKE_DL_LIBS}
 ${LibArchive_LIBRARIES} ${IMPORT_EXPORT_LIBRARIES} ${Arrow_LIBRARIES})

add_library(RowToColumn RowToColumnLoader.cpp RowToColumnLoader.h DelimitedParserUtils.cpp DelimitedParserUtils.h)
target_link_libraries(RowToColumn ThriftClient)

add_executable(StreamImporter StreamImporter.cpp)
target_link_libraries(StreamImporter RowToColumn mapd_thrift Logger Shared ${CMAKE_DL_LIBS} ${Boost_LIBRARIES} ${PROFILER_LIBS})

add_executable(KafkaImporter KafkaImporter.cpp)
target_link_libraries(KafkaImporter RowToColumn mapd_thrift ${RdKafka_LIBRARIES} Logger Shared ${CMAKE_DL_LIBS} ${Boost_LIBRARIES} ${PROFILER_LIBS})

install(TARGETS StreamImporter KafkaImporter DESTINATION bin  COMPONENT "exe")
