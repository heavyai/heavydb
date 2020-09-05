set(query_runner_source_files
    QueryRunner.cpp
    TestProcessSignalHandler.cpp
)

set(QR_LIBRARIES mapd_thrift Catalog QueryState Parser QueryEngine ImportExport Shared)
if("${MAPD_EDITION_LOWER}" STREQUAL "ee")
  list(APPEND query_runner_source_files ee/DistributedQueryRunner.cpp)
  list(APPEND QR_LIBRARIES Distributed)
endif()

add_library(QueryRunner ${query_runner_source_files})
target_link_libraries(QueryRunner ${QR_LIBRARIES} ${Boost_LIBRARIES})
