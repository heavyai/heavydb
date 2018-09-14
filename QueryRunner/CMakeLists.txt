set(query_runner_source_files
    QueryRunner.cpp
)

set(MAPD_LIBRARIES Catalog mapd_thrift)
if("${MAPD_EDITION_LOWER}" STREQUAL "ee")
  list(APPEND MAPD_LIBRARIES Distributed)
endif()

add_library(QueryRunner ${query_runner_source_files})
add_dependencies(QueryRunner Parser)
target_link_libraries(QueryRunner ${MAPD_LIBRARIES} ${Boost_LIBRARIES})
