set(query_runner_source_files
    QueryRunner.cpp
)

add_library(QueryRunner ${query_runner_source_files})
target_link_libraries(QueryRunner ${Boost_LIBRARIES})
