set(arrow_query_runner_files
    ArrowSQLRunner.cpp
    SchemaJson.cpp
    SQLiteComparator.cpp
)

add_library(ArrowQueryRunner ${arrow_query_runner_files})
target_link_libraries(ArrowQueryRunner ${Boost_LIBRARIES})

