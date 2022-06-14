set(arrow_query_runner_files
    ArrowSQLRunner.cpp
    RelAlgCache.cpp
    SQLiteComparator.cpp
)

add_library(ArrowQueryRunner ${arrow_query_runner_files})
target_link_libraries(ArrowQueryRunner gtest ${Boost_LIBRARIES})

