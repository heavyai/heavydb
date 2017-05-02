if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set_source_files_properties(sqlite3.c PROPERTIES COMPILE_FLAGS "-O0 -Wno-unused-value -Wno-parentheses-equality")
else()
  set_source_files_properties(sqlite3.c PROPERTIES COMPILE_FLAGS "-O0 -Wno-unused-value")
endif()
set(sqliteconnector_source_files
    SqliteConnector.cpp
    SqliteConnector.h
    sqlite3.c
    sqlite3.h
)

add_library(SqliteConnector ${sqliteconnector_source_files})

target_link_libraries(SqliteConnector ${Boost_THREAD_LIBRARY})
