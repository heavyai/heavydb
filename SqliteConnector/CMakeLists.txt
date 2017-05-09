add_library(SqliteConnector SqliteConnector.cpp SqliteConnector.h)

target_link_libraries(SqliteConnector sqlite3 ${Boost_THREAD_LIBRARY})
