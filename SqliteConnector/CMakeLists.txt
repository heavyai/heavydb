add_library(SqliteConnector SqliteConnector.cpp SqliteConnector.h)

set(SQLITE_CONNECTOR_LIBS ${Boost_THREAD_LIBRARY})

if (MSVC)
    list(APPEND SQLITE_CONNECTOR_LIBS ${SQLite3_LIBRARIES})
else()
    list(APPEND SQLITE_CONNECTOR_LIBS sqlite3)
endif()

target_link_libraries(SqliteConnector ${SQLITE_CONNECTOR_LIBS})
