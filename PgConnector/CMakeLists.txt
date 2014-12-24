cmake_minimum_required(VERSION 2.8)

#SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

SET (pg_source_files
    PgConnector.cpp
    PgConnector.h
)

#find_package (PQXX REQUIRED)
#find_package (PostgreSQL REQUIRED)
include_directories("/usr/local/include")
ADD_LIBRARY (PgConnector ${pg_source_files})
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /usr/local/lib)

target_link_libraries (PgConnector /usr/local/lib/libpqxx.dylib)


