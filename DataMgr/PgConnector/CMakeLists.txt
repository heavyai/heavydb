cmake_minimum_required(VERSION 2.8)

#SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

SET (pg_source_files
    PgConnector.cpp
    PgConnector.h
)

#find_package (PQXX REQUIRED)
#find_package (PostgreSQL REQUIRED)
include_directories("/usr/local/Cellar/libpqxx/4.0.1/include")
ADD_LIBRARY (PgConnector ${pg_source_files})

target_link_libraries (PgConnector pqxx)


