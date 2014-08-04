cmake_minimum_required(VERSION 2.8)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

SET (output_source_files
    PgConnector.cpp
    PgConnector.h
)
#find_package (PQXX REQUIRED)
#find_package (PostgreSQL REQUIRED)
include_directories(${CMAKE_INCLUDE_PATH})
ADD_LIBRARY (PgConnector ${output_source_files})

target_link_libraries (PgConnector pqxx)


