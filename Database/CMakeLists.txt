cmake_minimum_required(VERSION 2.8)

SET (database_source_files
    Database.cpp
    Database.h
)

set (database_include_dirs
    ${CMAKE_SOURCE_DIR}/Server/Output/
    ${CMAKE_SOURCE_DIR}/Server/TcpServer/
    ${CMAKE_SOURCE_DIR}/QueryEngine/Parse/SQL/
    ${CMAKE_BINARY_DIR}/QueryEngine/Parse/SQL/
)
#message("CMAKE_INCLUDE_PATH 2 = " ${CMAKE_INCLUDE_PATH})
find_package(Boost 1.50.0 REQUIRED system thread)
include_directories(${database_include_dirs} ${Boost_INCLUDE_DIR})

ADD_LIBRARY (Database ${database_source_files})

TARGET_LINK_LIBRARIES (Database Parser pthread ${Boost_THREAD_LIBRARY} )
