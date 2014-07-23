cmake_minimum_required(VERSION 2.8)

SET (database_source_files
    Database.cpp
    Database.h
)

find_package(Boost 1.50.0 REQUIRED system thread)
set (database_include_dirs
    ${CMAKE_SOURCE_DIR}/Server/Output/
    ${CMAKE_SOURCE_DIR}/Server/TcpServer/
    ${CMAKE_SOURCE_DIR}/QueryEngine/Parse/SQL/
)
include_directories(${database_include_dirs})

ADD_LIBRARY (Database ${database_source_files})


TARGET_LINK_LIBRARIES (Database pthread ${Boost_THREAD_LIBRARY} )
