cmake_minimum_required(VERSION 2.8)

PROJECT			(mapd_exe)

set( CMAKE_CXX_FLAGS "-g -std=c++0x ${CMAKE_CXX_FLAGS}" )
FIND_PACKAGE( Boost COMPONENTS filesystem-mt program_options system-mt REQUIRED )
#set( Boost_SYSTEM_LIBRARY "-lboost_filesystem-mt -lboost_system-mt")
include_directories(${Boost_INCLUDE_DIR} "/usr/local/include" ${CMAKE_CURRENT_SOURCE_DIR}/Parser ${CMAKE_CURRENT_BINARY_DIR})
ADD_SUBDIRECTORY    (SqliteConnector)
ADD_SUBDIRECTORY    (Catalog/)
ADD_SUBDIRECTORY    (Parser)
ADD_SUBDIRECTORY    (Analyzer)
ADD_SUBDIRECTORY    (Planner)

set(mapd_source_files main.cpp)
add_executable(mapd ${mapd_source_files})

set(initdb_source_files initdb.cpp)
add_executable(initdb ${initdb_source_files})

MESSAGE( STATUS "Boost_LIBRARIES: " ${Boost_LIBRARIES} )
target_link_libraries (mapd Catalog SqliteConnector Parser Analyzer Planner ${Boost_LIBRARIES} )
target_link_libraries (initdb Catalog SqliteConnector ${Boost_LIBRARIES} )
