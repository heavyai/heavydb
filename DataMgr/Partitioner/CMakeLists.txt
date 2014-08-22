cmake_minimum_required(VERSION 2.8)

#SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

set( CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}" )

SET (partitioner_source_files
    Partitioner.h
    LinearTablePartitioner.cpp
    LinearTablePartitioner.h
    AbstractTablePartitioner.h
    TablePartitionMgr.h
    TablePartitionMgr.cpp
)

set (partitioner_include_dirs
    ${CMAKE_CURRENT_SOURCE_DIR}/../Buffer
    ${CMAKE_CURRENT_SOURCE_DIR}/../Metadata
    ${CMAKE_CURRENT_SOURCE_DIR}/../PgConnector
    "/usr/local/include"
)

include_directories(${partitioner_include_dirs})

ADD_LIBRARY (Partitioner ${partitioner_source_files})

TARGET_LINK_LIBRARIES (Partitioner PgConnector)

if (BUILD_TEST)
     
    message( "Including PgConnector" )
    ADD_SUBDIRECTORY (${CMAKE_CURRENT_SOURCE_DIR}/../PgConnector ${CMAKE_CURRENT_BINARY_DIR}/PgConnector)
    message( "Including Buffer" )
    ADD_SUBDIRECTORY (${CMAKE_CURRENT_SOURCE_DIR}/../Buffer ${CMAKE_CURRENT_BINARY_DIR}/Buffer)
    message( "Including Catalog" )
    ADD_SUBDIRECTORY (${CMAKE_CURRENT_SOURCE_DIR}/../Metadata ${CMAKE_CURRENT_BINARY_DIR}/Metadata)
    #ADD_EXECUTABLE (PartitionTest LinearTablePartitionerTest.cpp)
    #TARGET_LINK_LIBRARIES(PartitionTest TablePartitioner PgConnector Buffer) 
    ADD_EXECUTABLE (TablePartitionerTest TablePartitionerTest.cpp)
    TARGET_LINK_LIBRARIES(TablePartitionerTest TablePartitioner Catalog PgConnector Buffer) 
endif(BUILD_TEST)
