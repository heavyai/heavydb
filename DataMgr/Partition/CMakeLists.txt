cmake_minimum_required(VERSION 2.8)

#SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

set( CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}" )

SET (partitioner_source_files
    LinearTablePartitioner.cpp
    LinearTablePartitioner.h
)

set (partitioner_include_dirs
    ${CMAKE_CURRENT_SOURCE_DIR}/../Buffer
)

include_directories(${partitioner_include_dirs})

ADD_LIBRARY (LinearTablePartitioner ${partitioner_source_files})

if (BUILD_TEST)
    message( "Including PgConnector" )
    ADD_SUBDIRECTORY (${CMAKE_CURRENT_SOURCE_DIR}/../PgConnector ${CMAKE_CURRENT_BINARY_DIR}/PgConnector)
    message( "Including Buffer" )
    ADD_SUBDIRECTORY (${CMAKE_CURRENT_SOURCE_DIR}/../Buffer ${CMAKE_CURRENT_BINARY_DIR}/Buffer)
    ADD_EXECUTABLE (PartitionTest LinearTablePartitionerTest.cpp)
    TARGET_LINK_LIBRARIES(PartitionTest LinearTablePartitioner PgConnector Buffer) 
endif(BUILD_TEST)
