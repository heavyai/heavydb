cmake_minimum_required(VERSION 2.8)

set( CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}" )

SET (buffer_source_files
    BufferMgr.cpp
    BufferMgr.h
    Buffer.cpp
    Buffer.h
)

ADD_LIBRARY(Buffer ${buffer_source_files})

if (BUILD_TEST)
    message( "Buffer BUILD_TEST" )
    ADD_SUBDIRECTORY (${CMAKE_CURRENT_SOURCE_DIR}/../File ${CMAKE_BINARY_DIR}/File)
    Target_Link_Libraries (Buffer File) 
endif(BUILD_TEST)








