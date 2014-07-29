cmake_minimum_required(VERSION 2.8)

set( CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}" )

SET (file_source_files
    FileMgr.cpp
    FileMgr.h
    File.cpp
    File.h
)

ADD_LIBRARY(File ${file_source_files})

#PROJECT				(File)

#INCLUDE_DIRECTORIES	("../../Shared")

#ADD_EXECUTABLE		(fileTest fileTest.cpp File.cpp)
#ADD_EXECUTABLE		(fileMgrTest fileMgrTest.cpp)

#INSTALL				(TARGETS fileTest DESTINATION bin)
#INSTALL				(TARGETS fileMgrTest DESTINATION bin)
