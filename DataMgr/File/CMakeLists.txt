cmake_minimum_required(VERSION 2.8)

PROJECT				(File)

INCLUDE_DIRECTORIES	("../../Shared")

ADD_EXECUTABLE		(fileTest fileTest.cpp File.cpp)
#ADD_EXECUTABLE		(fileMgrTest fileMgrTest.cpp)

INSTALL				(TARGETS fileTest DESTINATION bin)
#INSTALL				(TARGETS fileMgrTest DESTINATION bin)
