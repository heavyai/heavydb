cmake_minimum_required(VERSION 2.8)

PROJECT(Server)

ADD_LIBRARY(requestHandler RequestHandler.cpp)
ADD_LIBRARY(tcpConnection TcpConnection.cpp)
ADD_LIBRARY(tcpServer TcpServer.cpp)

#SET(CMAKE_C_COMPILER mpicc)
#SET(CMAKE_CXX_COMPILER mpicxx)

find_package(Boost 1.54.0 REQUIRED system thread)
include_directories(BOOST_INCLUDE_PATH)

target_link_libraries(tcpServer ${Boost_THREAD_LIBRARY})

ADD_EXECUTABLE(serverTest ServerTest.cpp)
target_link_libraries(serverTest pthread ${Boost_SYSTEM_LIBRARY} tcpServer tcpConnection requestHandler ) 

