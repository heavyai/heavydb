cmake_minimum_required(VERSION 2.8)

PROJECT(TcpServer)
ADD_EXECUTABLE(serv main.cpp)
ADD_LIBRARY(tcpServer TcpServer.cpp)

#SET(CMAKE_C_COMPILER mpicc)
#SET(CMAKE_CXX_COMPILER mpicxx)

find_package(Boost 1.54.0 REQUIRED system)
include_directories(BOOST_INCLUDE_PATH)

target_link_libraries(serv tcpServer ${Boost_SYSTEM_LIBRARY})

