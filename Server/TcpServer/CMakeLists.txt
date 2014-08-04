cmake_minimum_required(VERSION 2.8)

SET (tcpserver_source_files
    TcpServer.cpp
    TcpServer.h
    TcpConnection.cpp
    TcpConnection.h
)

find_package(Boost 1.50.0 COMPONENTS system thread)
include_directories(${Boost_INCLUDE_DIR})

ADD_LIBRARY (TcpServer ${tcpserver_source_files})

target_link_libraries(TcpServer ${Boost_THREAD_LIBRARY} ${Boost_SYSTEM_LIBRARY} )


