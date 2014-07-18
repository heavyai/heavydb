cmake_minimum_required(VERSION 2.8)

PROJECT			(MapD_exe)

ADD_SUBDIRECTORY    (QueryEngine/Parse/SQL)
ADD_SUBDIRECTORY	(Server/TcpServer)
ADD_SUBDIRECTORY	(Database)

set(mapd_source_files main.cpp)
add_executable(MapD ${mapd_source_files})

target_link_libraries (MapD Database TcpServer ${Boost_SYSTEM_LIBRARY} )
