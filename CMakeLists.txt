cmake_minimum_required(VERSION 2.8)

PROJECT			(MapD_exe)

set( CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}" )
include_directories(${Boost_INCLUDE_DIR} "/usr/local/include")
ADD_SUBDIRECTORY    (DataMgr/PgConnector)
ADD_SUBDIRECTORY    (DataMgr/Partitioner)
ADD_SUBDIRECTORY    (DataMgr/Metadata/)
ADD_SUBDIRECTORY    (DataMgr/Memory/)
ADD_SUBDIRECTORY    (QueryEngine/Parse/SQL)
ADD_SUBDIRECTORY	(Server)
ADD_SUBDIRECTORY	(Database)

set(mapd_source_files main.cpp)
add_executable(MapD ${mapd_source_files})

target_link_libraries (MapD Database Catalog Partitioner TcpServer Output  ${Boost_SYSTEM_LIBRARY} )

# add a target to generate API documentation with Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
add_custom_target(doc
${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
COMMENT "Generating API documentation with Doxygen" VERBATIM
)
endif(DOXYGEN_FOUND)

