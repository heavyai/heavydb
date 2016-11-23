cmake_minimum_required(VERSION 2.8)

SET (calcite_source_files
    Calcite.cpp
    Calcite.h
    ${CMAKE_BINARY_DIR}/gen-cpp/CalciteServer.cpp
    ${CMAKE_BINARY_DIR}/gen-cpp/calciteserver_constants.cpp
    ${CMAKE_BINARY_DIR}/gen-cpp/calciteserver_types.cpp
)

find_package(JNI REQUIRED)
include_directories(${JNI_INCLUDE_DIRS})

include_directories("/usr/local/include")

ADD_LIBRARY (Calcite ${calcite_source_files})

TARGET_LINK_LIBRARIES (Calcite Catalog ${Thrift_LIBRARIES} ${JAVA_JVM_LIBRARY})

add_custom_command(
    DEPENDS ${CMAKE_SOURCE_DIR}/java/thrift/calciteserver.thrift
    OUTPUT ${CMAKE_BINARY_DIR}/gen-cpp/CalciteServer.cpp ${CMAKE_BINARY_DIR}/gen-cpp/calciteserver_constants.cpp ${CMAKE_BINARY_DIR}/gen-cpp/calciteserver_types.cpp
    COMMAND ${Thrift_EXECUTABLE}
    ARGS -gen cpp -o ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/java/thrift/calciteserver.thrift)
