add_custom_command(
    DEPENDS ${CMAKE_SOURCE_DIR}/java/thrift/calciteserver.thrift ${CMAKE_SOURCE_DIR}/completion_hints.thrift ${CMAKE_SOURCE_DIR}/QueryEngine/extension_functions.thrift
    OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/CalciteServer.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/calciteserver_types.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/completion_hints_types.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/extension_functions_types.cpp
    COMMAND ${Thrift_EXECUTABLE}
    ARGS -gen cpp -r -I ${CMAKE_SOURCE_DIR} -o ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_SOURCE_DIR}/java/thrift/calciteserver.thrift)

add_library(calciteserver_thrift
    ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/CalciteServer.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/calciteserver_types.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/completion_hints_types.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/extension_functions_types.cpp
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

get_target_property(StringDictionary_BINARY_DIR StringDictionary BINARY_DIR)
include_directories(${StringDictionary_BINARY_DIR}
${StringDictionary_BINARY_DIR}/gen-cpp)

target_link_libraries(calciteserver_thrift ${Thrift_LIBRARIES})

add_library(Calcite Calcite.cpp Calcite.h)

target_link_libraries(Calcite Catalog calciteserver_thrift ${JAVA_JVM_LIBRARY})
