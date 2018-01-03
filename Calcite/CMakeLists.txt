add_custom_command(
    DEPENDS ${CMAKE_SOURCE_DIR}/java/thrift/calciteserver.thrift ${CMAKE_SOURCE_DIR}/completion_hints.thrift
    OUTPUT
        ${CMAKE_BINARY_DIR}/gen-cpp/CalciteServer.cpp
        ${CMAKE_BINARY_DIR}/gen-cpp/calciteserver_constants.cpp
        ${CMAKE_BINARY_DIR}/gen-cpp/calciteserver_types.cpp
        ${CMAKE_BINARY_DIR}/gen-cpp/completion_hints_constants.cpp
        ${CMAKE_BINARY_DIR}/gen-cpp/completion_hints_types.cpp
    COMMAND ${Thrift_EXECUTABLE}
    ARGS -gen cpp -r -I ${CMAKE_SOURCE_DIR} -o ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/java/thrift/calciteserver.thrift)

add_library(calciteserver_thrift
    ${CMAKE_BINARY_DIR}/gen-cpp/CalciteServer.cpp
    ${CMAKE_BINARY_DIR}/gen-cpp/calciteserver_constants.cpp
    ${CMAKE_BINARY_DIR}/gen-cpp/calciteserver_types.cpp
    ${CMAKE_BINARY_DIR}/gen-cpp/completion_hints_constants.cpp
    ${CMAKE_BINARY_DIR}/gen-cpp/completion_hints_types.cpp
)

get_target_property(StringDictionary_BINARY_DIR StringDictionary BINARY_DIR)
include_directories(${StringDictionary_BINARY_DIR}
${StringDictionary_BINARY_DIR}/gen-cpp)

target_link_libraries(calciteserver_thrift ${Thrift_LIBRARIES})

add_library(Calcite Calcite.cpp Calcite.h ${CMAKE_SOURCE_DIR}/Shared/ConfigResolve.h )

target_link_libraries(Calcite Catalog calciteserver_thrift ${JAVA_JVM_LIBRARY})
