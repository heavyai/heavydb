set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

find_package(Bisonpp REQUIRED)
find_package(Flexpp REQUIRED)

set(BisonppOutput ${CMAKE_CURRENT_BINARY_DIR}/Parser.cpp)

add_custom_command(
    OUTPUT
      ${BisonppOutput}
      ${CMAKE_CURRENT_BINARY_DIR}/parser.h
    COMMAND ${BISONPP_EXECUTABLE}
    -d
    -h${CMAKE_CURRENT_BINARY_DIR}/parser.h
        -o${BisonppOutput}
        ${CMAKE_CURRENT_SOURCE_DIR}/parser.y
    DEPENDS parser.y
    COMMENT "Generating Parser.cpp"
)

set(FlexppOutput ${CMAKE_CURRENT_BINARY_DIR}/Scanner.cpp)

set(FlexppOutput Scanner.cpp)
add_custom_command(
    OUTPUT ${FlexppOutput}
    COMMAND ${FLEXPP_EXECUTABLE}
        -d
        -i
        -o${FlexppOutput}
        ${CMAKE_CURRENT_SOURCE_DIR}/scanner.l
    DEPENDS scanner.l
    COMMENT "Generating Scanner.cpp"
)

add_custom_target(ParserFiles DEPENDS ${BisonppOutput})
add_custom_target(ScannerFiles DEPENDS ${FlexppOutput})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-register -Wno-write-strings -Wno-unused-function -Wno-unused-label -Wno-sign-compare")

set(parser_source_files
    ParserNode.cpp
    ParserNode.h
    ParserWrapper.cpp
    ParserWrapper.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../QueryEngine/ExtractFromTime.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../QueryEngine/DateTruncate.cpp
)

add_library(Parser ${BisonppOutput} ${FlexppOutput} ${CMAKE_CURRENT_BINARY_DIR}/parser.h ${parser_source_files})
add_dependencies(Parser ParserFiles ScannerFiles)
target_link_libraries(Parser Shared CsvImport QueryEngine)
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
