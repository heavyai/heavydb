set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

find_package(Bisonpp REQUIRED)
find_package(Flexpp REQUIRED)

set(BisonppOutput ${CMAKE_CURRENT_BINARY_DIR}/Parser_with_register.cpp)

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

set(FlexppOutput ${CMAKE_CURRENT_BINARY_DIR}/Scanner_with_register.cpp)

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

# C++17 removes the 'register' keyword and it is no longer allowed to 
# suppress the warning, so patch the generated scanner and parser files
# to remove the register keyword, and also patch embedded filename strings
# so they match the final file
set(ParserPatched ${CMAKE_CURRENT_BINARY_DIR}/Parser.cpp)
set(ScannerPatched ${CMAKE_CURRENT_BINARY_DIR}/Scanner.cpp)

add_custom_command(
    OUTPUT ${ParserPatched}
    COMMAND sed 
        -e "s/register //g" 
        -e "s/Parser_with_register/Parser/g" 
        ${BisonppOutput} > ${ParserPatched}
    DEPENDS ${BisonppOutput}
    COMMENT "Patching Parser.cpp to remove register keyword"
)

add_custom_command(
    OUTPUT ${ScannerPatched}
    COMMAND sed 
      -e "s/register //g" 
      -e "s/Scanner_with_register/Scanner/g" 
      ${FlexppOutput} > ${ScannerPatched}
    DEPENDS ${FlexppOutput}
    COMMENT "Patching Scanner.cpp to remove register keyword"
)

add_custom_target(ParserFiles DEPENDS ${BisonppOutput})
add_custom_target(ScannerFiles DEPENDS ${FlexppOutput})	

add_custom_target(PatchParser DEPENDS ${ParserPatched})
add_custom_target(PatchScanner DEPENDS ${ScannerPatched})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-write-strings -Wno-unused-function -Wno-unused-label -Wno-sign-compare")

set(parser_source_files
    ParserNode.cpp
    ParserNode.h
    ParserWrapper.cpp
    ParserWrapper.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../QueryEngine/ExtractFromTime.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../QueryEngine/DateTruncate.cpp
)

add_library(ParserGenerated ${ParserPatched} ${ScannerPatched})
add_library(Parser ${CMAKE_CURRENT_BINARY_DIR}/parser.h ${parser_source_files})

add_dependencies(PatchParser ParserFiles)
add_dependencies(PatchScanner ScannerFiles)

add_dependencies(Parser PatchParser PatchScanner)

target_link_libraries(Parser ParserGenerated Shared QueryEngine Catalog DataMgr LockMgr)
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
