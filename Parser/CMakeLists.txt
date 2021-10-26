set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

if(NOT MSVC)

find_package(Bisonpp REQUIRED)
find_package(Flexpp REQUIRED)

set(BisonppOutput ${CMAKE_CURRENT_BINARY_DIR}/Parser_with_register.cpp)

add_custom_command(
    OUTPUT
      ${BisonppOutput}
      ${CMAKE_CURRENT_BINARY_DIR}/parser.h
    COMMAND ${Bisonpp_EXECUTABLE}
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
    COMMAND ${Flexpp_EXECUTABLE}
        -d
        -i
        -o${FlexppOutput}
        ${CMAKE_CURRENT_SOURCE_DIR}/scanner.l
    DEPENDS scanner.l
    COMMENT "Generating Scanner.cpp"
)

endif()

# C++17 removes the 'register' keyword and it is no longer allowed to
# suppress the warning, so patch the generated scanner and parser files
# to remove the register keyword, and also patch embedded filename strings
# so they match the final file
set(ParserPatched ${CMAKE_CURRENT_BINARY_DIR}/Parser.cpp)
set(ScannerPatched ${CMAKE_CURRENT_BINARY_DIR}/Scanner.cpp)

if(MSVC)

file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/parser_wnd_pregen.h ${CMAKE_CURRENT_BINARY_DIR}/parser.h SYMBOLIC COPY_ON_ERROR)
file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/Parser_wnd_pregen.cpp ${CMAKE_CURRENT_BINARY_DIR}/Parser.cpp SYMBOLIC COPY_ON_ERROR)
file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/Scanner_wnd_pregen.cpp ${CMAKE_CURRENT_BINARY_DIR}/Scanner.cpp SYMBOLIC COPY_ON_ERROR)
file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/FlexLexer_wnd.h ${CMAKE_CURRENT_BINARY_DIR}/FlexLexer.h SYMBOLIC COPY_ON_ERROR)

else()

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
      -e "s/delete yy_state_buf\;/delete[] yy_state_buf\;/"
      ${FlexppOutput} > ${ScannerPatched}
    DEPENDS ${FlexppOutput}
    COMMENT "Patching Scanner.cpp to remove register keyword"
)

add_custom_target(ParserFiles DEPENDS ${BisonppOutput})
add_custom_target(ScannerFiles DEPENDS ${FlexppOutput})

add_custom_target(PatchParser DEPENDS ${ParserPatched})
add_custom_target(PatchScanner DEPENDS ${ScannerPatched})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-write-strings -Wno-unused-function -Wno-unused-label -Wno-sign-compare")

endif()

set(parser_source_files
    ParserNode.cpp
    ParserNode.h
    ParserWrapper.cpp
    ParserWrapper.h
)

add_library(ParserGenerated ${ParserPatched} ${ScannerPatched})
add_library(Parser ${CMAKE_CURRENT_BINARY_DIR}/parser.h ${parser_source_files})

if("${MAPD_EDITION_LOWER}" STREQUAL "ee")
    include_directories(Distributed/ee)
endif()

if(NOT MSVC)
add_dependencies(PatchParser ParserFiles)
add_dependencies(PatchScanner ScannerFiles)

add_dependencies(Parser PatchParser PatchScanner)
endif()

if("${MAPD_EDITION_LOWER}" STREQUAL "ee")
    target_link_libraries(Parser ParserGenerated QueryState ImportExport Shared Analyzer QueryEngine Catalog TableArchiver DataMgr LockMgr Distributed)
else()
    target_link_libraries(Parser ParserGenerated QueryState ImportExport Shared Analyzer QueryEngine Catalog TableArchiver DataMgr LockMgr)
endif()
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
