set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

if(NOT MSVC)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-write-strings -Wno-unused-function -Wno-unused-label -Wno-sign-compare")

endif()

set(parser_source_files
    ParserNode.cpp
    ParserNode.h
    ParserWrapper.cpp
    ParserWrapper.h
)

add_library(Parser ${parser_source_files})

if("${MAPD_EDITION_LOWER}" STREQUAL "ee")
    include_directories(Distributed/ee)
endif()

if("${MAPD_EDITION_LOWER}" STREQUAL "ee")
    target_link_libraries(Parser QueryState ImportExport Shared Analyzer QueryEngine Catalog TableArchiver DataMgr LockMgr Distributed)
else()
    target_link_libraries(Parser QueryState ImportExport Shared Analyzer QueryEngine Catalog TableArchiver DataMgr LockMgr)
endif()
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
