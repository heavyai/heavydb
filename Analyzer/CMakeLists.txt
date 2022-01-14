if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes")
endif()

set(analyzer_source_files
    Analyzer.cpp
    Analyzer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../QueryEngine/DateTruncate.cpp
    RangeTableEntry.cpp
)

add_library(Analyzer ${analyzer_source_files})

target_link_libraries(Analyzer Catalog Geospatial SchemaMgr)
