set(analyzer_source_files
    Analyzer.cpp
    Analyzer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../QueryEngine/DateTruncate.cpp
)

add_library(Analyzer ${analyzer_source_files})

target_link_libraries(Analyzer Catalog)
