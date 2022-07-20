set(ir_source_files
    Expr.cpp
    Expr.h
)

add_library(IR ${ir_source_files})

target_link_libraries(Analyzer Shared)
