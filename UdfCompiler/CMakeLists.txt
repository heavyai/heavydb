set(udf_compiler_source_files
    UdfCompiler.cpp)

# Suppress multi-line comment warning which is not disabled by a diagnostic pragma
# for some reason.
set_source_files_properties(UdfCompiler.cpp PROPERTIES COMPILE_OPTIONS "-Wno-comment")

add_library(UdfCompiler ${udf_compiler_source_files})

# Clang
find_package(Clang REQUIRED)
include_directories(${CLANG_INCLUDE_DIRS})
add_definitions(${CLANG_DEFINITIONS})

target_link_libraries(UdfCompiler Logger ${clang_libs})

