set(udf_compiler_source_files
    UdfCompiler.cpp)

add_library(UdfCompiler ${udf_compiler_source_files})

# Clang
find_package(Clang REQUIRED)
include_directories(${CLANG_INCLUDE_DIRS})
add_definitions(${CLANG_DEFINITIONS})

target_link_libraries(UdfCompiler Logger ${clang_libs})

