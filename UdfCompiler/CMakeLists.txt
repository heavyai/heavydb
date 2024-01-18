set(udf_compiler_source_files
  UdfCompiler.cpp)

add_library(UdfCompiler ${udf_compiler_source_files})

# Clang
find_package(Clang REQUIRED)
include_directories(SYSTEM ${CLANG_INCLUDE_DIRS})
add_definitions(${CLANG_DEFINITIONS})

# g++ 13.2.1 when building in RelWithDebInfo mode emits an error which requires suppression:
# /usr/lib/llvm14/include/clang/AST/ExternalASTSource.h:378:54: error: ‘this’ pointer is null [-Werror=nonnull]
#   378 |       Ptr = reinterpret_cast<uint64_t>((Source->*Get)(Ptr >> 1));
#       |                                        ~~~~~~~~~~~~~~^~~~~~~~~~
if (ENABLE_WARNINGS_AS_ERRORS
    AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU"
    AND "${CMAKE_BUILD_TYPE_LOWER}" STREQUAL "relwithdebinfo")
  set_source_files_properties(UdfCompiler.cpp PROPERTIES COMPILE_FLAGS -Wno-error=nonnull)
endif()

target_link_libraries(UdfCompiler Logger ${clang_libs})
