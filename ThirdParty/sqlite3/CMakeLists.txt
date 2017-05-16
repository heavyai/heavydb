if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set_source_files_properties(sqlite3.c PROPERTIES COMPILE_FLAGS "-O0 -Wno-unused-value -Wno-parentheses-equality")
else()
  set_source_files_properties(sqlite3.c PROPERTIES COMPILE_FLAGS "-O0 -Wno-unused-value")
endif()

add_library(sqlite3 sqlite3.c sqlite3.h)
target_link_libraries(sqlite3 ${CMAKE_DL_LIBS})
