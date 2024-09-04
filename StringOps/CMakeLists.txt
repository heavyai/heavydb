if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes")
endif()

set(string_ops_source_files
    StringOpInfo.cpp
    StringOpInfo.h
    StringOps.cpp
    StringOps.h
)

add_library(StringOps ${string_ops_source_files})

target_link_libraries(StringOps cpr)

if (NOT cpr_FOUND)
  target_include_directories(StringOps PRIVATE ${CMAKE_SOURCE_DIR}/ThirdParty/cpr/cpr/include)
endif()
