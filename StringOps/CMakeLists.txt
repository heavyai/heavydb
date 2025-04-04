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

if (CPR_FOUND)
  # link new cpr from deps
  target_link_libraries(StringOps ${CPR_LIBRARY})
else()
  # link old cpr_1.2 from ThirdParty sub-project
  target_link_libraries(StringOps cpr)
  target_include_directories(StringOps PRIVATE ${CMAKE_SOURCE_DIR}/ThirdParty/cpr_1.2/cpr/include)
endif()
