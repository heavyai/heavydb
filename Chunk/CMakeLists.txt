set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-result")

add_library(Chunk Chunk.cpp)
target_link_libraries(Chunk DataMgr)
