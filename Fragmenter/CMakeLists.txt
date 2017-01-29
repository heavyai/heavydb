add_library(Fragmenter InsertOrderFragmenter.cpp)

target_link_libraries(Fragmenter ${Boost_THREAD_LIBRARY})
