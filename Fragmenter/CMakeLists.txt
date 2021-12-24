add_library(Fragmenter InsertOrderFragmenter.cpp SortedOrderFragmenter.cpp TargetValueConvertersFactories.cpp InsertDataLoader.cpp)

add_dependencies(Fragmenter Calcite)
target_link_libraries(Fragmenter ${Boost_THREAD_LIBRARY})
