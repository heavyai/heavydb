add_library(Fragmenter InsertOrderFragmenter.cpp SortedOrderFragmenter.cpp UpdelStorage.cpp TargetValueConvertersFactories.cpp)

target_link_libraries(Fragmenter ${Boost_THREAD_LIBRARY})
