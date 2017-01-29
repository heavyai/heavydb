
add_library(StringDictionary StringDictionary.cpp StringDictionaryProxy.cpp)
target_link_libraries(StringDictionary Utils ${Glog_LIBRARIES} ${Thrift_LIBRARIES})

