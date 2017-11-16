add_library(StringDictionary StringDictionary.cpp StringDictionaryProxy.cpp)

if(ENABLE_FOLLY)
  target_link_libraries(StringDictionary Utils ${Glog_LIBRARIES} ${Thrift_LIBRARIES} ${Folly_LIBRARIES})
else()
  target_link_libraries(StringDictionary Utils ${Glog_LIBRARIES} ${Thrift_LIBRARIES})
endif()
