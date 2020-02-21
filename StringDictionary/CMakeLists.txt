add_library(StringDictionary StringDictionary.cpp StringDictionaryProxy.cpp)

if(ENABLE_FOLLY)
  target_link_libraries(StringDictionary Utils ${Boost_LIBRARIES} ${Thrift_LIBRARIES} ${PROFILER_LIBS} ThriftClient ${Folly_LIBRARIES} TBB::tbb)
else()
  target_link_libraries(StringDictionary Utils ${Boost_LIBRARIES} ${Thrift_LIBRARIES} ${PROFILER_LIBS} ThriftClient TBB::tbb)
endif()
