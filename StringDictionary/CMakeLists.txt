add_library(StringDictionary StringDictionary.cpp StringDictionaryProxy.cpp)

target_link_libraries(StringDictionary StringOps OSDependent UtilsStandalone ${Boost_LIBRARIES} ${Thrift_LIBRARIES} ${PROFILER_LIBS} ThriftClient ${TBB_LIBS})
