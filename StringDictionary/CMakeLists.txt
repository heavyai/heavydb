add_library(StringDictionary StringDictionary.cpp StringDictionaryProxy.cpp)

if(ENABLE_FOLLY)
  target_link_libraries(StringDictionary OSDependent Utils ${Boost_LIBRARIES} ${PROFILER_LIBS} ${Folly_LIBRARIES} ${TBB_LIBS})
else()
  target_link_libraries(StringDictionary OSDependent Utils ${Boost_LIBRARIES} ${PROFILER_LIBS} ${TBB_LIBS})
endif()
