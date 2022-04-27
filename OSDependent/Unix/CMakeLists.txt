if(BUILD_SHARED_LIBS)
  add_definitions("-DOSDEPENDENT_SHARED_OBJECT_NAME=\"${CMAKE_SHARED_LIBRARY_PREFIX}OSDependent${CMAKE_SHARED_LIBRARY_SUFFIX}\"")
endif()

add_library(OSDependent ${OSDEPENDENT_SOURCE_FILES})

target_link_libraries(OSDependent Logger)
