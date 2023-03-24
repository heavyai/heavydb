if("${MAPD_EDITION_LOWER}" STREQUAL "ee")
  list(APPEND OSDEPENDENT_SOURCE_FILES ../ee/heavyai_locks.cpp)
endif()
add_library(OSDependent ${OSDEPENDENT_SOURCE_FILES})
