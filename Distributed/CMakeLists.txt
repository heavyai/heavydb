if("${MAPD_EDITION_LOWER}" STREQUAL "ee" AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ee")
  add_library(Distributed ee/LeafHostInfo.cpp ee/DistributedLoader.cpp ee/LeafAggregator.cpp)
endif()
