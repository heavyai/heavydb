if(ENABLE_L0)
  add_library(L0Mgr L0Mgr.cpp L0Exception.cpp)
  target_link_libraries(L0Mgr ${LevelZero_LIBRARY})
endif()