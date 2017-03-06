  add_library(thrift_handler MapDHandler.cpp MapDHandler.h)
  target_link_libraries(thrift_handler mapd_thrift Shared ${Glog_LIBRARIES} ${CMAKE_DL_LIBS})
