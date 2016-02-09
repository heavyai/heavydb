cmake_minimum_required(VERSION 2.8)
#set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wno-attributes -O3")

add_library(StringDictionary StringDictionary.cpp)
target_link_libraries(StringDictionary Utils ${Glog_LIBRARIES})
