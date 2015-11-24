cmake_minimum_required(VERSION 2.8)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})
FIND_PACKAGE( Boost COMPONENTS system program_options regex REQUIRED )
ADD_EXECUTABLE (StreamInsertSimple StreamInsertSimple.cpp ${CMAKE_BINARY_DIR}/gen-cpp/MapD.cpp ${CMAKE_BINARY_DIR}/gen-cpp/mapd_constants.cpp ${CMAKE_BINARY_DIR}/gen-cpp/mapd_types.cpp)
ADD_EXECUTABLE (DataGen DataGen.cpp ${CMAKE_BINARY_DIR}/gen-cpp/MapD.cpp ${CMAKE_BINARY_DIR}/gen-cpp/mapd_constants.cpp ${CMAKE_BINARY_DIR}/gen-cpp/mapd_types.cpp)
ADD_EXECUTABLE (StreamInsert StreamInsert.cpp ${CMAKE_BINARY_DIR}/gen-cpp/MapD.cpp ${CMAKE_BINARY_DIR}/gen-cpp/mapd_constants.cpp ${CMAKE_BINARY_DIR}/gen-cpp/mapd_types.cpp)

set_source_files_properties(${mapd_thrift_gen_files} GENERATED)

target_link_libraries (StreamInsertSimple ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} ${Thrift_LIBRARIES})
target_link_libraries (StreamInsert ${CMAKE_DL_LIBS} ${Boost_LIBRARIES} ${Thrift_LIBRARIES})
target_link_libraries (DataGen ${CMAKE_DL_LIBS} ${Thrift_LIBRARIES})

install(TARGETS StreamInsertSimple StreamInsert DataGen DESTINATION SampleCode)
install(DIRECTORY . DESTINATION SampleCode)
