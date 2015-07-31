cmake_minimum_required(VERSION 2.8)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})
#set( CMAKE_CXX_FLAGS "-g -std=c++11 -Wall -O3 -pthread ${CMAKE_CXX_FLAGS} -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -L/usr/local/cuda/nvvm/lib -L/usr/local/cuda/nvvm/lib64")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-strict-aliasing")
#FIND_PACKAGE( Boost COMPONENTS filesystem program_options thread system REQUIRED )
set(llvm_config_cmd "llvm-config")
set(llvm_config_inc_arg "--includedir")
set(llvm_config_ld_arg "--ldflags")
set(llvm_config_lib_arg "--libs")
execute_process(COMMAND ${llvm_config_cmd} ${llvm_config_inc_arg}
    OUTPUT_VARIABLE LLVM_INC_FLAGS)
execute_process(COMMAND ${llvm_config_cmd} ${llvm_config_ld_arg} ${llvm_config_lib_arg}
    OUTPUT_VARIABLE LLVM_LIB_FLAGS)
string(REPLACE "\n" "" LLVM_LINKER_FLAGS "${LLVM_LIB_FLAGS} ${LLVM_LD_FLAGS}")
include_directories(${Boost_INCLUDE_DIR} "/usr/local/include" ${CMAKE_CURRENT_SOURCE_DIR}/Parser ${CMAKE_CURRENT_BINARY_DIR} ${LLVM_INC_FLAGS} "/usr/local/cuda/nvvm/include/" "/usr/local/cuda/include/")

add_executable(ExecuteTest ExecuteTest.cpp QueryRunner.cpp)
add_executable(RunQueryLoop RunQueryLoop.cpp QueryRunner.cpp)
add_executable(StringDictionaryTest StringDictionaryTest.cpp)
ADD_EXECUTABLE (PlanTest PlanTest.cpp)
ADD_EXECUTABLE (UtilTest UtilTest.cpp)
ADD_EXECUTABLE (StorageTest StorageTest.cpp PopulateTableRandom.cpp ScanTable.cpp)
ADD_EXECUTABLE (StoragePerfTest StoragePerfTest.cpp PopulateTableRandom.cpp ScanTable.cpp)

target_link_libraries (PlanTest gtest ${MAPD_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} "${LLVM_LINKER_FLAGS} -lnvvm -lcuda -lcudart -lglog -lz -lncurses -ldl -pthread")
target_link_libraries (UtilTest Utils gtest)
target_link_libraries (StorageTest gtest ${MAPD_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} "${LLVM_LINKER_FLAGS} -lnvvm -lcuda -lcudart -lglog -lz -lncurses -ldl -pthread")
target_link_libraries (StoragePerfTest gtest ${MAPD_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} "${LLVM_LINKER_FLAGS} -lnvvm -lcuda -lcudart -lglog -lz -lncurses -ldl -pthread")
target_link_libraries (ExecuteTest gtest ${MAPD_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} "${LLVM_LINKER_FLAGS} -lnvvm -lcuda -lcudart -lglog -lz -lncurses -ldl -pthread")
target_link_libraries (RunQueryLoop gtest ${MAPD_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} "${LLVM_LINKER_FLAGS} -lnvvm -lcuda -lcudart -lglog -lz -lncurses -ldl -pthread")
target_link_libraries (StringDictionaryTest StringDictionary gtest ${Boost_LIBRARIES})

install(TARGETS ExecuteTest RunQueryLoop StringDictionaryTest PlanTest
    UtilTest StorageTest StoragePerfTest DESTINATION Tests)
