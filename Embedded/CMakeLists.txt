add_library(DBEngine SHARED DBEngine.cpp DBEngine.h)
target_link_libraries(DBEngine tinfo ${LLVM_LINKER_FLAGS} QueryRunner DataMgr ${Arrow_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} ${TBB_LIBS})
