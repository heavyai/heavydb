add_library(DBEngine SHARED DBEngine.cpp)
target_link_libraries(DBEngine tinfo ${LLVM_LINKER_FLAGS} QueryRunner DataMgr)

