set(migration_mgr_source_files
    MigrationMgr.cpp
)

add_library(MigrationMgr ${migration_mgr_source_files})
if(WIN32)
  target_compile_definitions(MigrationMgr PRIVATE _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING)
endif(WIN32)

target_link_libraries(MigrationMgr Shared QueryEngine ${Boost_LIBRARIES})

