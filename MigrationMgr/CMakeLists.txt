set(migration_mgr_source_files
    MigrationMgr.cpp
)

add_library(MigrationMgr ${migration_mgr_source_files})

target_link_libraries(MigrationMgr Shared QueryEngine ${Boost_LIBRARIES})

