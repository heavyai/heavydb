set(schema_manager_source_files
    ColumnInfo.cpp
)

add_library(SchemaMgr ${schema_manager_source_files})

target_link_libraries(SchemaMgr Shared)