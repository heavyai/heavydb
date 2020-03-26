set(table_archive_source_files
    TableArchiver.cpp
)

add_library(TableArchiver ${table_archive_source_files})

target_link_libraries(TableArchiver Catalog Parser Shared)

