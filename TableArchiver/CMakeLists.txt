
if(MSVC)
    set(table_archive_source_files
        TableArchiverWindows.cpp
    )

    add_library(TableArchiver ${table_archive_source_files})
else()
    set(table_archive_source_files
        TableArchiver.cpp
    )

    add_library(TableArchiver ${table_archive_source_files})
endif()

target_link_libraries(TableArchiver Catalog Parser Shared)

