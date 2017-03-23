set(catalog_source_files
    Catalog.cpp
    Catalog.h
)

add_library(Catalog ${catalog_source_files})

if(ENABLE_CALCITE)
  target_link_libraries(Catalog SqliteConnector Fragmenter Calcite)
else()
  target_link_libraries(Catalog SqliteConnector Fragmenter)
endif()
