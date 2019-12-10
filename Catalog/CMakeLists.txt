set(catalog_source_files
    Catalog.cpp
    Catalog.h
    DBObject.cpp
    DumpRestore.cpp
    Grantee.cpp
    Grantee.h
    SessionInfo.cpp
    SharedDictionaryValidator.cpp
    SysCatalog.cpp
    SysCatalog.h
)

get_target_property(StringDictionary_BINARY_DIR StringDictionary BINARY_DIR)
include_directories(${StringDictionary_BINARY_DIR})

add_library(Catalog ${catalog_source_files})

if("${MAPD_EDITION_LOWER}" STREQUAL "ee")
  add_subdirectory(ee)
endif()

target_link_libraries(Catalog SqliteConnector StringDictionary Fragmenter QueryEngine ${AUTH_LIBRARIES} Calcite bcrypt)
if(ENABLE_KRB5)
  target_link_libraries(Catalog krb5_gss)
endif()
