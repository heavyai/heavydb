set(catalog_source_files
    Catalog.cpp
    Catalog.h
    CatalogSchemaProvider.cpp
    CatalogSchemaProvider.h
    DBObject.cpp
    Grantee.cpp
    Grantee.h
    SessionInfo.cpp
    SharedDictionaryValidator.cpp
    SysCatalog.cpp
    SysCatalog.h
    ForeignServer.h
    ForeignTable.cpp
    DdlCommandExecutor.cpp
    DdlCommandExecutor.h
)

get_target_property(StringDictionary_BINARY_DIR StringDictionary BINARY_DIR)
include_directories(${StringDictionary_BINARY_DIR})

add_library(Catalog ${catalog_source_files})

if("${MAPD_EDITION_LOWER}" STREQUAL "ee")
  add_subdirectory(ee)
endif()

target_link_libraries(Catalog SqliteConnector StringDictionary Fragmenter MigrationMgr ${AUTH_LIBRARIES} Calcite bcrypt)
if(ENABLE_KRB5)
  target_link_libraries(Catalog krb5_gss)
endif()
if((${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.1) OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  target_link_libraries(Catalog stdc++fs)
endif()
