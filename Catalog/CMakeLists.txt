set(catalog_source_files
    Catalog.cpp
    Catalog.h
    DBObject.cpp
    Grantee.cpp
    Grantee.h
    SessionInfo.cpp
    SessionsStore.cpp
    SharedDictionaryValidator.cpp
    SysCatalog.cpp
    SysCatalog.h
    ForeignServer.h
    ForeignTable.cpp
    DdlCommandExecutor.cpp
    DdlCommandExecutor.h
    AlterColumnRecovery.cpp
    AlterColumnRecovery.h
    RefreshTimeCalculator.cpp
)

get_target_property(StringDictionary_BINARY_DIR StringDictionary BINARY_DIR)
include_directories(${StringDictionary_BINARY_DIR})

if("${MAPD_EDITION_LOWER}" STREQUAL "ee")
  list(APPEND catalog_source_files ee/UserMapping.cpp)
endif()

add_library(Catalog ${catalog_source_files})
add_dependencies(Catalog Calcite StringDictionary thrift_gen)

if(WIN32)
  message(STATUS "Defining BOOST_UUID_FORCE_AUTO_LINK for Catalog")
  target_compile_definitions(Catalog PRIVATE BOOST_UUID_FORCE_AUTO_LINK)
endif()

if("${MAPD_EDITION_LOWER}" STREQUAL "ee")
  add_subdirectory(ee)
endif()

target_link_libraries(Catalog SqliteConnector StringDictionary Fragmenter MigrationMgr ${AUTH_LIBRARIES} Calcite rg3_bcrypt)
if(ENABLE_KRB5)
  target_link_libraries(Catalog krb5_gss)
endif()
if((${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.1) OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  target_link_libraries(Catalog stdc++fs)
endif()
