# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2014-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

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
    RWLocks.cpp
)

get_target_property(StringDictionary_BINARY_DIR StringDictionary BINARY_DIR)
include_directories(${StringDictionary_BINARY_DIR})

list(APPEND catalog_source_files UserMapping.cpp)

add_library(Catalog ${catalog_source_files})
add_dependencies(Catalog Calcite StringDictionary thrift_gen)

set(AUTH_LIBRARIES)

if(ENABLE_LDAP)
  add_library(LdapServer LdapServer.cpp LdapServer.h)
  find_library(LDAP_LIBRARY ldap REQUIRED)
  find_library(LBER_LIBRARY lber REQUIRED)
  target_link_libraries(LdapServer ${LDAP_LIBRARY} ${LBER_LIBRARY})
  list(APPEND AUTH_LIBRARIES LdapServer)
endif()

if(ENABLE_SAML)
  add_library(SamlServer SamlServer.cpp SamlServer.h)
  target_link_libraries(SamlServer ${OpenSaml_LIBRARIES})
  list(APPEND AUTH_LIBRARIES SamlServer)
  include_directories(SYSTEM ${OpenSaml_INCLUDE_DIRS})
endif()

install(DIRECTORY ${CMAKE_SOURCE_DIR}/Catalog/SystemDashboards/DashboardContent
        DESTINATION "Catalog/SystemDashboards" COMPONENT "data")
add_custom_target(system_dashboard_data ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_SOURCE_DIR}/Catalog/SystemDashboards/DashboardContent"
    "${CMAKE_BINARY_DIR}/Catalog/SystemDashboards/DashboardContent")
add_dependencies(Catalog system_dashboard_data)

target_link_libraries(Catalog SqliteConnector StringDictionary Fragmenter MigrationMgr ${AUTH_LIBRARIES} Calcite rg3_bcrypt)
if((${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.1) OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  target_link_libraries(Catalog stdc++fs)
endif()
