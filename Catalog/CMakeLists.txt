set(catalog_source_files
    Catalog.cpp
    Catalog.h
    DBObject.cpp
    Role.cpp
    SharedDictionaryValidator.cpp
)

get_target_property(StringDictionary_BINARY_DIR StringDictionary BINARY_DIR)
include_directories(${StringDictionary_BINARY_DIR})

add_library(Catalog ${catalog_source_files})

if("${MAPD_EDITION_LOWER}" STREQUAL "ee")
  find_library(LDAP_LIBRARIES NAMES ldap)
endif()

target_link_libraries(Catalog SqliteConnector StringDictionary Fragmenter ${LDAP_LIBRARIES} Calcite)
