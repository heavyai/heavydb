set(catalog_source_files
    Catalog.cpp
    Catalog.h
)

get_target_property(StringDictionary_BINARY_DIR StringDictionary BINARY_DIR)
include_directories(${StringDictionary_BINARY_DIR})

add_library(Catalog ${catalog_source_files})

if("${MAPD_EDITION_LOWER}" STREQUAL "ee")
  find_library(LDAP_LIBRARIES NAMES)
endif()

if(ENABLE_CALCITE)
  target_link_libraries(Catalog SqliteConnector StringDictionary Fragmenter ${LDAP_LIBRARIES} Calcite)
else()
  target_link_libraries(Catalog SqliteConnector StringDictionary Fragmenter ${LDAP_LIBRARIES})
endif()
