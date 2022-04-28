find_package(JNI REQUIRED)
include_directories(${JNI_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_BINARY_DIR})

## mvn process for java code
find_program(MVN_EXECUTABLE NAMES mvn)
if(NOT MVN_EXECUTABLE)
  message(FATAL_ERROR "mvn not found. Install Apache Maven.")
endif()
file(GLOB_RECURSE JAVA_POM RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} java/**/pom.xml)
file(GLOB_RECURSE JAVA_FTL RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} java/calcite/src/main/codegen/includes/*.ftl)
file(GLOB_RECURSE JAVA_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} java/**/*.java)
list(FILTER JAVA_SOURCES EXCLUDE REGEX ".*/gen/.*")
list(FILTER JAVA_SOURCES EXCLUDE REGEX ".*/generated-sources/.*")

set(OMNISCI_JAR_RELEASE_VERSION "${MAPD_VERSION_MAJOR}.${MAPD_VERSION_MINOR}.${MAPD_VERSION_PATCH}")
if("${MAPD_VERSION_EXTRA}" STREQUAL "dev")
  set (OMNISCI_JAR_RELEASE_VERSION "${OMNISCI_JAR_RELEASE_VERSION}-SNAPSHOT")
endif()

set(MVN_PATH_COMMAND "")
if(NOT MSVC)
    set(MVN_PATH_COMMAND "MVNPATH=${CMAKE_CURRENT_SOURCE_DIR}/java")
endif()

add_custom_command(
  OUTPUT
    ${CMAKE_BINARY_DIR}/bin/calcite-1.0-SNAPSHOT-jar-with-dependencies.jar
    COMMAND ${MVN_PATH_COMMAND} ${MVN_EXECUTABLE} -l ${CMAKE_BINARY_DIR}/mvn_build.log -e clean install -Dmaven.compiler.showDeprecation=true -Dmaven.compiler.showWarnings=true -Domnisci.release.version="${OMNISCI_JAR_RELEASE_VERSION}" -Djava.net.preferIPv4Stack=true -Dmaven.wagon.http.retryHandler.count=3 -DMAPD_LOG_DIR="${CMAKE_BINARY_DIR}"
    COMMAND mkdir -p ${CMAKE_BINARY_DIR}/bin
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/java/calcite/target/calcite-1.0-SNAPSHOT-jar-with-dependencies.jar ${CMAKE_BINARY_DIR}/bin
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/java
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/java/calcite/src/main/codegen/config.fmpp
    ${CMAKE_CURRENT_SOURCE_DIR}/java/pom.xml
    ${JAVA_POM}
    ${JAVA_SOURCES}
    ${JAVA_FTL}
  )
add_custom_target(calcite_java_lib ALL DEPENDS
  ${CMAKE_BINARY_DIR}/bin/calcite-1.0-SNAPSHOT-jar-with-dependencies.jar)
add_custom_target(calcite_java_clean
  COMMAND ${MVN_PATH_COMMAND} ${MVN_EXECUTABLE} -q clean -Domnisci.release.version="${OMNISCI_JAR_RELEASE_VERSION}"
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/java
  )

add_custom_target(maven_populate_cache
  COMMAND ${MVN_PATH_COMMAND} ${MVN_EXECUTABLE} -q verify -Domnisci.release.version="${OMNISCI_JAR_RELEASE_VERSION}"
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/java
  )

add_library(Calcite CalciteJNI.cpp SchemaJson.cpp)

add_dependencies(Calcite calcite_java_lib)

target_link_libraries(Calcite OSDependent Shared ${JAVA_JVM_LIBRARY})

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/java/calcite/target/calcite-1.0-SNAPSHOT-jar-with-dependencies.jar DESTINATION bin COMPONENT "jar")
