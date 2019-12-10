cmake_minimum_required(VERSION 3.2)

find_package(Doxygen)

configure_file(
  ${CMAKE_CURRENT_LIST_DIR}/../Doxyfile.in ${CMAKE_BINARY_DIR}/Doxyfile @ONLY)
add_custom_target(doxygen ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/
  COMMENT "Generate API documentation with Doxygen"
  VERBATIM
  USES_TERMINAL
)
add_custom_target(doxygen_zip zip -r doxygen-${MAPD_GIT_HASH}.zip doxygen
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/
  COMMENT "Generating .zip with API documentation from Doxygen"
  VERBATIM
  USES_TERMINAL
)
add_dependencies(doxygen_zip doxygen)
