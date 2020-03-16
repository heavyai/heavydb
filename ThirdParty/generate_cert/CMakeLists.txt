find_program(GO_EXECUTABLE NAMES go)
if(GO_EXECUTABLE)
  # add_custom_command + add_custom_target ensures this gets built on full builds, but only when necessary for incremental builds
  add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/bin/generate_cert
    COMMAND GOPATH=${CMAKE_BINARY_DIR}/go ${GO_EXECUTABLE} build -o ${CMAKE_BINARY_DIR}/bin/generate_cert ${CMAKE_CURRENT_SOURCE_DIR}/generate_cert.go
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS generate_cert.go
    )
  add_custom_target(generate_cert_target ALL DEPENDS ${CMAKE_BINARY_DIR}/bin/generate_cert)
  install(PROGRAMS ${CMAKE_BINARY_DIR}/bin/generate_cert DESTINATION bin)
  list(APPEND ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_BINARY_DIR}/bin/generate_cert)
  set(ADDITIONAL_MAKE_CLEAN_FILES ${ADDITIONAL_MAKE_CLEAN_FILES} PARENT_SCOPE)
endif()
