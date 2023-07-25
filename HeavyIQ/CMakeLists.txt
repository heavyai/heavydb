set(HEAVYIQ_BUILD_ID "41" CACHE STRING "HeavyIQ Build ID")
set(HEAVYIQ_URL "http://builds.mapd.com/heavyiq/heavyiq-${HEAVYIQ_BUILD_ID}/dist.tgz")

include(ExternalProject)
externalproject_add(heavyiq
  URL ${HEAVYIQ_URL}
  HTTP_USERNAME mapd
  HTTP_PASSWORD HyperInteractive
  PREFIX external
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  LOG_DOWNLOAD on
  DOWNLOAD_EXTRACT_TIMESTAMP true
  )

externalproject_get_property(heavyiq source_dir)
install(DIRECTORY ${source_dir}/ DESTINATION "heavyiq/")
add_custom_command(TARGET heavyiq COMMAND ${CMAKE_COMMAND} -E copy_directory ${source_dir} ${CMAKE_BINARY_DIR}/heavyiq)
list(APPEND ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_BINARY_DIR}/heavyiq)
set(ADDITIONAL_MAKE_CLEAN_FILES ${ADDITIONAL_MAKE_CLEAN_FILES} PARENT_SCOPE)

# Cleans the HeavyIQ directory so that the next build forces a rebuild
externalproject_get_property(heavyiq prefix)
add_custom_target(clean-heavyiq COMMAND ${CMAKE_COMMAND} -E remove_directory "${prefix}")
list(APPEND ADDITIONAL_MAKE_CLEAN_FILES ${prefix})

add_dependencies(clean-all clean-heavyiq)
