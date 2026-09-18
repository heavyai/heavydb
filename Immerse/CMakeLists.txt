# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

set(MAPD_WEBSERVER_URL "${MAPD_WEBSERVER_BASE_URL}${CMAKE_HOST_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}-heavy_web_server.tar.gz" CACHE STRING "URL to download web server binary")
set(MAPD_WEBSERVER_FILE "${MAPD_BUILDS_REPO_PATH}/frontend/${MAPD_WEBSERVER_BUILD_ID}/${CMAKE_HOST_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}-heavy_web_server.tar.gz" CACHE STRING "Local path to web server binary archive")
include(ExternalProject)
include(heavyai_functions)
externalproject_add_url_or_file(web-server ${MAPD_WEBSERVER_URL} ${MAPD_WEBSERVER_FILE})
externalproject_get_property(web-server source_dir)
add_custom_command(TARGET web-server COMMAND strip ${source_dir}/heavy_web_server)
install(PROGRAMS ${source_dir}/heavy_web_server DESTINATION "bin")
add_custom_command(TARGET web-server COMMAND ${CMAKE_COMMAND} -E copy_directory ${source_dir} "${CMAKE_BINARY_DIR}/bin/")
list(APPEND ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_BINARY_DIR}/bin/heavy_web_server)
set(ADDITIONAL_MAKE_CLEAN_FILES ${ADDITIONAL_MAKE_CLEAN_FILES} PARENT_SCOPE)

# Cleans the web server so that the next build forces a rebuild
externalproject_get_property(web-server prefix)
add_custom_target(clean-web-server
  COMMAND ${CMAKE_COMMAND} -E remove_directory "${prefix}"
  )
list(APPEND ADDITIONAL_MAKE_CLEAN_FILES ${prefix})

add_dependencies(clean-all clean-web-server)
