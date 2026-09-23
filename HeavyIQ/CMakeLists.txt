# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

set(HEAVYIQ_BUILD_ID "master" CACHE STRING "HeavyIQ Build ID")
set(HEAVYIQ_URL "${MAPD_BUILDS_REPO_URL}/heavyiq/heavyiq-${HEAVYIQ_BUILD_ID}/dist.tgz")
set(HEAVYIQ_FILE "${MAPD_BUILDS_REPO_PATH}/heavyiq/heavyiq-${HEAVYIQ_BUILD_ID}/dist.tgz" CACHE STRING "Local path to HeavyIQ dist.tgz")

include(ExternalProject)
include(heavyai_functions)
externalproject_add_url_or_file(heavyiq ${HEAVYIQ_URL} ${HEAVYIQ_FILE})

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
