# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

#
# Wrapper around ExternalProject_Add to handle local and remote archives
#
# Args:
#   target_name   - name of the external project target to create
#   file_url      - URL to download the archive with HTTP
#   file_path     - NFS path to copy the archive from if available
#

function (externalproject_add_url_or_file target_name file_url file_path)

  message(STATUS "In externalproject_add_url_or_file for ${target_name}")
  message(STATUS "  file_url      : ${file_url}")
  message(STATUS "  file_path     : ${file_path}")

  include(ExternalProject)

  if(EXISTS "${file_path}")

    message(STATUS "  File accessible at ${file_path}, will copy")

    externalproject_add(
      ${target_name}
      URL ${file_path} # URL works with files too!
      PREFIX external
      CONFIGURE_COMMAND ""
      UPDATE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND ""
    )

  else()

    message(STATUS "  File not accessible at ${file_path}, will download from ${file_url}")

    externalproject_add(
      ${target_name}
      URL ${file_url}
      HTTP_USERNAME mapd
      HTTP_PASSWORD HyperInteractive
      PREFIX external
      CONFIGURE_COMMAND ""
      UPDATE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND ""
      LOG_DOWNLOAD on
      DOWNLOAD_EXTRACT_TIMESTAMP true
    )

  endif()

  # parent code needs this for cleaning up
  set(source_dir "${CMAKE_BINARY_DIR}/external/src/${target_name}")
  message(STATUS "Setting source_dir to ${source_dir}")

  message(STATUS "  Done")

endfunction()

