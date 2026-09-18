# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

function(generateShaderTemplates scriptsDir shaderManifest outputDir outputLib namespace)
  find_package(Python REQUIRED)

  message(STATUS "Generating Shader Templates...")
  message(STATUS "  shaderManifest : " ${shaderManifest})
  message(STATUS "  outputDir      : " ${outputDir})
  message(STATUS "  namespace      : " ${namespace})

  set(GEN_CPP_COMMAND
    ${Python_EXECUTABLE} ${scriptsDir}/generate_shader_templates.py -m "${shaderManifest}" -o ${outputDir} -n ${namespace}
    )

  # VERBOSE flag is only available in cmake 3.15, so just comment this out for now
  # until I have a better solution for the rerun cmake noise
  # message(VERBOSE "Configuring shader template generator")
  execute_process(
    COMMAND ${GEN_CPP_COMMAND} --print-dependencies
    OUTPUT_VARIABLE SHADER_CPP_DEPENDENCIES
    RESULT_VARIABLE RET
  )
  if (NOT RET EQUAL 0)
    message(FATAL_ERROR "Failed to get the shader to cpp dependencies")
  endif()

  execute_process(
    COMMAND ${GEN_CPP_COMMAND} --print-outputs
    OUTPUT_VARIABLE SHADER_CPP_OUTPUTS
    RESULT_VARIABLE RET
  )
  if (NOT RET EQUAL 0)
    message(FATAL_ERROR "Failed to get the shader to cpp outputs")
  endif()

  add_custom_command(
    COMMAND ${GEN_CPP_COMMAND}
    DEPENDS
      ${scriptsDir}/generate_shader_templates.py
      ${SHADER_CPP_DEPENDENCIES}
    OUTPUT ${SHADER_CPP_OUTPUTS}
    COMMENT "Generating shader cpp files"
  )

  add_library(${outputLib} ${SHADER_CPP_OUTPUTS})

endfunction()
