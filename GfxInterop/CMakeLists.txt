# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# Pick up the defs from GFXDRIVER. Some are unnecesary, like SHADER_ARTIFACTS_ENABLED,
# but easier to just grab all to make sure the ones we care about are set
set(GFXINTEROP_DEFS ${GFXDRIVER_DEFS})
set(GFXINTEROP_INCLUDE_DIRS "")
set(GFXINTEROP_LIBS "")

# always compile this .cpp even if CUDA off
# @TODO(se) move this elsewhere and deprecate this library completely
set(GFXINTEROP_SOURCES Utils/CudaErrorCheck.cpp)

add_library(GfxInterop ${GFXINTEROP_SOURCES})
target_include_directories(GfxInterop PRIVATE ${GFXINTEROP_INCLUDE_DIRS})
target_link_libraries(GfxInterop ${GFXINTEROP_LIBS})
target_compile_definitions(GfxInterop PRIVATE ${GFXINTEROP_DEFS})
