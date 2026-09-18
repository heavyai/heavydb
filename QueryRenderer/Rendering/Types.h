/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <functional>
#include <memory>
#include <set>

#include "GfxDriver/Types.h"
#include "QueryRenderer/Scales/Types.h"
#include "QueryRenderer/Types.h"

namespace QueryRenderer {

class QueryFramebuffer;
using QueryFramebufferUqPtr = std::unique_ptr<QueryFramebuffer>;
using QueryFramebufferShPtr = std::shared_ptr<QueryFramebuffer>;

template <typename T>
class QueryIdMapPixelBuffer;

using QueryIdMapPixelBufferShPtr = std::shared_ptr<QueryIdMapPixelBuffer<uint32_t>>;
using QueryIdMapPixelBufferWkPtr = std::weak_ptr<QueryIdMapPixelBuffer<uint32_t>>;

template <typename T>
class QueryIdMapPboPoolT;
using QueryIdMapPboPool = QueryIdMapPboPoolT<uint32_t>;
using QueryIdMapPboPoolUqPtr = std::unique_ptr<QueryIdMapPboPool>;

class MultiGpuCompositor;

class QueryRenderSMAAPass;
using QueryRenderSMAAPassUqPtr = std::unique_ptr<QueryRenderSMAAPass>;

class SeparateMultiSamplesPass;
using SeparateMultiSamplesPassUqPtr = std::unique_ptr<SeparateMultiSamplesPass>;

class AccumRenderer;
using AccumRendererUqPtr = std::unique_ptr<AccumRenderer>;

using PerPassGpuCBFunc = std::function<void(const gfx::DeviceContext&,
                                            QueryFramebuffer&,
                                            const QueryRendererContext&,
                                            const bool,  // should clear
                                            const bool,  // should comp
                                            ScaleAccumRenderState*,
                                            const int)>;  // accumulator index

using PassCompleteCBFunc = std::function<void(const std::set<GpuId>&,  // render used gpus
                                              const std::set<GpuId>&,  // pass used gpus
                                              const QueryRendererContext&,
                                              const int,  // pass index
                                              ScaleAccumRenderState*)>;

}  // namespace QueryRenderer
