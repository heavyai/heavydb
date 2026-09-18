/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <optional>

namespace QueryRenderer {

template <typename T>
using OptionalT = std::optional<T>;

using OptionalStr = OptionalT<std::string>;

using GpuId = uint32_t;

class PngData;

struct RenderSessionKey;

class RootPerGpuData;
using RootPerGpuDataUqPtr = std::unique_ptr<RootPerGpuData>;

class QueryRenderManager;

class QueryDataLayout;
using QueryDataLayoutShPtr = std::shared_ptr<QueryDataLayout>;

class Renderer;

class QueryRendererContext;
using QueryRendererContextUqPtr = std::unique_ptr<QueryRendererContext>;

class GlobalRenderContext;

struct RowIdHitTestOffsetData;
struct QueryResultCacheItem;
using QueryResultCacheItemShPtr = std::shared_ptr<QueryResultCacheItem>;

class HitTestBuffers;
using HitTestBuffersUqPtr = std::unique_ptr<HitTestBuffers>;

}  // namespace QueryRenderer
