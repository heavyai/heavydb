/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ExecuteRenderInterface/RenderQueryUtils/RelScanTree.h"
#include "QueryEngine/Rendering/RenderInfo.h"

namespace QueryRenderer {

struct RenderRelAlgUtils {
  static void alterRAForRender(RelScanTree* rel_scan_tree, const RenderInfo& render_info);
};

}  // namespace QueryRenderer
