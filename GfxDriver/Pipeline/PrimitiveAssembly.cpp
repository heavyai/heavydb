/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Pipeline/PrimitiveAssembly.h"

#include "GfxDriver/Pipeline/PrimitiveAssemblyDependency.h"

namespace gfx {

PrimitiveAssembly::PrimitiveAssembly(const PrimitiveTopology topology)
    : topology_{topology}, vertex_buffer_{nullptr}, index_buffer_{nullptr} {}

PrimitiveAssembly::PrimitiveAssembly(const PrimitiveTopology topology,
                                     const PrimitiveAssemblyAttrInfo& attr_info,
                                     const IndexBuffer* index_buffer)
    : topology_{topology}
    , vertex_buffer_{attr_info.vbo_and_layout.vertex_buffer}
    , index_buffer_{index_buffer} {}

PrimitiveAssembly::~PrimitiveAssembly() {
  for (auto const& dependency : dependencies_) {
    dependency->removeDependent(this);
  }
}

void PrimitiveAssembly::addDependency(const PrimitiveAssemblyDependency* dependency) {
  if (dependency) {
    dependency->addDependent(this);
    dependencies_.insert(dependency);
  }
}

void PrimitiveAssembly::removeDependency(const PrimitiveAssemblyDependency* dependency) {
  dependencies_.erase(dependency);
}

}  // namespace gfx
