/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <unordered_set>

#include "GfxDriver/Pipeline/PrimitiveAssembly.h"

namespace gfx {

class PrimitiveAssemblyDependency {
 public:
  ~PrimitiveAssemblyDependency() { removeAllDependents(); }

  void markPrimitiveAssembliesDirty() {
    for (auto const& dependent : dependents_) {
      dependent->markDirty();
    }
  }

 private:
  void addDependent(PrimitiveAssembly* dependent) const { dependents_.insert(dependent); }
  void removeDependent(PrimitiveAssembly* dependent) const {
    dependents_.erase(dependent);
  }
  void removeAllDependents() {
    for (auto const& dependent : dependents_) {
      dependent->removeDependency(this);
    }
    dependents_.clear();
  }

  mutable std::unordered_set<PrimitiveAssembly*> dependents_;

  friend class PrimitiveAssembly;
};

}  // namespace gfx
