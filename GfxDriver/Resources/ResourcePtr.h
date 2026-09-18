/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * @file   ResourcePtr.h
 * @author Steve Blackmon <steve.blackmon@omnisci.com>
 * @brief  Specialization of sinked_ptr to use ResourceManager as sink
 */
#pragma once

#include "GfxDriver/Resources/SinkedPtr.h"

namespace gfx {

class ResourceManager;

template <typename RESOURCE_TYPE>
using resource_ptr = sinked_ptr<RESOURCE_TYPE, ResourceManager>;

//
// make_resource_ptr helpers
//

// Wrap a constructed resource in a resource_ptr of the same type (will deduce
// RESOURCE_TYPE from the passed pointer's type)
template <typename RESOURCE_TYPE>
inline auto make_resource_ptr(RESOURCE_TYPE* r) {
  return make_sinked_ptr<RESOURCE_TYPE, ResourceManager>(r);
}

// Wrap a constructed resource of BASE type in resource_ptr of RESOURCE type with
// automatic static_casting. Requires explicit RESOURCE_TYPE declaration
template <typename RESOURCE_TYPE, typename RESOURCE_BASE>
inline auto make_resource_ptr(RESOURCE_BASE* r) {
  return make_sinked_ptr<RESOURCE_TYPE, RESOURCE_BASE, ResourceManager>(r);
}

}  // namespace gfx
