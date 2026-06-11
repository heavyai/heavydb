/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <boost/serialization/shared_ptr.hpp>

#include "QueryEngine/RelAlgDag.h"

namespace boost {
namespace serialization {

template <class Archive>
void serialize(Archive& ar,
               RexWindowFunctionOperator::RexWindowBound& window_bound,
               const unsigned int version) {
  (ar & window_bound.unbounded);
  (ar & window_bound.preceding);
  (ar & window_bound.following);
  (ar & window_bound.is_current_row);
  (ar & window_bound.bound_expr);
  (ar & window_bound.order_key);
}

}  // namespace serialization
}  // namespace boost
