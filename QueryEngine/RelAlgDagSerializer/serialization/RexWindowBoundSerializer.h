/*
 * Copyright 2022 HEAVY.AI, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
