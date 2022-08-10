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

#include "QueryEngine/RelAlgDag.h"

#include <boost/serialization/tracking.hpp>

// Because RexSubQuery::result_ is a shared_ptr of a shared_ptr, boost by default will not
// track it appropriately during serialization. We need to change the default behavior and
// let boost track the shared_pt<ExecutionResult> so that it can be serialized and loaded.
// Without the track_selectively registration below, a compilation error would be thrown
// when attempting to serialize RexSubQuery::result_
BOOST_CLASS_TRACKING(RexSubQuery::ExecutionResultShPtr,
                     boost::serialization::track_selectively)

namespace boost {
namespace serialization {

template <class Archive>
void serialize(Archive& ar,
               RexSubQuery::ExecutionResultShPtr& result,
               const unsigned int version) {
  // Noop - this should do nothing. Results should never be serialized. Serialization
  // should happen before RelAlgExecutor gets its hands on the dag.
  // This serialize method is only used to avoid compilation errors.
  CHECK(result == nullptr);
}

}  // namespace serialization
}  // namespace boost
