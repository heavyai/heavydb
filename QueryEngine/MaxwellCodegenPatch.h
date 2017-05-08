/*
 * Copyright 2017 MapD Technologies, Inc.
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

#ifndef QUERYENGINE_MAXWELLCODEGENPATCH_H
#define QUERYENGINE_MAXWELLCODEGENPATCH_H

#include "Execute.h"
#include "../CudaMgr/CudaMgr.h"

inline bool need_patch_unnest_double(const SQLTypeInfo& ti, const bool is_maxwell, const bool mem_shared) {
  return is_maxwell && mem_shared && ti.is_fp() && ti.get_type() == kDOUBLE;
}

inline std::string patch_agg_fname(const std::string& agg_name) {
  const auto new_name = agg_name + "_slow";
  CHECK_EQ("agg_id_double_shared_slow", new_name);
  return new_name;
}

#endif  // QUERYENGINE_MAXWELLCODEGENPATCH_H
