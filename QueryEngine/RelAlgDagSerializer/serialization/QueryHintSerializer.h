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

#include "QueryEngine/QueryHint.h"

#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/vector.hpp>

namespace boost {
namespace serialization {

template <class Archive>
void serialize(Archive& ar, RegisteredQueryHint& query_hint, const unsigned int version) {
  (ar & query_hint.cpu_mode);
  (ar & query_hint.columnar_output);
  (ar & query_hint.rowwise_output);
  (ar & query_hint.keep_result);
  (ar & query_hint.keep_table_function_result);
  (ar & query_hint.watchdog);
  (ar & query_hint.dynamic_watchdog);
  (ar & query_hint.query_time_limit);
  (ar & query_hint.cuda_block_size);
  (ar & query_hint.cuda_grid_size_multiplier);
  (ar & query_hint.aggregate_tree_fanout);
  (ar & query_hint.overlaps_bucket_threshold);
  (ar & query_hint.overlaps_max_size);
  (ar & query_hint.overlaps_allow_gpu_build);
  (ar & query_hint.overlaps_no_cache);
  (ar & query_hint.overlaps_keys_per_bin);
  (ar & query_hint.use_loop_join);
  (ar & query_hint.max_join_hash_table_size);
  (ar & query_hint.loop_join_inner_table_max_num_rows);
  (ar & query_hint.registered_hint);
}

template <class Archive>
void serialize(Archive& ar, ExplainedQueryHint& query_hint, const unsigned int version) {
  // need to split serialization into separate load/store methods since members are only
  // accessible through getters/setters. See:
  // https://www.boost.org/doc/libs/1_74_0/libs/serialization/doc/serialization.html#splitting
  split_free(ar, query_hint, version);
}

template <class Archive>
void save(Archive& ar, const ExplainedQueryHint& query_hint, const unsigned int version) {
  ar << query_hint.getInteritPath();
  ar << query_hint.getListOptions();
  ar << query_hint.getKVOptions();
}

template <class Archive>
void load(Archive& ar, ExplainedQueryHint& query_hint, const unsigned int version) {
  std::vector<int> inherit_paths;
  std::vector<std::string> list_options;
  std::unordered_map<std::string, std::string> kv_options;
  ar >> inherit_paths;
  query_hint.setInheritPaths(inherit_paths);
  ar >> list_options;
  query_hint.setListOptions(list_options);
  ar >> kv_options;
  query_hint.setKVOptions(kv_options);
}

template <class Archive>
inline void save_construct_data(Archive& ar,
                                const ExplainedQueryHint* query_hint,
                                const unsigned int version) {
  ar << query_hint->getHint();
  ar << query_hint->isGlobalHint();
  ar << query_hint->hasOptions();
  ar << query_hint->hasKvOptions();
}

template <class Archive>
inline void load_construct_data(Archive& ar,
                                ExplainedQueryHint* query_hint,
                                const unsigned int version) {
  QueryHint hint;
  bool global_hint;
  bool is_marker;
  bool has_kv_type_options;
  ar >> hint;
  ar >> global_hint;
  ar >> is_marker;
  ar >> has_kv_type_options;
  ::new (query_hint)
      ExplainedQueryHint(hint, global_hint, is_marker, has_kv_type_options);
}

}  // namespace serialization
}  // namespace boost
