/*
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

#include <memory>

struct WatchdogConfig {
  bool enable = false;
  bool enable_dynamic = false;
  size_t time_limit = 10'000;
  size_t baseline_max_groups = 120'000'000;
};

struct CpuSubTasksConfig {
  bool enable = false;
  size_t sub_task_size = 500'000;
};

struct JoinConfig {
  bool allow_loop_joins = false;
  unsigned trivial_loop_join_threshold = 1'000;
  bool inner_join_fragment_skipping = true;
  size_t huge_join_hash_threshold = 1'000'000;
  size_t huge_join_hash_min_load = 10;
};

struct GroupByConfig {
  bool bigint_count = false;
  size_t default_max_groups_buffer_entry_guess = 16384;
  size_t big_group_threshold = 16384;
  bool use_groupby_buffer_desc = false;
};

struct ExecutionConfig {
  WatchdogConfig watchdog;
  CpuSubTasksConfig sub_tasks;
  JoinConfig join;
  GroupByConfig group_by;
};

struct FilterPushdownConfig {
  bool enable = false;
  float low_frac = -1.0f;
  float high_frac = -1.0f;
  size_t passing_row_ubound = 0;
};

struct OptimizationsConfig {
  FilterPushdownConfig filter_pushdown;
};

struct Config {
  ExecutionConfig exec;
  OptimizationsConfig opts;
};

using ConfigPtr = std::shared_ptr<Config>;
