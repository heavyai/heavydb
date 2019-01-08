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

#ifndef QUERYENGINE_COMPILATIONOPTIONS_H
#define QUERYENGINE_COMPILATIONOPTIONS_H

enum class ExecutorDeviceType { CPU, GPU };

enum class ExecutorOptLevel { Default, LoopStrengthReduction };

struct CompilationOptions {
  ExecutorDeviceType device_type_;
  const bool hoist_literals_;
  const ExecutorOptLevel opt_level_;
  const bool with_dynamic_watchdog_;
};

struct ExecutionOptions {
  const bool output_columnar_hint;
  const bool allow_multifrag;
  const bool just_explain;
  const bool allow_loop_joins;
  const bool with_watchdog;  // Per work unit, not global.
  const bool jit_debug;
  const bool just_validate;
  const bool with_dynamic_watchdog;  // Per work unit, not global.
  const unsigned
      dynamic_watchdog_time_limit;  // Dynamic watchdog time limit, in milliseconds.
  const bool find_push_down_candidates;
  const bool just_calcite_explain;
  const double gpu_input_mem_limit_percent;  // punt to CPU if input memory exceeds this
};

#endif  // QUERYENGINE_COMPILATIONOPTIONS_H
