/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Catalog/SessionInfo.h"
#include "QueryEngine/CompilationOptions.h"
#include "Shared/SystemParameters.h"

extern bool g_enable_watchdog;
extern bool g_enable_dynamic_watchdog;
extern unsigned g_dynamic_watchdog_time_limit;

namespace QueryRenderer {

// Pared down version of ExecutionOptions for render queries
struct RenderQueryExecutionOptions {
  const bool allow_multifrag = true;
  const bool allow_loop_joins = false;
  const bool with_enable_watchdog = g_enable_watchdog;
  const bool with_enable_dynamic_watchdog = g_enable_dynamic_watchdog;
  const unsigned dynamic_watchdog_time_limit = g_dynamic_watchdog_time_limit;
  const double gpu_input_mem_limit_percent = SystemParameters().gpu_input_mem_limit;
  const bool jit_debug = false;

  inline std::pair<CompilationOptions, ExecutionOptions> convertToRelAlgExecutorOptions(
      const Catalog_Namespace::SessionInfo& session_info,
      const bool just_validate) const {
    // Allow lazy fetch will be disabled for the last step of all render queries in
    // RelAlgExecutor
    CompilationOptions compilation_opts = {
        .device_type = just_validate ? ExecutorDeviceType::CPU
                                     : session_info.get_executor_device_type(),
        .hoist_literals = true,  // always true - see DBHandler::execute_rel_alg
        .opt_level = ExecutorOptLevel::Default,
        .with_dynamic_watchdog = with_enable_dynamic_watchdog,
        .allow_lazy_fetch = true,
        .just_validate = just_validate};

    // Some notes regarding the execution options below:
    // #1) output_columnar_hint:
    //       TODO(croot): Need to experiment with columnar output for render queries. This
    //       can be supported with sequential layouts, but performance implications are
    //       uncertain. Setting to false for now.
    // #2) allow_multifrag:
    //       TODO(croot): It's probably worth investigating any implications with this
    //       always being true and not abiding by the global config parameter. Not sure
    //       whether there are any implications here with the bump allocator work pending
    //       (as of 12/03/19).
    // #3) just_explain:
    //       we currently do a validate only really to extract the resulting column names
    //       and types for validation. This could be done by doing an explain instead
    //       (setting this to true). However, as of 02/15/2018 inside
    //       RelAlgExecutor::executeSort, when explain is turned on, the sort returns
    //       without target meta info, which the render relies on for name/type info. It's
    //       not immediately clear why this is done. We could return the meta info there
    //       if a render query is being explained, and therefore wouldn't disturb the
    //       existing logic. The one nice thing about using validate is that it could
    //       provide some pre-query checks such as memory that explain wouldn't provide,
    //       but it is more of a work horse.
    // #4) allow_loop_joins:
    //       This can always be true for validate queries - similar to
    //       DBHandler::execute_rel_alg
    // #5) All other defaulted options are determined by examining the execute_rel_alg()
    //     calls inside DBHandler::sql_execute_impl & DBHandler::validate_rel_alg
    ExecutionOptions exec_opts = {
        .output_columnar_hint = false,                          // See note above
        .allow_multifrag = true,                                // See note above
        .just_explain = false,                                  // See note above
        .allow_loop_joins = just_validate || allow_loop_joins,  // See note above
        .with_watchdog = with_enable_watchdog,
        .jit_debug = jit_debug,
        .just_validate = just_validate,
        .with_dynamic_watchdog = with_enable_dynamic_watchdog,
        .dynamic_watchdog_time_limit = dynamic_watchdog_time_limit,
        .find_push_down_candidates = false,
        .just_calcite_explain = false,
        .gpu_input_mem_limit_percent = gpu_input_mem_limit_percent,
        .allow_runtime_query_interrupt = false};

    return std::make_pair(compilation_opts, exec_opts);
  }
};

}  // namespace QueryRenderer
