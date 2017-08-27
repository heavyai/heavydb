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

#ifndef QUERY_RUNNER_H
#define QUERY_RUNNER_H

#include "../Catalog/Catalog.h"
#include "../QueryEngine/Execute.h"
#include "../QueryEngine/RelAlgExecutionDescriptor.h"

#include <memory>
#include <string>

Catalog_Namespace::SessionInfo* get_session(const char* db_path);

ExecutionResult run_select_query(const std::string& query_str,
                                 const std::unique_ptr<Catalog_Namespace::SessionInfo>& session,
                                 const ExecutorDeviceType device_type,
                                 const bool hoist_literals,
                                 const bool allow_loop_joins);

std::shared_ptr<ResultSet> run_multiple_agg(const std::string& query_str,
                                            const std::unique_ptr<Catalog_Namespace::SessionInfo>& session,
                                            const ExecutorDeviceType device_type,
                                            const bool hoist_literals,
                                            const bool allow_loop_joins);

#endif  // QUERY_RUNNER_H
