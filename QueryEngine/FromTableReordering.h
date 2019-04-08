/*
 * Copyright 2018, OmniSci, Inc.
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

#include "InputMetadata.h"
#include "RelAlgExecutionUnit.h"

// Returns a FROM permutation for the given join qualifiers and table sizes.
std::vector<size_t> get_node_input_permutation(
    const JoinQualsPerNestingLevel& left_deep_join_quals,
    const std::vector<InputTableInfo>& table_infos,
    const Executor* executor);
