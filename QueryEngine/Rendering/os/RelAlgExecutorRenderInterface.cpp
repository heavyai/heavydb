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

#include "../../RelAlgExecutor.h"

ExecutionResult RelAlgExecutor::renderWorkUnit(const RelAlgExecutor::WorkUnit& work_unit,
                                               const std::vector<TargetMetaInfo>& targets_meta,
                                               RenderInfo* render_info,
                                               const int32_t error_code,
                                               const int64_t queue_time_ms) {
  CHECK(false);
  return ExecutionResult(std::shared_ptr<ResultSet>(nullptr), {});
}
