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

#ifndef LEAFAGGREGATOR_H
#define LEAFAGGREGATOR_H

#include "LeafHostInfo.h"
#include "gen-cpp/MapD.h"
#include "QueryEngine/CompilationOptions.h"
#include "QueryEngine/TargetMetaInfo.h"

#include <glog/logging.h>

namespace Catalog_Namespace {
class SessionInfo;
}  // Catalog_Namespace

class ResultSet;

struct AggregatedResult {
  std::shared_ptr<ResultSet> rs;
  const std::vector<TargetMetaInfo> targets_meta;
};

class LeafAggregator {
 public:
  LeafAggregator(const std::vector<LeafHostInfo>& leaves) { CHECK(leaves.empty()); }

  AggregatedResult execute(const Catalog_Namespace::SessionInfo& parent_session_info,
                           const std::string& query_ra,
                           const ExecutionOptions& eo) {
    CHECK(false);
    return {nullptr, {}};
  }

  void connect(const Catalog_Namespace::SessionInfo& parent_session_info,
               const std::string& user,
               const std::string& passwd,
               const std::string& dbname) {
    CHECK(false);
  }

  void disconnect(const TSessionId session) { CHECK(false); }

  void interrupt(const TSessionId session) { CHECK(false); }

  size_t leafCount() const { return 0; }

  void set_execution_mode(const TSessionId session, const TExecuteMode::type mode) { CHECK(false); }
};

#endif  // LEAFAGGREGATOR_H
