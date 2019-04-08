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

#include "Execute.h"
#include "QueryRewrite.h"

#include "Shared/scope.h"

// The legacy way of executing queries. Don't change it, it's going away.

/*
 * x64 benchmark: "SELECT COUNT(*) FROM test WHERE x > 41;"
 *                x = 42, 64-bit column, 1-byte encoding
 *                3B rows in 1.2s on a i7-4870HQ core
 *
 * TODO(alex): check we haven't introduced a regression with the new translator.
 */

std::shared_ptr<ResultSet> Executor::execute(
    const Planner::RootPlan* root_plan,
    const Catalog_Namespace::SessionInfo& session,
    const bool hoist_literals,
    const ExecutorDeviceType device_type,
    const ExecutorOptLevel opt_level,
    const bool allow_multifrag,
    const bool allow_loop_joins,
    RenderInfo* render_info) {
  catalog_ = &root_plan->getCatalog();
  const auto stmt_type = root_plan->get_stmt_type();
  // capture the lock acquistion time
  auto clock_begin = timer_start();
  std::lock_guard<std::mutex> lock(execute_mutex_);
  if (g_enable_dynamic_watchdog) {
    resetInterrupt();
  }
  ScopeGuard restore_metainfo_cache = [this] { clearMetaInfoCache(); };
  int64_t queue_time_ms = timer_stop(clock_begin);
  ScopeGuard row_set_holder = [this] { row_set_mem_owner_ = nullptr; };
  switch (stmt_type) {
    case kSELECT: {
      throw std::runtime_error("The legacy SELECT path has been fully deprecated.");
    }
    case kINSERT: {
      if (root_plan->get_plan_dest() == Planner::RootPlan::kEXPLAIN) {
        auto explanation_rs = std::make_shared<ResultSet>("No explanation available.");
        explanation_rs->setQueueTime(queue_time_ms);
        return explanation_rs;
      }
      auto& cat = session.getCatalog();
      auto& sys_cat = Catalog_Namespace::SysCatalog::instance();
      auto user_metadata = session.get_currentUser();
      const int table_id = root_plan->get_result_table_id();
      auto td = cat.getMetadataForTable(table_id);
      DBObject dbObject(td->tableName, TableDBObjectType);
      dbObject.loadKey(cat);
      dbObject.setPrivileges(AccessPrivileges::INSERT_INTO_TABLE);
      std::vector<DBObject> privObjects;
      privObjects.push_back(dbObject);
      if (Catalog_Namespace::SysCatalog::instance().arePrivilegesOn() &&
          !sys_cat.checkPrivileges(user_metadata, privObjects)) {
        throw std::runtime_error(
            "Violation of access privileges: user " + user_metadata.userName +
            " has no insert privileges for table " + td->tableName + ".");
        break;
      }
      executeSimpleInsert(root_plan);
      auto empty_rs = std::make_shared<ResultSet>(std::vector<TargetInfo>{},
                                                  ExecutorDeviceType::CPU,
                                                  QueryMemoryDescriptor(),
                                                  nullptr,
                                                  this);
      empty_rs->setQueueTime(queue_time_ms);
      return empty_rs;
    }
    default:
      CHECK(false);
  }
  CHECK(false);
  return nullptr;
}
