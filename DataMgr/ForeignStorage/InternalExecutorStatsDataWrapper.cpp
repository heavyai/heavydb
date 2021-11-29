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

#include "InternalExecutorStatsDataWrapper.h"

#include "Catalog/SysCatalog.h"
#include "ImportExport/Importer.h"
#include "QueryEngine/Execute.h"

namespace foreign_storage {
InternalExecutorStatsDataWrapper::InternalExecutorStatsDataWrapper()
    : InternalSystemDataWrapper() {}

InternalExecutorStatsDataWrapper::InternalExecutorStatsDataWrapper(
    const int db_id,
    const ForeignTable* foreign_table)
    : InternalSystemDataWrapper(db_id, foreign_table) {}

namespace {
void populate_import_buffers_for_executor_resource_pool_summary(
    const ExecutorResourceMgr_Namespace::ResourcePoolInfo& resource_pool_info,
    std::map<std::string, import_export::TypedImportBuffer*>& import_buffers) {
  if (auto itr = import_buffers.find("total_cpu_slots"); itr != import_buffers.end()) {
    itr->second->addInt(resource_pool_info.total_cpu_slots);
  }
  if (auto itr = import_buffers.find("total_gpu_slots"); itr != import_buffers.end()) {
    itr->second->addInt(resource_pool_info.total_gpu_slots);
  }
  if (auto itr = import_buffers.find("total_cpu_result_mem");
      itr != import_buffers.end()) {
    itr->second->addInt(resource_pool_info.total_cpu_result_mem);
  }
  if (auto itr = import_buffers.find("total_cpu_buffer_pool_mem");
      itr != import_buffers.end()) {
    itr->second->addBigint(resource_pool_info.total_cpu_buffer_pool_mem);
  }
  if (auto itr = import_buffers.find("total_gpu_buffer_pool_mem");
      itr != import_buffers.end()) {
    itr->second->addBigint(resource_pool_info.total_gpu_buffer_pool_mem);
  }

  if (auto itr = import_buffers.find("allocated_cpu_slots");
      itr != import_buffers.end()) {
    itr->second->addInt(resource_pool_info.allocated_cpu_slots);
  }
  if (auto itr = import_buffers.find("allocated_gpu_slots");
      itr != import_buffers.end()) {
    itr->second->addInt(resource_pool_info.allocated_cpu_slots);
  }
  if (auto itr = import_buffers.find("allocated_cpu_result_mem");
      itr != import_buffers.end()) {
    itr->second->addInt(resource_pool_info.allocated_cpu_result_mem);
  }
  if (auto itr = import_buffers.find("allocated_cpu_buffer_pool_mem");
      itr != import_buffers.end()) {
    itr->second->addBigint(resource_pool_info.allocated_cpu_buffer_pool_mem);
  }
  if (auto itr = import_buffers.find("allocated_gpu_buffer_pool_mem");
      itr != import_buffers.end()) {
    itr->second->addBigint(resource_pool_info.allocated_gpu_buffer_pool_mem);
  }
  if (auto itr = import_buffers.find("allocated_cpu_buffers");
      itr != import_buffers.end()) {
    itr->second->addInt(resource_pool_info.allocated_cpu_buffers);
  }
  if (auto itr = import_buffers.find("allocated_gpu_buffers");
      itr != import_buffers.end()) {
    itr->second->addInt(resource_pool_info.allocated_gpu_buffers);
  }
  if (auto itr = import_buffers.find("allocated_temp_cpu_buffer_pool_mem");
      itr != import_buffers.end()) {
    itr->second->addBigint(resource_pool_info.allocated_temp_cpu_buffer_pool_mem);
  }
  if (auto itr = import_buffers.find("allocated_temp_gpu_buffer_pool_mem");
      itr != import_buffers.end()) {
    itr->second->addBigint(resource_pool_info.allocated_temp_gpu_buffer_pool_mem);
  }

  if (auto itr = import_buffers.find("total_requests"); itr != import_buffers.end()) {
    itr->second->addBigint(resource_pool_info.total_requests);
  }
  if (auto itr = import_buffers.find("outstanding_requests");
      itr != import_buffers.end()) {
    itr->second->addInt(resource_pool_info.outstanding_requests);
  }
  if (auto itr = import_buffers.find("outstanding_cpu_slots_requests");
      itr != import_buffers.end()) {
    itr->second->addInt(resource_pool_info.outstanding_cpu_slots_requests);
  }
  if (auto itr = import_buffers.find("outstanding_gpu_slots_requests");
      itr != import_buffers.end()) {
    itr->second->addInt(resource_pool_info.outstanding_gpu_slots_requests);
  }
  if (auto itr = import_buffers.find("outstanding_cpu_result_mem_requests");
      itr != import_buffers.end()) {
    itr->second->addInt(resource_pool_info.outstanding_cpu_result_mem_requests);
  }
  if (auto itr = import_buffers.find("outstanding_cpu_buffer_pool_mem_requests");
      itr != import_buffers.end()) {
    itr->second->addInt(resource_pool_info.outstanding_cpu_buffer_pool_mem_requests);
  }
  if (auto itr = import_buffers.find("outstanding_gpu_buffer_pool_mem_requests");
      itr != import_buffers.end()) {
    itr->second->addInt(resource_pool_info.outstanding_gpu_buffer_pool_mem_requests);
  }
}

}  // namespace

void InternalExecutorStatsDataWrapper::initializeObjectsForTable(
    const std::string& table_name) {
  CHECK_EQ(foreign_table_->tableName, table_name);
  CHECK_EQ(foreign_table_->tableName,
           Catalog_Namespace::EXECUTOR_RESOURCE_POOL_SUMMARY_SYS_TABLE_NAME)
      << "Unexpected table name: " << foreign_table_->tableName;
  executor_resource_pool_info_ = Executor::get_executor_resource_pool_info();
  row_count_ = 1;
}

void InternalExecutorStatsDataWrapper::populateChunkBuffersForTable(
    const std::string& table_name,
    std::map<std::string, import_export::TypedImportBuffer*>& import_buffers) {
  CHECK_EQ(foreign_table_->tableName, table_name);
  CHECK_EQ(foreign_table_->tableName,
           Catalog_Namespace::EXECUTOR_RESOURCE_POOL_SUMMARY_SYS_TABLE_NAME)
      << "Unexpected table name: " << foreign_table_->tableName;
  populate_import_buffers_for_executor_resource_pool_summary(executor_resource_pool_info_,
                                                             import_buffers);
}

}  // namespace foreign_storage
