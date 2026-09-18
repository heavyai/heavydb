/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/QueryDataTableQueues.h"

#include <vector>

#include "GfxDriver/RenderLogger.h"
#include "QueryRenderer/Data/BaseQueryDataTable.h"
#include "QueryRenderer/Data/QueryDataTableSQL.h"
#include "QueryRenderer/Data/Transforms/BaseXform.h"
#include "QueryRenderer/QueryRendererContext.h"
#include "Shared/scope.h"

namespace QueryRenderer {

namespace {
struct QueryQueueEntry {
  QueryDataTableSQLJSONShPtr data_table;
  JSONLocation json_loc;
  const OptionalStr sql_query_override;
  const heavyai::InSituFlags insitu_flags;

  explicit QueryQueueEntry(QueryDataTableSQLJSONShPtr& data_table,
                           const JSONLocation& json_loc,
                           const OptionalStr& sql_query_override,
                           const heavyai::InSituFlags insitu_flags)
      : data_table{data_table}
      , json_loc{json_loc}
      , sql_query_override{sql_query_override}
      , insitu_flags{insitu_flags} {}
};

struct SourceTableQueueEntry {
  BaseDataTableShPtr data_table;
  JSONLocation json_loc;
  bool is_initializing;
  explicit SourceTableQueueEntry(BaseDataTableShPtr data_table,
                                 const JSONLocation& json_loc,
                                 bool is_initializing)
      : data_table{data_table}, json_loc{json_loc}, is_initializing{is_initializing} {}
};

struct NotifyQueueEntry {
  QueryDataTableSQLJSONShPtr data_table;
  RefEventType ref_event_type;
  explicit NotifyQueueEntry(QueryDataTableSQLJSONShPtr& data_table,
                            RefEventType ref_event_type)
      : data_table{data_table}, ref_event_type{ref_event_type} {}
};
}  // namespace

struct QueryDataTableQueues::Queues {
  std::vector<QueryQueueEntry> query;
  std::vector<SourceTableQueueEntry> source_table;
  std::vector<NotifyQueueEntry> notify;
};

//
// QueryDataTableQueues
//
QueryDataTableQueues::QueryDataTableQueues(QueryRendererContext& render_context)
    : ctx_{render_context}, queues_{std::make_unique<Queues>()} {}

QueryDataTableQueues::~QueryDataTableQueues() {}

//
// Add to queues
//
bool QueryDataTableQueues::addToQueryQueue(QueryDataTableSQLJSONShPtr data_table,
                                           const JSONLocation& json_loc,
                                           const OptionalStr& sql_query_override,
                                           const heavyai::InSituFlags insitu_flags) {
  RENDER_LOG_SCOPE();
  queues_->query.emplace_back(data_table, json_loc, sql_query_override, insitu_flags);
  return data_table->getQuerySQL().hasExecutableSql();
}

void QueryDataTableQueues::addToSourceTableQueue(BaseDataTableShPtr data_table,
                                                 const JSONLocation& json_loc,
                                                 bool is_initializing) {
  RENDER_LOG_SCOPE();
  queues_->source_table.emplace_back(data_table, json_loc, is_initializing);
}

void QueryDataTableQueues::addToNotifyQueue(QueryDataTableSQLJSONShPtr& data_table,
                                            RefEventType event_type) {
  RENDER_LOG_SCOPE();
  queues_->notify.emplace_back(data_table, event_type);
}

//
// isTableInNotifyQueue
//
bool QueryDataTableQueues::isTableInNotifyQueue(const std::string& table_name) {
  auto const& queue = queues_->notify;
  return std::find_if(queue.cbegin(), queue.cend(), [&](const NotifyQueueEntry& entry) {
           return entry.data_table->getNameRef() == table_name;
         }) != queue.end();
}

//
// Process queues
//
void QueryDataTableQueues::processQueryQueue() {
  RENDER_LOG_SCOPE();
  auto& queue = queues_->query;
  ScopeGuard clear_queue_guard = [&] { queue.clear(); };
  for (auto const& queue_entry : queue) {
    bool did_query_execute = ctx_.executeQuery(*queue_entry.data_table,
                                               &queue_entry.json_loc,
                                               queue_entry.sql_query_override,
                                               queue_entry.insitu_flags);
    queue_entry.data_table->postRunQuery(did_query_execute);
  }
}

void QueryDataTableQueues::processSourceTableQueue() {
  RENDER_LOG_SCOPE();
  auto& queue = queues_->source_table;
  ScopeGuard clear_queue_guard = [&] { queue.clear(); };
  for (auto const& queue_entry : queue) {
    if (queue_entry.is_initializing) {
      auto xform = std::dynamic_pointer_cast<BaseXform>(queue_entry.data_table);
      xform->initialize(xform, queue_entry.json_loc);
    } else {
      auto data_table_json =
          std::dynamic_pointer_cast<BaseQueryDataTableSQLJSON>(queue_entry.data_table);
      data_table_json->updateFromJSONObjInternal(queue_entry.json_loc, true);
    }
  }
}

void QueryDataTableQueues::processNotifyQueue() {
  RENDER_LOG_SCOPE();
  auto& queue = queues_->notify;
  ScopeGuard clear_queue_guard = [&] { queue.clear(); };
  for (auto const& queue_entry : queue) {
    ctx_.notifyRefEvent(queue_entry.ref_event_type, queue_entry.data_table);
  }
}

void QueryDataTableQueues::clear() {
  RENDER_LOG_SCOPE();
  queues_->query.clear();
  queues_->source_table.clear();
  queues_->notify.clear();
}

}  // namespace QueryRenderer
