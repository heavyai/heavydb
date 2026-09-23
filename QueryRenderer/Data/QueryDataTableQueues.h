/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/Types.h"
#include "QueryRenderer/Events/RefEvent.h"
#include "QueryRenderer/Types.h"
#include "QueryRenderer/Utils/RapidJSONUtils.h"
#include "Shared/Rendering/InSituFlags.h"

namespace QueryRenderer {

//
// QueryDataTableQueues
//
class QueryDataTableQueues {
 public:
  explicit QueryDataTableQueues(QueryRendererContext& render_context);
  ~QueryDataTableQueues();

  bool addToQueryQueue(
      QueryDataTableSQLJSONShPtr data_table,
      const JSONLocation& json_loc,
      const OptionalStr& sql_query_override = std::nullopt,
      const heavyai::InSituFlags insitu_flags = heavyai::InSituFlags::kInSitu);

  void addToSourceTableQueue(BaseDataTableShPtr data_table,
                             const JSONLocation& json_loc,
                             bool is_initializing);

  void addToNotifyQueue(QueryDataTableSQLJSONShPtr& data_table,
                        const RefEventType event_type);
  bool isTableInNotifyQueue(const std::string& table_name);

  void processQueryQueue();
  void processSourceTableQueue();
  void processNotifyQueue();

  void clear();

 private:
  QueryRendererContext& ctx_;
  struct Queues;
  std::unique_ptr<Queues> queues_;
};

}  // namespace QueryRenderer
