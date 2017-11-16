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

/*
 * File:   MapDDistributedHandler.h
 * Author: Chris Root
 *
 * Created on Nov 6, 2017, 10:00 AM
 */

#ifndef MAPDDISTRIBUTEDHANDLER_H_
#define MAPDDISTRIBUTEDHANDLER_H_

#include "../MapDHandler.h"

class MapDAggHandler {
 public:
  ~MapDAggHandler() {}

 private:
  MapDAggHandler(MapDHandler* mapd_handler) { CHECK(false); }

  void cluster_execute(TQueryResult& _return,
                       const Catalog_Namespace::SessionInfo& session_info,
                       const std::string& query_str,
                       const bool column_format,
                       const std::string& nonce,
                       const int32_t first_n,
                       const int32_t at_most_n) {
    CHECK(false);
  }
  friend class MapDHandler;
};

class MapDLeafHandler {
 public:
  ~MapDLeafHandler() {}

 private:
  MapDLeafHandler(MapDHandler* mapd_handler) { CHECK(false); }

  void start_query(TPendingQuery& _return,
                   const TSessionId& session,
                   const std::string& query_ra,
                   const bool just_explain) {
    CHECK(false);
  }

  void execute_first_step(TStepResult& _return, const TPendingQuery& pending_query) { CHECK(false); }

  void broadcast_serialized_rows(const std::string& serialized_rows,
                                 const TRowDescriptor& row_desc,
                                 const TQueryId query_id) {
    CHECK(false);
  }

  void flush_queue() { CHECK(false); }

  friend class MapDHandler;
};

#endif /* MAPDDISTRIBUTEDHANDLER_H_ */
