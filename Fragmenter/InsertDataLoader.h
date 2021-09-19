/*
 * Copyright 2019, OmniSci, Inc.
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

#ifndef INSERT_DATA_LOADER_H_
#define INSERT_DATA_LOADER_H_

#include "../Catalog/Catalog.h"
#include "Fragmenter.h"

namespace Fragmenter_Namespace {

struct InsertDataLoader {
 public:
  struct DistributedConnector {
    virtual size_t leafCount() = 0;
    virtual void insertDataToLeaf(
        const Catalog_Namespace::SessionInfo& parent_session_info,
        const size_t leaf_idx,
        Fragmenter_Namespace::InsertData& insert_data) = 0;
    virtual void checkpoint(const Catalog_Namespace::SessionInfo& parent_session_info,
                            int tableId) = 0;
    virtual void rollback(const Catalog_Namespace::SessionInfo& parent_session_info,
                          int tableId) = 0;
  };

  InsertDataLoader(DistributedConnector& connector)
      : leaf_count_(connector.leafCount())
      , current_leaf_index_(0)
      , connector_(connector){};

  void insertData(const Catalog_Namespace::SessionInfo& session_info,
                  InsertData& insert_data);

 private:
  void moveToNextLeaf() {
    current_leaf_index_++;
    if (current_leaf_index_ >= leaf_count_) {
      current_leaf_index_ = 0;
    }
  }

  size_t leaf_count_;
  size_t current_leaf_index_;
  DistributedConnector& connector_;
};

}  // namespace Fragmenter_Namespace

#endif
