/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef INSERT_DATA_LOADER_H_
#define INSERT_DATA_LOADER_H_

#include "../Catalog/Catalog.h"
#include "Fragmenter.h"

namespace Fragmenter_Namespace {

class InsertDataLoader {
 public:
  class InsertConnector {
   public:
    virtual size_t shardCount() = 0;
    virtual void insertChunksToLeaf(
        const Catalog_Namespace::SessionInfo& parent_session_info,
        const size_t shard_idx,
        const Fragmenter_Namespace::InsertChunks& insert_chunks) = 0;
    virtual void insertDataToLeaf(
        const Catalog_Namespace::SessionInfo& parent_session_info,
        const size_t shard_idx,
        Fragmenter_Namespace::InsertData& insert_data) = 0;
    virtual void checkpoint(const Catalog_Namespace::SessionInfo& parent_session_info,
                            int tableId) = 0;
    virtual void rollback(const Catalog_Namespace::SessionInfo& parent_session_info,
                          int tableId) = 0;

    virtual ~InsertConnector() = default;
  };

  InsertDataLoader(InsertConnector& connector)
      : shard_count_(connector.shardCount())
      , current_shard_index_(0)
      , connector_(connector) {}

  void insertData(const Catalog_Namespace::SessionInfo& session_info,
                  InsertData& insert_data);

  void insertChunks(const Catalog_Namespace::SessionInfo& session_info,
                    const InsertChunks& insert_chunks);

  size_t getShardCount() const { return shard_count_; }

 private:
  /**
   * Move to the next available shard index internally. Done under a lock
   * to prevent contention.
   *
   * @return the current shard index (prior to moving to the next index)
   */
  size_t moveToNextShard();

  size_t shard_count_;
  size_t current_shard_index_;
  InsertConnector& connector_;
  std::shared_mutex current_shard_index_mutex_;
};

class LocalInsertConnector : public InsertDataLoader::InsertConnector {
 public:
  size_t shardCount() override { return 1; }
  void insertChunksToLeaf(
      const Catalog_Namespace::SessionInfo& parent_session_info,
      const size_t shard_idx,
      const Fragmenter_Namespace::InsertChunks& insert_chunks) override;
  void insertDataToLeaf(const Catalog_Namespace::SessionInfo& parent_session_info,
                        const size_t shard_idx,
                        Fragmenter_Namespace::InsertData& insert_data) override;
  void checkpoint(const Catalog_Namespace::SessionInfo& parent_session_info,
                  int tableId) override;
  void rollback(const Catalog_Namespace::SessionInfo& parent_session_info,
                int tableId) override;
};

}  // namespace Fragmenter_Namespace

#endif
