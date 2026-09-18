/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "BaseQueryRsrcPool.h"

namespace QueryRenderer {

template <typename T>
class QueryBufferPool : public BaseQueryRsrcPool<T, 300000, size_t> {
 public:
  QueryBufferPool(QueryBufferManager& query_buffer_mgr)
      : query_buffer_mgr_(query_buffer_mgr), total_used_bytes_(0) {}
  ~QueryBufferPool() override {}

  size_t getTotalUsedBytes() const { return total_used_bytes_; }

 private:
  std::shared_ptr<T> initializeRsrc(size_t num_bytes) final {
    total_used_bytes_ += num_bytes;
    return std::make_shared<T>(this->query_buffer_mgr_, num_bytes);
  }

  void updateRsrc(std::shared_ptr<T>& rsrc_ptr, size_t num_bytes) final {
    auto curr_bytes = rsrc_ptr->getNumBytes();
    rsrc_ptr->rebuild(num_bytes);
    if (curr_bytes > num_bytes) {
      total_used_bytes_ -= (curr_bytes - num_bytes);
    } else {
      total_used_bytes_ += (num_bytes - curr_bytes);
    }
  }

  void inactivateRsrc(T* inactivated_rsrc) noexcept final {
    if constexpr (std::is_base_of_v<QueryLayoutBuffer, T>) {  // NOLINT
      inactivated_rsrc->reset();
    }
  }

  void deleteRsrc(const T* deleted_rsrc_ptr) final {
    total_used_bytes_ -= deleted_rsrc_ptr->getNumBytes();
  }

  QueryBufferManager& query_buffer_mgr_;
  size_t total_used_bytes_;
};

}  // namespace QueryRenderer
