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

#pragma once

#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "ExecutorResourceMgrCommon.h"
#include "Logger/Logger.h"
#include "Shared/BinarySemaphore.h"

namespace ExecutorResourceMgr_Namespace {

/**
 * @type OutstandingQueueRequests
 * @brief Stores and allows access to a binary semaphore per
 * RequestId (using an `std::unordered_map`), as well as accessing
 * all outstanding RequestIds for waiting requests
 */
class OutstandingQueueRequests {
  using OutstandingQueueRequestsMap =
      std::unordered_map<RequestId, SemaphoreShim_Namespace::BinarySemaphore>;

 public:
  /**
   * @brief Submits a request with id `request_id` into the queue, waiting on a
   * `BinarySemaphore` until `ExecutorResourceMgr` decides to grant the request
   * resources and wakes the waiting thread.
   *
   * @param request_id - `RequestId` for this request
   */
  void queue_request_and_wait(const RequestId request_id) {
    auto& wait_semaphore = get_semaphore_for_request(request_id);
    wait_semaphore.try_acquire();
    delete_semaphore_for_request(request_id);
  }

  /**
   * @brief Submits a request with id `request_id` into the queue, waiting on a
   * `BinarySemaphore` until `ExecutorResourceMgr` decides to grant the request
   * resources and wakes the waiting thread. If it waits for a period longer
   * than `max_wait_in_ms`, a `QueryTimedOutWaitingInQueue` exception is thrown.
   *
   * @param request_id - `RequestId` for this request
   */
  void queue_request_and_wait_with_timeout(const RequestId request_id,
                                           const size_t max_wait_in_ms) {
    CHECK_GT(max_wait_in_ms, size_t(0));
    auto& wait_semaphore = get_semaphore_for_request(request_id);
    // Binary semaphore returns false if it was not unblocked before the specified timeout
    const bool did_timeout = !(wait_semaphore.try_acquire_for(max_wait_in_ms));
    delete_semaphore_for_request(request_id);
    if (did_timeout) {
      throw QueryTimedOutWaitingInQueue(max_wait_in_ms);
    }
  }

  /**
   * @brief Get the `RequestId`s of all outsanding requests in the queue
   *
   * @return std::vector<RequestId> - The vector of request ids for outstanding requests
   * in the queue
   */
  std::vector<RequestId> get_outstanding_request_ids() {
    std::vector<RequestId> outstanding_request_ids;
    std::shared_lock<std::shared_mutex> requests_read_lock(requests_map_mutex_);
    outstanding_request_ids.reserve(outstanding_requests_map_.size());
    for (const auto& request_entry : outstanding_requests_map_) {
      outstanding_request_ids.emplace_back(request_entry.first);
    }
    return outstanding_request_ids;
  }

  /**
   * @brief Wakes a waiting thread in the queue.
   * Invoked by `ExecutorResourceMgr::process_queue_loop()`
   *
   * @param request_id - `RequestId` of the request/thread that should be awoken
   */
  void wake_request_by_id(const RequestId request_id) {
    std::unique_lock<std::shared_mutex> requests_write_lock(requests_map_mutex_);
    const auto request_itr = outstanding_requests_map_.find(request_id);
    if (request_itr == outstanding_requests_map_.end()) {
      outstanding_requests_map_[request_id].set_ready();
    } else {
      request_itr->second.release();
    }
  }

 private:
  /**
   * @brief Creates a new entry in `outstanding_requests_map_`, assigning
   * a `BinarySemaphore` for the given requesting thread with id `request_id`
   *
   * @param request_id - `RequestId` for the requesting thread - will be used
   * as a key in `outstanding_requests_map_`
   * @return SemaphoreShim_Namespace::BinarySemaphore& - a reference to the
   * `BinarySemaphore` that was added to `outstanding_requests_map_` for this
   * request
   */
  SemaphoreShim_Namespace::BinarySemaphore& get_semaphore_for_request(
      const RequestId request_id) {
    std::unique_lock<std::shared_mutex> requests_write_lock(requests_map_mutex_);
    return outstanding_requests_map_[request_id];
  }

  /**
   * @brief Internal method: removes a `RequestId`-`BinarySemaphore` entry
   * from `outstanding_requests_map_`. Invoked after a request thread is awoken
   * (including on timeout).
   *
   * @param request_id - `RequestId` key to be removed from the
   * `outstanding_requests_map_`
   */
  void delete_semaphore_for_request(const RequestId request_id) {
    std::unique_lock<std::shared_mutex> requests_write_lock(requests_map_mutex_);
    CHECK_EQ(outstanding_requests_map_.erase(request_id),
             size_t(1));  // Ensure the erase call returns 1, meaning there was actually
                          // an entry in the map matching this request_id to delete
  }

  /**
   * @brief Stores a map of `RequestId` to `BinarySemaphore`
   */
  OutstandingQueueRequestsMap outstanding_requests_map_;

  /**
   * @brief Read-write lock protecting the `outstanding_requests_map_`
   */
  mutable std::shared_mutex requests_map_mutex_;
};

}  // namespace ExecutorResourceMgr_Namespace