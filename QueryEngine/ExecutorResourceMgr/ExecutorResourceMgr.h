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

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <set>
#include <shared_mutex>

#include "ExecutorResourceMgrCommon.h"
#include "ExecutorResourcePool.h"
#include "OutstandingQueueRequests.h"

namespace ExecutorResourceMgr_Namespace {

/**
 * @brief Stores current key statistics relating to `ExecutorResourceMgr`
 * state, particularly around the number of requests in queue and currently
 * executing.
 *
 * Some of these stats are used to populate the system table
 * `executor_pool_summary`, and in the future they may be used along
 * with the `RequestStats` vector to learn from query patterns for
 * more efficient scheduling.
 */
struct ExecutorStats {
  size_t requests{0};
  size_t cpu_requests{0};
  size_t gpu_requests{0};
  size_t queue_length{0};
  size_t cpu_queue_length{0};
  size_t gpu_queue_length{0};
  size_t total_queue_time_ms{0};
  size_t total_cpu_queue_time_ms{0};
  size_t total_gpu_queue_time_ms{0};
  size_t requests_actually_queued{0};
  size_t cpu_requests_actually_queued{0};
  size_t gpu_requests_actually_queued{0};
  size_t sum_queue_size_at_entry{0};
  size_t sum_cpu_queue_size_at_entry{0};
  size_t sum_gpu_queue_size_at_entry{0};
  size_t requests_executing{0};
  size_t cpu_requests_executing{0};
  size_t gpu_requests_executing{0};
  size_t requests_executed{0};
  size_t cpu_requests_executed{0};
  size_t gpu_requests_executed{0};
  size_t total_execution_time_ms{0};
  size_t total_cpu_execution_time_ms{0};
  size_t total_gpu_execution_time_ms{0};
  size_t total_time_ms{0};
  size_t total_cpu_time_ms{0};
  size_t total_gpu_time_ms{0};
  size_t requests_with_timeouts{0};
  size_t requests_timed_out{0};
};

/**
 * @brief Stores info pertaining to a single request made to
 * `ExecutorResourceMgr`, including its `request_id`, min and max possible
 * resource grants, actual resource_grant, and various timing stats.
 *
 * In future work it is planned to use these stats to allow the `ExecutorResourceMgr`
 * to learn from query patterns and timings over time to enable more efficient
 * query scheduling.
 */
struct RequestStats {
  const RequestId request_id;
  const RequestInfo request_info;
  const ResourceGrant min_resource_grant;
  const ResourceGrant max_resource_grant;
  ResourceGrant actual_resource_grant;
  std::chrono::steady_clock::time_point enqueue_time;             // in ms
  std::chrono::steady_clock::time_point deque_time;               // in ms
  std::chrono::steady_clock::time_point execution_finished_time;  // in ms
  size_t queue_length_at_entry;
  size_t device_type_queue_length_at_entry;
  bool finished_queueing{false};
  bool finished_executing{false};
  size_t queue_time_ms{0};
  size_t execution_time_ms{0};
  size_t total_time_ms{0};
  size_t timeout_in_ms{0};
  bool timed_out{false};
  // this variable will be filled w/ a corresponding msg when an error is occurred
  // when processing the resource allocation request by ERM
  std::optional<std::string> error;

  RequestStats(const RequestId request_id,
               const RequestInfo& request_info,
               const ResourceGrant& min_resource_grant,
               const ResourceGrant& max_resource_grant,
               const std::chrono::steady_clock::time_point& enqueue_time,
               const size_t queue_length_at_entry,
               const size_t device_type_queue_length_at_entry,
               const size_t timeout_in_ms)
      : request_id(request_id)
      , request_info(request_info)
      , min_resource_grant(min_resource_grant)
      , max_resource_grant(max_resource_grant)
      , enqueue_time(enqueue_time)
      , queue_length_at_entry(queue_length_at_entry)
      , device_type_queue_length_at_entry(device_type_queue_length_at_entry)
      , timeout_in_ms(timeout_in_ms) {}
};

enum ExecutionRequestStage { QUEUED, EXECUTING };

class ExecutorResourceHandle;  // forward declaration

/**
 * @brief `ExecutorResourceMgr` is the central manager for resources available to all
 * executors in the system. It manages an `ExecutorResourcePool` to keep track of
 * available and allocated resources (currently CPU slots/threads, GPUs, CPU result
 * memory, and CPU and GPU buffer pool memory). It also manages a thread queue which keeps
 * requesting threads (from `Executor::launchKernelsViaResourceMgr`) waiting until there
 * it can schedule them. At that point, it gives the calling executor thread a
 * `ResourceHandle` detailing the resources granted to the query, which once it goes out
 * of scope will return the granted resources to the ExecutorResourcePool.
 */
class ExecutorResourceMgr : public std::enable_shared_from_this<ExecutorResourceMgr> {
 public:
  /**
   * @brief The constructor instantiates an `ExecutorResourcePool` with the provided
   * parameters, and starts the process queue by launching a thread to invoke
   * process_queue_loop.
   */
  ExecutorResourceMgr(
      const std::vector<std::pair<ResourceType, size_t>>& total_resources,
      const std::vector<ConcurrentResourceGrantPolicy>&
          concurrent_resource_grant_policies,
      const std::vector<ResourceGrantPolicy>& max_per_request_resource_grant_policies,
      const double max_available_resource_use_ratio,
      const CPUResultMemResourceType cpu_result_memory_resource_type);

  /**
   * @brief The destructor ensures that the process queue thread (`process_queue_thread`)
   * is stopped and that any threads waiting for resources are joined.
   * Currently only called on database shutdown.
   */
  ~ExecutorResourceMgr();

  /**
   * @brief Requests resources from `ExecutorResourceMgr`, will throw if request takes
   * longer than time specified by timeout_in_ms
   *
   * @param request_info  - Details the resources requested
   * @param timeout_in_ms - Specifies the max time in ms the requesting thread
   * should wait for a request before an exception is thrown
   * @return std::unique_ptr<ExecutorResourceHandle> - The actual resource grant, whcih
   * when it goes out of scope (and the destructor is called), will return the allocated
   * resources to the `ExecutorResourcePool`
   */
  std::unique_ptr<ExecutorResourceHandle> request_resources_with_timeout(
      const RequestInfo& request_info,
      const size_t timeout_in_ms);

  /**
   * @brief Requests resources from `ExecutorResourceMgr`, with no timeout
   * (unlike `request_resources_with_timeout`)
   *
   * Internally calls `request_resources_with_timeout` with 0 timeout, specifying
   * no-timeout.
   *
   * @param request_info  - Details the resources requested
   * @return std::unique_ptr<ExecutorResourceHandle> - The actual resource grant,
   * which when it goes out of scope (and the destructor is called), will return the
   * allocated resources to the `ExecutorResourcePool`
   */
  std::unique_ptr<ExecutorResourceHandle> request_resources(
      const RequestInfo& request_info);

  /**
   * @brief Instructs `ExecutorResourceMgr` that the resources held by the
   * requestor with the given `request_id` can be freed/returned to the
   * `ExecutorResourcePool`.
   *
   * This method is only called automatically in the destructor of
   * `ExecutorResourceHandle` (i.e. when it goes out of scope along with
   * the parent executor thread)
   *
   * @param request_id  - `RequestId` for the query step that requested the resources
   * @param resource_grant  - The resources that were granted from `ExecutorResourcePool`
   * and that now will be freed/returned to the pool
   */
  void release_resources(const RequestId request_id, const ResourceGrant& resource_grant);

  /**
   * @brief Returns a copy of the `ExecutorStats` struct held by `ExecutorResourceMgr`.
   * Used for testing currently.
   */
  ExecutorStats get_executor_stats() const;

  /**
   * @brief Prints the `ExecutorStats` struct. Use for debugging.
   */
  void print_executor_stats() const;

  /**
   * @brief Returns the allocated and total available amount of the resource
   * specified.
   *
   * Interally queries `ExecutorResourcePool`
   *
   * @return std::pair<size_t, size_t> - First member is the allocated amount of the
   * resource, the second member is the total amount of the resource (including allocated
   * and available)
   */
  std::pair<size_t, size_t> get_resource_info(const ResourceType resource_type) const {
    return executor_resource_pool_.get_resource_info(resource_type);
  }

  /**
   * @brief Returns a struct containing the total and allocated amounts of all
   * resources tracked by `ExecutorResourceMgr`/`ExecutorResourcePool`, as well as the
   * number of outstanding requests (in total and by resource type)
   *
   * @return ResourcePoolInfo - Defined in `ExecutorResourcePool.h`,
   * contains total and allocated amounts for all resources, and total
   * requests outstanding and per type of resource
   */
  ResourcePoolInfo get_resource_info() const {
    return executor_resource_pool_.get_resource_info();
  }

  /**
   * @brief Used to change the total amount available of a specified resource after
   * construction of `ExecutorResourceMgr`
   *
   * Currently only used for testing, but could also be used to provide adminstrator
   * DDL commands to change resource availability in a running server
   *
   * @param resource_type - Type of resource to alter the quantity of
   * @param resoure_quantity - The new quantity available of the resource
   */
  void set_resource(const ResourceType resource_type, const size_t resoure_quantity);

  /**
   * @brief Get the concurrent resource grant policy for a given resource type
   *
   * Queries the ExecutorResourcePool for the current concurrency policies (including
   * normal and oversubscribed) for a resource type
   * @param resource_type - Type of resource to get the concurrency policy for
   * @return const ConcurrentResourceGrantPolicy& - Specifies the chosen
   * concurrency policy for normal operation and when the specified resource
   * is oversubscribed
   */
  ConcurrentResourceGrantPolicy get_concurrent_resource_grant_policy(
      const ResourceType resource_type) const;

  /**
   * @brief Set the concurrent resource grant policy for a given resource type (stored
   * in `ConcurrentResourceGrantPolicy`)
   *
   * @param concurrent_resource_grant_policy - Object containing the resource type and
   * the concurrency policies for when the resource is undersubscribed and oversubscribed
   */
  void set_concurrent_resource_grant_policy(
      const ConcurrentResourceGrantPolicy& concurrent_resource_grant_policy);

  /**
   * @brief Pauses the process queue in a thread-safe manner, waiting for all queries
   * in the executing stage to finish before yielding to the caller (ensuring that the
   * ExecutorResourcePool has no outstanding allocations). If the process queue is
   * already paused, the call is a no-op. This method is used to live-change parameters
   * associated with ExecutorResourcePool.
   *
   * Note that when the queue is fully paused, there will be no executing requests but
   * there can be one or more queued requests.
   */
  void pause_process_queue();

  /**
   * @brief Resumes the process queue in a thread-safe manner. If the process queue
   * is already paused, the call is a no-op.
   */
  void resume_process_queue();

 private:
  /**
   * @brief Internal method: A thread is assigned to run this function in the constructor
   * of `ExecutorResourceMgr`, where it loops continuously waiting for changes
   * to the process queue (i.e. the introduction of a new resource request
   * or the finish of an existing request).
   */
  void process_queue_loop();

  /**
   * @brief Internal method: Returns the `RequestStats` for a request specified by
   * `request_id`.
   *
   * Takes a read lock on `queue_stats_mutex_` for thread safety.
   * Note this method should not be used in internal methods where
   * a lock is already taken on `queue_stats_mutex_`.
   *
   * @param request_id - The `RequestId` for the request we want to
   * retrieve stats for
   * @return RequestStats - The stats for the specified request,
   * including resources granted
   */
  RequestStats get_request_for_id(const RequestId request_id) const;

  void mark_request_error(const RequestId request_id, std::string error_msg);

  /**
   * @brief Internal method: Invoked from `process_queue_loop`, chooses the next
   * resource request to grant.
   *
   * Currently based on FIFO logic, choosing the oldest request in the queue
   * that we have enough resources in `ExecutorResourcePool` to fulfill.
   * Future variants could add more sophisticated logic for more optimal
   * query scheduling (i.e. non-FIFO).
   *
   * @return RequestId - The id of the request we will grant resources for.
   */
  RequestId choose_next_request();

  /**
   * @brief Internal method: Invoked from request_resource/request_resource_with_timeout,
   * places request in the request queue where it will be wait to be granted the
   * resources requested.
   *
   * Note that this method assigns a `RequestId` to the request,
   * adds the request to the `RequestStats`, and adds the assigned
   * request_id to the QUEUED stage, but does not actually notify the
   * process queue or queue the request in the `OutstandQueueRequests` object.
   * Those tasks are done subsequently to invocation of this method in
   * the parent call (`request_resources_with_timeout`)
   *
   * @param request_info - info for the current resource request
   * @param timeout_in_ms - request timeout
   * @param min_resource_grant - min allowable resource request, calculated in caller
   * `request_resources_with_timeout` method from request_info
   * @param max_resource_grant - max (ideal) resource request, calculated in caller
   * `request_resources_with_timeout` method from request_info
   * @return RequestId - generated request id for this request
   */
  RequestId enqueue_request(const RequestInfo& request_info,
                            const size_t timeout_in_ms,
                            const ResourceGrant& min_resource_grant,
                            const ResourceGrant& max_resource_grant);

  /**
   * @brief Internal method: Moves the request from the `QUEUED` stage to `EXECUTING`
   * stage and performs other bookkeeping.
   *
   * Invoked by process_queue_loop after determing the next resource request
   * to serve (via `choose_next_request`).
   *
   * @param request_id - `RequestId` of the request being moved out of the request
   * queue (to be executed)
   */
  void mark_request_dequed(const RequestId request_id);

  /**
   * @brief Internal method: Called if the request times out (i.e. request was made via
   * `request_resources_with_timeout`), moves request out of `QUEUED` stage and does other
   * bookkeeping on the request's `RequestStats` and on `ExecutorStats`.
   *
   * Invoked from `request_resources_with_timeout` if a `QueryTimedOutWaitingInQueue`
   * exception is thrown
   *
   * @param request_id - `RequestId` for the resource request that has timed out
   */
  void mark_request_timed_out(const RequestId request_id);

  /**
   * @brief Internal method: Invoked on successful completion of a query step from
   * release_resources method, removes request from `EXECUTING` stage and performs various
   * bookkeeping, including recording execution and final times in request_stats
   *
   * @param request_id - `RequestId` for the resource request that has finished executing
   */
  void mark_request_finished(const RequestId request_id);

  /**
   * @brief Internal method: Invoked when we have `ResourceStat` error in the middle of
   * handling the request, removes request from `EXECUTING` stage and modify related
   * executor_stats
   *
   * @param request_id - `RequestId` for the resource request that has finished executing
   */
  void handle_resource_stat_error(const RequestId request_id);

  /**
   * @brief Internal method: Set the `should_process_queue_` flag to true, signifying that
   * the queue should be processed. Protected by a lock on `processor_queue_mutex_`.
   *
   * Invoked in two places:
   * 1) `release_resources` (as the resources have been returned to the pool, potentially
   * permitting granting of resources to another request)
   * 2) After `enqueue_request`, as we have a new request to evaluate whether
   * if it can be served
   */
  void set_process_queue_flag() {
    std::unique_lock<std::mutex> queue_lock(processor_queue_mutex_);
    should_process_queue_ = true;
  }

  /**
   * @brief Internal method: Invoked from `ExecutorResourceMgr` destructor, sets
   * `stop_process_queue_thread_` to true (behind a lock on `processor_queue_mutex_`) and
   * then attempts to join all threads left in the request queue on server shutdown.
   *
   */
  void stop_process_queue_thread();

  /**
   * @brief Internal method: Get the request ids for a given stage (`QUEUED` or
   * `EXECUTING`)
   *
   * Invoked from `choose_next_request` to get all outstanding queued requests
   *
   * @param request_status - request stage type to fetch request ids for
   * @return std::vector<RequestId> - vector of request ids in the requested stage
   */
  std::vector<RequestId> get_requests_for_stage(
      const ExecutionRequestStage request_status) const;

  /**
   * @brief Internal method: Adds the request specified by the provided `request_id`
   * to the specified stage
   *
   * @param request_id - Request id to add to the specified stage
   * @param request_status - Stage (`QUEUED` or `EXECUTING`) to add the specified request
   * to
   */
  void add_request_to_stage(const RequestId request_id,
                            const ExecutionRequestStage request_status);

  /**
   * @brief Internal method: Removes the request specified by the provided `request_id`
   * from the specified stage
   *
   * @param request_id - Request id to remove from the specified stage
   * @param request_status - Stage (`QUEUED` or `EXECUTING`) to remove the specified
   * request from
   */
  void remove_request_from_stage(const RequestId request_id,
                                 const ExecutionRequestStage request_status);
  /**
   * @brief Get the `DataMgr` chunk ids and associated sizes pertaining to the input data
   * needed by a request
   *
   * @param request_id - Request id to fetch the needed chunks for
   * @return ChunkRequestInfo - struct containing vector of `ChunkKey` and byte sizes as
   * well as device memory space (CPU or GPU), total bytes, etc
   */
  ChunkRequestInfo get_chunk_request_info(const RequestId request_id);

  /**
   * @brief Keeps track of available resources for execution
   *
   */
  ExecutorResourcePool executor_resource_pool_;

  /**
   * @brief An atomic that is incremented with each incoming request,
   * and used to assign `RequestIds` to incoming request
   */

  std::atomic<size_t> requests_count_{0};

  /**
   * @brief Holds a single `ExecutorStats` struct that pertains to cummulative
   * stats for `ExecutorResourceMgr`, i.e. number of requests, queue length,
   * total execution time, etc
   */
  ExecutorStats executor_stats_;

  const size_t ACTUALLY_QUEUED_MIN_MS{2};

  /**
   * @brief The thread started in the `ExecutorResourceMgr` constructor
   * that continuously loops inside of `process_queue_loop` to determine
   * the next resource request that should be granted
   *
   * When the database is stopped/shut down, join will be called on this
   * thread to clean up outstanding threads waiting on resource requests
   */
  std::thread process_queue_thread_;

  /**
   * @brief RW mutex that protects access to `stop_process_queue_thread_`
   * and `pause_processor_queue_`
   */
  mutable std::mutex processor_queue_mutex_;

  mutable std::mutex pause_processor_queue_mutex_;
  mutable std::mutex print_mutex_;

  /**
   * @brief RW mutex that protects access to `executor_stats_`
   * and `request_stats_`
   */
  mutable std::shared_mutex queue_stats_mutex_;

  /**
   * @brief RW mutex that protects access to `queued_requests_`
   */
  mutable std::shared_mutex queued_set_mutex_;

  /**
   * @brief RW mutex that protects access to `executing_requests_`
   */
  mutable std::shared_mutex executing_set_mutex_;

  std::condition_variable processor_queue_condition_;
  std::condition_variable pause_processor_queue_condition_;

  bool should_process_queue_{false};
  bool stop_process_queue_thread_{false};
  bool pause_process_queue_{false};
  bool process_queue_is_paused_{false};
  size_t process_queue_counter_{0};

  /**
   * @brief Stores and manages a map of request ids to `BinarySemaphore` objects
   * to allow threads waiting for resources to be selectively queued/blocked
   * and then when they are choosen for resource grants/execution, woken
   */
  OutstandingQueueRequests outstanding_queue_requests_;

  /**
   * @brief Set of all request ids that are currently queued. Protected by
   * `queued_set_mutex_`.
   */
  std::set<RequestId> queued_requests_;

  /**
   * @brief Set of all request ids that are currently executing (i.e.
   * post-granting of resources). Protected by `executing_set_mutex_`.
   */
  std::set<RequestId> executing_requests_;

  /**
   * @brief Stores a vector of all requests that have been seen
   * by `ExecutorResourceMgr`, with each incoming request appending
   * a `RequestStats` struct to this vector. Protected by `queue_stats_mutex_`.
   *
   * With a long-running server this vector could become quite long, but
   * the `RequestStats` objects are light enough where the total memory needed
   * should still be negligible compared to all the other things stored
   * in the server (even 100K requests would only total to a handful of MB).
   * The longer-term goal for this state is for
   * `ExecutorResourceMgr` to use it as historical data to optimize
   * query/request scheduling based on usage/request patterns.
   */
  std::vector<RequestStats> requests_stats_;

  const bool enable_stats_printing_{false};
  const bool enable_debug_printing_{false};

  const RequestId INVALID_REQUEST_ID{std::numeric_limits<size_t>::max()};

  const double max_available_resource_use_ratio_;
};

/**
 * @brief Convenience factory-esque method that allows us to use the
 * same logic to generate an `ExecutorResourceMgr` both internally
 * and for `ExecutorResourceMgr` tests
 */
std::shared_ptr<ExecutorResourceMgr> generate_executor_resource_mgr(
    const size_t num_cpu_slots,
    const size_t num_gpu_slots,
    const size_t cpu_result_mem,
    const bool use_cpu_mem_pool_for_output_buffers,
    const size_t cpu_buffer_pool_mem,
    const size_t gpu_buffer_pool_mem,
    const double per_query_max_cpu_slots_ratio,
    const double per_query_max_cpu_result_mem_ratio,
    const double per_query_max_pinned_cpu_buffer_pool_mem_ratio,
    const double per_query_max_pageable_cpu_buffer_pool_mem_ratio,
    const bool allow_cpu_kernel_concurrency,
    const bool allow_cpu_gpu_kernel_concurrency,
    const bool allow_cpu_slot_oversubscription_concurrency,
    const bool allow_gpu_slot_oversubscription,
    const bool allow_cpu_result_mem_oversubscription_concurrency,
    const double max_available_resource_use_ratio);

/**
 * @brief A wrapper returned by `ExecutorResourceMgr` to the requestee, containing
 * the ResourceGrant that was granted. When this class goes out of scope (i.e.
 * the executing thread finishes its query step, the destructor will automatically
 * instruct `ExecutorResourceMgr` to release the granted resources back to the
 * `ExecutorResourcePool`.
 */
class ExecutorResourceHandle {
 public:
  ExecutorResourceHandle(std::shared_ptr<ExecutorResourceMgr> resource_mgr,
                         const RequestId request_id,
                         const ResourceGrant& resource_grant)
      : resource_mgr_(resource_mgr)
      , request_id_(request_id)
      , resource_grant_(resource_grant) {}

  ~ExecutorResourceHandle() {
    resource_mgr_->release_resources(request_id_, resource_grant_);
  }

  inline RequestId get_request_id() const { return request_id_; }
  inline ResourceGrant get_resource_grant() const { return resource_grant_; }

 private:
  std::shared_ptr<ExecutorResourceMgr> resource_mgr_;
  const RequestId request_id_;
  const ResourceGrant resource_grant_;
};

}  // namespace ExecutorResourceMgr_Namespace
