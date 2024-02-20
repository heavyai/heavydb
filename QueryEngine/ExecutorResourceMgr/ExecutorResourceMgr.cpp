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

#include <iostream>
#include <thread>

#include "ExecutorResourceMgr.h"
#include "Logger/Logger.h"

static const char* REQUEST_STATS_ERROR_MSG_PREFIX = "RequestStats error: ";

namespace ExecutorResourceMgr_Namespace {

ExecutorResourceMgr::ExecutorResourceMgr(
    const std::vector<std::pair<ResourceType, size_t>>& total_resources,
    const std::vector<ConcurrentResourceGrantPolicy>& concurrent_resource_grant_policies,
    const std::vector<ResourceGrantPolicy>& max_per_request_resource_grant_policies,
    const double max_available_resource_use_ratio,
    const CPUResultMemResourceType cpu_result_memory_resource_type)
    : executor_resource_pool_(total_resources,
                              concurrent_resource_grant_policies,
                              max_per_request_resource_grant_policies,
                              cpu_result_memory_resource_type)
    , max_available_resource_use_ratio_(max_available_resource_use_ratio) {
  CHECK_GT(max_available_resource_use_ratio_, 0.0);
  CHECK_LE(max_available_resource_use_ratio_, 1.0);
  process_queue_thread_ = std::thread(&ExecutorResourceMgr::process_queue_loop, this);
  LOG(EXECUTOR) << "Executor Resource Manager queue processing thread started";
}

ExecutorResourceMgr::~ExecutorResourceMgr() {
  stop_process_queue_thread();
}

std::unique_ptr<ExecutorResourceHandle>
ExecutorResourceMgr::request_resources_with_timeout(const RequestInfo& request_info,
                                                    const size_t timeout_in_ms) {
  std::pair<ResourceGrant, ResourceGrant> min_max_resource_grants;

  // Following can throw
  // Should we put in stats to track errors?
  min_max_resource_grants =
      executor_resource_pool_.calc_min_max_resource_grants_for_request(request_info);

  const auto request_id = enqueue_request(request_info,
                                          timeout_in_ms,
                                          min_max_resource_grants.first,
                                          min_max_resource_grants.second);

  if (enable_debug_printing_) {
    std::unique_lock<std::mutex> print_lock(print_mutex_);
    std::cout << std::endl << "Min resource grant";
    min_max_resource_grants.first.print();
    std::cout << std::endl << "Max resource grant";
    min_max_resource_grants.second.print();
  }

  set_process_queue_flag();
  processor_queue_condition_.notify_one();

  // Following queue_request methods will block until ExecutorResourceMgr lets them
  // execute
  if (timeout_in_ms > 0) {
    try {
      outstanding_queue_requests_.queue_request_and_wait_with_timeout(request_id,
                                                                      timeout_in_ms);
    } catch (QueryTimedOutWaitingInQueue& timeout_exception) {
      // Need to annotate request and executor stats accordingly
      mark_request_timed_out(request_id);
      throw;
    }
  } else {
    outstanding_queue_requests_.queue_request_and_wait(request_id);
  }

  auto this_ptr = shared_from_this();
  std::shared_lock<std::shared_mutex> queue_stats_read_lock(queue_stats_mutex_);
  RequestStats const& request_stats = requests_stats_[request_id];
  if (request_stats.error) {
    // we throw `ExecutorResourceMgrError` to carry request_id
    // instead of std::runtime_error to revert ERM's status outside from this function
    // after releasing `queue_stats_mutex_` lock
    throw ExecutorResourceMgrError(request_id,
                                   REQUEST_STATS_ERROR_MSG_PREFIX + *request_stats.error);
  }
  const ResourceGrant& actual_resource_grant = request_stats.actual_resource_grant;
  // Ensure each resource granted was at least the minimum requested
  CHECK_GE(actual_resource_grant.cpu_slots, min_max_resource_grants.first.cpu_slots);
  CHECK_GE(actual_resource_grant.gpu_slots, min_max_resource_grants.first.gpu_slots);
  CHECK_GE(actual_resource_grant.cpu_result_mem,
           min_max_resource_grants.first.cpu_result_mem);
  return std::make_unique<ExecutorResourceHandle>(
      this_ptr, request_id, actual_resource_grant);
}

std::unique_ptr<ExecutorResourceHandle> ExecutorResourceMgr::request_resources(
    const RequestInfo& request_info) {
  try {
    return request_resources_with_timeout(
        request_info,
        static_cast<size_t>(0));  // 0 signifies no timeout
  } catch (ExecutorResourceMgrError const& e) {
    if (e.getErrorMsg().find(REQUEST_STATS_ERROR_MSG_PREFIX) == 0u) {
      handle_resource_stat_error(e.getRequestId());
    }
    throw std::runtime_error(e.getErrorMsg());
  }
}

void ExecutorResourceMgr::release_resources(const RequestId request_id,
                                            const ResourceGrant& resource_grant) {
  if (!resource_grant.is_empty()) {  // Should only be empty if request times out, should
                                     // we CHECK for this
    const auto chunk_request_info = get_chunk_request_info(request_id);
    executor_resource_pool_.deallocate_resources(resource_grant, chunk_request_info);
  }
  mark_request_finished(request_id);
  set_process_queue_flag();
  processor_queue_condition_.notify_one();
}

RequestStats ExecutorResourceMgr::get_request_for_id(const RequestId request_id) const {
  std::shared_lock<std::shared_mutex> queue_stats_read_lock(queue_stats_mutex_);
  CHECK_LT(request_id, requests_stats_.size());
  return requests_stats_[request_id];
}

void ExecutorResourceMgr::mark_request_error(const RequestId request_id,
                                             std::string error_msg) {
  std::unique_lock<std::shared_mutex> queue_stats_write_lock(queue_stats_mutex_);
  CHECK_LT(request_id, requests_stats_.size());
  requests_stats_[request_id].error = std::move(error_msg);
}

RequestId ExecutorResourceMgr::choose_next_request() {
  const auto request_ids = get_requests_for_stage(ExecutionRequestStage::QUEUED);
  LOG(EXECUTOR) << "ExecutorResourceMgr Queue Itr: " << process_queue_counter_ - 1
                << " Queued requests: " << request_ids.size();
  std::unique_lock<std::shared_mutex> queue_stats_lock(queue_stats_mutex_);
  for (const auto request_id : request_ids) {
    auto& request_stats = requests_stats_[request_id];
    try {
      const auto actual_resource_grant =
          executor_resource_pool_.determine_dynamic_resource_grant(
              request_stats.min_resource_grant,
              request_stats.max_resource_grant,
              request_stats.request_info.chunk_request_info,
              max_available_resource_use_ratio_);
      // boolean sentinel first member of returned pair says whether
      // a resource grant was able to be made at all
      if (actual_resource_grant.first) {
        request_stats.actual_resource_grant = actual_resource_grant.second;
        LOG(EXECUTOR) << "ExecutorResourceMgr Queue chosen request ID: " << request_id
                      << " from " << request_ids.size() << " queued requests.";
        LOG(EXECUTOR) << "Request grant: " << actual_resource_grant.second.to_string();
        if (enable_debug_printing_) {
          std::unique_lock<std::mutex> print_lock(print_mutex_);
          std::cout << std::endl << "Actual grant";
          actual_resource_grant.second.print();
        }
        return request_id;
      }
    } catch (std::runtime_error const& e) {
      throw ExecutorResourceMgrError(request_id, e.what());
    }
  }
  return INVALID_REQUEST_ID;
}

ExecutorStats ExecutorResourceMgr::get_executor_stats() const {
  std::shared_lock<std::shared_mutex> queue_stats_read_lock(queue_stats_mutex_);
  return executor_stats_;  // Will make copy
}

void ExecutorResourceMgr::print_executor_stats() const {
  // Get atomic copy of executor_stats_ first
  const auto executor_stats = get_executor_stats();
  std::unique_lock<std::mutex> print_lock(print_mutex_);
  std::cout << std::endl << "Executor Stats" << std::endl;
  std::cout << "Requests: " << executor_stats.requests << std::endl;
  std::cout << "CPU Requests: " << executor_stats.cpu_requests << std::endl;
  std::cout << "GPU Requests: " << executor_stats.gpu_requests << std::endl;
  std::cout << "Queue Length: " << executor_stats.queue_length << std::endl;
  std::cout << "CPU Queue Length: " << executor_stats.cpu_queue_length << std::endl;
  std::cout << "GPU Queue Length: " << executor_stats.gpu_queue_length << std::endl;
  std::cout << "Total Queue Time(ms): " << executor_stats.total_queue_time_ms
            << std::endl;
  std::cout << "Total CPU Queue Time(ms): " << executor_stats.total_cpu_queue_time_ms
            << std::endl;
  std::cout << "Total GPU Queue Time(ms): " << executor_stats.total_gpu_queue_time_ms
            << std::endl;
  std::cout << "Requests Actually Queued: " << executor_stats.requests_actually_queued
            << std::endl;
  std::cout << "Requests Executing: " << executor_stats.requests_executing << std::endl;
  std::cout << "Requests Executed: " << executor_stats.requests_executed << std::endl;
  std::cout << "Total Execution Time(ms): " << executor_stats.total_execution_time_ms
            << std::endl;
  std::cout << "Total CPU Execution Time(ms): "
            << executor_stats.total_cpu_execution_time_ms << std::endl;
  std::cout << "Total GPU Execution Time(ms): "
            << executor_stats.total_gpu_execution_time_ms << std::endl;
  std::cout << "Total Time(ms): " << executor_stats.total_time_ms << std::endl;
  std::cout << "Total CPU Time(ms): " << executor_stats.total_cpu_time_ms << std::endl;
  std::cout << "Total GPU Time(ms): " << executor_stats.total_gpu_time_ms << std::endl;

  // Below technically not thread safe, but called from process_queue_loop for now so ok

  const double avg_execution_time_ms =
      executor_stats.total_execution_time_ms /
      std::max(executor_stats.requests_executed, size_t(1));
  const double avg_cpu_execution_time_ms =
      executor_stats.total_cpu_execution_time_ms /
      std::max(executor_stats.cpu_requests_executed, size_t(1));
  const double avg_gpu_execution_time_ms =
      executor_stats.total_gpu_execution_time_ms /
      std::max(executor_stats.gpu_requests_executed, size_t(1));
  const double avg_total_time_ms = executor_stats.total_time_ms /
                                   std::max(executor_stats.requests_executed, size_t(1));
  const double avg_cpu_total_time_ms =
      executor_stats.total_cpu_time_ms /
      std::max(executor_stats.cpu_requests_executed, size_t(1));
  const double avg_gpu_total_time_ms =
      executor_stats.total_gpu_time_ms /
      std::max(executor_stats.gpu_requests_executed, size_t(1));

  std::cout << "Avg Execution Time(ms): " << avg_execution_time_ms << std::endl;
  std::cout << "Avg CPU Execution Time(ms): " << avg_cpu_execution_time_ms << std::endl;
  std::cout << "Avg GPU Execution Time(ms): " << avg_gpu_execution_time_ms << std::endl;

  std::cout << "Avg Total Time(ms): " << avg_total_time_ms << std::endl;
  std::cout << "Avg CPU Total Time(ms): " << avg_cpu_total_time_ms << std::endl;
  std::cout << "Avg GPU Total Time(ms): " << avg_gpu_total_time_ms << std::endl;

  std::cout << "Process queue loop counter: " << process_queue_counter_ << std::endl
            << std::endl;
}

void ExecutorResourceMgr::stop_process_queue_thread() {
  {
    std::unique_lock<std::mutex> queue_lock(processor_queue_mutex_);
    stop_process_queue_thread_ = true;
  }
  processor_queue_condition_.notify_one();
  process_queue_thread_.join();
}

void ExecutorResourceMgr::pause_process_queue() {
  {
    std::unique_lock<std::mutex> queue_lock(processor_queue_mutex_);
    if (pause_process_queue_ || process_queue_is_paused_) {  // Was already true, abort
      LOG(INFO)
          << "Pause of ExecutorResourceMgr queue was called, but was already paused. "
             "Taking no action.";
      return;
    }
    pause_process_queue_ = true;
  }
  processor_queue_condition_.notify_one();

  std::unique_lock<std::mutex> pause_queue_lock(pause_processor_queue_mutex_);
  pause_processor_queue_condition_.wait(pause_queue_lock,
                                        [=] { return process_queue_is_paused_; });

  CHECK_EQ(executor_stats_.requests_executing, size_t(0));
}

void ExecutorResourceMgr::resume_process_queue() {
  {
    std::unique_lock<std::mutex> queue_lock(processor_queue_mutex_);
    if (!process_queue_is_paused_) {
      LOG(INFO)
          << "Resume of ExecutorResourceMgr queue was called, but was not paused. Taking "
             "no action.";
      return;
    }
    CHECK_EQ(executor_stats_.requests_executing, size_t(0));
    process_queue_is_paused_ = false;
    pause_process_queue_ = false;
    should_process_queue_ = true;
  }
  processor_queue_condition_.notify_one();
}

void ExecutorResourceMgr::set_resource(const ResourceType resource_type,
                                       const size_t resource_quantity) {
  pause_process_queue();
  CHECK_EQ(get_resource_info(resource_type).first, size_t(0));
  executor_resource_pool_.set_resource(resource_type, resource_quantity);
  const auto resource_info = get_resource_info(resource_type);
  CHECK_EQ(resource_info.first, size_t(0));
  CHECK_EQ(resource_info.second, resource_quantity);
  resume_process_queue();
}

ConcurrentResourceGrantPolicy ExecutorResourceMgr::get_concurrent_resource_grant_policy(
    const ResourceType resource_type) const {
  return executor_resource_pool_.get_concurrent_resource_grant_policy(resource_type);
}

void ExecutorResourceMgr::set_concurrent_resource_grant_policy(
    const ConcurrentResourceGrantPolicy& concurrent_resource_grant_policy) {
  pause_process_queue();
  executor_resource_pool_.set_concurrent_resource_grant_policy(
      concurrent_resource_grant_policy);
  const auto applied_concurrent_resource_grant_policy =
      executor_resource_pool_.get_concurrent_resource_grant_policy(
          concurrent_resource_grant_policy.resource_type);
  CHECK(concurrent_resource_grant_policy.concurrency_policy ==
        applied_concurrent_resource_grant_policy.concurrency_policy);
  CHECK(concurrent_resource_grant_policy.oversubscription_concurrency_policy ==
        applied_concurrent_resource_grant_policy.oversubscription_concurrency_policy);
  resume_process_queue();
}

void ExecutorResourceMgr::process_queue_loop() {
  const size_t min_ms_between_print_stats{5000};  // 5 sec
  if (enable_stats_printing_) {
    print_executor_stats();
  }
  std::chrono::steady_clock::time_point last_print_time =
      std::chrono::steady_clock::now();
  while (true) {
    std::unique_lock<std::mutex> queue_lock(processor_queue_mutex_);
    processor_queue_condition_.wait(queue_lock, [=] {
      return should_process_queue_ || stop_process_queue_thread_ || pause_process_queue_;
    });
    // Use the following flag to know when to exit
    // (to prevent leaving this thread dangling at server shutdown)
    if (stop_process_queue_thread_) {
      should_process_queue_ =
          false;  // not strictly neccessary, but would be if we add threads
      return;
    }

    if (pause_process_queue_) {
      should_process_queue_ = false;
      if (executor_stats_.requests_executing == 0) {
        {
          std::unique_lock<std::mutex> pause_queue_lock(pause_processor_queue_mutex_);
          process_queue_is_paused_ = true;
        }
        pause_processor_queue_condition_.notify_one();
      }
      continue;
    }

    process_queue_counter_++;
    RequestId chosen_request_id;
    try {
      chosen_request_id = choose_next_request();
    } catch (ExecutorResourceMgrError const& e) {
      chosen_request_id = e.getRequestId();
      mark_request_error(chosen_request_id, e.getErrorMsg());
    }
    if (enable_debug_printing_) {
      std::unique_lock<std::mutex> print_lock(print_mutex_);
      std::cout << "Process loop iteration: " << process_queue_counter_ - 1 << std::endl;
      std::cout << "Process loop chosen request_id: " << chosen_request_id << std::endl;
    }
    if (chosen_request_id == INVALID_REQUEST_ID) {
      // Means no query was found that could be currently run
      // Below is safe as we hold an exclusive lock on processor_queue_mutex_
      should_process_queue_ = false;
      continue;
    }
    // If here we have a valid request id
    mark_request_dequed(chosen_request_id);
    const auto request_stats = get_request_for_id(chosen_request_id);
    if (!request_stats.error) {
      executor_resource_pool_.allocate_resources(
          request_stats.actual_resource_grant,
          request_stats.request_info.chunk_request_info);
    }
    outstanding_queue_requests_.wake_request_by_id(chosen_request_id);

    if (enable_stats_printing_) {
      std::chrono::steady_clock::time_point current_time =
          std::chrono::steady_clock::now();
      const size_t ms_since_last_print_stats =
          std::chrono::duration_cast<std::chrono::milliseconds>(current_time -
                                                                last_print_time)
              .count();
      if (ms_since_last_print_stats >= min_ms_between_print_stats) {
        print_executor_stats();
        last_print_time = current_time;
      }
    }
    // Leave should_process_queue_ as true to see if we can allocate resources for another
    // request
  }
}

RequestId ExecutorResourceMgr::enqueue_request(const RequestInfo& request_info,
                                               const size_t timeout_in_ms,
                                               const ResourceGrant& min_resource_grant,
                                               const ResourceGrant& max_resource_grant) {
  const std::chrono::steady_clock::time_point enqueue_time =
      std::chrono::steady_clock::now();
  std::unique_lock<std::shared_mutex> queue_stats_write_lock(queue_stats_mutex_);
  const RequestId request_id = requests_count_.fetch_add(1, std::memory_order_relaxed);
  executor_stats_.requests++;
  if (timeout_in_ms > 0) {
    executor_stats_.requests_with_timeouts++;
  }
  const size_t queue_length_at_entry = executor_stats_.queue_length++;
  executor_stats_.sum_queue_size_at_entry += queue_length_at_entry;
  size_t device_type_queue_length_at_entry{0};
  switch (request_info.request_device_type) {
    case ExecutorDeviceType::CPU: {
      executor_stats_.cpu_requests++;
      device_type_queue_length_at_entry = executor_stats_.cpu_queue_length++;
      executor_stats_.sum_cpu_queue_size_at_entry += device_type_queue_length_at_entry;
      break;
    }
    case ExecutorDeviceType::GPU: {
      executor_stats_.gpu_requests++;
      device_type_queue_length_at_entry = executor_stats_.gpu_queue_length++;
      executor_stats_.sum_gpu_queue_size_at_entry += device_type_queue_length_at_entry;
      break;
    }
    default:
      UNREACHABLE();
  }

  requests_stats_.emplace_back(RequestStats(request_id,
                                            request_info,
                                            min_resource_grant,
                                            max_resource_grant,
                                            enqueue_time,
                                            queue_length_at_entry,
                                            device_type_queue_length_at_entry,
                                            timeout_in_ms));
  add_request_to_stage(request_id, ExecutionRequestStage::QUEUED);
  return request_id;
}

void ExecutorResourceMgr::mark_request_dequed(const RequestId request_id) {
  const std::chrono::steady_clock::time_point deque_time =
      std::chrono::steady_clock::now();
  // Below is only to CHECK our request_id against high water mark... should be
  // relatively inexpensive though
  const size_t current_request_count = requests_count_.load(std::memory_order_relaxed);
  CHECK_LT(request_id, current_request_count);
  {
    std::unique_lock<std::shared_mutex> queue_stats_write_lock(queue_stats_mutex_);
    RequestStats& request_stats = requests_stats_[request_id];
    request_stats.deque_time = deque_time;
    request_stats.finished_queueing = true;
    request_stats.queue_time_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(request_stats.deque_time -
                                                              request_stats.enqueue_time)
            .count();
  }
  remove_request_from_stage(request_id, ExecutionRequestStage::QUEUED);
  add_request_to_stage(request_id, ExecutionRequestStage::EXECUTING);

  std::shared_lock<std::shared_mutex> queue_stats_read_lock(queue_stats_mutex_);
  const RequestStats& request_stats = requests_stats_[request_id];
  executor_stats_.queue_length--;
  executor_stats_.requests_executing++;
  if (request_stats.queue_time_ms <= ACTUALLY_QUEUED_MIN_MS) {
    executor_stats_.total_queue_time_ms += request_stats.queue_time_ms;
    executor_stats_.requests_actually_queued++;
  }
  switch (request_stats.request_info.request_device_type) {
    case ExecutorDeviceType::CPU:
      executor_stats_.cpu_queue_length--;
      executor_stats_.cpu_requests_executing++;
      if (request_stats.queue_time_ms <= ACTUALLY_QUEUED_MIN_MS) {
        executor_stats_.total_cpu_queue_time_ms += request_stats.queue_time_ms;
        executor_stats_.cpu_requests_actually_queued++;
      }
      break;
    case ExecutorDeviceType::GPU:
      executor_stats_.gpu_queue_length--;
      executor_stats_.gpu_requests_executing++;
      if (request_stats.queue_time_ms <= ACTUALLY_QUEUED_MIN_MS) {
        executor_stats_.total_gpu_queue_time_ms += request_stats.queue_time_ms;
        executor_stats_.gpu_requests_actually_queued++;
      }
      break;
    default:
      UNREACHABLE();
  }
}

void ExecutorResourceMgr::mark_request_timed_out(const RequestId request_id) {
  const size_t current_request_count = requests_count_.load(std::memory_order_relaxed);
  CHECK_LT(request_id, current_request_count);
  {
    std::unique_lock<std::shared_mutex> queue_stats_write_lock(queue_stats_mutex_);
    RequestStats& request_stats = requests_stats_[request_id];
    CHECK(!request_stats.finished_queueing);
    CHECK_GT(request_stats.timeout_in_ms, size_t(0));
    request_stats.timed_out = true;
  }
  remove_request_from_stage(request_id, ExecutionRequestStage::QUEUED);
  std::shared_lock<std::shared_mutex> queue_stats_read_lock(queue_stats_mutex_);
  const RequestStats& request_stats = requests_stats_[request_id];
  CHECK_GT(executor_stats_.queue_length, size_t(0));
  executor_stats_.queue_length--;
  executor_stats_.requests_timed_out++;
  switch (request_stats.request_info.request_device_type) {
    case ExecutorDeviceType::CPU: {
      CHECK_GT(executor_stats_.cpu_queue_length, size_t(0));
      executor_stats_.cpu_queue_length--;
      break;
    }
    case ExecutorDeviceType::GPU: {
      CHECK_GT(executor_stats_.gpu_queue_length, size_t(0));
      executor_stats_.gpu_queue_length--;
      break;
    }
    default:
      UNREACHABLE();
  }
}

void ExecutorResourceMgr::handle_resource_stat_error(const RequestId request_id) {
  const std::chrono::steady_clock::time_point execution_finished_time =
      std::chrono::steady_clock::now();
  const size_t current_request_count = requests_count_.load(std::memory_order_relaxed);
  CHECK_LT(request_id, current_request_count);
  std::unique_lock<std::shared_mutex> queue_stats_write_lock(queue_stats_mutex_);
  RequestStats& request_stats = requests_stats_[request_id];
  request_stats.execution_finished_time = execution_finished_time;
  request_stats.finished_executing = true;
  request_stats.execution_time_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          request_stats.execution_finished_time - request_stats.deque_time)
          .count();
  request_stats.total_time_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          request_stats.execution_finished_time - request_stats.enqueue_time)
          .count();
  remove_request_from_stage(request_id, ExecutionRequestStage::EXECUTING);

  executor_stats_.requests_executing--;
  executor_stats_.requests_executed++;
  executor_stats_.total_execution_time_ms += request_stats.execution_time_ms;
  executor_stats_.total_time_ms += request_stats.total_time_ms;
}

void ExecutorResourceMgr::mark_request_finished(const RequestId request_id) {
  const std::chrono::steady_clock::time_point execution_finished_time =
      std::chrono::steady_clock::now();
  // Below is only to CHECK our request_id against high water mark... should be
  // relatively inexpensive though
  const size_t current_request_count = requests_count_.load(std::memory_order_relaxed);
  CHECK_LT(request_id, current_request_count);
  std::unique_lock<std::shared_mutex> queue_stats_write_lock(queue_stats_mutex_);
  RequestStats& request_stats = requests_stats_[request_id];
  request_stats.execution_finished_time = execution_finished_time;
  request_stats.finished_executing = true;
  request_stats.execution_time_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          request_stats.execution_finished_time - request_stats.deque_time)
          .count();
  request_stats.total_time_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          request_stats.execution_finished_time - request_stats.enqueue_time)
          .count();
  remove_request_from_stage(request_id, ExecutionRequestStage::EXECUTING);

  executor_stats_.requests_executing--;
  executor_stats_.requests_executed++;
  executor_stats_.total_execution_time_ms += request_stats.execution_time_ms;
  executor_stats_.total_time_ms += request_stats.total_time_ms;
  switch (request_stats.request_info.request_device_type) {
    case ExecutorDeviceType::CPU: {
      executor_stats_.cpu_requests_executing--;
      executor_stats_.cpu_requests_executed++;
      executor_stats_.total_cpu_execution_time_ms += request_stats.execution_time_ms;
      executor_stats_.total_cpu_time_ms += request_stats.total_time_ms;
      break;
    }
    case ExecutorDeviceType::GPU: {
      executor_stats_.gpu_requests_executing--;
      executor_stats_.gpu_requests_executed++;
      executor_stats_.total_gpu_execution_time_ms += request_stats.execution_time_ms;
      executor_stats_.total_gpu_time_ms += request_stats.total_time_ms;
      break;
    }
    default:
      UNREACHABLE();
  }
}

std::vector<RequestId> ExecutorResourceMgr::get_requests_for_stage(
    const ExecutionRequestStage request_stage) const {
  auto& chosen_set = request_stage == ExecutionRequestStage::QUEUED ? queued_requests_
                                                                    : executing_requests_;
  auto& chosen_mutex = request_stage == ExecutionRequestStage::QUEUED
                           ? queued_set_mutex_
                           : executing_set_mutex_;
  std::shared_lock<std::shared_mutex> set_read_lock(chosen_mutex);

  const std::vector<RequestId> request_ids_for_stage(chosen_set.begin(),
                                                     chosen_set.end());
  return request_ids_for_stage;
}

void ExecutorResourceMgr::add_request_to_stage(
    const RequestId request_id,
    const ExecutionRequestStage request_stage) {
  auto& chosen_set = request_stage == ExecutionRequestStage::QUEUED ? queued_requests_
                                                                    : executing_requests_;
  auto& chosen_mutex = request_stage == ExecutionRequestStage::QUEUED
                           ? queued_set_mutex_
                           : executing_set_mutex_;
  std::unique_lock<std::shared_mutex> set_write_lock(chosen_mutex);

  CHECK(chosen_set.insert(request_id)
            .second);  // Should return true as element should not exist in set
}

void ExecutorResourceMgr::remove_request_from_stage(
    const RequestId request_id,
    const ExecutionRequestStage request_stage) {
  auto& chosen_set = request_stage == ExecutionRequestStage::QUEUED ? queued_requests_
                                                                    : executing_requests_;
  auto& chosen_mutex = request_stage == ExecutionRequestStage::QUEUED
                           ? queued_set_mutex_
                           : executing_set_mutex_;
  std::unique_lock<std::shared_mutex> set_write_lock(chosen_mutex);

  CHECK_EQ(chosen_set.erase(request_id),
           size_t(1));  // Should return 1 as element must be in set
}

ChunkRequestInfo ExecutorResourceMgr::get_chunk_request_info(const RequestId request_id) {
  std::shared_lock<std::shared_mutex> queue_stats_read_lock(queue_stats_mutex_);
  return requests_stats_[request_id].request_info.chunk_request_info;
}

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
    const double max_available_resource_use_ratio) {
  CHECK_GT(num_cpu_slots, size_t(0));
  const auto cpu_result_mem_resource_type =
      use_cpu_mem_pool_for_output_buffers
          ? CPUResultMemResourceType{ResourceType::CPU_BUFFER_POOL_MEM,
                                     ResourceSubtype::PINNED_CPU_BUFFER_POOL_MEM}
          : CPUResultMemResourceType{ResourceType::CPU_RESULT_MEM,
                                     ResourceSubtype::CPU_RESULT_MEM};
  const size_t cpu_result_mem_bytes =
      use_cpu_mem_pool_for_output_buffers ? 0u : cpu_result_mem;
  if (!use_cpu_mem_pool_for_output_buffers) {
    CHECK_GT(cpu_result_mem_bytes, 0u);
  }
  CHECK_GT(cpu_buffer_pool_mem, size_t(0));
  CHECK_GT(per_query_max_cpu_slots_ratio, size_t(0));
  CHECK_EQ(!(allow_cpu_kernel_concurrency || allow_cpu_gpu_kernel_concurrency) &&
               allow_cpu_slot_oversubscription_concurrency,
           false);
  CHECK_EQ(!(allow_cpu_kernel_concurrency || allow_cpu_gpu_kernel_concurrency) &&
               allow_cpu_result_mem_oversubscription_concurrency,
           false);
  CHECK_GT(max_available_resource_use_ratio, 0.0);
  CHECK_LE(max_available_resource_use_ratio, 1.0);

  const std::vector<std::pair<ResourceType, size_t>> total_resources = {
      std::make_pair(ResourceType::CPU_SLOTS, num_cpu_slots),
      std::make_pair(ResourceType::GPU_SLOTS, num_gpu_slots),
      std::make_pair(ResourceType::CPU_RESULT_MEM, cpu_result_mem_bytes),
      std::make_pair(ResourceType::CPU_BUFFER_POOL_MEM, cpu_buffer_pool_mem),
      std::make_pair(ResourceType::GPU_BUFFER_POOL_MEM, gpu_buffer_pool_mem)};

  const auto max_per_request_cpu_slots_grant_policy = gen_ratio_resource_grant_policy(
      ResourceSubtype::CPU_SLOTS, per_query_max_cpu_slots_ratio);

  // Use unlimited policy for now as some GPU query plans can need more kernels than gpus
  const auto max_per_request_gpu_slots_grant_policy =
      gen_unlimited_resource_grant_policy(ResourceSubtype::GPU_SLOTS);
  const auto max_per_request_cpu_result_mem_grant_policy =
      gen_ratio_resource_grant_policy(cpu_result_mem_resource_type.resource_subtype,
                                      per_query_max_cpu_result_mem_ratio);

  const auto max_per_request_pinned_cpu_buffer_pool_mem =
      gen_ratio_resource_grant_policy(ResourceSubtype::PINNED_CPU_BUFFER_POOL_MEM,
                                      per_query_max_pinned_cpu_buffer_pool_mem_ratio);
  const auto max_per_request_pageable_cpu_buffer_pool_mem =
      gen_ratio_resource_grant_policy(ResourceSubtype::PAGEABLE_CPU_BUFFER_POOL_MEM,
                                      per_query_max_pageable_cpu_buffer_pool_mem_ratio);

  const std::vector<ResourceGrantPolicy> max_per_request_resource_grant_policies = {
      max_per_request_cpu_slots_grant_policy,
      max_per_request_gpu_slots_grant_policy,
      max_per_request_cpu_result_mem_grant_policy,
      max_per_request_pinned_cpu_buffer_pool_mem,
      max_per_request_pageable_cpu_buffer_pool_mem};

  const auto cpu_slots_undersubscription_concurrency_policy =
      allow_cpu_kernel_concurrency ? ResourceConcurrencyPolicy::ALLOW_CONCURRENT_REQUESTS
                                   : ResourceConcurrencyPolicy::ALLOW_SINGLE_REQUEST;
  // Whether a single query can oversubscribe CPU slots should be controlled with
  // per_query_max_cpu_slots_ratio
  const auto cpu_slots_oversubscription_concurrency_policy =
      allow_cpu_slot_oversubscription_concurrency
          ? ResourceConcurrencyPolicy::ALLOW_CONCURRENT_REQUESTS
          : ResourceConcurrencyPolicy::ALLOW_SINGLE_REQUEST;
  const auto gpu_slots_undersubscription_concurrency_policy =
      allow_cpu_gpu_kernel_concurrency
          ? ResourceConcurrencyPolicy::ALLOW_SINGLE_REQUEST
          : ResourceConcurrencyPolicy::ALLOW_SINGLE_REQUEST_GLOBALLY;
  const auto gpu_slots_oversubscription_concurrency_policy =
      !allow_gpu_slot_oversubscription
          ? ResourceConcurrencyPolicy::DISALLOW_REQUESTS
          : (allow_cpu_gpu_kernel_concurrency
                 ? ResourceConcurrencyPolicy::ALLOW_SINGLE_REQUEST
                 : ResourceConcurrencyPolicy::ALLOW_SINGLE_REQUEST_GLOBALLY);

  // Whether a single query can oversubscribe CPU memory should be controlled with
  // per_query_max_cpu_result_mem_ratio
  const auto cpu_result_mem_oversubscription_concurrency_policy =
      allow_cpu_result_mem_oversubscription_concurrency
          ? ResourceConcurrencyPolicy::ALLOW_CONCURRENT_REQUESTS
          : ResourceConcurrencyPolicy::ALLOW_SINGLE_REQUEST;

  const auto concurrent_cpu_slots_grant_policy =
      ConcurrentResourceGrantPolicy(ResourceType::CPU_SLOTS,
                                    cpu_slots_undersubscription_concurrency_policy,
                                    cpu_slots_oversubscription_concurrency_policy);
  const ConcurrentResourceGrantPolicy concurrent_gpu_slots_grant_policy(
      ResourceType::GPU_SLOTS,
      gpu_slots_undersubscription_concurrency_policy,
      gpu_slots_oversubscription_concurrency_policy);

  const auto concurrent_cpu_result_mem_grant_policy =
      ConcurrentResourceGrantPolicy(cpu_result_mem_resource_type.resource_type,
                                    ResourceConcurrencyPolicy::ALLOW_CONCURRENT_REQUESTS,
                                    cpu_result_mem_oversubscription_concurrency_policy);

  const std::vector<ConcurrentResourceGrantPolicy> concurrent_resource_grant_policies{
      concurrent_cpu_slots_grant_policy,
      concurrent_gpu_slots_grant_policy,
      concurrent_cpu_result_mem_grant_policy};

  return std::make_shared<ExecutorResourceMgr>(total_resources,
                                               concurrent_resource_grant_policies,
                                               max_per_request_resource_grant_policies,
                                               max_available_resource_use_ratio,
                                               cpu_result_mem_resource_type);
}

}  // namespace ExecutorResourceMgr_Namespace
