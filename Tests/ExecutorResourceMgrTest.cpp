/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <future>
#include <thread>

#include "../QueryEngine/ExecutorResourceMgr/ExecutorResourceMgr.h"
#include "../Shared/scope.h"
#include "TestHelpers.h"

using namespace ExecutorResourceMgr_Namespace;

extern bool g_executor_resource_mgr_allow_auto_shrink_num_cpu_slot_for_groupby_query;

const size_t default_cpu_slots{16};
const size_t default_gpu_slots{2};
const size_t default_cpu_result_mem{1UL << 34};       // 16GB
const size_t default_cpu_buffer_pool_mem{1UL << 37};  // 128GB
const size_t default_gpu_buffer_pool_mem{1UL << 35};  // 32GB
const double default_per_query_max_cpu_slots_ratio{0.8};
const double default_per_query_max_cpu_result_mem_ratio{0.8};
const double default_per_query_max_pinned_cpu_buffer_pool_mem_ratio{0.75};
const double default_per_query_max_pageable_cpu_buffer_pool_mem_ratio{0.5};
const bool default_allow_cpu_kernel_concurrency{true};
const bool default_allow_cpu_gpu_kernel_concurrency{true};
const bool default_allow_cpu_slot_oversubscription_concurrency{false};
const bool default_allow_gpu_slot_oversubscription{false};
const bool default_allow_cpu_result_mem_oversubscription_concurrency{false};
const double default_max_available_resource_use_ratio{0.8};

const size_t default_priority_level_request{0};
const size_t default_cpu_slots_request{4};
const size_t default_min_cpu_slots_request{2};
const size_t default_gpu_slots_request{0};
const size_t default_min_gpu_slots_request{0};
const size_t default_cpu_result_mem_request{1UL << 30};      // 1GB
const size_t default_min_cpu_result_mem_request{1UL << 28};  // 256MB

const ChunkRequestInfo default_chunk_request_info;  // Empty request info to satisfy
                                                    // request_resources signature
const bool default_output_buffers_reusable_intra_thread{false};

const size_t ZERO_SIZE{0UL};

const char* QUERY_NEEDS_TOO_MANY_CPU_SLOTS_ERR_MSG = "Query requested more CPU slots";
const char* QUERY_NEEDS_TOO_MANY_GPU_SLOTS_ERR_MSG = "Query requested more GPU slots";
const char* QUERY_NEEDS_TOO_MUCH_CPU_RESULT_MEM_ERR_MSG =
    "Query requested more CPU result memory";
const char* QUERY_NEEDS_TOO_MUCH_CPU_BUFFER_POOP_MEM_ERR_MSG =
    "Query requested more CPU buffer pool mem";
const char* QUERY_NEEDS_TOO_MUCH_GPU_BUFFER_POOP_MEM_ERR_MSG =
    "Query requested more GPU buffer pool mem";

std::shared_ptr<ExecutorResourceMgr> gen_resource_mgr_with_defaults(
    bool enable_cpu_buffer_pool = false) {
  return generate_executor_resource_mgr(
      default_cpu_slots,
      default_gpu_slots,
      default_cpu_result_mem,
      enable_cpu_buffer_pool,
      default_cpu_buffer_pool_mem,
      default_gpu_buffer_pool_mem,
      default_per_query_max_cpu_slots_ratio,
      default_per_query_max_cpu_result_mem_ratio,
      default_per_query_max_pinned_cpu_buffer_pool_mem_ratio,
      default_per_query_max_pageable_cpu_buffer_pool_mem_ratio,
      default_allow_cpu_kernel_concurrency,
      default_allow_cpu_gpu_kernel_concurrency,
      default_allow_cpu_slot_oversubscription_concurrency,
      default_allow_gpu_slot_oversubscription,
      default_allow_cpu_result_mem_oversubscription_concurrency,
      default_max_available_resource_use_ratio);
}

// Pool-backed output buffers (the production default), where CPU result memory and
// pinned input chunks both draw on the CPU buffer pool
std::shared_ptr<ExecutorResourceMgr> gen_pool_backed_resource_mgr(
    const size_t cpu_buffer_pool_mem,
    const size_t num_cpu_slots = default_cpu_slots) {
  return generate_executor_resource_mgr(
      num_cpu_slots,
      default_gpu_slots,
      0u,  // result memory is drawn from the cpu buffer pool instead
      true,
      cpu_buffer_pool_mem,
      default_gpu_buffer_pool_mem,
      default_per_query_max_cpu_slots_ratio,
      default_per_query_max_cpu_result_mem_ratio,
      default_per_query_max_pinned_cpu_buffer_pool_mem_ratio,
      default_per_query_max_pageable_cpu_buffer_pool_mem_ratio,
      default_allow_cpu_kernel_concurrency,
      default_allow_cpu_gpu_kernel_concurrency,
      default_allow_cpu_slot_oversubscription_concurrency,
      default_allow_gpu_slot_oversubscription,
      default_allow_cpu_result_mem_oversubscription_concurrency,
      default_max_available_resource_use_ratio);
}

ChunkRequestInfo gen_cpu_chunk_request_info(const size_t num_chunks,
                                            const size_t bytes_per_chunk,
                                            const bool bytes_scales_per_kernel) {
  ChunkRequestInfo chunk_request_info;
  chunk_request_info.device_memory_pool_type = ExecutorDeviceType::CPU;
  chunk_request_info.bytes_scales_per_kernel = bytes_scales_per_kernel;
  for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    chunk_request_info.chunks_with_byte_sizes.emplace_back(
        ChunkKey{1, 1, 1, static_cast<int>(chunk_idx)}, bytes_per_chunk);
    chunk_request_info.bytes_per_kernel.emplace_back(bytes_per_chunk);
    chunk_request_info.total_bytes += bytes_per_chunk;
  }
  chunk_request_info.num_chunks = num_chunks;
  chunk_request_info.max_bytes_per_kernel = bytes_per_chunk;
  return chunk_request_info;
}

RequestInfo gen_default_request_info() {
  const RequestInfo request_info(ExecutorDeviceType::CPU,
                                 default_priority_level_request,
                                 default_cpu_slots_request,
                                 default_min_cpu_slots_request,
                                 default_gpu_slots_request,
                                 default_min_gpu_slots_request,
                                 default_cpu_result_mem_request,
                                 default_min_cpu_result_mem_request,
                                 default_chunk_request_info,
                                 default_output_buffers_reusable_intra_thread);
  return request_info;
}

ChunkRequestInfo gen_chunk_request_info(const int table_id,
                                        const int num_fragments,
                                        const int start_fragment_id,
                                        const int num_columns,
                                        const int start_column_id,
                                        const size_t bytes_per_chunk,
                                        const bool bytes_scales_per_kernel) {
  ChunkRequestInfo chunk_request_info;
  chunk_request_info.device_memory_pool_type = ExecutorDeviceType::CPU;
  chunk_request_info.num_chunks = 0;
  chunk_request_info.total_bytes = 0;
  chunk_request_info.max_bytes_per_kernel = 0;
  chunk_request_info.bytes_scales_per_kernel = bytes_scales_per_kernel;
  const int db_id{1};
  for (int fragment_id = start_fragment_id;
       fragment_id < start_fragment_id + num_fragments;
       ++fragment_id) {
    chunk_request_info.bytes_per_kernel.emplace_back(static_cast<size_t>(0));
    for (int column_id = start_column_id; column_id < start_column_id + num_columns;
         ++column_id) {
      const std::vector<int> chunk_key{db_id, table_id, column_id, fragment_id};
      chunk_request_info.chunks_with_byte_sizes.emplace_back(
          std::make_pair(chunk_key, bytes_per_chunk));
      chunk_request_info.num_chunks++;
      chunk_request_info.total_bytes += bytes_per_chunk;
      chunk_request_info.bytes_per_kernel.back() += bytes_per_chunk;
    }
  }
  for (const auto bytes_per_kernel : chunk_request_info.bytes_per_kernel) {
    if (bytes_per_kernel > chunk_request_info.max_bytes_per_kernel) {
      chunk_request_info.max_bytes_per_kernel = bytes_per_kernel;
    }
  }
  return chunk_request_info;
}

void check_resources(std::shared_ptr<ExecutorResourceMgr> executor_resource_mgr,
                     const size_t expected_allocated_cpu_slots,
                     const size_t expected_total_cpu_slots,
                     const size_t expected_allocated_gpu_slots,
                     const size_t expected_total_gpu_slots,
                     const size_t expected_allocated_cpu_result_mem,
                     const size_t expected_total_cpu_result_mem,
                     const size_t expected_allocated_cpu_buffer_mem,
                     const size_t expected_total_cpu_buffer_mem,
                     const size_t expected_allocated_gpu_buffer_mem,
                     const size_t expected_total_gpu_buffer_mem) {
  ASSERT_TRUE(executor_resource_mgr != nullptr);

  const auto cpu_slot_info =
      executor_resource_mgr->get_resource_info(ResourceType::CPU_SLOTS);
  EXPECT_EQ(cpu_slot_info.first, expected_allocated_cpu_slots);
  EXPECT_EQ(cpu_slot_info.second, expected_total_cpu_slots);

  const auto gpu_slot_info =
      executor_resource_mgr->get_resource_info(ResourceType::GPU_SLOTS);
  EXPECT_EQ(gpu_slot_info.first, expected_allocated_gpu_slots);
  EXPECT_EQ(gpu_slot_info.second, expected_total_gpu_slots);

  const auto cpu_result_mem_info =
      executor_resource_mgr->get_resource_info(ResourceType::CPU_RESULT_MEM);
  EXPECT_EQ(cpu_result_mem_info.first, expected_allocated_cpu_result_mem);
  EXPECT_EQ(cpu_result_mem_info.second, expected_total_cpu_result_mem);

  const auto cpu_buffer_mem_info =
      executor_resource_mgr->get_resource_info(ResourceType::CPU_BUFFER_POOL_MEM);
  EXPECT_EQ(cpu_buffer_mem_info.first, expected_allocated_cpu_buffer_mem);
  EXPECT_EQ(cpu_buffer_mem_info.second, expected_total_cpu_buffer_mem);

  const auto gpu_buffer_mem_info =
      executor_resource_mgr->get_resource_info(ResourceType::GPU_BUFFER_POOL_MEM);
  EXPECT_EQ(gpu_buffer_mem_info.first, expected_allocated_gpu_buffer_mem);
  EXPECT_EQ(gpu_buffer_mem_info.second, expected_total_gpu_buffer_mem);
}

TEST(ExecutorResourceMgr, SetupAndParams) {
  auto executor_resource_mgr = gen_resource_mgr_with_defaults();
  ASSERT_TRUE(executor_resource_mgr != nullptr);
  check_resources(executor_resource_mgr,
                  ZERO_SIZE,
                  default_cpu_slots,
                  ZERO_SIZE,
                  default_gpu_slots,
                  ZERO_SIZE,
                  default_cpu_result_mem,
                  ZERO_SIZE,
                  default_cpu_buffer_pool_mem,
                  ZERO_SIZE,
                  default_gpu_buffer_pool_mem);
}

TEST(ExecutorResourceMgr, RequestResourceAndRelease) {
  auto executor_resource_mgr = gen_resource_mgr_with_defaults();
  ASSERT_TRUE(executor_resource_mgr != nullptr);
  const auto request_info = gen_default_request_info();
  {
    // executor_resource_handle will release resources when it goes out of scope
    const auto executor_resource_handle =
        executor_resource_mgr->request_resources(request_info);
    EXPECT_EQ(executor_resource_handle->get_request_id(),
              size_t(0));  // ids start monotonically at 0, a bit of an implementation
                           // detail but a good sanity check nonetheless
    const auto resource_grant = executor_resource_handle->get_resource_grant();
    EXPECT_EQ(resource_grant.cpu_slots, default_cpu_slots_request);
    EXPECT_EQ(resource_grant.gpu_slots, default_gpu_slots_request);
    EXPECT_EQ(resource_grant.cpu_result_mem, default_cpu_result_mem_request);

    // Allocated resources should be same size as request
    check_resources(executor_resource_mgr,
                    default_cpu_slots_request,
                    default_cpu_slots,
                    default_gpu_slots_request,
                    default_gpu_slots,
                    default_cpu_result_mem_request,
                    default_cpu_result_mem,
                    ZERO_SIZE,
                    default_cpu_buffer_pool_mem,
                    ZERO_SIZE,
                    default_gpu_buffer_pool_mem);

    const auto pre_resource_release_executor_stats =
        executor_resource_mgr->get_executor_stats();
    EXPECT_EQ(pre_resource_release_executor_stats.requests, size_t(1));
    EXPECT_EQ(pre_resource_release_executor_stats.cpu_requests, size_t(1));
    EXPECT_EQ(pre_resource_release_executor_stats.gpu_requests, size_t(0));
    EXPECT_EQ(pre_resource_release_executor_stats.queue_length,
              size_t(0));  // previous request should be out of queue already as resource
                           // was granted
    EXPECT_EQ(pre_resource_release_executor_stats.cpu_queue_length, size_t(0));
    EXPECT_EQ(pre_resource_release_executor_stats.gpu_queue_length, size_t(0));
    EXPECT_EQ(pre_resource_release_executor_stats.requests_executing, size_t(1));
    EXPECT_EQ(pre_resource_release_executor_stats.cpu_requests_executing, size_t(1));
    EXPECT_EQ(pre_resource_release_executor_stats.gpu_requests_executing, size_t(0));
    EXPECT_EQ(pre_resource_release_executor_stats.requests_executed, size_t(0));
    EXPECT_EQ(pre_resource_release_executor_stats.cpu_requests_executed, size_t(0));
    EXPECT_EQ(pre_resource_release_executor_stats.gpu_requests_executed, size_t(0));
  }
  // Allocated resources should be back to 0
  check_resources(executor_resource_mgr,
                  ZERO_SIZE,
                  default_cpu_slots,
                  ZERO_SIZE,
                  default_gpu_slots,
                  ZERO_SIZE,
                  default_cpu_result_mem,
                  ZERO_SIZE,
                  default_cpu_buffer_pool_mem,
                  ZERO_SIZE,
                  default_gpu_buffer_pool_mem);

  const auto post_resource_release_executor_stats =
      executor_resource_mgr->get_executor_stats();
  EXPECT_EQ(post_resource_release_executor_stats.requests, size_t(1));
  EXPECT_EQ(post_resource_release_executor_stats.cpu_requests, size_t(1));
  EXPECT_EQ(post_resource_release_executor_stats.gpu_requests, size_t(0));
  EXPECT_EQ(
      post_resource_release_executor_stats.queue_length,
      size_t(
          0));  // previous request should be out of queue already as resource was granted
  EXPECT_EQ(post_resource_release_executor_stats.cpu_queue_length, size_t(0));
  EXPECT_EQ(post_resource_release_executor_stats.gpu_queue_length, size_t(0));
  // Request should be moved out of executing
  EXPECT_EQ(post_resource_release_executor_stats.requests_executing, size_t(0));
  EXPECT_EQ(post_resource_release_executor_stats.cpu_requests_executing, size_t(0));
  EXPECT_EQ(post_resource_release_executor_stats.gpu_requests_executing, size_t(0));
  // And into executed
  EXPECT_EQ(post_resource_release_executor_stats.requests_executed, size_t(1));
  EXPECT_EQ(post_resource_release_executor_stats.cpu_requests_executed, size_t(1));
  EXPECT_EQ(post_resource_release_executor_stats.gpu_requests_executed, size_t(0));
}

TEST(ExecutorResourceMgr, RequestTimeout) {
  auto executor_resource_mgr = gen_resource_mgr_with_defaults();
  ASSERT_TRUE(executor_resource_mgr != nullptr);

  const auto default_request_info = gen_default_request_info();
  auto executor_resource_handle1 =
      executor_resource_mgr->request_resources(default_request_info);

  const size_t many_cpu_slots{13};
  RequestInfo many_cpu_slots_request_info = default_request_info;
  many_cpu_slots_request_info.cpu_slots = many_cpu_slots;
  many_cpu_slots_request_info.min_cpu_slots = many_cpu_slots;
  // Below should be queued as there are not enough cpu_slot resource (16 available vs 8
  // allocated and 12 requested)

  const size_t timeout_ms{100UL};
  // Won't be able to enter queue before timeout as we're still holding
  // onto executor_resource_handle1 which has 4 cpu slots out of a total of
  // 16 available, and we need 13

  EXPECT_THROW(
      {
        auto executor_resource_handle2 =
            executor_resource_mgr->request_resources_with_timeout(
                many_cpu_slots_request_info, timeout_ms);
      },
      QueryTimedOutWaitingInQueue);

  auto future_executor_resource_handle3 =
      std::async(std::launch::async,
                 &ExecutorResourceMgr::request_resources_with_timeout,
                 executor_resource_mgr.get(),
                 many_cpu_slots_request_info,
                 timeout_ms);
  executor_resource_handle1.reset();
  // By deleting executor_resource_handle1 and so releasing its resources
  // the folowing should succeed before the 100ms timeout
  EXPECT_NO_THROW(future_executor_resource_handle3.get());
}

TEST(ExecutorResourceMgr, ErrorOnPerQueryLargeRequests) {
  auto executor_resource_mgr = gen_resource_mgr_with_defaults();
  ASSERT_TRUE(executor_resource_mgr != nullptr);

  const size_t jumbo_cpu_slots_request{
      15};  // Less than total available (16) but more than 80% max ratio per query
            // policy, should throw

  const RequestInfo too_many_cpu_slots_request_info(
      ExecutorDeviceType::CPU,
      default_priority_level_request,
      jumbo_cpu_slots_request,
      jumbo_cpu_slots_request,
      default_gpu_slots_request,
      default_min_gpu_slots_request,
      default_cpu_result_mem_request,
      default_min_cpu_result_mem_request,
      default_chunk_request_info,
      default_output_buffers_reusable_intra_thread);

  try {
    executor_resource_mgr->request_resources(too_many_cpu_slots_request_info);
    EXPECT_TRUE(false) << "We should throw an exception.";
  } catch (std::runtime_error const& e) {
    std::string err_msg(e.what());
    EXPECT_TRUE(err_msg.find(QUERY_NEEDS_TOO_MANY_CPU_SLOTS_ERR_MSG) !=
                std::string::npos);
  }

  check_resources(executor_resource_mgr,
                  ZERO_SIZE,
                  default_cpu_slots,
                  ZERO_SIZE,
                  default_gpu_slots,
                  ZERO_SIZE,
                  default_cpu_result_mem,
                  ZERO_SIZE,
                  default_cpu_buffer_pool_mem,
                  ZERO_SIZE,
                  default_gpu_buffer_pool_mem);

  const size_t jumbo_gpu_slots_request{3};
  const RequestInfo too_many_gpu_slots_request_info(
      ExecutorDeviceType::GPU,
      default_priority_level_request,
      default_cpu_slots_request,
      default_min_cpu_slots_request,
      jumbo_gpu_slots_request,
      jumbo_gpu_slots_request,
      default_cpu_result_mem_request,
      default_min_cpu_result_mem_request,
      default_chunk_request_info,
      default_output_buffers_reusable_intra_thread);

  try {
    executor_resource_mgr->request_resources(too_many_gpu_slots_request_info);
    EXPECT_TRUE(false) << "We should throw an exception.";
  } catch (std::runtime_error const& e) {
    std::string err_msg(e.what());
    EXPECT_TRUE(err_msg.find(QUERY_NEEDS_TOO_MANY_GPU_SLOTS_ERR_MSG) !=
                std::string::npos);
  }

  check_resources(executor_resource_mgr,
                  ZERO_SIZE,
                  default_cpu_slots,
                  ZERO_SIZE,
                  default_gpu_slots,
                  ZERO_SIZE,
                  default_cpu_result_mem,
                  ZERO_SIZE,
                  default_cpu_buffer_pool_mem,
                  ZERO_SIZE,
                  default_gpu_buffer_pool_mem);

  const size_t jumbo_cpu_result_mem_request{1UL << 38};
  const RequestInfo too_much_cpu_result_mem_request_info(
      ExecutorDeviceType::CPU,
      default_priority_level_request,
      default_cpu_slots_request,
      default_min_cpu_slots_request,
      default_gpu_slots_request,
      default_min_gpu_slots_request,
      jumbo_cpu_result_mem_request,
      jumbo_cpu_result_mem_request,
      default_chunk_request_info,
      default_output_buffers_reusable_intra_thread);

  try {
    executor_resource_mgr->request_resources(too_much_cpu_result_mem_request_info);
    EXPECT_TRUE(false) << "We should throw an exception.";
  } catch (std::runtime_error const& e) {
    std::string err_msg(e.what());
    EXPECT_TRUE(err_msg.find(QUERY_NEEDS_TOO_MUCH_CPU_RESULT_MEM_ERR_MSG) !=
                std::string::npos);
  }

  check_resources(executor_resource_mgr,
                  ZERO_SIZE,
                  default_cpu_slots,
                  ZERO_SIZE,
                  default_gpu_slots,
                  ZERO_SIZE,
                  default_cpu_result_mem,
                  ZERO_SIZE,
                  default_cpu_buffer_pool_mem,
                  ZERO_SIZE,
                  default_gpu_buffer_pool_mem);

  const auto post_exceptions_executor_stats = executor_resource_mgr->get_executor_stats();
  // We currently don't record executor stats for queries that statically error, so stats
  // should show no queries as having been enqueued/currently executing

  EXPECT_EQ(post_exceptions_executor_stats.requests, size_t(0));
  EXPECT_EQ(post_exceptions_executor_stats.cpu_requests, size_t(0));
  EXPECT_EQ(post_exceptions_executor_stats.gpu_requests, size_t(0));
  EXPECT_EQ(
      post_exceptions_executor_stats.queue_length,
      size_t(
          0));  // previous request should be out of queue already as resource was granted
  EXPECT_EQ(post_exceptions_executor_stats.cpu_queue_length, size_t(0));
  EXPECT_EQ(post_exceptions_executor_stats.gpu_queue_length, size_t(0));
  // Request should be moved out of executing
  EXPECT_EQ(post_exceptions_executor_stats.requests_executing, size_t(0));
  EXPECT_EQ(post_exceptions_executor_stats.cpu_requests_executing, size_t(0));
  EXPECT_EQ(post_exceptions_executor_stats.gpu_requests_executing, size_t(0));
  // And into executed
  EXPECT_EQ(post_exceptions_executor_stats.requests_executed, size_t(0));
  EXPECT_EQ(post_exceptions_executor_stats.cpu_requests_executed, size_t(0));
  EXPECT_EQ(post_exceptions_executor_stats.gpu_requests_executed, size_t(0));
}

TEST(ExecutorResourceMgr, AllowPerQueryLargeRequestsWithSmallMinRequests) {
  {  // Scope the following so we can test with other resource mgr configs
    auto executor_resource_mgr = gen_resource_mgr_with_defaults();
    ASSERT_TRUE(executor_resource_mgr != nullptr);
    const size_t jumbo_cpu_slots_request{
        15};  // Less than total available (16) but more than 80% max ratio per query
              // policy, should throw
    const size_t min_cpu_slots_request{
        8};  // Less than 80% of 16 available slots, should run

    const RequestInfo min_cpu_slots_fallback_request_info(
        ExecutorDeviceType::CPU,
        default_priority_level_request,
        jumbo_cpu_slots_request,
        min_cpu_slots_request,
        default_gpu_slots_request,
        default_min_gpu_slots_request,
        default_cpu_result_mem_request,
        default_min_cpu_result_mem_request,
        default_chunk_request_info,
        default_output_buffers_reusable_intra_thread);

    ASSERT_NO_THROW(
        executor_resource_mgr->request_resources(min_cpu_slots_fallback_request_info));

    const auto executor_resource_handle =
        executor_resource_mgr->request_resources(min_cpu_slots_fallback_request_info);
    const auto resource_grant = executor_resource_handle->get_resource_grant();
    // Should be ceil(16 * 0.8) -> 13, where 16 is total cpu slots and 0.8
    // default_per_query_max_cpu_result_mem_ratio which we initialized the resource mgr
    // with
    const size_t expected_cpu_slots_grant{13};
    EXPECT_EQ(resource_grant.cpu_slots, expected_cpu_slots_grant);
    EXPECT_EQ(resource_grant.gpu_slots, default_gpu_slots_request);
    EXPECT_EQ(resource_grant.cpu_result_mem, default_cpu_result_mem_request);
    check_resources(executor_resource_mgr,
                    expected_cpu_slots_grant,
                    default_cpu_slots,
                    default_gpu_slots_request,
                    default_gpu_slots,
                    default_cpu_result_mem_request,
                    default_cpu_result_mem,
                    ZERO_SIZE,
                    default_cpu_buffer_pool_mem,
                    ZERO_SIZE,
                    default_gpu_buffer_pool_mem);
  }

  // Test to ensure that ResourceConcurrencyPolicy::DISALLOW_REQUESTS for a resource's
  // oversubscription_concurrency_policy successfully trumps a max per query resource
  // grant ratio greater than 1.0
  {  // Scope the following so we can test with other resource mgr configs
    const double per_query_max_cpu_result_mem_ratio{2.0};
    const bool allow_cpu_result_mem_oversubscription_concurrency{false};
    // const ConcurrentResourceGrantPolicy concurrent_cpu_result_mem_grant_policy
    // (ResourceConcurrencyPolicy::ALLOW_CONCURRENT_REQUESTS,
    // ResourceConcurrencyPolicy::DISALLOW_REQUESTS);
    auto executor_resource_mgr = generate_executor_resource_mgr(
        default_cpu_slots,
        default_gpu_slots,
        default_cpu_result_mem,
        false,
        default_cpu_buffer_pool_mem,
        default_gpu_buffer_pool_mem,
        default_per_query_max_cpu_slots_ratio,
        default_per_query_max_pinned_cpu_buffer_pool_mem_ratio,
        default_per_query_max_pageable_cpu_buffer_pool_mem_ratio,
        per_query_max_cpu_result_mem_ratio,
        default_allow_cpu_kernel_concurrency,
        default_allow_cpu_gpu_kernel_concurrency,
        default_allow_cpu_slot_oversubscription_concurrency,
        default_allow_gpu_slot_oversubscription,
        allow_cpu_result_mem_oversubscription_concurrency,
        default_max_available_resource_use_ratio);

    const size_t jumbo_cpu_result_mem_request{default_cpu_result_mem * 4};
    const RequestInfo jumbo_cpu_result_mem_request_info(
        ExecutorDeviceType::CPU,
        default_priority_level_request,
        default_cpu_slots_request,
        default_min_cpu_slots_request,
        default_gpu_slots_request,
        default_min_gpu_slots_request,
        jumbo_cpu_result_mem_request,
        default_cpu_result_mem,
        default_chunk_request_info,
        default_output_buffers_reusable_intra_thread);

    //// Should be ceil(16 * 0.8) -> 13, where 16 is total cpu slots and 0.8
    //// default_per_query_max_cpu_result_mem_ratio which we initialized the resource mgr
    //// with
  }
}

TEST(ExecutorResourceMgr, TooBigBufferPoolRequests) {
  auto executor_resource_mgr = gen_resource_mgr_with_defaults();
  ASSERT_TRUE(executor_resource_mgr != nullptr);
  // Ensure exception is thrown for too large of a request
  {
    auto cpu_big_request_info = gen_default_request_info();
    cpu_big_request_info.chunk_request_info =
        gen_chunk_request_info(1, 2, 1, 2, 1, static_cast<size_t>(1L << 37), false);
    ScopeGuard reset_flag =
        [orig =
             g_executor_resource_mgr_allow_auto_shrink_num_cpu_slot_for_groupby_query]() {
          g_executor_resource_mgr_allow_auto_shrink_num_cpu_slot_for_groupby_query = orig;
        };

    g_executor_resource_mgr_allow_auto_shrink_num_cpu_slot_for_groupby_query = false;
    try {
      executor_resource_mgr->request_resources(cpu_big_request_info);
      EXPECT_TRUE(false) << "We should throw an exception.";
    } catch (std::runtime_error const& e) {
      std::string err_msg(e.what());
      EXPECT_TRUE(err_msg.find(QUERY_NEEDS_TOO_MUCH_CPU_BUFFER_POOP_MEM_ERR_MSG) !=
                  std::string::npos);
    }

    // ResourcePool should be empty
    check_resources(executor_resource_mgr,
                    ZERO_SIZE,
                    default_cpu_slots,
                    ZERO_SIZE,
                    default_gpu_slots,
                    ZERO_SIZE,
                    default_cpu_result_mem,
                    ZERO_SIZE,
                    default_cpu_buffer_pool_mem,
                    ZERO_SIZE,
                    default_gpu_buffer_pool_mem);

    auto gpu_big_request_info = cpu_big_request_info;
    gpu_big_request_info.chunk_request_info.device_memory_pool_type =
        ExecutorDeviceType::GPU;
    gpu_big_request_info.chunk_request_info.bytes_scales_per_kernel = false;

    try {
      executor_resource_mgr->request_resources(cpu_big_request_info);
      EXPECT_TRUE(false) << "We should throw an exception.";
    } catch (std::runtime_error const& e) {
      std::string err_msg(e.what());
      EXPECT_TRUE(err_msg.find(QUERY_NEEDS_TOO_MUCH_CPU_BUFFER_POOP_MEM_ERR_MSG) !=
                  std::string::npos);
    }

    // ResourcePool should be empty
    check_resources(executor_resource_mgr,
                    ZERO_SIZE,
                    default_cpu_slots,
                    ZERO_SIZE,
                    default_gpu_slots,
                    ZERO_SIZE,
                    default_cpu_result_mem,
                    ZERO_SIZE,
                    default_cpu_buffer_pool_mem,
                    ZERO_SIZE,
                    default_gpu_buffer_pool_mem);
  }
}

TEST(ExecutorResourceMgr, SimpleResourceRequestsQueueAsync) {
  auto executor_resource_mgr = gen_resource_mgr_with_defaults();
  ASSERT_TRUE(executor_resource_mgr != nullptr);
  check_resources(executor_resource_mgr,
                  ZERO_SIZE,
                  default_cpu_slots,
                  ZERO_SIZE,
                  default_gpu_slots,
                  ZERO_SIZE,
                  default_cpu_result_mem,
                  ZERO_SIZE,
                  default_cpu_buffer_pool_mem,
                  ZERO_SIZE,
                  default_gpu_buffer_pool_mem);

  const auto executor_stats0 = executor_resource_mgr->get_executor_stats();
  EXPECT_EQ(executor_stats0.requests, size_t(0));
  EXPECT_EQ(executor_stats0.queue_length, size_t(0));
  EXPECT_EQ(executor_stats0.requests_executing, size_t(0));
  EXPECT_EQ(executor_stats0.requests_executed, size_t(0));

  const auto default_request_info = gen_default_request_info();
  auto executor_resource_handle1 =
      executor_resource_mgr->request_resources(default_request_info);
  check_resources(executor_resource_mgr,
                  default_cpu_slots_request,
                  default_cpu_slots,
                  default_gpu_slots_request,
                  default_gpu_slots,
                  default_cpu_result_mem_request,
                  default_cpu_result_mem,
                  ZERO_SIZE,
                  default_cpu_buffer_pool_mem,
                  ZERO_SIZE,
                  default_gpu_buffer_pool_mem);

  const auto executor_stats1 = executor_resource_mgr->get_executor_stats();
  EXPECT_EQ(executor_stats1.requests, size_t(1));
  EXPECT_EQ(executor_stats1.queue_length, size_t(0));
  EXPECT_EQ(executor_stats1.requests_executing, size_t(1));
  EXPECT_EQ(executor_stats1.requests_executed, size_t(0));

  auto executor_resource_handle2 =
      executor_resource_mgr->request_resources(default_request_info);
  check_resources(executor_resource_mgr,
                  2 * default_cpu_slots_request,
                  default_cpu_slots,
                  2 * default_gpu_slots_request,
                  default_gpu_slots,
                  2 * default_cpu_result_mem_request,
                  default_cpu_result_mem,
                  ZERO_SIZE,
                  default_cpu_buffer_pool_mem,
                  ZERO_SIZE,
                  default_gpu_buffer_pool_mem);
  const auto executor_stats2 = executor_resource_mgr->get_executor_stats();
  EXPECT_EQ(executor_stats2.requests, size_t(2));
  EXPECT_EQ(executor_stats2.queue_length, size_t(0));
  EXPECT_EQ(executor_stats2.requests_executing, size_t(2));
  EXPECT_EQ(executor_stats2.requests_executed, size_t(0));

  const size_t many_cpu_slots{12};
  RequestInfo many_cpu_slots_request_info = default_request_info;
  many_cpu_slots_request_info.cpu_slots = many_cpu_slots;
  many_cpu_slots_request_info.min_cpu_slots = many_cpu_slots;
  // Below should be queued as there are not enough cpu_slot resource (16 available vs 8
  // allocated and 12 requested)
  auto future_executor_resource_handle3 =
      std::async(std::launch::async,
                 &ExecutorResourceMgr::request_resources,
                 executor_resource_mgr.get(),
                 many_cpu_slots_request_info);
  check_resources(executor_resource_mgr,
                  2 * default_cpu_slots_request,
                  default_cpu_slots,
                  2 * default_gpu_slots_request,
                  default_gpu_slots,
                  2 * default_cpu_result_mem_request,
                  default_cpu_result_mem,
                  ZERO_SIZE,
                  default_cpu_buffer_pool_mem,
                  ZERO_SIZE,
                  default_gpu_buffer_pool_mem);

  // Now free request 2 by deleting its resource handle, causing it to give its resources
  // back to the queue
  executor_resource_handle2.reset();
  // Now the third request (big_cpu_slots_request_info, stored in
  // executor_future_resource_handle3, should have space to execute)
  auto executor_resource_handle3 = future_executor_resource_handle3.get();

  check_resources(executor_resource_mgr,
                  default_cpu_slots_request + many_cpu_slots,
                  default_cpu_slots,
                  2 * default_gpu_slots_request,
                  default_gpu_slots,
                  2 * default_cpu_result_mem_request,
                  default_cpu_result_mem,
                  ZERO_SIZE,
                  default_cpu_buffer_pool_mem,
                  ZERO_SIZE,
                  default_gpu_buffer_pool_mem);
  const auto executor_stats3 = executor_resource_mgr->get_executor_stats();
  EXPECT_EQ(executor_stats3.requests, size_t(3));
  // request 3 exited queue and is executing
  EXPECT_EQ(executor_stats3.queue_length, size_t(0));
  EXPECT_EQ(executor_stats3.requests_executing, size_t(2));
  // request 2 has executed
  EXPECT_EQ(executor_stats3.requests_executed, size_t(1));
}

TEST(ExecutorResourceMgr, BufferPoolRequestsQueueAsync) {
  auto executor_resource_mgr = gen_resource_mgr_with_defaults();
  ASSERT_TRUE(executor_resource_mgr != nullptr);

  {
    // First buffer allocation - 16GB of 128GB
    auto request1_info = gen_default_request_info();
    request1_info.chunk_request_info =
        gen_chunk_request_info(1,
                               1,
                               0,
                               1,
                               0,
                               static_cast<size_t>(1UL << 34),
                               false);  // 1X16GB chunk (16GB total)
    auto executor_resource_handle1 =
        executor_resource_mgr->request_resources(request1_info);

    check_resources(executor_resource_mgr,
                    default_cpu_slots_request,
                    default_cpu_slots,
                    default_gpu_slots_request,
                    default_gpu_slots,
                    default_cpu_result_mem_request,
                    default_cpu_result_mem,
                    request1_info.chunk_request_info.total_bytes,
                    default_cpu_buffer_pool_mem,
                    ZERO_SIZE,
                    default_gpu_buffer_pool_mem);

    // Make a second request for two fragments (64GB of chunk data)
    // Should be completely non-overlapping request1, meaning CPU buffer pool
    // will have 64GB + 16GB -> 72GB of 128GB allocated

    auto request2_info = gen_default_request_info();
    request2_info.chunk_request_info =
        gen_chunk_request_info(1,
                               2,
                               1,
                               2,
                               1,
                               static_cast<size_t>(1UL << 34),
                               false);  // 4X16GB chunks (64GB total)
    auto executor_resource_handle2 =
        executor_resource_mgr->request_resources(request2_info);

    check_resources(executor_resource_mgr,
                    default_cpu_slots_request * 2,
                    default_cpu_slots,
                    default_gpu_slots_request * 2,
                    default_gpu_slots,
                    default_cpu_result_mem_request * 2,
                    default_cpu_result_mem,
                    request1_info.chunk_request_info.total_bytes +
                        request2_info.chunk_request_info.total_bytes,
                    default_cpu_buffer_pool_mem,
                    ZERO_SIZE,
                    default_gpu_buffer_pool_mem);

    // Make a third request for two fragments - completely overlapping the second request
    // Buffer pool should still have 72GB of 128GB allocated

    auto request3_info = gen_default_request_info();
    request3_info.chunk_request_info =
        gen_chunk_request_info(1,
                               2,
                               1,
                               2,
                               1,
                               static_cast<size_t>(1UL << 34),
                               false);  // 4X16GB chunks (64GB total)
    auto executor_resource_handle3 =
        executor_resource_mgr->request_resources(request2_info);

    check_resources(executor_resource_mgr,
                    default_cpu_slots_request * 3,
                    default_cpu_slots,
                    default_gpu_slots_request * 3,
                    default_gpu_slots,
                    default_cpu_result_mem_request * 3,
                    default_cpu_result_mem,
                    request1_info.chunk_request_info.total_bytes +
                        request2_info.chunk_request_info
                            .total_bytes,  // request 3 should add no allocation size
                    default_cpu_buffer_pool_mem,
                    ZERO_SIZE,
                    default_gpu_buffer_pool_mem);

    // Now ask for 4 more chunks not overlapping with the last two requests
    auto request4_info = gen_default_request_info();
    request4_info.chunk_request_info =
        gen_chunk_request_info(1,
                               2,
                               3,
                               2,
                               3,
                               static_cast<size_t>(1UL << 34),
                               false);  // 4X16GB chunks (64GB total)
    // Below should be queued as there is not enough buffer pool memory (128GB) for 16GB +
    // 64GB + 64GB

    auto future_executor_resource_handle4 =
        std::async(std::launch::async,
                   &ExecutorResourceMgr::request_resources,
                   executor_resource_mgr.get(),
                   request4_info);

    // We should have same resources allocated since last check as request4 should still
    // be in queue waiting for resources
    check_resources(executor_resource_mgr,
                    default_cpu_slots_request * 3,
                    default_cpu_slots,
                    default_gpu_slots_request * 3,
                    default_gpu_slots,
                    default_cpu_result_mem_request * 3,
                    default_cpu_result_mem,
                    request1_info.chunk_request_info.total_bytes +
                        request2_info.chunk_request_info
                            .total_bytes,  // request 3 should add no allocation size
                    default_cpu_buffer_pool_mem,
                    ZERO_SIZE,
                    default_gpu_buffer_pool_mem);

    // Now reset request2, which releases its resources
    // However, since the same chunks are requested by request3,
    // allocated cpu buffer pool should remain the same, and
    // request4 should still remain gated

    executor_resource_handle2.reset();

    check_resources(
        executor_resource_mgr,
        default_cpu_slots_request * 2,
        default_cpu_slots,
        default_gpu_slots_request * 2,
        default_gpu_slots,
        default_cpu_result_mem_request * 2,
        default_cpu_result_mem,
        request1_info.chunk_request_info.total_bytes +
            request2_info.chunk_request_info
                .total_bytes,  // request2 is gone but request3 still holds same chunks
        default_cpu_buffer_pool_mem,
        ZERO_SIZE,
        default_gpu_buffer_pool_mem);

    // Now reset request3, which releases its resources, including the
    // chunks it (and formerly request2 before its reset above) were
    // holding, which should allow request4 to run

    executor_resource_handle3.reset();

    auto executor_resource_handle4 = future_executor_resource_handle4.get();

    check_resources(executor_resource_mgr,
                    default_cpu_slots_request * 2,  // request1 + request4
                    default_cpu_slots,
                    default_gpu_slots_request * 2,  // request1 + request4
                    default_gpu_slots,
                    default_cpu_result_mem_request * 2,  // request1 + request4
                    default_cpu_result_mem,
                    request1_info.chunk_request_info.total_bytes +
                        request4_info.chunk_request_info
                            .total_bytes,  // Note request2|3's chunks requests are gone
                    default_cpu_buffer_pool_mem,
                    ZERO_SIZE,
                    default_gpu_buffer_pool_mem);
  }
}

TEST(ExecutorResourceMgr, BufferPoolPageableRequest) {
  auto executor_resource_mgr = gen_resource_mgr_with_defaults();
  ASSERT_TRUE(executor_resource_mgr != nullptr);
  check_resources(executor_resource_mgr,
                  ZERO_SIZE,
                  default_cpu_slots,
                  ZERO_SIZE,
                  default_gpu_slots,
                  ZERO_SIZE,
                  default_cpu_result_mem,
                  ZERO_SIZE,
                  default_cpu_buffer_pool_mem,
                  ZERO_SIZE,
                  default_gpu_buffer_pool_mem);

  const size_t num_frags{8};
  // First buffer allocation - 128GB of 128GB, but per request cap of 75% or 96GB
  auto pageable_request1_info = gen_default_request_info();
  pageable_request1_info.cpu_slots = num_frags;
  pageable_request1_info.min_cpu_slots = 1;
  pageable_request1_info.chunk_request_info = gen_chunk_request_info(
      1,
      num_frags,
      0,
      1,
      0,
      static_cast<size_t>(1UL << 34),
      true /* bytes_scales_per_kernel */);  // 16X16GB chunks (256GB total)

  const auto executor_pageable_request1_resource_handle =
      executor_resource_mgr->request_resources(pageable_request1_info);
  const auto pageable_request1_resource_grant =
      executor_pageable_request1_resource_handle->get_resource_grant();
  // max pageable cpu buffer pool mem ratio is 0.5, so expect 25% of cpu slots requests
  // granted
  EXPECT_EQ(pageable_request1_resource_grant.cpu_slots,
            size_t(4));  // 4X16GB->64GB = 50% 128GB cpu buffer pool
  EXPECT_EQ(pageable_request1_resource_grant.gpu_slots, default_gpu_slots_request);
  EXPECT_EQ(pageable_request1_resource_grant.cpu_result_mem,
            default_cpu_result_mem_request);
  EXPECT_EQ(pageable_request1_resource_grant.buffer_mem_gated_per_slot, true);
  EXPECT_EQ(pageable_request1_resource_grant.buffer_mem_per_slot,
            static_cast<size_t>(1UL << 34));
  EXPECT_EQ(pageable_request1_resource_grant.buffer_mem_for_given_slots,
            4UL * static_cast<size_t>(1UL << 34));

  check_resources(executor_resource_mgr,
                  4UL,
                  default_cpu_slots,
                  default_gpu_slots_request,  // request1 + request4
                  default_gpu_slots,
                  default_cpu_result_mem_request,  // request1 + request4
                  default_cpu_result_mem,
                  4UL * static_cast<size_t>(1UL << 34),
                  default_cpu_buffer_pool_mem,
                  ZERO_SIZE,
                  default_gpu_buffer_pool_mem);
}

TEST(ExecutorResourceMgr, ChangeTotalResource) {
  auto executor_resource_mgr = gen_resource_mgr_with_defaults();
  ASSERT_TRUE(executor_resource_mgr != nullptr);
  check_resources(executor_resource_mgr,
                  ZERO_SIZE,
                  default_cpu_slots,
                  ZERO_SIZE,
                  default_gpu_slots,
                  ZERO_SIZE,
                  default_cpu_result_mem,
                  ZERO_SIZE,
                  default_cpu_buffer_pool_mem,
                  ZERO_SIZE,
                  default_gpu_buffer_pool_mem);
  {
    const auto request_info = gen_default_request_info();
    const auto executor_resource_handle =
        executor_resource_mgr->request_resources(request_info);
    check_resources(executor_resource_mgr,
                    default_cpu_slots_request,
                    default_cpu_slots,
                    default_gpu_slots_request,
                    default_gpu_slots,
                    default_cpu_result_mem_request,
                    default_cpu_result_mem,
                    ZERO_SIZE,
                    default_cpu_buffer_pool_mem,
                    ZERO_SIZE,
                    default_gpu_buffer_pool_mem);
  }
  check_resources(executor_resource_mgr,
                  ZERO_SIZE,
                  default_cpu_slots,
                  ZERO_SIZE,
                  default_gpu_slots,
                  ZERO_SIZE,
                  default_cpu_result_mem,
                  ZERO_SIZE,
                  default_cpu_buffer_pool_mem,
                  ZERO_SIZE,
                  default_gpu_buffer_pool_mem);

  const size_t new_cpu_slot_size{size_t(1)};
  executor_resource_mgr->set_resource(ResourceType::CPU_SLOTS, new_cpu_slot_size);
  check_resources(executor_resource_mgr,
                  ZERO_SIZE,
                  new_cpu_slot_size,
                  ZERO_SIZE,
                  default_gpu_slots,
                  ZERO_SIZE,
                  default_cpu_result_mem,
                  ZERO_SIZE,
                  default_cpu_buffer_pool_mem,
                  ZERO_SIZE,
                  default_gpu_buffer_pool_mem);
  const auto request_info = gen_default_request_info();
  // Now request should throw as we request a 4 CPU slots with a minimum fallback of 2,
  // but only have 1 in pool
  try {
    executor_resource_mgr->request_resources(request_info);
    EXPECT_TRUE(false) << "We should throw an exception.";
  } catch (std::runtime_error const& e) {
    std::string err_msg(e.what());
    EXPECT_TRUE(err_msg.find(QUERY_NEEDS_TOO_MANY_CPU_SLOTS_ERR_MSG) !=
                std::string::npos);
  }

  check_resources(executor_resource_mgr,
                  ZERO_SIZE,
                  new_cpu_slot_size,
                  ZERO_SIZE,
                  default_gpu_slots,
                  ZERO_SIZE,
                  default_cpu_result_mem,
                  ZERO_SIZE,
                  default_cpu_buffer_pool_mem,
                  ZERO_SIZE,
                  default_gpu_buffer_pool_mem);
}

TEST(ExecutorResourceMgr, HandleFailedResourceRequest) {
  // this is the same setup from problematic system's status
  auto executor_resource_mgr = generate_executor_resource_mgr(40,
                                                              1,
                                                              10732475596,
                                                              false,
                                                              50979258598,
                                                              6621341312,
                                                              0.9,
                                                              0.8,
                                                              1,
                                                              0.5,
                                                              true,
                                                              true,
                                                              false,
                                                              true,
                                                              false,
                                                              0.8);
  ASSERT_TRUE(executor_resource_mgr != nullptr);

  ScopeGuard reset_flag =
      [orig =
           g_executor_resource_mgr_allow_auto_shrink_num_cpu_slot_for_groupby_query]() {
        g_executor_resource_mgr_allow_auto_shrink_num_cpu_slot_for_groupby_query = orig;
      };

  g_executor_resource_mgr_allow_auto_shrink_num_cpu_slot_for_groupby_query = false;

  // Ensure exception is thrown for too large of a request
  bool detect_exception = false;
  try {
    // we try to send the same resource request for processing problematic query
    std::vector<std::pair<ChunkKey, size_t>> chunks_with_byte_sizes;
    std::vector<size_t> chunks_sizes{
        128000000, 128000000, 128000000, 128000000, 128000000, 128000000, 128000000,
        128000000, 128000000, 128000000, 118686880, 256000000, 256000000, 256000000,
        256000000, 256000000, 256000000, 256000000, 256000000, 256000000, 256000000,
        237373760, 256000000, 256000000, 256000000, 256000000, 256000000, 256000000,
        256000000, 256000000, 256000000, 237373760, 128000000, 128000000, 128000000,
        128000000, 128000000, 128000000, 128000000, 128000000, 128000000, 128000000,
        118686880, 64000000,  64000000,  64000000,  64000000,  64000000,  64000000,
        64000000,  64000000,  64000000,  64000000,  59343440};
    std::vector<size_t> bytes_per_kernel(11, 832000000);
    ChunkRequestInfo chunk_request_info{ExecutorDeviceType::CPU,
                                        chunks_with_byte_sizes,
                                        55,
                                        9091464720,
                                        bytes_per_kernel,
                                        832000000,
                                        true};
    RequestInfo request_info{ExecutorDeviceType::CPU,
                             0,
                             11,
                             1,
                             0,
                             0,
                             13017488704,
                             1183408064,
                             chunk_request_info,
                             true};
    // after sending the request_info, we expect to get the same exception from the issue
    // report
    auto const request_handle = executor_resource_mgr->request_resources(request_info);
    CHECK(request_handle);
  } catch (std::runtime_error const& e) {
    std::string error_msg(e.what());
    if (error_msg.find(QUERY_NEEDS_TOO_MUCH_CPU_RESULT_MEM_ERR_MSG) !=
        std::string::npos) {
      detect_exception = true;
    }
    // `requests_executing` should be zero, otherwise we cannot pause the ERM queue
    // which is the main source of hanging when calling `clear cpu/gpu memory`
    EXPECT_EQ(executor_resource_mgr->get_executor_stats().requests_executing,
              static_cast<size_t>(0));
  }
  EXPECT_TRUE(detect_exception);
}

TEST(ExecutorResourceMgr, EnableCPUBufferPool) {
  auto executor_resource_mgr = gen_resource_mgr_with_defaults(true);
  ASSERT_TRUE(executor_resource_mgr != nullptr);
  auto check_cpu_result_mem_stat = [executor_resource_mgr]() {
    // allocated CPU_RESULT_MEM should always zero, we track this by using
    // ResourceType::CPU_BUFFER_POOL_MEM
    EXPECT_EQ(
        executor_resource_mgr->get_resource_info(ResourceType::CPU_RESULT_MEM).first,
        static_cast<size_t>(0));
    // total size of CPU_RESULT_MEM pool should always zero
    EXPECT_EQ(
        executor_resource_mgr->get_resource_info(ResourceType::CPU_RESULT_MEM).second,
        static_cast<size_t>(0));
  };
  check_cpu_result_mem_stat();
  const auto request_info = gen_default_request_info();
  {
    const auto executor_resource_handle =
        executor_resource_mgr->request_resources(request_info);
    EXPECT_EQ(executor_resource_handle->get_request_id(), static_cast<size_t>(0));

    const auto resource_grant = executor_resource_handle->get_resource_grant();
    EXPECT_EQ(resource_grant.cpu_slots, default_cpu_slots_request);
    EXPECT_EQ(resource_grant.gpu_slots, default_gpu_slots_request);

    check_cpu_result_mem_stat();

    const auto cpu_buffer_mem_info =
        executor_resource_mgr->get_resource_info(ResourceType::CPU_BUFFER_POOL_MEM);
    // CPU buffer pool should allocate the requested cpu result mem
    EXPECT_EQ(cpu_buffer_mem_info.first, default_cpu_result_mem_request);
    EXPECT_EQ(cpu_buffer_mem_info.second, default_cpu_buffer_pool_mem);
  }

  const auto post_resource_release_executor_stats =
      executor_resource_mgr->get_executor_stats();
  EXPECT_EQ(post_resource_release_executor_stats.requests, size_t(1));
  EXPECT_EQ(post_resource_release_executor_stats.cpu_requests, size_t(1));
  EXPECT_EQ(post_resource_release_executor_stats.gpu_requests, size_t(0));
  EXPECT_EQ(
      post_resource_release_executor_stats.queue_length,
      size_t(
          0));  // previous request should be out of queue already as resource was granted
  EXPECT_EQ(post_resource_release_executor_stats.cpu_queue_length, size_t(0));
  EXPECT_EQ(post_resource_release_executor_stats.gpu_queue_length, size_t(0));
  // Request should be moved out of executing
  EXPECT_EQ(post_resource_release_executor_stats.requests_executing, size_t(0));
  EXPECT_EQ(post_resource_release_executor_stats.cpu_requests_executing, size_t(0));
  EXPECT_EQ(post_resource_release_executor_stats.gpu_requests_executing, size_t(0));
  // And into executed
  EXPECT_EQ(post_resource_release_executor_stats.requests_executed, size_t(1));
  EXPECT_EQ(post_resource_release_executor_stats.cpu_requests_executed, size_t(1));
  EXPECT_EQ(post_resource_release_executor_stats.gpu_requests_executed, size_t(0));
}

TEST(ExecutorResourceMgr, AdjustNumCPUSlotForCPUGroupbyQuery) {
  auto executor_resource_mgr = generate_executor_resource_mgr(
      32,
      1,
      0u,
      true,
      65536,
      default_gpu_buffer_pool_mem,
      default_per_query_max_cpu_slots_ratio,
      default_per_query_max_cpu_result_mem_ratio,
      default_per_query_max_pinned_cpu_buffer_pool_mem_ratio,
      default_per_query_max_pageable_cpu_buffer_pool_mem_ratio,
      default_allow_cpu_kernel_concurrency,
      default_allow_cpu_gpu_kernel_concurrency,
      default_allow_cpu_slot_oversubscription_concurrency,
      default_allow_gpu_slot_oversubscription,
      default_allow_cpu_result_mem_oversubscription_concurrency,
      default_max_available_resource_use_ratio);
  ASSERT_TRUE(executor_resource_mgr != nullptr);
  ChunkRequestInfo chunk_request_info{
      ExecutorDeviceType::CPU,
      std::vector<std::pair<std::vector<int>, unsigned long>>(50),
      50,
      796,
      std::vector<unsigned long>(50, 16),
      16,
      true};

  for (size_t i = 0; i < 50; ++i) {
    chunk_request_info.chunks_with_byte_sizes[i] = {{3, 220, 1, static_cast<int>(i)},
                                                    (i == 49) ? 12ull : 16ull};
  }

  RequestInfo resource_request_info{
      ExecutorDeviceType::CPU, 0, 40, 1, 0, 0, 63680, 1592, chunk_request_info, true};

  ScopeGuard reset_flag =
      [orig =
           g_executor_resource_mgr_allow_auto_shrink_num_cpu_slot_for_groupby_query]() {
        g_executor_resource_mgr_allow_auto_shrink_num_cpu_slot_for_groupby_query = orig;
      };

  g_executor_resource_mgr_allow_auto_shrink_num_cpu_slot_for_groupby_query = false;
  try {
    auto request_info = executor_resource_mgr->request_resources(resource_request_info);
    ASSERT_TRUE(false) << "We should throw an ERM \'RequestStat Error\' exception.";
  } catch (std::runtime_error const& e) {
    std::string const err_msg{e.what()};
    // Limit is the per-query result memory ratio (0.8) of the 65536 byte pool, less the
    // 796 bytes of headroom reserved for this request's input chunks: 52429 - 796
    std::string const expected_msg{
        "RequestStats error: Query requested more CPU result memory (62.18 KB) than "
        "available per query (50.42 KB) in executor resource pool"};
    ASSERT_EQ(err_msg, expected_msg);
  }

  g_executor_resource_mgr_allow_auto_shrink_num_cpu_slot_for_groupby_query = true;
  ASSERT_NO_THROW(executor_resource_mgr->request_resources(resource_request_info));
}

TEST(ExecutorResourceMgr, AdjustedNumCPUSlotBecomeZero) {
  auto executor_resource_mgr = generate_executor_resource_mgr(
      1,
      1,
      0u,
      true,
      1000,
      default_gpu_buffer_pool_mem,
      default_per_query_max_cpu_slots_ratio,
      default_per_query_max_cpu_result_mem_ratio,
      default_per_query_max_pinned_cpu_buffer_pool_mem_ratio,
      default_per_query_max_pageable_cpu_buffer_pool_mem_ratio,
      default_allow_cpu_kernel_concurrency,
      default_allow_cpu_gpu_kernel_concurrency,
      default_allow_cpu_slot_oversubscription_concurrency,
      default_allow_gpu_slot_oversubscription,
      default_allow_cpu_result_mem_oversubscription_concurrency,
      default_max_available_resource_use_ratio);
  ASSERT_TRUE(executor_resource_mgr != nullptr);
  ChunkRequestInfo chunk_request_info{
      ExecutorDeviceType::CPU,
      std::vector<std::pair<std::vector<int>, unsigned long>>(40),
      40,
      796,
      std::vector<unsigned long>(40, 16),
      16,
      true};

  for (size_t i = 0; i < 40; ++i) {
    chunk_request_info.chunks_with_byte_sizes[i] = {{3, 220, 1, static_cast<int>(i)}, 16};
  }

  RequestInfo resource_request_info{
      ExecutorDeviceType::CPU, 0, 40, 1, 0, 0, 63680, 1592, chunk_request_info, true};

  ScopeGuard reset_flag =
      [orig =
           g_executor_resource_mgr_allow_auto_shrink_num_cpu_slot_for_groupby_query]() {
        g_executor_resource_mgr_allow_auto_shrink_num_cpu_slot_for_groupby_query = orig;
      };

  g_executor_resource_mgr_allow_auto_shrink_num_cpu_slot_for_groupby_query = false;
  try {
    auto request_info = executor_resource_mgr->request_resources(resource_request_info);
    ASSERT_TRUE(false) << "We should throw an exception.";
  } catch (std::runtime_error const& e) {
    std::string const err_msg{e.what()};
    // Limit is the per-query result memory ratio (0.8) of the 1000 byte pool, less the
    // headroom for a gated slot's worth of chunk memory (16 bytes): 800 - 16
    std::string const expected_msg{
        "RequestStats error: Query requested more CPU result memory (1.55 KB) than "
        "available per query (784 bytes) in executor resource pool"};
    ASSERT_EQ(err_msg, expected_msg);
  }

  g_executor_resource_mgr_allow_auto_shrink_num_cpu_slot_for_groupby_query = true;
  try {
    auto request_info = executor_resource_mgr->request_resources(resource_request_info);
    ASSERT_TRUE(false) << "We should throw an exception.";
  } catch (std::runtime_error const& e) {
    std::string const err_msg{e.what()};
    std::string const expected_msg{
        "Failed to adjust CPU slots for \'QueryNeedsTooMuchCpuResultMem\' error: "
        "adjusted CPU slots should be larger than zero"};
    ASSERT_EQ(err_msg, expected_msg);
  }
}

// Was IdentitalAdjustedNumCPUSlot, which drove the auto-shrink retry to recompute the
// slot count it was already given and hit the "adjusted CPU slots is equal to the
// original resource request" guard. That is no longer reachable here: the retry now
// sizes slots against the same limit the pool enforces, so a request rejected for
// needing too much result memory always yields strictly fewer slots (or zero, which
// the other guard covers). The guard is kept as a defensive stop against unbounded
// recursion, so this test asserts the shrink instead.
TEST(ExecutorResourceMgr, AdjustedNumCPUSlotStrictlyShrinks) {
  auto executor_resource_mgr = generate_executor_resource_mgr(
      1,
      1,
      0u,
      true,
      131072,
      default_gpu_buffer_pool_mem,
      default_per_query_max_cpu_slots_ratio,
      default_per_query_max_cpu_result_mem_ratio,
      default_per_query_max_pinned_cpu_buffer_pool_mem_ratio,
      default_per_query_max_pageable_cpu_buffer_pool_mem_ratio,
      default_allow_cpu_kernel_concurrency,
      default_allow_cpu_gpu_kernel_concurrency,
      default_allow_cpu_slot_oversubscription_concurrency,
      default_allow_gpu_slot_oversubscription,
      default_allow_cpu_result_mem_oversubscription_concurrency,
      default_max_available_resource_use_ratio);
  ASSERT_TRUE(executor_resource_mgr != nullptr);
  ChunkRequestInfo chunk_request_info{
      ExecutorDeviceType::CPU,
      std::vector<std::pair<std::vector<int>, unsigned long>>(2),
      2,
      796,
      std::vector<unsigned long>(2, 16),
      16,
      true};

  for (size_t i = 0; i < 2; ++i) {
    chunk_request_info.chunks_with_byte_sizes[i] = {{3, 220, 1, static_cast<int>(i)}, 16};
  }

  RequestInfo resource_request_info{
      ExecutorDeviceType::CPU, 0, 2, 1, 0, 0, 131072, 1592, chunk_request_info, true};

  ScopeGuard reset_flag =
      [orig =
           g_executor_resource_mgr_allow_auto_shrink_num_cpu_slot_for_groupby_query]() {
        g_executor_resource_mgr_allow_auto_shrink_num_cpu_slot_for_groupby_query = orig;
      };

  g_executor_resource_mgr_allow_auto_shrink_num_cpu_slot_for_groupby_query = false;
  try {
    auto request_info = executor_resource_mgr->request_resources(resource_request_info);
    ASSERT_TRUE(false) << "We should throw an exception.";
  } catch (std::runtime_error const& e) {
    std::string const err_msg{e.what()};
    // Limit is the per-query result memory ratio (0.8) of the 131072 byte pool, less
    // the 796 bytes of headroom reserved for this request's input chunks: 104858 - 796
    std::string const expected_msg{
        "RequestStats error: Query requested more CPU result memory (128.0 KB) than "
        "available per query (101.62 KB) in executor resource pool"};
    ASSERT_EQ(err_msg, expected_msg);
  }

  g_executor_resource_mgr_allow_auto_shrink_num_cpu_slot_for_groupby_query = true;
  {
    auto const request_handle =
        executor_resource_mgr->request_resources(resource_request_info);
    ASSERT_TRUE(request_handle != nullptr);
    const auto resource_grant = request_handle->get_resource_grant();
    EXPECT_LT(resource_grant.cpu_slots, resource_request_info.cpu_slots);
    // Result memory and chunk memory together must fit the pool, which is the invariant
    // whose absence previously aborted the server
    const auto cpu_buffer_mem_info =
        executor_resource_mgr->get_resource_info(ResourceType::CPU_BUFFER_POOL_MEM);
    EXPECT_LE(cpu_buffer_mem_info.first, cpu_buffer_mem_info.second);
  }
}

// The field failure: with output buffers drawn from the CPU buffer pool, result memory
// and pinned input chunks each fit on their own but not together. The pool used to
// admit such a request and then abort the server on the CHECK_LE in
// add_chunk_requests_to_allocated_pool.
TEST(ExecutorResourceMgr, CpuResultMemAndChunksCompeteForPool) {
  constexpr size_t cpu_buffer_pool_mem{1UL << 20};  // 1MB
  // Per-query limits: result memory 0.8 -> 838861, pinned chunks 0.75 -> 786432
  const auto chunk_request_info = gen_cpu_chunk_request_info(4, 125000, false);
  ASSERT_EQ(chunk_request_info.total_bytes, size_t(500000));

  {
    auto executor_resource_mgr = gen_pool_backed_resource_mgr(cpu_buffer_pool_mem);
    // 700000 of result memory and 500000 of chunks are individually under their limits,
    // but 1200000 overcommits the 1048576 byte pool
    const RequestInfo resource_request_info(ExecutorDeviceType::CPU,
                                            default_priority_level_request,
                                            1,
                                            1,
                                            0,
                                            0,
                                            700000,
                                            700000,
                                            chunk_request_info,
                                            false);
    try {
      executor_resource_mgr->request_resources(resource_request_info);
      ASSERT_TRUE(false) << "Expected the request to be rejected.";
    } catch (std::runtime_error const& e) {
      ASSERT_TRUE(std::string(e.what()).find(
                      QUERY_NEEDS_TOO_MUCH_CPU_RESULT_MEM_ERR_MSG) != std::string::npos)
          << e.what();
    }
    // A rejected request must leave the pool untouched
    const auto cpu_buffer_mem_info =
        executor_resource_mgr->get_resource_info(ResourceType::CPU_BUFFER_POOL_MEM);
    EXPECT_EQ(cpu_buffer_mem_info.first, ZERO_SIZE);
  }

  {
    auto executor_resource_mgr = gen_pool_backed_resource_mgr(cpu_buffer_pool_mem);
    // Same chunks, but result memory now leaves room for them
    const RequestInfo resource_request_info(ExecutorDeviceType::CPU,
                                            default_priority_level_request,
                                            1,
                                            1,
                                            0,
                                            0,
                                            300000,
                                            300000,
                                            chunk_request_info,
                                            false);
    auto const request_handle =
        executor_resource_mgr->request_resources(resource_request_info);
    ASSERT_TRUE(request_handle != nullptr);
    EXPECT_EQ(request_handle->get_resource_grant().cpu_result_mem, size_t(300000));
    // Both the result memory and the chunks are accounted against the one pool
    const auto cpu_buffer_mem_info =
        executor_resource_mgr->get_resource_info(ResourceType::CPU_BUFFER_POOL_MEM);
    EXPECT_EQ(cpu_buffer_mem_info.first, size_t(800000));
    EXPECT_LE(cpu_buffer_mem_info.first, cpu_buffer_mem_info.second);
  }
}

// The same competition, but down the path where chunk memory is gated per CPU slot,
// which commits chunk bytes through a second code path
TEST(ExecutorResourceMgr, CpuResultMemAndGatedChunksCompeteForPool) {
  constexpr size_t cpu_buffer_pool_mem{1UL << 20};  // 1MB
  auto executor_resource_mgr = gen_pool_backed_resource_mgr(cpu_buffer_pool_mem);
  // 900000 bytes of chunks exceeds the 786432 byte pinned limit, so the chunks are
  // gated to 225000 bytes per CPU slot
  const auto chunk_request_info = gen_cpu_chunk_request_info(4, 225000, true);
  ASSERT_EQ(chunk_request_info.total_bytes, size_t(900000));

  const RequestInfo resource_request_info(ExecutorDeviceType::CPU,
                                          default_priority_level_request,
                                          4,
                                          1,
                                          0,
                                          0,
                                          600000,
                                          600000,
                                          chunk_request_info,
                                          false);
  auto const request_handle =
      executor_resource_mgr->request_resources(resource_request_info);
  ASSERT_TRUE(request_handle != nullptr);
  const auto resource_grant = request_handle->get_resource_grant();
  EXPECT_TRUE(resource_grant.buffer_mem_gated_per_slot);
  // Slots are gated down to what the pool can hold alongside the result memory
  EXPECT_EQ(resource_grant.cpu_slots, size_t(1));
  EXPECT_EQ(resource_grant.buffer_mem_for_given_slots, size_t(225000));
  const auto cpu_buffer_mem_info =
      executor_resource_mgr->get_resource_info(ResourceType::CPU_BUFFER_POOL_MEM);
  EXPECT_EQ(cpu_buffer_mem_info.first, size_t(825000));
  EXPECT_LE(cpu_buffer_mem_info.first, cpu_buffer_mem_info.second);
}

// Pool-backed result memory is limited by the per-query result memory ratio, not by the
// pinned chunk ratio that shares the same pool
TEST(ExecutorResourceMgr, PoolBackedCpuResultMemUsesResultMemRatio) {
  constexpr size_t cpu_buffer_pool_mem{1UL << 20};  // 1MB
  // 0.8 of the pool is 838861 bytes, versus 786432 for the 0.75 pinned chunk ratio
  auto executor_resource_mgr = gen_pool_backed_resource_mgr(cpu_buffer_pool_mem);

  auto gen_request = [](const size_t cpu_result_mem) {
    return RequestInfo(ExecutorDeviceType::CPU,
                       default_priority_level_request,
                       1,
                       1,
                       0,
                       0,
                       cpu_result_mem,
                       cpu_result_mem,
                       default_chunk_request_info,
                       false);
  };

  {
    auto const request_handle =
        executor_resource_mgr->request_resources(gen_request(800000));
    ASSERT_TRUE(request_handle != nullptr);
    EXPECT_EQ(request_handle->get_resource_grant().cpu_result_mem, size_t(800000));
  }

  try {
    executor_resource_mgr->request_resources(gen_request(900000));
    ASSERT_TRUE(false) << "Expected the request to be rejected.";
  } catch (std::runtime_error const& e) {
    ASSERT_TRUE(std::string(e.what()).find(
                    QUERY_NEEDS_TOO_MUCH_CPU_RESULT_MEM_ERR_MSG) != std::string::npos)
        << e.what();
  }
}

// When a group by query is retried with fewer CPU slots, the reduced result memory has
// to leave room for the query's input chunks in the shared pool
TEST(ExecutorResourceMgr, AdjustNumCPUSlotReservesChunkHeadroom) {
  constexpr size_t cpu_buffer_pool_mem{1UL << 20};  // 1MB
  auto executor_resource_mgr = gen_pool_backed_resource_mgr(cpu_buffer_pool_mem);
  const auto chunk_request_info = gen_cpu_chunk_request_info(2, 131072, false);
  ASSERT_EQ(chunk_request_info.total_bytes, size_t(262144));

  // 8 slots at 131072 bytes per slot asks for the whole pool
  const RequestInfo resource_request_info(ExecutorDeviceType::CPU,
                                          default_priority_level_request,
                                          8,
                                          1,
                                          0,
                                          0,
                                          1048576,
                                          131072,
                                          chunk_request_info,
                                          true);

  ScopeGuard reset_flag =
      [orig =
           g_executor_resource_mgr_allow_auto_shrink_num_cpu_slot_for_groupby_query]() {
        g_executor_resource_mgr_allow_auto_shrink_num_cpu_slot_for_groupby_query = orig;
      };
  g_executor_resource_mgr_allow_auto_shrink_num_cpu_slot_for_groupby_query = true;

  auto const request_handle =
      executor_resource_mgr->request_resources(resource_request_info);
  ASSERT_TRUE(request_handle != nullptr);
  // Result memory is capped at 838861 less the 262144 reserved for chunks, so the retry
  // gets 576717 / 131072 == 4 slots rather than the 8 it asked for
  EXPECT_EQ(request_handle->get_resource_grant().cpu_slots, size_t(4));
  const auto cpu_buffer_mem_info =
      executor_resource_mgr->get_resource_info(ResourceType::CPU_BUFFER_POOL_MEM);
  EXPECT_EQ(cpu_buffer_mem_info.first, size_t(786432));
  EXPECT_LE(cpu_buffer_mem_info.first, cpu_buffer_mem_info.second);
}

TEST(ExecutorResourceMgr, ResourceSubtypeMapping) {
  EXPECT_EQ(resource_subtype_to_string(ResourceSubtype::CPU_SLOTS), "cpu_slots");
  EXPECT_EQ(resource_subtype_to_string(ResourceSubtype::GPU_SLOTS), "gpu_slots");
  EXPECT_EQ(resource_subtype_to_string(ResourceSubtype::CPU_RESULT_MEM),
            "cpu_result_mem");
  EXPECT_EQ(resource_subtype_to_string(ResourceSubtype::GPU_RESULT_MEM),
            "gpu_result_mem");
  EXPECT_EQ(resource_subtype_to_string(ResourceSubtype::PINNED_CPU_BUFFER_POOL_MEM),
            "pinned_cpu_buffer_pool_mem");
  EXPECT_EQ(resource_subtype_to_string(ResourceSubtype::PAGEABLE_CPU_BUFFER_POOL_MEM),
            "pageable_cpu_buffer_pool_mem");
  EXPECT_EQ(resource_subtype_to_string(ResourceSubtype::PINNED_GPU_BUFFER_POOL_MEM),
            "pinned_gpu_buffer_pool_mem");
  EXPECT_EQ(resource_subtype_to_string(ResourceSubtype::PAGEABLE_GPU_BUFFER_POOL_MEM),
            "pageable_gpu_buffer_pool_mem");
  EXPECT_EQ(resource_subtype_to_string(ResourceSubtype::CPU_RESULT_MEM_IN_POOL),
            "cpu_result_mem_in_pool");

  // Every subtype must roll up under a type that lists it, or the pool will track a
  // subtype that no type-level total ever accounts for
  for (size_t subtype_idx = 0; subtype_idx < ResourceSubtypeSize; ++subtype_idx) {
    const auto subtype = static_cast<ResourceSubtype>(subtype_idx);
    const auto resource_type = map_resource_subtype_to_resource_type(subtype);
    EXPECT_NE(resource_type, ResourceType::INVALID_TYPE)
        << resource_subtype_to_string(subtype);
    const auto subtypes = map_resource_type_to_resource_subtypes(resource_type);
    EXPECT_NE(std::find(subtypes.begin(), subtypes.end(), subtype), subtypes.end())
        << resource_subtype_to_string(subtype);
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  namespace po = boost::program_options;
  po::options_description desc("Options");

  // these two are here to allow passing correctly google testing parameters
  desc.add_options()("gtest_list_tests", "list all test");
  desc.add_options()("gtest_filter", "filters tests, use --help for details");

  logger::LogOptions log_options(argv[0]);
  log_options.severity_ = logger::Severity::FATAL;
  log_options.set_options();  // update default values
  desc.add(log_options.get_options());

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  return err;
}
