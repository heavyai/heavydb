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

#include <future>
#include <thread>

#include "../QueryEngine/ExecutorResourceMgr/ExecutorResourceMgr.h"
#include "TestHelpers.h"

using namespace ExecutorResourceMgr_Namespace;

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

std::shared_ptr<ExecutorResourceMgr> gen_resource_mgr_with_defaults() {
  return generate_executor_resource_mgr(
      default_cpu_slots,
      default_gpu_slots,
      default_cpu_result_mem,
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

  EXPECT_THROW(executor_resource_mgr->request_resources(too_many_cpu_slots_request_info),
               QueryNeedsTooManyCpuSlots);

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

  EXPECT_THROW(executor_resource_mgr->request_resources(too_many_gpu_slots_request_info),
               QueryNeedsTooManyGpuSlots);
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

  EXPECT_THROW(
      executor_resource_mgr->request_resources(too_much_cpu_result_mem_request_info),
      QueryNeedsTooMuchCpuResultMem);
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
    EXPECT_THROW(executor_resource_mgr->request_resources(cpu_big_request_info),
                 QueryNeedsTooMuchBufferPoolMem);

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
    EXPECT_THROW(executor_resource_mgr->request_resources(gpu_big_request_info),
                 QueryNeedsTooMuchBufferPoolMem);

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
  EXPECT_THROW(executor_resource_mgr->request_resources(request_info),
               QueryNeedsTooManyCpuSlots);
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
