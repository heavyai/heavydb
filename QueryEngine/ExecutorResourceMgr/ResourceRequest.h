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

#include <vector>

#include "ExecutorResourceMgrCommon.h"

namespace ExecutorResourceMgr_Namespace {

/**
 * @brief Specifies all `DataMgr` chunks needed for a query step/request,
 * along with their sizes in bytes. It also keeps track of other metadata
 * to avoid having to recompute this info, such as total_bytes and a
 * vector of total byte sizes for each kernel. The latter is relevant
 * if `bytes_scales_per_kernel` is true, as the `ExecutorResourceMgr`/
 * `ExecutorResourcePool` can scale back the number of kernels allowed to run
 * simultaneously to ensure that a query step can run.
 */
struct ChunkRequestInfo {
  ExecutorDeviceType device_memory_pool_type{ExecutorDeviceType::CPU};
  std::vector<std::pair<ChunkKey, size_t>> chunks_with_byte_sizes;
  size_t num_chunks{0};
  size_t total_bytes{0};
  std::vector<size_t> bytes_per_kernel;
  size_t max_bytes_per_kernel{0};
  bool bytes_scales_per_kernel{false};
};

/**
 * @brief Specifies the minimum and maximum quanity either requested or granted
 * for a request of `resource_subtype`.
 */
struct ResourceRequest {
  ResourceSubtype resource_subtype{ResourceSubtype::INVALID_SUBTYPE};
  size_t max_quantity{0};
  size_t min_quantity{0};
};

/**
 * @brief Alias of ResourceRequest to ResourceRequestGrant to
 * better semantically differentiate between resource requests
 * and resource grants in `ExecutorResourcePool`
 */
using ResourceRequestGrant = ResourceRequest;

/**
 * @brief A container to store requested and minimum neccessary resource requests across
 * all resource types currently supported by `ExecutorResourceMgr`/`ExecutorResourcePool`.
 * It also includes a `ChunkRequestInfo` struct to denote which `DataMgr` chunks (with
 * their sizes in bytes) are neccesary for the query.
 *
 * `RequestInfo` is the principal data interface between
 * `Executor::launchKernelsViaResourceMgr` and `ExecutorResourceMgr`.
 */
struct RequestInfo {
  ExecutorDeviceType request_device_type;
  size_t priority_level{0};
  size_t cpu_slots{0};
  size_t min_cpu_slots{0};
  size_t gpu_slots{0};
  size_t min_gpu_slots{0};
  size_t cpu_result_mem{0};
  size_t min_cpu_result_mem{0};
  ChunkRequestInfo chunk_request_info;
  bool output_buffers_reusable_intra_thread{false};
  bool chunk_memory_scales_by_num_threads{false};
  bool request_must_run_alone{false};
  bool request_must_run_alone_for_device_type{false};

  RequestInfo(const ExecutorDeviceType request_device_type,
              const size_t priority_level,
              const size_t cpu_slots,
              const size_t min_cpu_slots,
              const size_t gpu_slots,
              const size_t min_gpu_slots,
              const size_t cpu_result_mem,
              const size_t min_cpu_result_mem,
              const ChunkRequestInfo& chunk_request_info)
      : request_device_type(request_device_type)
      , priority_level(priority_level)
      , cpu_slots(cpu_slots)
      , min_cpu_slots(min_cpu_slots)
      , gpu_slots(gpu_slots)
      , min_gpu_slots(min_gpu_slots)
      , cpu_result_mem(cpu_result_mem)
      , min_cpu_result_mem(min_cpu_result_mem)
      , chunk_request_info(chunk_request_info) {}

  /**
   * @brief Simple constructor assuming no difference between min and requested resources,
   * and no intra-thread cpu mem sharing
   */
  RequestInfo(const ExecutorDeviceType request_device_type,
              const size_t num_kernels,
              const size_t cpu_result_mem)
      : request_device_type(request_device_type)
      , priority_level(static_cast<size_t>(0))
      , cpu_slots(num_kernels)
      , min_cpu_slots(num_kernels)
      , gpu_slots(request_device_type == ExecutorDeviceType::GPU ? num_kernels
                                                                 : static_cast<size_t>(0))
      , min_gpu_slots(request_device_type == ExecutorDeviceType::GPU
                          ? num_kernels
                          : static_cast<size_t>(0))
      , cpu_result_mem(cpu_result_mem)
      , min_cpu_result_mem(cpu_result_mem) {}
};

}  // namespace  ExecutorResourceMgr_Namespace