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

#include <math.h>
#include <string>

#include "Logger/Logger.h"
#include "QueryEngine/ExecutorDeviceType.h"
#include "Shared/StringTransform.h"

namespace ExecutorResourceMgr_Namespace {

// Type declarations and using aliases
using ChunkKey = std::vector<int>;
using RequestId = size_t;

class ExecutorResourceMgrError {
 public:
  ExecutorResourceMgrError(RequestId const request_id, std::string error_msg)
      : request_id_(request_id), error_msg_(std::move(error_msg)) {}
  RequestId getRequestId() const { return request_id_; }
  std::string getErrorMsg() const { return error_msg_; }

 private:
  RequestId request_id_;
  std::string error_msg_;
};

class QueryTimedOutWaitingInQueue : public std::runtime_error {
 public:
  QueryTimedOutWaitingInQueue(const size_t timeout_ms)
      : std::runtime_error("Query exceeded queue timeout threshold " +
                           std::to_string(timeout_ms) + "ms.") {}
};

class QueryNeedsTooMuchBufferPoolMem : public std::runtime_error {
 public:
  QueryNeedsTooMuchBufferPoolMem(const size_t max_buffer_pool_mem,
                                 const size_t requested_buffer_pool_mem,
                                 const ExecutorDeviceType device_type)
      : std::runtime_error(
            "Query requested more " + get_device_type_string(device_type) +
            " buffer pool mem (" + format_num_bytes(requested_buffer_pool_mem) +
            ") than max available for query (" + format_num_bytes(max_buffer_pool_mem) +
            ") in executor resource pool") {}

 private:
  std::string get_device_type_string(const ExecutorDeviceType device_type) {
    switch (device_type) {
      case ExecutorDeviceType::CPU:
        return "CPU";
      case ExecutorDeviceType::GPU:
        return "GPU";
      default:
        UNREACHABLE();
        return "";
    }
  }
};

class QueryNeedsTooManyCpuSlots : public std::runtime_error {
 public:
  QueryNeedsTooManyCpuSlots(const size_t max_cpu_slots, const size_t requested_cpu_slots)
      : std::runtime_error(
            "Query requested more CPU slots (" + std::to_string(requested_cpu_slots) +
            ") than max available for query (" + std::to_string(max_cpu_slots) +
            ") in executor resource pool") {}
};

class QueryNeedsTooManyGpuSlots : public std::runtime_error {
 public:
  QueryNeedsTooManyGpuSlots(const size_t max_gpu_slots, const size_t requested_gpu_slots)
      : std::runtime_error(
            "Query requested more GPU slots (" + std::to_string(requested_gpu_slots) +
            ") than available per query (" + std::to_string(max_gpu_slots) +
            ") in executor resource pool") {}
};

class QueryNeedsTooMuchCpuResultMem : public std::runtime_error {
 public:
  QueryNeedsTooMuchCpuResultMem(const size_t max_cpu_result_mem,
                                const size_t requested_cpu_result_mem)
      : std::runtime_error(
            "Query requested more CPU result memory (" +
            format_num_bytes(requested_cpu_result_mem) + ") than available per query (" +
            format_num_bytes(max_cpu_result_mem) + ") in executor resource pool") {}
};

/**
 * @type ResourceType
 * @brief Stores the resource type for a ExecutorResourcePool request
 */
enum class ResourceType {
  CPU_SLOTS = 0,
  GPU_SLOTS = 1,
  CPU_RESULT_MEM = 2,
  GPU_RESULT_MEM = 3,
  CPU_BUFFER_POOL_MEM = 4,
  GPU_BUFFER_POOL_MEM = 5,
  INVALID_TYPE = 6,
  NUM_RESOURCE_TYPES = 6,
};

static constexpr size_t ResourceTypeSize =
    static_cast<size_t>(ResourceType::NUM_RESOURCE_TYPES);

static const char* ResourceTypeStrings[] = {"cpu_slots",
                                            "gpu_slots",
                                            "cpu_result_mem",
                                            "gpu_result_mem",
                                            "cpu_buffer_pool_mem",
                                            "gpu_buffer_pool_mem"};

inline std::string resource_type_to_string(const ResourceType resource_type) {
  return ResourceTypeStrings[static_cast<size_t>(resource_type)];
}

/**
 * @type ResourceSubtype
 * @brief Stores the resource sub-type for a ExecutorResourcePool request
 *
 * The concept of a ResourceSubtype needs to be distinguished from
 * ResourceType as certain categories of resource requests, i.e. buffer
 * pool mem, can be further distinguished by whether the memory
 * is pinned or pageable (i.e. input chunks, which can be evicted, are considered
 * pageable, while kernel result memory is considered pinned as it cannot currently
 * be evicted or deleted during a query).
 */
enum class ResourceSubtype {
  CPU_SLOTS = 0,
  GPU_SLOTS = 1,
  CPU_RESULT_MEM = 2,
  GPU_RESULT_MEM = 3,
  PINNED_CPU_BUFFER_POOL_MEM = 4,
  PAGEABLE_CPU_BUFFER_POOL_MEM = 5,
  PINNED_GPU_BUFFER_POOL_MEM = 6,
  PAGEABLE_GPU_BUFFER_POOL_MEM = 7,
  INVALID_SUBTYPE = 8,
  NUM_RESOURCE_SUBTYPES = 8,
};

static constexpr size_t ResourceSubtypeSize =
    static_cast<size_t>(ResourceSubtype::NUM_RESOURCE_SUBTYPES);

static const char* ResourceSubtypeStrings[] = {"cpu_slots",
                                               "gpu_slots",
                                               "cpu_result_mem",
                                               "gpu_result_mem",
                                               "pinned_cpu_buffer_pool_mem",
                                               "pinned_gpu_buffer_pool_mem",
                                               "pageable_cpu_buffer_pool_mem",
                                               "pageable_gpu_buffer_pool_mem",
                                               "invalid_type"};

inline std::string resource_subtype_to_string(const ResourceSubtype resource_subtype) {
  return ResourceSubtypeStrings[static_cast<size_t>(resource_subtype)];
}

}  // namespace ExecutorResourceMgr_Namespace
