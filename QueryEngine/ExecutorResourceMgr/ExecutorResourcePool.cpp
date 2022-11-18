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

#include <iostream>  // For debug print methods
#include <sstream>

#include "ExecutorResourcePool.h"

namespace ExecutorResourceMgr_Namespace {

const bool ENABLE_DEBUG_PRINTING{false};
static std::mutex debug_print_mutex_;

template <typename... Ts>
void debug_print(Ts&&... print_args) {
  std::unique_lock<std::mutex> print_lock(debug_print_mutex_);
  (std::cout << ... << print_args);
  std::cout << std::endl;
}

void ResourceGrant::print() const {
  std::cout << std::endl << "Request Grant Info" << std::endl;
  std::cout << "Grant CPU slots: " << cpu_slots << std::endl;
  std::cout << "Grant GPU slots: " << gpu_slots << std::endl;
  std::cout << "Grant CPU result mem: " << format_num_bytes(cpu_result_mem) << std::endl;
}

std::string ResourceGrant::to_string() const {
  std::ostringstream oss;
  oss << "Granted CPU Slots: " << cpu_slots << " GPU Slots: " << gpu_slots
      << " CPU result mem: " << format_num_bytes(cpu_result_mem);
  return oss.str();
}

ExecutorResourcePool::ExecutorResourcePool(
    const std::vector<std::pair<ResourceType, size_t>>& total_resources,
    const std::vector<ConcurrentResourceGrantPolicy>& concurrent_resource_grant_policies,
    const std::vector<ResourceGrantPolicy>& max_per_request_resource_grant_policies) {
  init(total_resources,
       concurrent_resource_grant_policies,
       max_per_request_resource_grant_policies);
  log_parameters();
}

void ExecutorResourcePool::init(
    const std::vector<std::pair<ResourceType, size_t>>& total_resources,
    const std::vector<ConcurrentResourceGrantPolicy>& concurrent_resource_grant_policies,
    const std::vector<ResourceGrantPolicy>& max_resource_grants_per_request_policies) {
  for (const auto& total_resource : total_resources) {
    if (total_resource.first == ResourceType::INVALID_TYPE) {
      continue;
    }
    total_resources_[static_cast<size_t>(total_resource.first)] = total_resource.second;
    resource_type_validity_[static_cast<size_t>(total_resource.first)] = true;
  }

  for (const auto& concurrent_resource_grant_policy :
       concurrent_resource_grant_policies) {
    const ResourceType resource_type = concurrent_resource_grant_policy.resource_type;
    if (resource_type == ResourceType::INVALID_TYPE) {
      continue;
    }
    concurrent_resource_grant_policies_[static_cast<size_t>(resource_type)] =
        concurrent_resource_grant_policy;
  }

  for (const auto& max_resource_grant_per_request_policy :
       max_resource_grants_per_request_policies) {
    const ResourceSubtype resource_subtype =
        max_resource_grant_per_request_policy.resource_subtype;
    if (resource_subtype == ResourceSubtype::INVALID_SUBTYPE) {
      continue;
    }
    max_resource_grants_per_request_policies_[static_cast<size_t>(resource_subtype)] =
        max_resource_grant_per_request_policy;
  }

  init_concurrency_policies();
  init_max_resource_grants_per_requests();
}

void ExecutorResourcePool::init_concurrency_policies() {
  size_t resource_type_idx = 0;
  for (auto& concurrent_resource_grant_policy : concurrent_resource_grant_policies_) {
    const auto resource_type = static_cast<ResourceType>(resource_type_idx);
    const auto concurrency_policy_resource_type =
        concurrent_resource_grant_policy.resource_type;
    CHECK(resource_type == concurrency_policy_resource_type ||
          concurrency_policy_resource_type == ResourceType::INVALID_TYPE);
    if (is_resource_valid(resource_type)) {
      if (concurrency_policy_resource_type == ResourceType::INVALID_TYPE) {
        concurrent_resource_grant_policy.resource_type = resource_type;
        concurrent_resource_grant_policy.concurrency_policy =
            ResourceConcurrencyPolicy::ALLOW_SINGLE_REQUEST;
        concurrent_resource_grant_policy.oversubscription_concurrency_policy =
            ResourceConcurrencyPolicy::DISALLOW_REQUESTS;
      }
    } else {
      concurrent_resource_grant_policy.resource_type = ResourceType::INVALID_TYPE;
    }
    resource_type_idx++;
  }
}

void ExecutorResourcePool::init_max_resource_grants_per_requests() {
  size_t resource_subtype_idx = 0;
  for (auto& max_resource_grant_per_request_policy :
       max_resource_grants_per_request_policies_) {
    const auto resource_subtype = static_cast<ResourceSubtype>(resource_subtype_idx);
    const auto resource_type = map_resource_subtype_to_resource_type(resource_subtype);
    const auto policy_resource_subtype =
        max_resource_grant_per_request_policy.resource_subtype;
    CHECK(resource_subtype == policy_resource_subtype ||
          policy_resource_subtype == ResourceSubtype::INVALID_SUBTYPE);
    if (is_resource_valid(resource_type)) {
      if (policy_resource_subtype == ResourceSubtype::INVALID_SUBTYPE) {
        max_resource_grant_per_request_policy.resource_subtype = resource_subtype;
        max_resource_grant_per_request_policy.policy_size_type =
            ResourceGrantPolicySizeType::UNLIMITED;
      }
      max_resource_grants_per_request_[static_cast<size_t>(
          max_resource_grant_per_request_policy.resource_subtype)] =
          max_resource_grant_per_request_policy.get_grant_quantity(
              get_total_resource(resource_type),
              get_concurrent_resource_grant_policy(resource_type)
                      .oversubscription_concurrency_policy ==
                  ResourceConcurrencyPolicy::DISALLOW_REQUESTS);
    } else {
      max_resource_grant_per_request_policy.resource_subtype =
          ResourceSubtype::INVALID_SUBTYPE;
    }
    resource_subtype_idx++;
  }
}

void ExecutorResourcePool::log_parameters() const {
  for (size_t resource_idx = 0; resource_idx < ResourceTypeSize; ++resource_idx) {
    const ResourceType resource_type = static_cast<ResourceType>(resource_idx);
    if (!is_resource_valid(resource_type)) {
      continue;
    }
    const auto total_resource = get_total_resource(resource_type);
    const auto resource_type_str = resource_type_to_string(resource_type);
    LOG(EXECUTOR) << "Resource: " << resource_type_str << ": " << total_resource;
    LOG(EXECUTOR) << "Concurrency Policy for " << resource_type_str << ": "
                  << get_concurrent_resource_grant_policy(resource_type).to_string();
    LOG(EXECUTOR) << "Max per-request resource grants for sub-types:";
    const auto resource_subtypes = map_resource_type_to_resource_subtypes(resource_type);
    for (const auto& resource_subtype : resource_subtypes) {
      LOG(EXECUTOR)
          << get_max_resource_grant_per_request_policy(resource_subtype).to_string();
    }
  }
}

size_t ExecutorResourcePool::get_allocated_resource_of_type(
    const ResourceType resource_type) const {
  const auto resource_subtypes = map_resource_type_to_resource_subtypes(resource_type);
  size_t resource_type_allocation_sum{0};
  for (const auto& resource_subtype : resource_subtypes) {
    resource_type_allocation_sum += get_allocated_resource_of_subtype(resource_subtype);
  }
  return resource_type_allocation_sum;
}

std::pair<size_t, size_t> ExecutorResourcePool::get_resource_info(
    const ResourceType resource_type) const {
  std::shared_lock<std::shared_mutex> resource_read_lock(resource_mutex_);
  return std::make_pair(get_allocated_resource_of_type(resource_type),
                        get_total_resource(resource_type));
}

ResourcePoolInfo ExecutorResourcePool::get_resource_info() const {
  std::shared_lock<std::shared_mutex> resource_read_lock(resource_mutex_);
  return ResourcePoolInfo(
      get_total_resource(ResourceType::CPU_SLOTS),
      get_total_resource(ResourceType::GPU_SLOTS),
      get_total_resource(ResourceType::CPU_RESULT_MEM),
      get_total_resource(ResourceType::CPU_BUFFER_POOL_MEM),
      get_total_resource(ResourceType::GPU_BUFFER_POOL_MEM),
      get_allocated_resource_of_type(ResourceType::CPU_SLOTS),
      get_allocated_resource_of_type(ResourceType::GPU_SLOTS),
      get_allocated_resource_of_type(ResourceType::CPU_RESULT_MEM),
      get_allocated_resource_of_type(ResourceType::CPU_BUFFER_POOL_MEM),
      get_allocated_resource_of_type(ResourceType::GPU_BUFFER_POOL_MEM),
      allocated_cpu_buffer_pool_chunks_.size(),
      allocated_gpu_buffer_pool_chunks_.size(),
      get_allocated_resource_of_subtype(ResourceSubtype::PAGEABLE_CPU_BUFFER_POOL_MEM),
      get_allocated_resource_of_subtype(ResourceSubtype::PAGEABLE_GPU_BUFFER_POOL_MEM),
      total_num_requests_,
      outstanding_num_requests_,
      get_outstanding_per_resource_num_requests(ResourceType::CPU_SLOTS),
      get_outstanding_per_resource_num_requests(ResourceType::GPU_SLOTS),
      get_outstanding_per_resource_num_requests(ResourceType::CPU_RESULT_MEM),
      get_outstanding_per_resource_num_requests(ResourceType::CPU_BUFFER_POOL_MEM),
      get_outstanding_per_resource_num_requests(ResourceType::GPU_BUFFER_POOL_MEM));
}

void ExecutorResourcePool::set_resource(const ResourceType resource_type,
                                        const size_t resource_quantity) {
  CHECK(resource_type != ResourceType::INVALID_TYPE);
  if (outstanding_num_requests_) {
    throw std::runtime_error(
        "Executor Pool must be clear of requests to change resources available.");
  }
  const std::vector<std::pair<ResourceType, size_t>> total_resources_vec = {
      std::make_pair(resource_type, resource_quantity)};
  init(total_resources_vec, {}, {});
}

void ExecutorResourcePool::set_concurrent_resource_grant_policy(
    const ConcurrentResourceGrantPolicy& concurrent_resource_grant_policy) {
  CHECK(concurrent_resource_grant_policy.resource_type != ResourceType::INVALID_TYPE);
  if (outstanding_num_requests_) {
    throw std::runtime_error(
        "Executor Pool must be clear of requests to change resource concurrent resource "
        "grant policies.");
  }
  init({}, {concurrent_resource_grant_policy}, {});
}

size_t ExecutorResourcePool::calc_max_resource_grant_for_request(
    const size_t requested_resource_quantity,
    const size_t min_requested_resource_quantity,
    const size_t max_grantable_resource_quantity) const {
  if (requested_resource_quantity <= max_grantable_resource_quantity) {
    return requested_resource_quantity;
  }
  if (min_requested_resource_quantity <= max_grantable_resource_quantity) {
    return max_grantable_resource_quantity;
  }
  return static_cast<size_t>(0);
}

std::pair<size_t, size_t>
ExecutorResourcePool::calc_min_dependent_resource_grant_for_request(
    const size_t min_requested_dependent_resource_quantity,
    const size_t min_requested_independent_resource_quantity,
    const size_t dependent_to_independent_resource_ratio) const {
  const size_t adjusted_min_independent_resource_quantity =
      std::max(static_cast<size_t>(
                   ceil(static_cast<double>(min_requested_dependent_resource_quantity) /
                        dependent_to_independent_resource_ratio)),
               min_requested_independent_resource_quantity);
  const size_t adjusted_min_dependent_resource_quantity =
      adjusted_min_independent_resource_quantity *
      dependent_to_independent_resource_ratio;
  return std::make_pair(adjusted_min_dependent_resource_quantity,
                        adjusted_min_independent_resource_quantity);
}

std::pair<size_t, size_t>
ExecutorResourcePool::calc_max_dependent_resource_grant_for_request(
    const size_t requested_dependent_resource_quantity,
    const size_t min_requested_dependent_resource_quantity,
    const size_t max_grantable_dependent_resource_quantity,
    const size_t min_requested_independent_resource_quantity,
    const size_t max_grantable_independent_resource_quantity,
    const size_t dependent_to_independent_resource_ratio) const {
  CHECK_LE(min_requested_dependent_resource_quantity,
           requested_dependent_resource_quantity);
  CHECK_LE(min_requested_independent_resource_quantity,
           max_grantable_independent_resource_quantity);

  if (requested_dependent_resource_quantity <=
      max_grantable_dependent_resource_quantity) {
    // Dependent resource request falls under max grantable limit, grant requested
    // resource
    return std::make_pair(requested_dependent_resource_quantity,
                          max_grantable_independent_resource_quantity);
  }
  // First member of pair returned is min resource grant, second is min dependent
  // resource grant
  const auto adjusted_min_dependent_and_independent_resource_grant =
      calc_min_dependent_resource_grant_for_request(
          min_requested_dependent_resource_quantity,
          min_requested_independent_resource_quantity,
          dependent_to_independent_resource_ratio);

  if (adjusted_min_dependent_and_independent_resource_grant.first >
      max_grantable_dependent_resource_quantity) {
    // If here the min grantable dependent resource is greater than what was to provided
    // to the function as grantable of the dependent resource
    return std::make_pair(static_cast<size_t>(0), static_cast<size_t>(0));
  }

  const size_t adjusted_max_independent_resource_quantity = std::min(
      max_grantable_dependent_resource_quantity / dependent_to_independent_resource_ratio,
      max_grantable_independent_resource_quantity);

  CHECK_GE(adjusted_max_independent_resource_quantity,
           adjusted_min_dependent_and_independent_resource_grant.second);

  const size_t granted_dependent_resource_quantity =
      dependent_to_independent_resource_ratio *
      adjusted_max_independent_resource_quantity;
  return std::make_pair(granted_dependent_resource_quantity,
                        adjusted_max_independent_resource_quantity);
}

void ExecutorResourcePool::throw_insufficient_resource_error(
    const ResourceSubtype resource_subtype,
    const size_t min_resource_requested) const {
  const size_t max_resource_grant_per_request =
      get_max_resource_grant_per_request(resource_subtype);

  switch (resource_subtype) {
    case ResourceSubtype::CPU_SLOTS:
      throw QueryNeedsTooManyCpuSlots(max_resource_grant_per_request,
                                      min_resource_requested);
    case ResourceSubtype::GPU_SLOTS:
      throw QueryNeedsTooManyGpuSlots(max_resource_grant_per_request,
                                      min_resource_requested);
    case ResourceSubtype::CPU_RESULT_MEM:
      throw QueryNeedsTooMuchCpuResultMem(max_resource_grant_per_request,
                                          min_resource_requested);
    default:
      throw std::runtime_error(
          "Insufficient resources for request");  // todo: just placeholder
  }
}

std::vector<ResourceRequestGrant>
ExecutorResourcePool::calc_static_resource_grant_ranges_for_request(
    const std::vector<ResourceRequest>& resource_requests) const {
  std::vector<ResourceRequestGrant> resource_request_grants;

  std::array<ResourceRequestGrant, ResourceSubtypeSize> all_resource_grants;
  for (const auto& resource_request : resource_requests) {
    CHECK(resource_request.resource_subtype != ResourceSubtype::INVALID_SUBTYPE);
    CHECK_LE(resource_request.min_quantity, resource_request.max_quantity);

    ResourceRequestGrant resource_grant;
    resource_grant.resource_subtype = resource_request.resource_subtype;
    resource_grant.max_quantity = calc_max_resource_grant_for_request(
        resource_request.max_quantity,
        resource_request.min_quantity,
        get_max_resource_grant_per_request(resource_request.resource_subtype));
    if (resource_grant.max_quantity < resource_request.min_quantity) {
      // Current implementation should always return 0 if it cannot grant requested amount
      CHECK_EQ(resource_grant.max_quantity, size_t(0));
      throw_insufficient_resource_error(resource_request.resource_subtype,
                                        resource_request.min_quantity);
    }
    all_resource_grants[static_cast<size_t>(resource_grant.resource_subtype)] =
        resource_grant;
  }
  return resource_request_grants;
}

std::pair<ResourceGrant, ResourceGrant>
ExecutorResourcePool::calc_min_max_resource_grants_for_request(
    const RequestInfo& request_info) const {
  ResourceGrant min_resource_grant, max_resource_grant;

  CHECK_LE(request_info.min_cpu_slots, request_info.cpu_slots);
  CHECK_LE(request_info.min_gpu_slots, request_info.gpu_slots);
  CHECK_LE(request_info.min_cpu_result_mem, request_info.cpu_result_mem);

  max_resource_grant.cpu_slots = calc_max_resource_grant_for_request(
      request_info.cpu_slots,
      request_info.min_cpu_slots,
      get_max_resource_grant_per_request(ResourceSubtype::CPU_SLOTS));
  if (max_resource_grant.cpu_slots == 0 && request_info.min_cpu_slots > 0) {
    throw QueryNeedsTooManyCpuSlots(
        get_max_resource_grant_per_request(ResourceSubtype::CPU_SLOTS),
        request_info.min_cpu_slots);
  }

  max_resource_grant.gpu_slots = calc_max_resource_grant_for_request(
      request_info.gpu_slots,
      request_info.min_gpu_slots,
      get_max_resource_grant_per_request(ResourceSubtype::GPU_SLOTS));
  if (max_resource_grant.gpu_slots == 0 && request_info.min_gpu_slots > 0) {
    throw QueryNeedsTooManyGpuSlots(
        get_max_resource_grant_per_request(ResourceSubtype::GPU_SLOTS),
        request_info.min_gpu_slots);
  }

  max_resource_grant.cpu_result_mem = calc_max_resource_grant_for_request(
      request_info.cpu_result_mem,
      request_info.min_cpu_result_mem,
      get_max_resource_grant_per_request(ResourceSubtype::CPU_RESULT_MEM));
  if (max_resource_grant.cpu_result_mem == 0 && request_info.min_cpu_result_mem > 0) {
    throw QueryNeedsTooMuchCpuResultMem(
        get_max_resource_grant_per_request(ResourceSubtype::CPU_RESULT_MEM),
        request_info.min_cpu_result_mem);
  }

  const auto& chunk_request_info = request_info.chunk_request_info;

  const size_t max_pinned_buffer_pool_grant_for_memory_level =
      get_max_resource_grant_per_request(
          chunk_request_info.device_memory_pool_type == ExecutorDeviceType::CPU
              ? ResourceSubtype::PINNED_CPU_BUFFER_POOL_MEM
              : ResourceSubtype::PINNED_GPU_BUFFER_POOL_MEM);

  if (chunk_request_info.total_bytes > max_pinned_buffer_pool_grant_for_memory_level) {
    if (!chunk_request_info.bytes_scales_per_kernel) {
      throw QueryNeedsTooMuchBufferPoolMem(max_pinned_buffer_pool_grant_for_memory_level,
                                           chunk_request_info.total_bytes,
                                           chunk_request_info.device_memory_pool_type);
    }
    // If here we have bytes_needed_scales_per_kernel
    // For now, this can only be for a CPU request, but that may be relaxed down the
    // road
    const size_t max_pageable_buffer_pool_grant_for_memory_level =
        get_max_resource_grant_per_request(
            chunk_request_info.device_memory_pool_type == ExecutorDeviceType::CPU
                ? ResourceSubtype::PAGEABLE_CPU_BUFFER_POOL_MEM
                : ResourceSubtype::PAGEABLE_GPU_BUFFER_POOL_MEM);
    CHECK(chunk_request_info.device_memory_pool_type == ExecutorDeviceType::CPU);
    const auto max_chunk_memory_and_cpu_slots_grant =
        calc_max_dependent_resource_grant_for_request(
            chunk_request_info.total_bytes,  // requested_dependent_resource_quantity
            chunk_request_info
                .max_bytes_per_kernel,  // min_requested_dependent_resource_quantity
            max_pageable_buffer_pool_grant_for_memory_level,  // max_grantable_dependent_resource_quantity
            request_info.min_cpu_slots,    // min_requested_independent_resource_quantity
            max_resource_grant.cpu_slots,  // max_grantable_indepndent_resource_quantity
            chunk_request_info
                .max_bytes_per_kernel);  // dependent_to_independent_resource_ratio

    CHECK_LE(max_chunk_memory_and_cpu_slots_grant.second, max_resource_grant.cpu_slots);
    if (max_chunk_memory_and_cpu_slots_grant.first == size_t(0)) {
      // Make sure cpu_slots is 0 as well
      CHECK_EQ(max_chunk_memory_and_cpu_slots_grant.second, size_t(0));
      // Get what min grant would have been if it was grantable so that we can present a
      // meaningful error message
      const auto adjusted_min_chunk_memory_and_cpu_slots_grant =
          calc_min_dependent_resource_grant_for_request(
              chunk_request_info
                  .max_bytes_per_kernel,   // min_requested_dependent_resource_quantity
              request_info.min_cpu_slots,  // min_requested_independent_resource_quantity
              chunk_request_info
                  .max_bytes_per_kernel);  // dependent_to_independent_resource_ratio
      // Ensure we would not have been able to satisfy this grant
      CHECK_GT(adjusted_min_chunk_memory_and_cpu_slots_grant.first,
               max_pageable_buffer_pool_grant_for_memory_level);
      // The logic for calc_min_dependent_resource_grant_for_request is constrained to
      // at least return at least the min dependent resource quantity requested, here
      // CPU slots
      CHECK_GE(adjusted_min_chunk_memory_and_cpu_slots_grant.second,
               request_info.min_cpu_slots);

      // May need additional error message as we could fail even though bytes per kernel
      // < total buffer pool bytes, if cpu slots < min requested cpu slots
      throw QueryNeedsTooMuchBufferPoolMem(
          max_pageable_buffer_pool_grant_for_memory_level,
          adjusted_min_chunk_memory_and_cpu_slots_grant
              .first,  // min chunk memory grant (without chunk grant constraints)
          chunk_request_info.device_memory_pool_type);
    }
    // If here query is allowed but cpu slots are gated to gate number of chunks
    // simultaneously pinned We should have been gated to a minimum of our request's
    // min_cpu_slots
    CHECK_GE(max_chunk_memory_and_cpu_slots_grant.second, request_info.min_cpu_slots);
    max_resource_grant.cpu_slots = max_chunk_memory_and_cpu_slots_grant.second;
    max_resource_grant.buffer_mem_gated_per_slot = true;
    min_resource_grant.buffer_mem_gated_per_slot = true;
    max_resource_grant.buffer_mem_per_slot = chunk_request_info.max_bytes_per_kernel;
    min_resource_grant.buffer_mem_per_slot = chunk_request_info.max_bytes_per_kernel;
    max_resource_grant.buffer_mem_for_given_slots =
        chunk_request_info.max_bytes_per_kernel * max_resource_grant.cpu_slots;
    min_resource_grant.buffer_mem_for_given_slots =
        chunk_request_info.max_bytes_per_kernel * request_info.min_cpu_slots;
  }

  min_resource_grant.cpu_slots = request_info.min_cpu_slots;
  min_resource_grant.gpu_slots = request_info.min_gpu_slots;
  min_resource_grant.cpu_result_mem = request_info.cpu_result_mem;

  return std::make_pair(min_resource_grant, max_resource_grant);
}

bool ExecutorResourcePool::check_request_against_global_policy(
    const size_t resource_total,
    const size_t resource_allocated,
    const ConcurrentResourceGrantPolicy& concurrent_resource_grant_policy) const {
  if (concurrent_resource_grant_policy.concurrency_policy ==
          ResourceConcurrencyPolicy::ALLOW_SINGLE_REQUEST_GLOBALLY &&
      resource_allocated > 0) {
    return false;
  }
  if (concurrent_resource_grant_policy.oversubscription_concurrency_policy ==
          ResourceConcurrencyPolicy::ALLOW_SINGLE_REQUEST_GLOBALLY &&
      resource_allocated > resource_total) {
    return false;
  }
  return true;
}

bool ExecutorResourcePool::check_request_against_policy(
    const size_t min_resource_request,
    const size_t resource_total,
    const size_t resource_allocated,
    const size_t global_outstanding_requests,
    const ConcurrentResourceGrantPolicy& concurrent_resource_grant_policy) const {
  auto test_request_against_policy =
      [min_resource_request, resource_allocated, global_outstanding_requests](
          const ResourceConcurrencyPolicy& resource_concurrency_policy) {
        switch (resource_concurrency_policy) {
          case ResourceConcurrencyPolicy::DISALLOW_REQUESTS: {
            // DISALLOW_REQUESTS for undersubscription policy doesn't make much sense as
            // a resource pool-wide policy (unless we are using it as a sanity check for
            // something like CPU mode), but planning to implement per-query or priority
            // level policies so will leave for now
            return min_resource_request == 0;
          }
          case ResourceConcurrencyPolicy::ALLOW_SINGLE_REQUEST_GLOBALLY: {
            // redundant with check_request_against_global_policy,
            // so considered CHECKing instead that the following cannot
            // be true, but didn't want to couple the two functions
            return global_outstanding_requests == 0;
          }
          case ResourceConcurrencyPolicy::ALLOW_SINGLE_REQUEST: {
            return min_resource_request == 0 || resource_allocated == 0;
          }
          case ResourceConcurrencyPolicy::ALLOW_CONCURRENT_REQUESTS: {
            return true;
          }
          default:
            UNREACHABLE();
        }
        return false;
      };

  if (!test_request_against_policy(concurrent_resource_grant_policy.concurrency_policy)) {
    return false;
  }
  if (min_resource_request + resource_allocated <= resource_total) {
    return true;
  }
  return test_request_against_policy(
      concurrent_resource_grant_policy.oversubscription_concurrency_policy);
}

/* Unlocked internal version */

bool ExecutorResourcePool::can_currently_satisfy_request_impl(
    const ResourceGrant& min_resource_grant,
    const ChunkRequestInfo& chunk_request_info) const {
  // Currently expects to be protected by mutex from ExecutorResourceMgr

  // Arguably exceptions below shouldn't happen as resource_grant,
  // if generated by ExecutorResourcePool per design, should be within
  // per query max limits. But since this is an external class api call and
  // the input could be anything provided by the caller, and we may want
  // to allow for dynamic per query limits, throwing instead of CHECKing
  // for now, but may re-evaluate.

  if (min_resource_grant.cpu_slots >
      get_max_resource_grant_per_request(ResourceSubtype::CPU_SLOTS)) {
    throw QueryNeedsTooManyCpuSlots(
        get_max_resource_grant_per_request(ResourceSubtype::CPU_SLOTS),
        min_resource_grant.cpu_slots);
  }
  if (min_resource_grant.gpu_slots >
      get_max_resource_grant_per_request(ResourceSubtype::GPU_SLOTS)) {
    throw QueryNeedsTooManyGpuSlots(
        get_max_resource_grant_per_request(ResourceSubtype::GPU_SLOTS),
        min_resource_grant.gpu_slots);
  }
  if (min_resource_grant.cpu_result_mem >
      get_max_resource_grant_per_request(ResourceSubtype::CPU_RESULT_MEM)) {
    throw QueryNeedsTooMuchCpuResultMem(
        get_max_resource_grant_per_request(ResourceSubtype::CPU_RESULT_MEM),
        min_resource_grant.cpu_result_mem);
  }

  // First check if request is in violation of any global
  // ALLOW_SINGLE_GLOBAL_REQUEST policies

  if (!check_request_against_global_policy(
          get_total_resource(ResourceType::CPU_SLOTS),
          get_allocated_resource_of_type(ResourceType::CPU_SLOTS),
          get_concurrent_resource_grant_policy(ResourceType::CPU_SLOTS))) {
    return false;
  }
  if (!check_request_against_global_policy(
          get_total_resource(ResourceType::GPU_SLOTS),
          get_allocated_resource_of_type(ResourceType::GPU_SLOTS),
          get_concurrent_resource_grant_policy(ResourceType::GPU_SLOTS))) {
    return false;
  }
  if (!check_request_against_global_policy(
          get_total_resource(ResourceType::CPU_RESULT_MEM),
          get_allocated_resource_of_type(ResourceType::CPU_RESULT_MEM),
          get_concurrent_resource_grant_policy(ResourceType::CPU_RESULT_MEM))) {
    return false;
  }

  const bool can_satisfy_cpu_slots_request = check_request_against_policy(
      min_resource_grant.cpu_slots,
      get_total_resource(ResourceType::CPU_SLOTS),
      get_allocated_resource_of_type(ResourceType::CPU_SLOTS),
      outstanding_num_requests_,
      get_concurrent_resource_grant_policy(ResourceType::CPU_SLOTS));

  const bool can_satisfy_gpu_slots_request = check_request_against_policy(
      min_resource_grant.gpu_slots,
      get_total_resource(ResourceType::GPU_SLOTS),
      get_allocated_resource_of_type(ResourceType::GPU_SLOTS),
      outstanding_num_requests_,
      get_concurrent_resource_grant_policy(ResourceType::GPU_SLOTS));

  const bool can_satisfy_cpu_result_mem_request = check_request_against_policy(
      min_resource_grant.cpu_result_mem,
      get_total_resource(ResourceType::CPU_RESULT_MEM),
      get_allocated_resource_of_type(ResourceType::CPU_RESULT_MEM),
      outstanding_num_requests_,
      get_concurrent_resource_grant_policy(ResourceType::CPU_RESULT_MEM));

  // Short circuit before heavier chunk check operation
  if (!(can_satisfy_cpu_slots_request && can_satisfy_gpu_slots_request &&
        can_satisfy_cpu_result_mem_request)) {
    return false;
  }

  return can_currently_satisfy_chunk_request(min_resource_grant, chunk_request_info);
}

ChunkRequestInfo ExecutorResourcePool::get_requested_chunks_not_in_pool(
    const ChunkRequestInfo& chunk_request_info) const {
  const BufferPoolChunkMap& chunk_map_for_memory_level =
      chunk_request_info.device_memory_pool_type == ExecutorDeviceType::CPU
          ? allocated_cpu_buffer_pool_chunks_
          : allocated_gpu_buffer_pool_chunks_;
  ChunkRequestInfo missing_chunk_info;
  missing_chunk_info.device_memory_pool_type = chunk_request_info.device_memory_pool_type;
  std::vector<std::pair<ChunkKey, size_t>> missing_chunks_with_byte_sizes;
  for (const auto& requested_chunk : chunk_request_info.chunks_with_byte_sizes) {
    if (chunk_map_for_memory_level.find(requested_chunk.first) ==
        chunk_map_for_memory_level.end()) {
      missing_chunk_info.chunks_with_byte_sizes.emplace_back(requested_chunk);
      missing_chunk_info.total_bytes += requested_chunk.second;
    }
  }
  missing_chunk_info.num_chunks = missing_chunk_info.chunks_with_byte_sizes.size();
  return missing_chunk_info;
}

size_t ExecutorResourcePool::get_chunk_bytes_not_in_pool(
    const ChunkRequestInfo& chunk_request_info) const {
  const BufferPoolChunkMap& chunk_map_for_memory_level =
      chunk_request_info.device_memory_pool_type == ExecutorDeviceType::CPU
          ? allocated_cpu_buffer_pool_chunks_
          : allocated_gpu_buffer_pool_chunks_;
  size_t chunk_bytes_not_in_pool{0};
  for (const auto& requested_chunk : chunk_request_info.chunks_with_byte_sizes) {
    const auto chunk_itr = chunk_map_for_memory_level.find(requested_chunk.first);
    if (chunk_itr == chunk_map_for_memory_level.end()) {
      chunk_bytes_not_in_pool += requested_chunk.second;
    } else if (requested_chunk.second > chunk_itr->second.second) {
      chunk_bytes_not_in_pool += requested_chunk.second - chunk_itr->second.second;
    }
  }
  return chunk_bytes_not_in_pool;
}

bool ExecutorResourcePool::can_currently_satisfy_chunk_request(
    const ResourceGrant& min_resource_grant,
    const ChunkRequestInfo& chunk_request_info) const {
  // Expects lock on resource_mutex_ already taken

  const size_t total_buffer_mem_for_memory_level =
      chunk_request_info.device_memory_pool_type == ExecutorDeviceType::CPU
          ? get_total_resource(ResourceType::CPU_BUFFER_POOL_MEM)
          : get_total_resource(ResourceType::GPU_BUFFER_POOL_MEM);
  const size_t allocated_buffer_mem_for_memory_level =
      get_total_allocated_buffer_pool_mem_for_level(
          chunk_request_info.device_memory_pool_type);

  if (min_resource_grant.buffer_mem_gated_per_slot) {
    CHECK_GT(min_resource_grant.buffer_mem_per_slot, size_t(0));
    // We only allow scaling back slots to cap buffer pool memory required on CPU
    // currently
    CHECK(chunk_request_info.device_memory_pool_type == ExecutorDeviceType::CPU);
    const size_t min_buffer_pool_mem_required =
        min_resource_grant.cpu_slots * min_resource_grant.buffer_mem_per_slot;
    // Below is a sanity check... we'll never be able to run the query if minimum pool
    // memory required is not <= the total buffer pool memory
    CHECK_LE(min_buffer_pool_mem_required, total_buffer_mem_for_memory_level);
    return allocated_buffer_mem_for_memory_level + min_buffer_pool_mem_required <=
           total_buffer_mem_for_memory_level;
  }

  // CHECK and not exception as parent should have checked this, can re-evaluate whether
  // should be exception
  CHECK_LE(chunk_request_info.total_bytes, total_buffer_mem_for_memory_level);
  const size_t chunk_bytes_not_in_pool = get_chunk_bytes_not_in_pool(chunk_request_info);
  if (ENABLE_DEBUG_PRINTING) {
    debug_print("Chunk bytes not in pool: ", format_num_bytes(chunk_bytes_not_in_pool));
  }
  return chunk_bytes_not_in_pool + allocated_buffer_mem_for_memory_level <=
         total_buffer_mem_for_memory_level;
}

void ExecutorResourcePool::add_chunk_requests_to_allocated_pool(
    const ResourceGrant& resource_grant,
    const ChunkRequestInfo& chunk_request_info) {
  // Expects lock on resource_mutex_ already taken

  if (resource_grant.buffer_mem_gated_per_slot) {
    CHECK(chunk_request_info.device_memory_pool_type == ExecutorDeviceType::CPU);
    CHECK_LE(get_total_allocated_buffer_pool_mem_for_level(
                 chunk_request_info.device_memory_pool_type) +
                 resource_grant.buffer_mem_for_given_slots,
             get_total_resource(ResourceType::CPU_BUFFER_POOL_MEM));
    allocated_resources_[static_cast<size_t>(
        ResourceSubtype::PAGEABLE_CPU_BUFFER_POOL_MEM)] +=
        resource_grant.buffer_mem_for_given_slots;

    const std::string& pool_level_string =
        chunk_request_info.device_memory_pool_type == ExecutorDeviceType::CPU ? "CPU"
                                                                              : "GPU";
    LOG(EXECUTOR) << "ExecutorResourePool " << pool_level_string
                  << " allocated_temp chunk addition: "
                  << format_num_bytes(resource_grant.buffer_mem_for_given_slots);
    LOG(EXECUTOR) << "ExecutorResourePool " << pool_level_string
                  << " pool state: Transient Allocations: "
                  << format_num_bytes(get_allocated_resource_of_subtype(
                         ResourceSubtype::PAGEABLE_CPU_BUFFER_POOL_MEM))
                  << " Total Allocations: "
                  << format_num_bytes(get_total_allocated_buffer_pool_mem_for_level(
                         chunk_request_info.device_memory_pool_type));
    return;
  }

  BufferPoolChunkMap& chunk_map_for_memory_level =
      chunk_request_info.device_memory_pool_type == ExecutorDeviceType::CPU
          ? allocated_cpu_buffer_pool_chunks_
          : allocated_gpu_buffer_pool_chunks_;
  size_t& pinned_buffer_mem_for_memory_level =
      chunk_request_info.device_memory_pool_type == ExecutorDeviceType::CPU
          ? allocated_resources_[static_cast<size_t>(
                ResourceSubtype::PINNED_CPU_BUFFER_POOL_MEM)]
          : allocated_resources_[static_cast<size_t>(
                ResourceSubtype::PINNED_GPU_BUFFER_POOL_MEM)];
  const size_t total_buffer_mem_for_memory_level =
      chunk_request_info.device_memory_pool_type == ExecutorDeviceType::CPU
          ? get_total_resource(ResourceType::CPU_BUFFER_POOL_MEM)
          : get_total_resource(ResourceType::GPU_BUFFER_POOL_MEM);

  // Following variables are for logging
  const size_t pre_pinned_chunks_for_memory_level = chunk_map_for_memory_level.size();
  const size_t pre_pinned_buffer_mem_for_memory_level =
      pinned_buffer_mem_for_memory_level;

  for (const auto& requested_chunk : chunk_request_info.chunks_with_byte_sizes) {
    auto chunk_itr = chunk_map_for_memory_level.find(requested_chunk.first);
    if (chunk_itr == chunk_map_for_memory_level.end()) {
      pinned_buffer_mem_for_memory_level += requested_chunk.second;
      chunk_map_for_memory_level.insert(
          std::make_pair(requested_chunk.first,
                         std::make_pair(size_t(1) /* initial reference count */,
                                        requested_chunk.second)));
    } else {
      if (requested_chunk.second > chunk_itr->second.second) {
        pinned_buffer_mem_for_memory_level +=
            requested_chunk.second - chunk_itr->second.second;
        chunk_itr->second.second = requested_chunk.second;
      }
      chunk_itr->second.first += 1;  // Add reference count
    }
  }
  const size_t post_pinned_chunks_for_memory_level = chunk_map_for_memory_level.size();
  const size_t net_new_allocated_chunks =
      post_pinned_chunks_for_memory_level - pre_pinned_chunks_for_memory_level;
  const size_t net_new_allocated_memory =
      pinned_buffer_mem_for_memory_level - pre_pinned_buffer_mem_for_memory_level;

  const std::string& pool_level_string =
      chunk_request_info.device_memory_pool_type == ExecutorDeviceType::CPU ? "CPU"
                                                                            : "GPU";
  LOG(EXECUTOR) << "ExecutorResourePool " << pool_level_string
                << " chunk allocation: " << chunk_request_info.num_chunks << " chunks | "
                << format_num_bytes(chunk_request_info.total_bytes);
  LOG(EXECUTOR) << "ExecutorResourePool " << pool_level_string
                << " pool delta: " << net_new_allocated_chunks << " chunks added | "
                << format_num_bytes(net_new_allocated_memory);
  LOG(EXECUTOR) << "ExecutorResourePool " << pool_level_string
                << " pool state: " << post_pinned_chunks_for_memory_level << " chunks | "
                << format_num_bytes(get_total_allocated_buffer_pool_mem_for_level(
                       chunk_request_info.device_memory_pool_type));

  if (ENABLE_DEBUG_PRINTING) {
    debug_print("After chunk allocation: ",
                format_num_bytes(pinned_buffer_mem_for_memory_level),
                " of ",
                format_num_bytes(total_buffer_mem_for_memory_level),
                ", with ",
                chunk_map_for_memory_level.size(),
                " chunks.");
  }
  CHECK_LE(pinned_buffer_mem_for_memory_level, total_buffer_mem_for_memory_level);
}

void ExecutorResourcePool::remove_chunk_requests_from_allocated_pool(
    const ResourceGrant& resource_grant,
    const ChunkRequestInfo& chunk_request_info) {
  // Expects lock on resource_mutex_ already taken

  if (resource_grant.buffer_mem_gated_per_slot) {
    CHECK(chunk_request_info.device_memory_pool_type == ExecutorDeviceType::CPU);
    CHECK_GE(
        get_allocated_resource_of_subtype(ResourceSubtype::PAGEABLE_CPU_BUFFER_POOL_MEM),
        resource_grant.buffer_mem_for_given_slots);
    CHECK_GE(
        get_allocated_resource_of_subtype(ResourceSubtype::PAGEABLE_CPU_BUFFER_POOL_MEM),
        resource_grant.buffer_mem_for_given_slots);
    allocated_resources_[static_cast<size_t>(
        ResourceSubtype::PAGEABLE_CPU_BUFFER_POOL_MEM)] -=
        resource_grant.buffer_mem_for_given_slots;
    const std::string& pool_level_string =
        chunk_request_info.device_memory_pool_type == ExecutorDeviceType::CPU ? "CPU"
                                                                              : "GPU";
    LOG(EXECUTOR) << "ExecutorResourePool " << pool_level_string
                  << " allocated_temp chunk removal: "
                  << format_num_bytes(resource_grant.buffer_mem_for_given_slots);
    LOG(EXECUTOR) << "ExecutorResourePool " << pool_level_string
                  << " pool state: Transient Allocations: "
                  << format_num_bytes(get_allocated_resource_of_subtype(
                         ResourceSubtype::PAGEABLE_CPU_BUFFER_POOL_MEM))
                  << " Total Allocations: "
                  << format_num_bytes(get_allocated_resource_of_type(
                         ResourceType::CPU_BUFFER_POOL_MEM));
    return;
  }

  BufferPoolChunkMap& chunk_map_for_memory_level =
      chunk_request_info.device_memory_pool_type == ExecutorDeviceType::CPU
          ? allocated_cpu_buffer_pool_chunks_
          : allocated_gpu_buffer_pool_chunks_;
  size_t& pinned_buffer_mem_for_memory_level =
      chunk_request_info.device_memory_pool_type == ExecutorDeviceType::CPU
          ? allocated_resources_[static_cast<size_t>(
                ResourceSubtype::PINNED_CPU_BUFFER_POOL_MEM)]
          : allocated_resources_[static_cast<size_t>(
                ResourceSubtype::PINNED_GPU_BUFFER_POOL_MEM)];

  // Following variables are for logging
  const size_t pre_remove_allocated_chunks_for_memory_level =
      chunk_map_for_memory_level.size();
  const size_t pre_remove_allocated_buffer_mem_for_memory_level =
      pinned_buffer_mem_for_memory_level;

  for (const auto& requested_chunk : chunk_request_info.chunks_with_byte_sizes) {
    auto chunk_itr = chunk_map_for_memory_level.find(requested_chunk.first);
    // Chunk must exist in pool
    CHECK(chunk_itr != chunk_map_for_memory_level.end());
    chunk_itr->second.first -= 1;
    if (chunk_itr->second.first == size_t(0)) {
      pinned_buffer_mem_for_memory_level -= chunk_itr->second.second;
      chunk_map_for_memory_level.erase(chunk_itr);
    }
  }
  const size_t total_buffer_mem_for_memory_level =
      chunk_request_info.device_memory_pool_type == ExecutorDeviceType::CPU
          ? get_total_resource(ResourceType::CPU_BUFFER_POOL_MEM)
          : get_total_resource(ResourceType::GPU_BUFFER_POOL_MEM);

  const size_t post_remove_allocated_chunks_for_memory_level =
      chunk_map_for_memory_level.size();
  const size_t net_removed_allocated_chunks =
      pre_remove_allocated_chunks_for_memory_level -
      post_remove_allocated_chunks_for_memory_level;
  const size_t net_removed_allocated_memory =
      pre_remove_allocated_buffer_mem_for_memory_level -
      pinned_buffer_mem_for_memory_level;

  const std::string& pool_level_string =
      chunk_request_info.device_memory_pool_type == ExecutorDeviceType::CPU ? "CPU"
                                                                            : "GPU";
  LOG(EXECUTOR) << "ExecutorResourePool " << pool_level_string
                << " chunk removal: " << chunk_request_info.num_chunks << " chunks | "
                << format_num_bytes(chunk_request_info.total_bytes);
  LOG(EXECUTOR) << "ExecutorResourePool " << pool_level_string
                << " pool delta: " << net_removed_allocated_chunks << " chunks removed | "
                << format_num_bytes(net_removed_allocated_memory);
  LOG(EXECUTOR) << "ExecutorResourePool " << pool_level_string
                << " pool state: " << post_remove_allocated_chunks_for_memory_level
                << " chunks | " << format_num_bytes(pinned_buffer_mem_for_memory_level);

  if (ENABLE_DEBUG_PRINTING) {
    debug_print("After chunk removal: ",
                format_num_bytes(pinned_buffer_mem_for_memory_level) + " of ",
                format_num_bytes(total_buffer_mem_for_memory_level),
                ", with ",
                chunk_map_for_memory_level.size(),
                " chunks.");
  }
}

bool ExecutorResourcePool::can_currently_satisfy_request(
    const ResourceGrant& min_resource_grant,
    const ChunkRequestInfo& chunk_request_info) const {
  std::shared_lock<std::shared_mutex> resource_read_lock(resource_mutex_);
  return can_currently_satisfy_request_impl(min_resource_grant, chunk_request_info);
}

size_t ExecutorResourcePool::determine_dynamic_single_resource_grant(
    const size_t min_resource_requested,
    const size_t max_resource_requested,
    const size_t resource_allocated,
    const size_t total_resource,
    const double max_request_backoff_ratio) const {
  CHECK_LE(min_resource_requested, max_resource_requested);
  if (min_resource_requested + resource_allocated >= total_resource) {
    return min_resource_requested;
  }
  // The below is safe in unsigned math as we know that resource_allocated <
  // total_resource from the above conditional
  const size_t resource_remaining = total_resource - resource_allocated;
  return std::max(min_resource_requested,
                  std::min(max_resource_requested,
                           static_cast<size_t>(
                               round(max_request_backoff_ratio * resource_remaining))));
}

std::pair<bool, ResourceGrant> ExecutorResourcePool::determine_dynamic_resource_grant(
    const ResourceGrant& min_resource_grant,
    const ResourceGrant& max_resource_grant,
    const ChunkRequestInfo& chunk_request_info,
    const double max_request_backoff_ratio) const {
  std::unique_lock<std::shared_mutex> resource_write_lock(resource_mutex_);
  CHECK_LE(max_request_backoff_ratio, 1.0);
  const bool can_satisfy_request =
      can_currently_satisfy_request_impl(min_resource_grant, chunk_request_info);
  ResourceGrant actual_resource_grant;
  if (!can_satisfy_request) {
    return std::make_pair(false, actual_resource_grant);
  }
  actual_resource_grant.cpu_slots = determine_dynamic_single_resource_grant(
      min_resource_grant.cpu_slots,
      max_resource_grant.cpu_slots,
      get_allocated_resource_of_type(ResourceType::CPU_SLOTS),
      get_total_resource(ResourceType::CPU_SLOTS),
      max_request_backoff_ratio);
  actual_resource_grant.gpu_slots = determine_dynamic_single_resource_grant(
      min_resource_grant.gpu_slots,
      max_resource_grant.gpu_slots,
      get_allocated_resource_of_type(ResourceType::GPU_SLOTS),
      get_total_resource(ResourceType::GPU_SLOTS),
      max_request_backoff_ratio);
  actual_resource_grant.cpu_result_mem = determine_dynamic_single_resource_grant(
      min_resource_grant.cpu_result_mem,
      max_resource_grant.cpu_result_mem,
      get_allocated_resource_of_type(ResourceType::CPU_RESULT_MEM),
      get_total_resource(ResourceType::CPU_RESULT_MEM),
      max_request_backoff_ratio);
  if (min_resource_grant.buffer_mem_gated_per_slot) {
    CHECK(chunk_request_info.device_memory_pool_type == ExecutorDeviceType::CPU);
    // Below is quite redundant, but can revisit
    CHECK_EQ(chunk_request_info.max_bytes_per_kernel,
             min_resource_grant.buffer_mem_per_slot);
    CHECK_EQ(chunk_request_info.max_bytes_per_kernel,
             max_resource_grant.buffer_mem_per_slot);

    const size_t allocated_buffer_mem_for_memory_level =
        chunk_request_info.device_memory_pool_type == ExecutorDeviceType::CPU
            ? get_allocated_resource_of_type(ResourceType::CPU_BUFFER_POOL_MEM)
            : get_allocated_resource_of_type(ResourceType::GPU_BUFFER_POOL_MEM);
    const size_t total_buffer_mem_for_memory_level =
        chunk_request_info.device_memory_pool_type == ExecutorDeviceType::CPU
            ? get_total_resource(ResourceType::CPU_BUFFER_POOL_MEM)
            : get_total_resource(ResourceType::GPU_BUFFER_POOL_MEM);

    CHECK_LE(allocated_buffer_mem_for_memory_level, total_buffer_mem_for_memory_level);

    const size_t remaining_buffer_mem_for_memory_level =
        total_buffer_mem_for_memory_level - allocated_buffer_mem_for_memory_level;

    CHECK_LE(min_resource_grant.buffer_mem_for_given_slots,
             remaining_buffer_mem_for_memory_level);
    const size_t max_grantable_mem =
        std::min(remaining_buffer_mem_for_memory_level,
                 max_resource_grant.buffer_mem_for_given_slots);
    const auto granted_buffer_mem_and_cpu_slots =
        calc_max_dependent_resource_grant_for_request(
            chunk_request_info.total_bytes,  // requested_dependent_resource_quantity
            min_resource_grant
                .buffer_mem_for_given_slots,  // min_requested_dependent_resource_quantity
            max_grantable_mem,                // max_grantable_dependent_resource_quantity
            min_resource_grant.cpu_slots,  // min_requested_independent_resource_quantity
            max_resource_grant.cpu_slots,  // max_grantable_independent_resource_quantity
            chunk_request_info
                .max_bytes_per_kernel);  // dependent_to_independent_resource_ratio
    const size_t granted_buffer_mem = granted_buffer_mem_and_cpu_slots.first;
    const size_t granted_cpu_slots = granted_buffer_mem_and_cpu_slots.second;
    CHECK_EQ(granted_buffer_mem,
             granted_cpu_slots * chunk_request_info.max_bytes_per_kernel);
    CHECK_GE(granted_cpu_slots, min_resource_grant.cpu_slots);
    CHECK_LE(granted_cpu_slots, max_resource_grant.cpu_slots);
    actual_resource_grant.buffer_mem_gated_per_slot = true;
    actual_resource_grant.buffer_mem_per_slot = chunk_request_info.max_bytes_per_kernel;
    actual_resource_grant.buffer_mem_for_given_slots = granted_buffer_mem;
    actual_resource_grant.cpu_slots =
        granted_cpu_slots;  // Override cpu slots with restricted dependent resource
                            // calc
  }
  return std::make_pair(true, actual_resource_grant);
}

void ExecutorResourcePool::allocate_resources(
    const ResourceGrant& resource_grant,
    const ChunkRequestInfo& chunk_request_info) {
  std::unique_lock<std::shared_mutex> resource_write_lock(resource_mutex_);

  // Caller (ExecutorResourceMgr) should never request resource allocation for a request
  // it knows cannot be granted, however use below as a sanity check Use unlocked
  // internal method as we already hold lock above
  const bool can_satisfy_request =
      can_currently_satisfy_request_impl(resource_grant, chunk_request_info);
  CHECK(can_satisfy_request);

  allocated_resources_[static_cast<size_t>(ResourceSubtype::CPU_SLOTS)] +=
      resource_grant.cpu_slots;
  allocated_resources_[static_cast<size_t>(ResourceSubtype::GPU_SLOTS)] +=
      resource_grant.gpu_slots;
  allocated_resources_[static_cast<size_t>(ResourceSubtype::CPU_RESULT_MEM)] +=
      resource_grant.cpu_result_mem;

  total_num_requests_++;
  outstanding_num_requests_++;
  if (resource_grant.cpu_slots > 0) {
    increment_outstanding_per_resource_num_requests(ResourceType::CPU_SLOTS);
    increment_total_per_resource_num_requests(ResourceType::CPU_SLOTS);
  }
  if (resource_grant.gpu_slots > 0) {
    increment_outstanding_per_resource_num_requests(ResourceType::GPU_SLOTS);
    increment_total_per_resource_num_requests(ResourceType::GPU_SLOTS);
  }
  if (resource_grant.cpu_result_mem > 0) {
    increment_outstanding_per_resource_num_requests(ResourceType::CPU_RESULT_MEM);
    increment_total_per_resource_num_requests(ResourceType::CPU_RESULT_MEM);
  }
  if (chunk_request_info.device_memory_pool_type == ExecutorDeviceType::CPU) {
    if (resource_grant.buffer_mem_gated_per_slot || chunk_request_info.num_chunks > 0) {
      increment_outstanding_per_resource_num_requests(ResourceType::CPU_BUFFER_POOL_MEM);
      increment_total_per_resource_num_requests(ResourceType::CPU_BUFFER_POOL_MEM);
    }
  } else if (chunk_request_info.device_memory_pool_type == ExecutorDeviceType::GPU) {
    if (resource_grant.buffer_mem_gated_per_slot || chunk_request_info.num_chunks > 0) {
      increment_outstanding_per_resource_num_requests(ResourceType::GPU_BUFFER_POOL_MEM);
      increment_total_per_resource_num_requests(ResourceType::GPU_BUFFER_POOL_MEM);
    }
  }

  LOG(EXECUTOR) << "ExecutorResourcePool allocation: " << outstanding_num_requests_
                << " requests ("
                << get_outstanding_per_resource_num_requests(ResourceType::CPU_SLOTS)
                << " CPU | "
                << get_outstanding_per_resource_num_requests(ResourceType::GPU_SLOTS)
                << " GPU)";
  LOG(EXECUTOR) << "ExecutorResourcePool state: CPU slots: "
                << get_allocated_resource_of_type(ResourceType::CPU_SLOTS) << " of "
                << get_total_resource(ResourceType::CPU_SLOTS) << " | GPU slots: "
                << get_allocated_resource_of_type(ResourceType::GPU_SLOTS) << " of "
                << get_total_resource(ResourceType::GPU_SLOTS) << " CPU result mem: "
                << format_num_bytes(
                       get_allocated_resource_of_type(ResourceType::CPU_RESULT_MEM))
                << " of "
                << format_num_bytes(get_total_resource(ResourceType::CPU_RESULT_MEM));
  add_chunk_requests_to_allocated_pool(resource_grant, chunk_request_info);
}

void ExecutorResourcePool::deallocate_resources(
    const ResourceGrant& resource_grant,
    const ChunkRequestInfo& chunk_request_info) {
  std::unique_lock<std::shared_mutex> resource_write_lock(resource_mutex_);

  // Caller (ExecutorResourceMgr) should never request resource allocation for a request
  // it knows cannot be granted, however use below as a sanity check

  CHECK_LE(resource_grant.cpu_slots,
           get_allocated_resource_of_type(ResourceType::CPU_SLOTS));
  CHECK_LE(resource_grant.gpu_slots,
           get_allocated_resource_of_type(ResourceType::GPU_SLOTS));
  CHECK_LE(resource_grant.cpu_result_mem,
           get_allocated_resource_of_type(ResourceType::CPU_RESULT_MEM));

  allocated_resources_[static_cast<size_t>(ResourceSubtype::CPU_SLOTS)] -=
      resource_grant.cpu_slots;
  allocated_resources_[static_cast<size_t>(ResourceSubtype::GPU_SLOTS)] -=
      resource_grant.gpu_slots;
  allocated_resources_[static_cast<size_t>(ResourceSubtype::CPU_RESULT_MEM)] -=
      resource_grant.cpu_result_mem;

  outstanding_num_requests_--;
  if (resource_grant.cpu_slots > 0) {
    decrement_outstanding_per_resource_num_requests(ResourceType::CPU_SLOTS);
  }
  if (resource_grant.gpu_slots > 0) {
    decrement_outstanding_per_resource_num_requests(ResourceType::GPU_SLOTS);
  }
  if (resource_grant.cpu_result_mem > 0) {
    decrement_outstanding_per_resource_num_requests(ResourceType::CPU_RESULT_MEM);
  }
  if (chunk_request_info.device_memory_pool_type == ExecutorDeviceType::CPU) {
    if (resource_grant.buffer_mem_gated_per_slot || chunk_request_info.num_chunks > 0) {
      decrement_outstanding_per_resource_num_requests(ResourceType::CPU_BUFFER_POOL_MEM);
    }
  } else if (chunk_request_info.device_memory_pool_type == ExecutorDeviceType::GPU) {
    if (resource_grant.buffer_mem_gated_per_slot || chunk_request_info.num_chunks > 0) {
      decrement_outstanding_per_resource_num_requests(ResourceType::GPU_BUFFER_POOL_MEM);
    }
  }

  LOG(EXECUTOR) << "ExecutorResourcePool de-allocation: " << outstanding_num_requests_
                << " requests ("
                << get_outstanding_per_resource_num_requests(ResourceType::CPU_SLOTS)
                << " CPU | "
                << get_outstanding_per_resource_num_requests(ResourceType::GPU_SLOTS)
                << " GPU)";
  LOG(EXECUTOR) << "ExecutorResourcePool state: CPU slots: "
                << get_allocated_resource_of_type(ResourceType::CPU_SLOTS) << " of "
                << get_total_resource(ResourceType::CPU_SLOTS) << " | GPU slots: "
                << get_allocated_resource_of_type(ResourceType::GPU_SLOTS) << " of "
                << get_total_resource(ResourceType::GPU_SLOTS) << " CPU result mem: "
                << format_num_bytes(
                       get_allocated_resource_of_type(ResourceType::CPU_RESULT_MEM))
                << " of "
                << format_num_bytes(get_total_resource(ResourceType::CPU_RESULT_MEM));
  remove_chunk_requests_from_allocated_pool(resource_grant, chunk_request_info);

  if (sanity_check_pool_state_on_deallocations_) {
    sanity_check_requests_against_allocations();
  }
}

void ExecutorResourcePool::sanity_check_requests_against_allocations() const {
  const size_t sum_resource_requests =
      get_outstanding_per_resource_num_requests(ResourceType::CPU_SLOTS) +
      get_outstanding_per_resource_num_requests(ResourceType::GPU_SLOTS) +
      get_outstanding_per_resource_num_requests(ResourceType::CPU_RESULT_MEM);

  CHECK_LE(outstanding_num_requests_, total_num_requests_);
  CHECK_LE(outstanding_num_requests_, sum_resource_requests);
  const bool has_outstanding_resource_requests = sum_resource_requests > 0;
  const bool has_outstanding_num_requests_globally = outstanding_num_requests_ > 0;
  CHECK_EQ(has_outstanding_resource_requests, has_outstanding_num_requests_globally);

  CHECK_EQ(get_outstanding_per_resource_num_requests(ResourceType::CPU_SLOTS) > 0,
           get_allocated_resource_of_type(ResourceType::CPU_SLOTS) > 0);
  CHECK_LE(get_outstanding_per_resource_num_requests(ResourceType::CPU_SLOTS),
           get_allocated_resource_of_type(ResourceType::CPU_SLOTS));

  CHECK_EQ(get_outstanding_per_resource_num_requests(ResourceType::GPU_SLOTS) > 0,
           get_allocated_resource_of_type(ResourceType::GPU_SLOTS) > 0);
  CHECK_LE(get_outstanding_per_resource_num_requests(ResourceType::GPU_SLOTS),
           get_allocated_resource_of_type(ResourceType::GPU_SLOTS));

  CHECK_EQ(get_outstanding_per_resource_num_requests(ResourceType::CPU_RESULT_MEM) > 0,
           get_allocated_resource_of_type(ResourceType::CPU_RESULT_MEM) > 0);
  CHECK_LE(get_outstanding_per_resource_num_requests(ResourceType::CPU_RESULT_MEM),
           get_allocated_resource_of_type(ResourceType::CPU_RESULT_MEM));

  CHECK_EQ(
      get_outstanding_per_resource_num_requests(ResourceType::CPU_BUFFER_POOL_MEM) > 0,
      get_allocated_resource_of_type(ResourceType::CPU_BUFFER_POOL_MEM) > 0);

  CHECK_EQ(
      get_outstanding_per_resource_num_requests(ResourceType::GPU_BUFFER_POOL_MEM) > 0,
      get_allocated_resource_of_type(ResourceType::GPU_BUFFER_POOL_MEM) > 0);

  if (outstanding_num_requests_ == static_cast<size_t>(0)) {
    CHECK_EQ(get_allocated_resource_of_type(ResourceType::CPU_BUFFER_POOL_MEM),
             size_t(0));
    CHECK_EQ(get_allocated_resource_of_type(ResourceType::GPU_BUFFER_POOL_MEM),
             size_t(0));
    CHECK(allocated_cpu_buffer_pool_chunks_.empty());
    CHECK(allocated_gpu_buffer_pool_chunks_.empty());
  }
}

}  // namespace ExecutorResourceMgr_Namespace
