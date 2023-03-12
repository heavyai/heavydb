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

#include <array>
#include <map>
#include <shared_mutex>

#include "ExecutorResourceMgrCommon.h"
#include "ResourceGrantPolicy.h"
#include "ResourceRequest.h"

namespace ExecutorResourceMgr_Namespace {

/**
 * @brief Returns the `ResourceType` associated with a given `ResourceSubtype`
 *
 * @param resource_subtype - The `ResourceSubtype` to find the parent `ResourceType` for
 * @return ResourceType - The resource type that subsumes (as a category) the specified
 * `resource_subtype`
 */
inline ResourceType map_resource_subtype_to_resource_type(
    const ResourceSubtype resource_subtype) {
  switch (resource_subtype) {
    case ResourceSubtype::CPU_SLOTS:
      return ResourceType::CPU_SLOTS;
    case ResourceSubtype::GPU_SLOTS:
      return ResourceType::GPU_SLOTS;
    case ResourceSubtype::CPU_RESULT_MEM:
      return ResourceType::CPU_RESULT_MEM;
    case ResourceSubtype::GPU_RESULT_MEM:
      return ResourceType::GPU_RESULT_MEM;
    case ResourceSubtype::PINNED_CPU_BUFFER_POOL_MEM:
      return ResourceType::CPU_BUFFER_POOL_MEM;
    case ResourceSubtype::PINNED_GPU_BUFFER_POOL_MEM:
      return ResourceType::GPU_BUFFER_POOL_MEM;
    case ResourceSubtype::PAGEABLE_CPU_BUFFER_POOL_MEM:
      return ResourceType::CPU_BUFFER_POOL_MEM;
    case ResourceSubtype::PAGEABLE_GPU_BUFFER_POOL_MEM:
      return ResourceType::GPU_BUFFER_POOL_MEM;
    default:
      UNREACHABLE();
      return ResourceType::INVALID_TYPE;
  }
}

/**
 * @brief Returns the 1-or-more `ResourceSubtype`s associated with a
 * given `ResourceType`.
 *
 * @param resource_type - The parent `ResourceType`
 * @return std::vector<ResourceSubtype> - A vector of one or more `ResourceSubtype`s
 * associated with the `ResourceType`
 */
inline std::vector<ResourceSubtype> map_resource_type_to_resource_subtypes(
    const ResourceType resource_type) {
  switch (resource_type) {
    case ResourceType::CPU_SLOTS:
      return {ResourceSubtype::CPU_SLOTS};
    case ResourceType::GPU_SLOTS:
      return {ResourceSubtype::GPU_SLOTS};
    case ResourceType::CPU_RESULT_MEM:
      return {ResourceSubtype::CPU_RESULT_MEM};
    case ResourceType::GPU_RESULT_MEM:
      return {ResourceSubtype::GPU_RESULT_MEM};
    case ResourceType::CPU_BUFFER_POOL_MEM:
      return {ResourceSubtype::PINNED_CPU_BUFFER_POOL_MEM,
              ResourceSubtype::PAGEABLE_CPU_BUFFER_POOL_MEM};
    case ResourceType::GPU_BUFFER_POOL_MEM:
      return {ResourceSubtype::PINNED_GPU_BUFFER_POOL_MEM,
              ResourceSubtype::PAGEABLE_GPU_BUFFER_POOL_MEM};
    default:
      UNREACHABLE();
      return {ResourceSubtype::INVALID_SUBTYPE};
  }
}

/**
 * @brief A container for various stats about the current state of the
 * `ExecutorResourcePool`. Note that `ExecutorResourcePool` does not persist
 * a struct of this type, but rather builds one on the fly when
 * `ExecutorResourcePool::get_resource_info()` is called.
 */
struct ResourcePoolInfo {
  size_t total_cpu_slots{0};
  size_t total_gpu_slots{0};
  size_t total_cpu_result_mem{0};
  size_t total_cpu_buffer_pool_mem{0};
  size_t total_gpu_buffer_pool_mem{0};

  size_t allocated_cpu_slots{0};
  size_t allocated_gpu_slots{0};
  size_t allocated_cpu_result_mem{0};
  size_t allocated_cpu_buffer_pool_mem{0};
  size_t allocated_gpu_buffer_pool_mem{0};

  size_t allocated_cpu_buffers{0};
  size_t allocated_gpu_buffers{0};

  size_t allocated_temp_cpu_buffer_pool_mem{0};
  size_t allocated_temp_gpu_buffer_pool_mem{0};

  size_t total_requests{0};
  size_t outstanding_requests{0};
  size_t outstanding_cpu_slots_requests{0};
  size_t outstanding_gpu_slots_requests{0};
  size_t outstanding_cpu_result_mem_requests{0};
  size_t outstanding_cpu_buffer_pool_mem_requests{0};
  size_t outstanding_gpu_buffer_pool_mem_requests{0};

  ResourcePoolInfo() {}

  ResourcePoolInfo(const size_t total_cpu_slots,
                   const size_t total_gpu_slots,
                   const size_t total_cpu_result_mem,
                   const size_t total_cpu_buffer_pool_mem,
                   const size_t total_gpu_buffer_pool_mem,
                   const size_t allocated_cpu_slots,
                   const size_t allocated_gpu_slots,
                   const size_t allocated_cpu_result_mem,
                   const size_t allocated_cpu_buffer_pool_mem,
                   const size_t allocated_gpu_buffer_pool_mem,
                   const size_t allocated_cpu_buffers,
                   const size_t allocated_gpu_buffers,
                   const size_t allocated_temp_cpu_buffer_pool_mem,
                   const size_t allocated_temp_gpu_buffer_pool_mem,
                   const size_t total_requests,
                   const size_t outstanding_requests,
                   const size_t outstanding_cpu_slots_requests,
                   const size_t outstanding_gpu_slots_requests,
                   const size_t outstanding_cpu_result_mem_requests,
                   const size_t outstanding_cpu_buffer_pool_mem_requests,
                   const size_t outstanding_gpu_buffer_pool_mem_requests)
      : total_cpu_slots(total_cpu_slots)
      , total_gpu_slots(total_gpu_slots)
      , total_cpu_result_mem(total_cpu_result_mem)
      , total_cpu_buffer_pool_mem(total_cpu_buffer_pool_mem)
      , total_gpu_buffer_pool_mem(total_gpu_buffer_pool_mem)
      , allocated_cpu_slots(allocated_cpu_slots)
      , allocated_gpu_slots(allocated_gpu_slots)
      , allocated_cpu_result_mem(allocated_cpu_result_mem)
      , allocated_cpu_buffer_pool_mem(allocated_cpu_buffer_pool_mem)
      , allocated_gpu_buffer_pool_mem(allocated_gpu_buffer_pool_mem)
      , allocated_cpu_buffers(allocated_cpu_buffers)
      , allocated_gpu_buffers(allocated_gpu_buffers)
      , allocated_temp_cpu_buffer_pool_mem(allocated_temp_cpu_buffer_pool_mem)
      , allocated_temp_gpu_buffer_pool_mem(allocated_temp_gpu_buffer_pool_mem)
      , total_requests(total_requests)
      , outstanding_requests(outstanding_requests)
      , outstanding_cpu_slots_requests(outstanding_cpu_slots_requests)
      , outstanding_gpu_slots_requests(outstanding_gpu_slots_requests)
      , outstanding_cpu_result_mem_requests(outstanding_cpu_result_mem_requests)
      , outstanding_cpu_buffer_pool_mem_requests(outstanding_cpu_buffer_pool_mem_requests)
      , outstanding_gpu_buffer_pool_mem_requests(
            outstanding_gpu_buffer_pool_mem_requests) {}
};

/**
 * @brief Specifies the resources of each type for a given
 * resource grant
 */
struct ResourceGrant {
  size_t cpu_slots{0};
  size_t gpu_slots{0};
  size_t cpu_result_mem{0};

  // Below is only relevant if buffer_mem_gated_per_slot is true
  bool buffer_mem_gated_per_slot{false};
  size_t buffer_mem_per_slot{0};
  size_t buffer_mem_for_given_slots{0};

  bool is_empty() const {
    return cpu_slots == 0 && gpu_slots == 0 && cpu_result_mem == 0;
  }

  void print() const;

  std::string to_string() const;
};

using BufferPoolChunkMap =
    std::map<ChunkKey, std::pair<size_t, size_t>>;  // Value is reference count and size

/**
 * @brief ExecutorResourcePool keeps track of available compute and memory resources and
 * can be queried to get the min and max resources grantable (embodied in a ResourceGrant)
 * for a request, given a ResourceRequest.
 *
 * ExecutorResourcePool keeps track of logical resources available to the executor,
 * categorized and typed by the ResourceType enum. Current valid categories of
 * ResourceType include CPU_SLOTS, GPU_SLOTS, CPU_RESULT_MEM, CPU_BUFFER_POOL_MEM, and
 * GPU_BUFFER_POOL_MEM. Furthermore, a ResourceSubtype enum is used to represent more
 * granular sub-categories of the above. Namely, there exists ResourceSubtype
 * PINNED_CPU_BUFFER_POOL_MEM and PINNED_GPU_BUFFER_POOL_MEM to represent non-pageable
 * memory (specifcally for kernel results), and PAGEABLE_CPU_BUFFER_POOL_MEM and
 * PAGEABLE_GPU_BUFFER_POOL_MEM to represent data that could be evicted as neccessary.
 *
 * Currently, a singleton ExecutorResourcePool is managed by ExecutorResourceMgr and is
 * initialized by the latter in the ExecutorResourceMgr constructor. Various parameters
 * driving behavior of ExecutorResourcePool are passed to its constructor, comprising the
 * total resources available in the pool in each of the above categories, policies around
 * concurrent requests to the pool for each of the resources (embodied in a vector of
 * ConcurrentResourceGrantPolicy), and policies around limits to individual resource
 * grants (embodied in a vector of ResourceGrantPolicy).
 *
 * Generally for a given resource request, the following lifecycle is prescribed, as
 * can be seen in the various invocations of ExecutorResourcePool methods by
 * ExecutorResourceMgr:
 * 1. call_min_max_resource_grants_for_request: Get min and max possible resource
 * grant, given a resource_request. If it is determined to be impossible to grant
 * even the minimum requests specified in resource_request, this will throw an error.
 * 2. determine_dynamic_resource_grant: Given the min and max possible resource
 * grants determined from #1, the ExecutorResourcePool calculates an actual
 * grant to give a query based on current resource availability in the pool.
 * 3. allocate_resources: Allocate the actual resource grant computed in #2
 * from the ExecutorResourcePool, marking the resources as used/allocated
 * so they cannot be used by other queries/requestors until deallocated
 * 4. deallocate_resources: Ultimately invoked from the destructor of the resource
 * handle given to the executing thread, this returns the allocated resources
 * to the pool for use by other queries.
 *
 */
class ExecutorResourcePool {
 public:
  ExecutorResourcePool(
      const std::vector<std::pair<ResourceType, size_t>>& total_resources,
      const std::vector<ConcurrentResourceGrantPolicy>&
          concurrent_resource_grant_policies,
      const std::vector<ResourceGrantPolicy>& max_per_request_resource_grant_policies);

  void log_parameters() const;

  std::vector<ResourceRequestGrant> calc_static_resource_grant_ranges_for_request(
      const std::vector<ResourceRequest>& resource_requests) const;

  /**
   * @brief Given the provided resource_request, statically calculate the minimum and
   * maximum grantable resources for that request. Note that the max resource grant
   * may be less that requested by the query.
   *
   * Note that this method only looks at static total available resources as well as
   * the ideal and minimum resources requested (in resource_request) to determine
   * the max grants, and does not evaluate the current state of resource use in the pool.
   * That is done in a later call, detrmine_dynamic_resource_grant.
   *
   * @param resource_request - Details the resources a query would like to have as well
   * as the minimum resources it can run with
   * @return std::pair<ResourceGrant, ResourceGrant> - A pair of the minimum and maximum
   * grantable resources that could be potentially granted for this request,
   */
  std::pair<ResourceGrant, ResourceGrant> calc_min_max_resource_grants_for_request(
      const RequestInfo& resource_request) const;

  bool can_currently_satisfy_request(const ResourceGrant& min_resource_grant,
                                     const ChunkRequestInfo& chunk_request_info) const;

  /**
   * @brief Determines the actual resource grant to give a query (which will be somewhere
   * between the provided min_resource_grant and max_resource_grant, unless it is
   * determined that the request cannot be currently satisfied).
   *
   * Generally the resources granted of each type are computed independently, but if
   * buffer_mem_gated_per_slot is set on min_resource_grant, other resources such as
   * threads granted may be scaled back to match the amount of buffer pool mem available.
   *
   * @param min_resource_grant - The min resource grant allowable for this request,
   * determined in calc_min_max_resource_grants_for_request
   * @param max_resource_grant - The max resource grant possible for this request,
   * determined in calc_min_max_resource_grants_for_request
   * @param chunk_request_info - The DataMgr chunks with associated sizes needed for this
   * query
   * @param max_request_backoff_ratio - The fraction from 0 to 1 of each resource we will
   * leave in the pool, even if the resources are available to satisfy the
   * max_resource_grant (so that there will be resources available for other queries).
   * @return std::pair<bool, ResourceGrant> - the first boolean member of the pair
   * specifies whether the request can currently be satisfied given current resources in
   * the pool, the second is the actual resource grant a requestor will receive.
   */
  std::pair<bool, ResourceGrant> determine_dynamic_resource_grant(
      const ResourceGrant& min_resource_grant,
      const ResourceGrant& max_resource_grant,
      const ChunkRequestInfo& chunk_request_info,
      const double max_request_backoff_ratio) const;

  /**
   * @brief Given a resource grant (assumed to be computed in
   * determine_dynamic_resource_grant), actually allocate (reserve) the resources in the
   * pool so other requestors (queries) cannot use those resources until returned to the
   * pool
   *
   * Note that the chunk requests do not and should not neccessarily match the state of
   * the BufferMgrs (where evictions can happen etc), but are just used to keep track of
   * what chunks are pledged for running queries. In the future we may try to get all of
   * this info from the BufferMgr directly, but would need to add a layer of state there
   * that would keep track of both what is currently allocated and what is pledged to
   * queries. For now, this effort was not deemed worth the complexity and risk it would
   * introduce.
   *
   * @param resource_grant - Granted resource_grant, assumed to be determined previously
   * in determine_dynamic_resource_grant
   * @param chunk_request_info - The DataMgr chunk keys and other associated info needed
   * by this query. The ExecutorResourcePool must keep track of chunks that are in the
   * pool so it can properly determine whether queries can execute (given chunks can be
   * shared resources across requestors/queries).
   */
  void allocate_resources(const ResourceGrant& resource_grant,
                          const ChunkRequestInfo& chunk_request_info);

  /**
   * @brief Deallocates resources granted to a requestor such that they can be used
   * for other requests
   *
   * @param resource_grant - Resources granted to the request that should be deallocated
   * @param chunk_request_info - The DataMgr chunk keys (and other associated info)
   * granted to this query that should be deallocated.
   */
  void deallocate_resources(const ResourceGrant& resource_grant,
                            const ChunkRequestInfo& chunk_request_info);

  /**
   * @brief Returns the allocated and total available amount of the resource
   * specified.
   *
   * @return std::pair<size_t, size_t> - First member is the allocated amount of the
   * resource, the second member is the total amount of the resource (including allocated
   * and available)
   */
  std::pair<size_t, size_t> get_resource_info(const ResourceType resource_type) const;

  /**
   * @brief Returns a struct detailing the allocated and total available resources
   * of each type tracked in ExecutorResourcePool
   *
   * @return ResourcePoolInfo - Struct detailining the allocaated and total available
   * resources of each typed tracked in ExecutorResourcePool
   */
  ResourcePoolInfo get_resource_info() const;

  /**
   * @brief Sets the quantity of resource_type to resource_quantity. If pool has
   * outstanding requests, will throw. Responsibility of allowing the pool to empty and
   * preventing concurrent requests while this operation is running is left to the caller
   * (in particular, ExecutorResourceMgr::set_resource pauses the process queue, which
   * waits until all executing requests are finished before yielding to the caller, before
   * calling this method).
   *
   * Currently only used for testing, but a SQL interface to live-change resources
   * available in the pool could be added.
   *
   * @param resource_type - type of resource to change the quanity of
   * @param resource_quantity - new quantity of resource for given resource_type
   */
  void set_resource(const ResourceType resource_type, const size_t resource_quantity);

  inline ConcurrentResourceGrantPolicy get_concurrent_resource_grant_policy(
      const ResourceType resource_type) const {
    return concurrent_resource_grant_policies_[static_cast<size_t>(resource_type)];
  }

  inline const ResourceGrantPolicy& get_max_resource_grant_per_request_policy(
      const ResourceSubtype resource_subtype) const {
    return max_resource_grants_per_request_policies_[static_cast<size_t>(
        resource_subtype)];
  }

  /**
   * @brief Resets the concurrent resource grant policy object, which specifies a
   * ResourceType as well as normal and oversubscription concurrency policies. If pool has
   * outstanding requests, will throw. Responsibility of allowing the pool to empty and
   * preventing concurrent requests while this operation is running is left to the caller
   * (in particular, ExecutorResourceMgr::set_concurent_resource_grant_policy pauses the
   * process queue, which waits until all executing requests are finished before yielding
   * to the caller, before calling this method).
   *
   * Currently only used for testing, but a SQL interface to live-change concurrency
   * policies for the pool could be added.
   *
   * @param concurrent_resource_grant_policy - new concurrent resource policy (which
   * encompasses the type of resource)
   */
  void set_concurrent_resource_grant_policy(
      const ConcurrentResourceGrantPolicy& concurrent_resource_grant_policy);

 private:
  void init(
      const std::vector<std::pair<ResourceType, size_t>>& total_resources,
      const std::vector<ConcurrentResourceGrantPolicy>&
          concurrent_resource_grant_policies,
      const std::vector<ResourceGrantPolicy>& max_per_request_resource_grant_policies);

  void init_concurrency_policies();
  void init_max_resource_grants_per_requests();

  void throw_insufficient_resource_error(const ResourceSubtype resource_subtype,
                                         const size_t min_resource_requested) const;

  size_t calc_max_resource_grant_for_request(
      const size_t requested_resource_quantity,
      const size_t min_requested_resource_quantity,
      const size_t max_grantable_resource_quantity) const;

  std::pair<size_t, size_t> calc_min_dependent_resource_grant_for_request(
      const size_t min_requested_dependent_resource_quantity,
      const size_t min_requested_independent_resource_quantity,
      const size_t dependent_to_independent_resource_ratio) const;

  std::pair<size_t, size_t> calc_max_dependent_resource_grant_for_request(
      const size_t requested_dependent_resource_quantity,
      const size_t min_requested_dependent_resource_quantity,
      const size_t max_grantable_dependent_resource_quantity,
      const size_t min_requested_independent_resource_quantity,
      const size_t max_grantable_independent_resource_quantity,
      const size_t dependent_to_independent_resource_ratio) const;

  bool check_request_against_global_policy(
      const size_t resource_total,
      const size_t resource_allocated,
      const ConcurrentResourceGrantPolicy& concurrent_resource_grant_policy) const;

  bool check_request_against_policy(
      const size_t resource_request,
      const size_t resource_total,
      const size_t resource_allocated,
      const size_t global_outstanding_requests,
      const ConcurrentResourceGrantPolicy& concurrent_resource_grant_policy) const;

  // Unlocked internal version
  bool can_currently_satisfy_request_impl(
      const ResourceGrant& min_resource_grant,
      const ChunkRequestInfo& chunk_request_info) const;

  bool can_currently_satisfy_chunk_request(
      const ResourceGrant& min_resource_grant,
      const ChunkRequestInfo& chunk_request_info) const;
  ChunkRequestInfo get_requested_chunks_not_in_pool(
      const ChunkRequestInfo& chunk_request_info) const;
  size_t get_chunk_bytes_not_in_pool(const ChunkRequestInfo& chunk_request_info) const;
  void add_chunk_requests_to_allocated_pool(const ResourceGrant& resource_grant,
                                            const ChunkRequestInfo& chunk_request_info);
  void remove_chunk_requests_from_allocated_pool(
      const ResourceGrant& resource_grant,
      const ChunkRequestInfo& chunk_request_info);

  size_t determine_dynamic_single_resource_grant(
      const size_t min_resource_requested,
      const size_t max_resource_requested,
      const size_t resource_allocated,
      const size_t total_resource,
      const double max_request_backoff_ratio) const;

  void sanity_check_requests_against_allocations() const;

  inline size_t get_total_allocated_buffer_pool_mem_for_level(
      const ExecutorDeviceType memory_pool_type) const {
    return memory_pool_type == ExecutorDeviceType::CPU
               ? get_allocated_resource_of_type(ResourceType::CPU_BUFFER_POOL_MEM)
               : get_allocated_resource_of_type(ResourceType::GPU_BUFFER_POOL_MEM);
  }

  inline bool is_resource_valid(const ResourceType resource_type) const {
    return resource_type_validity_[static_cast<size_t>(resource_type)];
  }

  inline size_t get_total_resource(const ResourceType resource_type) const {
    return total_resources_[static_cast<size_t>(resource_type)];
  }

  inline size_t get_allocated_resource_of_subtype(
      const ResourceSubtype resource_subtype) const {
    return allocated_resources_[static_cast<size_t>(resource_subtype)];
  }

  size_t get_allocated_resource_of_type(const ResourceType resource_type) const;

  inline size_t get_max_resource_grant_per_request(
      const ResourceSubtype resource_subtype) const {
    return max_resource_grants_per_request_[static_cast<size_t>(resource_subtype)];
  }

  inline size_t get_total_per_resource_num_requests(
      const ResourceType resource_type) const {
    return total_per_resource_num_requests_[static_cast<size_t>(resource_type)];
  }

  inline size_t increment_total_per_resource_num_requests(
      const ResourceType resource_type) {
    return ++total_per_resource_num_requests_[static_cast<size_t>(resource_type)];
  }

  inline size_t decrement_total_per_resource_num_requests(
      const ResourceType resource_type) {
    return --total_per_resource_num_requests_[static_cast<size_t>(resource_type)];
  }

  inline size_t get_outstanding_per_resource_num_requests(
      const ResourceType resource_type) const {
    return outstanding_per_resource_num_requests_[static_cast<size_t>(resource_type)];
  }

  inline size_t increment_outstanding_per_resource_num_requests(
      const ResourceType resource_type) {
    return ++outstanding_per_resource_num_requests_[static_cast<size_t>(resource_type)];
  }

  inline size_t decrement_outstanding_per_resource_num_requests(
      const ResourceType resource_type) {
    return --outstanding_per_resource_num_requests_[static_cast<size_t>(resource_type)];
  }

  std::array<size_t, ResourceTypeSize>
      total_resources_{};  // Will be value initialized to 0s
  std::array<bool, ResourceTypeSize> resource_type_validity_{
      false};  // Will be value initialized to false
  std::array<size_t, ResourceSubtypeSize>
      allocated_resources_{};  // Will be value initialized to 0s
  std::array<ResourceGrantPolicy, ResourceSubtypeSize>
      max_resource_grants_per_request_policies_{};
  std::array<size_t, ResourceSubtypeSize> max_resource_grants_per_request_{};
  std::array<ConcurrentResourceGrantPolicy, ResourceTypeSize>
      concurrent_resource_grant_policies_;

  size_t total_num_requests_{0};
  size_t outstanding_num_requests_{0};

  std::array<size_t, ResourceTypeSize>
      total_per_resource_num_requests_{};  // Will be value initialized to 0s

  std::array<size_t, ResourceTypeSize>
      outstanding_per_resource_num_requests_{};  // Will be value initialized to 0s

  BufferPoolChunkMap allocated_cpu_buffer_pool_chunks_;
  BufferPoolChunkMap allocated_gpu_buffer_pool_chunks_;

  const bool sanity_check_pool_state_on_deallocations_{false};
  mutable std::shared_mutex resource_mutex_;
};

}  // namespace ExecutorResourceMgr_Namespace
