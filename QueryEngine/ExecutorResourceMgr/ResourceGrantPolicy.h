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

#include "ExecutorResourceMgrCommon.h"

namespace ExecutorResourceMgr_Namespace {

/**
 * @brief The sizing type for a `ResourceGrantPolicy` of a given resource type, specifying
 * whether resource grants for a request should be `UNLIMITED`, `CONSTANT`, or
 * `RATIO_TO_TOTAL` (i.e. a ratio of the total available resource).
 */
enum ResourceGrantPolicySizeType { UNLIMITED, CONSTANT, RATIO_TO_TOTAL };

union ResourceGrantSizeInfo {
  size_t constant;
  double ratio_to_total;
  ResourceGrantSizeInfo() : constant(0) {}
  ResourceGrantSizeInfo(const size_t constant) : constant(constant) {}
  ResourceGrantSizeInfo(const double ratio_to_total) : ratio_to_total(ratio_to_total) {}
};

/**
 * @brief Specifies the policy for granting a resource of a specific `ResourceSubtype`.
 * Note that this policy only pertains to resource grants on an isolated basis, and that
 * grant policy with respect to concurrent requests is controlled by
 * `ConcurrentResourceGrantPolicy`.
 *
 * `ExecutorResourcePool` uses `get_grant_quantity` to determine the maximum allowable
 * grants for each type of resource (on a static basis) when being initialized.
 */
struct ResourceGrantPolicy {
  ResourceSubtype resource_subtype{ResourceSubtype::INVALID_SUBTYPE};
  ResourceGrantPolicySizeType policy_size_type{ResourceGrantPolicySizeType::CONSTANT};
  ResourceGrantSizeInfo size_info;

  ResourceGrantPolicy() {}

  ResourceGrantPolicy(const ResourceSubtype resource_subtype,
                      const ResourceGrantPolicySizeType policy_size_type,
                      const ResourceGrantSizeInfo size_info);

  /**
   * @brief Used to calculate the maximum quantity that can be granted of the resource.
   *
   * If `policy_size_type` is a ratio, the max grantable quantity will be the ratio
   * specified in the `size_info` union times the specified `total_resource_quantity`.
   * If `policy_size_type` is a constant, the max grantable quantity will be that
   * constant, regardless of what is specified by `total_resource_quantity`.
   * If `policy_size_type` is unlimited, the max grantable quantity will be
   * unlimited, specifically the max allowed number for the `size_t` type.
   *
   * @param total_resource_quantity - the total quantity available of the resource
   * specified by `ResourceGrantPolicy::resource_subtype`
   * @return size_t - The max quantity of the resource specified by
   * `ResourceGrantPolicy::resource_subtype` that can be granted
   */
  size_t get_grant_quantity(const size_t total_resource_quantity) const;

  /**
   * @brief Used to calculate the maximum quantity that can be granted of the resource,
   * optionally capping at the specified `total_resource_quantity` .
   *
   *
   * If `policy_size_type` is a ratio, the max grantable quantity will be the ratio
   * specified in the `size_info` union times the specified `total_resource_quantity`.
   * If `policy_size_type` is a constant, the max grantable quantity will be that
   * constant, regardless of what is specified by `total_resource_quantity`.
   * If `policy_size_type` is unlimited, the max grantable quantity will be
   * unlimited, specifically the max allowed number for the `size_t` type.
   * Regardless of the above, if `cap_at_total_available` is set to true, then
   * the return size will be the minimum of the above and `total_resource_quantity`.
   * This flag is used by `ExecutorResourcePool::init_max_resource_grants_per_requests()`
   * if oversubscription is not allowed (ResourceConcurrencyPolicy::DISALLOW_REQUESTS)
   * as specified by `ConcurrentResourceGrantPolicy::oversubscription_concurrency_policy)
   *
   * @param total_resource_quantity - the total quantity available of the resource
   * specified by `ResourceGrantPolicy::resource_subtype`
   * @return size_t - The max quantity of the resource specified by
   * `ResourceGrantPolicy::resource_subtype` that can be granted
   */
  size_t get_grant_quantity(const size_t total_resource_quantity,
                            const bool cap_at_total_available) const;

  std::string to_string() const;
};

/**
 * @brief Generates a `ResourceGrantPolicy` with `ResourceGrantPolicySizeType::UNLIMITED`
 *
 * @param resource_subtype - type of resource subtype to create a `ResourceGrantPolicy`
 * for
 * @return ResourceGrantPolicy - The generated "unlimited" `ResourceGrantPolicy`
 */
ResourceGrantPolicy gen_unlimited_resource_grant_policy(
    const ResourceSubtype resource_subtype);

/**
 * @brief Generates a `ResourceGrantPolicy` with `ResourceGrantPolicySizeType::CONSTANT`
 *
 * @param resource_subtype - type of resource subtype to create a `ResourceGrantPolicy`
 * for
 * @return ResourceGrantPolicy - The generated "constant-sized" `ResourceGrantPolicy`
 */
ResourceGrantPolicy gen_constant_resource_grant_policy(
    const ResourceSubtype resource_subtype,
    size_t constant_grant);

/**
 * @brief Generates a `ResourceGrantPolicy` with
 * `ResourceGrantPolicySizeType::RATIO_TO_TOTAL`
 *
 * @param resource_subtype - type of resource subtype to create a `ResourceGrantPolicy`
 * for
 * @return ResourceGrantPolicy - The generated "ratio-to-total" `ResourceGrantPolicy`
 */
ResourceGrantPolicy gen_ratio_resource_grant_policy(
    const ResourceSubtype resource_subtype,
    const double ratio_grant);

/**
 * @brief Specifies whether grants for a specified resource can be made concurrently
 * (`ALLOW_CONCURRENT_REQEUSTS`), can only be made to one request at a time
 * (`ALOW_SINGLE_REQUEST`), can be made only if there are no other requests in flight,
 * regardless of type (`ALLOW_SINGLE_REQUEST_GLOBALLY`), or are not allowed at all
 * (`DISALLOW_REQUESTS`).
 *
 * The policy is qualified with a `ResourceType` and whether the resource is
 * under-subscribed or over-subscribed in `ConcurrentResourceGrantPolicy`. For example,
 * `DISALLOW_REQUESTS` would generally only be used to specify resource oversubscription
 * policy, as if it was used for the under-subscription policy for that resource, the
 * resource could not be used at all.
 */
enum ResourceConcurrencyPolicy {
  ALLOW_CONCURRENT_REQUESTS,
  ALLOW_SINGLE_REQUEST,
  ALLOW_SINGLE_REQUEST_GLOBALLY,
  DISALLOW_REQUESTS
};

std::string get_resouce_concurrency_policy_string(
    const ResourceConcurrencyPolicy& resource_concurrency_policy);

/**
 * @brief Specifies the policies for resource grants in the presence of other
 * requests, both under situations of resource undersubscription (i.e there are
 * still resources of the given type in the pool) and oversubscription.
 *
 * For example, it may make sense to allow concurrent grants for CPU
 * threads (via `ResourceConcurrencyPolicy::ALLOW_CONCURRENT_REQUESTS`), but
 * if there would be oversubscription of CPU threads, to only allow a single
 * request for that type to run (via `ResourceConcurrencyPolicy::ALLOW_SINGLE_REQUEST).
 * Or if the desired policy was to have GPU requests run alone in the system,
 * `ConcurrentResourceGrantPolicy::concurrency_policy` could be set to
 * `ResourceConcurrencyPolicy::ALLOW_SINGLE_REQUEST_GLOBALLY1
 */
struct ConcurrentResourceGrantPolicy {
  /**
   * @brief The type of a resource this concurrent resource grant policy pertains to
   */
  ResourceType resource_type{ResourceType::INVALID_TYPE};

  /**
   * @brief The grant policy in effect when there are concurrent requests for the resource
   * specified by `resource_type`, but the resource is not oversubscribed (i.e. requested
   * in greater quantity between all requests than is available)
   */
  ResourceConcurrencyPolicy concurrency_policy{DISALLOW_REQUESTS};

  /**
   * @brief The grant policy in effect when there are concurrent requests for the resource
   * specified by `resource_type`, and the resource is oversubscribed (i.e. requested
   * in greater quantity between all requests than is available)
   */
  ResourceConcurrencyPolicy oversubscription_concurrency_policy{DISALLOW_REQUESTS};

  ConcurrentResourceGrantPolicy() {}

  ConcurrentResourceGrantPolicy(
      const ResourceType resource_type,
      const ResourceConcurrencyPolicy concurrency_policy,
      const ResourceConcurrencyPolicy oversubscription_concurrency_policy)
      : resource_type(resource_type)
      , concurrency_policy(concurrency_policy)
      , oversubscription_concurrency_policy(oversubscription_concurrency_policy) {}

  bool operator==(const ConcurrentResourceGrantPolicy& rhs) const;

  std::string to_string() const;
};

}  // namespace ExecutorResourceMgr_Namespace
