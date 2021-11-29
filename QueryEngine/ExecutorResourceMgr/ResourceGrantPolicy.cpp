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

#include <math.h>
#include <limits>
#include <sstream>

#include "Logger/Logger.h"
#include "ResourceGrantPolicy.h"

namespace ExecutorResourceMgr_Namespace {

ResourceGrantPolicy::ResourceGrantPolicy(
    const ResourceSubtype resource_subtype,
    const ResourceGrantPolicySizeType policy_size_type,
    const ResourceGrantSizeInfo size_info)
    : resource_subtype(resource_subtype)
    , policy_size_type(policy_size_type)
    , size_info(size_info) {}

size_t ResourceGrantPolicy::get_grant_quantity(
    const size_t total_resource_quantity) const {
  switch (policy_size_type) {
    case ResourceGrantPolicySizeType::UNLIMITED:
      return std::numeric_limits<size_t>::max();
    case ResourceGrantPolicySizeType::CONSTANT:
      return size_info.constant;
    case ResourceGrantPolicySizeType::RATIO_TO_TOTAL:
      return static_cast<size_t>(
          ceil(size_info.ratio_to_total * total_resource_quantity));
    default:
      UNREACHABLE();
  }
  UNREACHABLE();
  return std::numeric_limits<size_t>::max();
}
// Optionally caps quantity at total available, for case of DISALLOW_REQUESTS
// for ConcurrentResourceGrantPolicy oversubscription sub-policy
size_t ResourceGrantPolicy::get_grant_quantity(const size_t total_resource_quantity,
                                               const bool cap_at_total_available) const {
  const size_t max_per_query_grant_without_cap =
      get_grant_quantity(total_resource_quantity);
  if (cap_at_total_available) {
    return std::min(max_per_query_grant_without_cap, total_resource_quantity);
  }
  return max_per_query_grant_without_cap;
}

std::string ResourceGrantPolicy::to_string() const {
  std::ostringstream oss;
  oss << "RESOURCE TYPE: "
      << ResourceSubtypeStrings[static_cast<size_t>(resource_subtype)] << " ";
  switch (policy_size_type) {
    case ResourceGrantPolicySizeType::UNLIMITED:
      oss << "SIZE TYPE: Unlimited";
      break;
    case ResourceGrantPolicySizeType::CONSTANT:
      oss << "SIZE TYPE: Constant (" << size_info.constant << ")";
      break;
    case ResourceGrantPolicySizeType::RATIO_TO_TOTAL:
      oss << "SIZE TYPE: Ratio-to-total (" << size_info.ratio_to_total << ")";
      break;
    default:
      UNREACHABLE();
  }
  return oss.str();
}

ResourceGrantPolicy gen_unlimited_resource_grant_policy(
    const ResourceSubtype resource_subtype) {
  // Make a dummy size_info union
  const ResourceGrantSizeInfo size_info;
  const ResourceGrantPolicy resource_grant_policy(
      resource_subtype, ResourceGrantPolicySizeType::UNLIMITED, size_info);
  return resource_grant_policy;
}

ResourceGrantPolicy gen_constant_resource_grant_policy(
    const ResourceSubtype resource_subtype,
    const size_t constant_grant) {
  const ResourceGrantSizeInfo size_info(constant_grant);
  const ResourceGrantPolicy resource_grant_policy(
      resource_subtype, ResourceGrantPolicySizeType::CONSTANT, size_info);
  return resource_grant_policy;
}

ResourceGrantPolicy gen_ratio_resource_grant_policy(
    const ResourceSubtype resource_subtype,
    const double ratio_grant) {
  const ResourceGrantSizeInfo size_info(ratio_grant);
  const ResourceGrantPolicy resource_grant_policy(
      resource_subtype, ResourceGrantPolicySizeType::RATIO_TO_TOTAL, size_info);
  return resource_grant_policy;
}

std::string get_resource_concurrency_policy_string(
    const ResourceConcurrencyPolicy& resource_concurrency_policy) {
  std::ostringstream oss;
  switch (resource_concurrency_policy) {
    case ALLOW_CONCURRENT_REQUESTS:
      oss << "Allow concurrent requests";
      break;
    case ALLOW_SINGLE_REQUEST:
      oss << "Allow single request for resource type";
      break;
    case ALLOW_SINGLE_REQUEST_GLOBALLY:
      oss << "Allow single request globally";
      break;
    case DISALLOW_REQUESTS:
      oss << "Disallow requests";
      break;
    default:
      UNREACHABLE();
  }
  return oss.str();
}

bool ConcurrentResourceGrantPolicy::operator==(
    const ConcurrentResourceGrantPolicy& rhs) const {
  return resource_type == rhs.resource_type &&
         concurrency_policy == rhs.concurrency_policy &&
         oversubscription_concurrency_policy == rhs.oversubscription_concurrency_policy;
}

std::string ConcurrentResourceGrantPolicy::to_string() const {
  std::ostringstream oss;
  oss << "Resource Type: " << resource_type_to_string(resource_type)
      << "Undersubscribed Policy: "
      << get_resource_concurrency_policy_string(concurrency_policy)
      << " Oversubscribed Policy: "
      << get_resource_concurrency_policy_string(oversubscription_concurrency_policy);
  return oss.str();
}

}  // namespace ExecutorResourceMgr_Namespace
