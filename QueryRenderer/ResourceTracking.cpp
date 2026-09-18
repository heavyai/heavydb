/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/ResourceTracking.h"

namespace QueryRenderer {

std::string ResourceTrackingString(std::string_view resource_creator,
                                   const int resource_index) {
  // placeholder for now
  // we need to get the SessionID into this
  std::string resource_tracking_string(resource_creator);
  if (resource_index >= 0) {
    resource_tracking_string += "[" + std::to_string(resource_index) + "]";
  }
  return resource_tracking_string;
}

}  // namespace QueryRenderer