/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Utils/AnyDataType.h"

namespace QueryRenderer {

class JSONLocation;

struct DistinctHistogramProps {
  const bool approximate;
  const size_t num_bins;

  DistinctHistogramProps(const bool in_approximate = defaultApproximate(),
                         const size_t in_num_bins = defaultNumBins());
  DistinctHistogramProps(const JSONLocation& json_loc);
  DistinctHistogramProps(const std::vector<AnyDataType>& deserialized_props);

  void serialize(std::stringstream& ss) const;
  static std::vector<AnyDataType> deserialize(std::istringstream& ss);

  static decltype(approximate) getApproximateFromJSONObj(const JSONLocation& json_loc);
  static decltype(num_bins) getNumBinsFromJSONObj(const JSONLocation& json_loc);

  static inline decltype(approximate) defaultApproximate() { return true; }
  static inline decltype(num_bins) defaultNumBins() { return 1000; }

 private:
  void validate();
};

}  // namespace QueryRenderer
