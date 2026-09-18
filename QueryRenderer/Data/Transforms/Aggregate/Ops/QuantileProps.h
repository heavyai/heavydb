/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/Transforms/Aggregate/Ops/DistinctHistogramProps.h"

namespace QueryRenderer {

struct QuantileProps : public DistinctHistogramProps {
  uint16_t num_quantiles;
  bool include_extrema;

  QuantileProps(const uint16_t in_num_quantiles = defaultNumQuantiles(),
                const bool in_include_extrema = defaultIncludeExtrema(),
                const bool in_approximate = DistinctHistogramProps::defaultApproximate(),
                const size_t in_num_bins = DistinctHistogramProps::defaultNumBins());
  QuantileProps(const JSONLocation& json_loc);

  void serialize(std::stringstream& ss) const;
  static std::vector<AnyDataType> deserialize(std::istringstream& ss);

  static decltype(num_quantiles) getNumQuantilesFromJSONObj(const JSONLocation& json_loc);
  static decltype(include_extrema) getIncludeExtremaFromJSONObj(
      const JSONLocation& json_loc);

  static inline decltype(num_quantiles) defaultNumQuantiles() { return 2; }
  static inline decltype(include_extrema) defaultIncludeExtrema() { return false; }

 private:
  void validate();
};

}  // namespace QueryRenderer
