#pragma once

#include <optional>
#include <string>

namespace shared {

struct LonLatBoundingBox {
  double min_lon = 0.0;
  double min_lat = 0.0;
  double max_lon = 0.0;
  double max_lat = 0.0;
  static std::optional<LonLatBoundingBox> parse(const std::string& str);

  bool hasNoOverlapWith(const LonLatBoundingBox& other) const;
};

}  // namespace shared
