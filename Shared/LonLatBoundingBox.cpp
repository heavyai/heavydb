#include "Shared/LonLatBoundingBox.h"

#include <stdexcept>
#include <string>

#include <boost/tokenizer.hpp>

namespace shared {

std::optional<LonLatBoundingBox> LonLatBoundingBox::parse(const std::string& str) {
  if (str.empty()) {
    return std::nullopt;
  }
  boost::char_separator<char> separators(" ,");
  boost::tokenizer<boost::char_separator<char>> tokens(str, separators);
  auto const num_tokens = std::distance(tokens.begin(), tokens.end());
  if (num_tokens != 4) {
    throw std::runtime_error("Failed to parse lon/lat bounding box from '" + str +
                             "': must be 4 comma-separated numeric values");
  }
  auto parse_double = [&](const std::string& token) -> double {
    double value{};
    try {
      value = std::stod(token);
    } catch (std::exception& e) {
      throw std::runtime_error(std::string("Failed to parse lon/lat bounding box from '" +
                                           str + "': value '" + token +
                                           "' is not numeric"));
    };
    return value;
  };
  LonLatBoundingBox bb;
  auto itr = tokens.begin();
  bb.min_lon = parse_double(*itr++);
  bb.min_lat = parse_double(*itr++);
  bb.max_lon = parse_double(*itr++);
  bb.max_lat = parse_double(*itr++);
  if (bb.min_lon < -180.0 || bb.max_lon < -180.0 || bb.min_lon > 180.0 ||
      bb.max_lon > 180.0 || bb.min_lat < -90.0 || bb.max_lat < -90.0 ||
      bb.min_lat > 90.0 || bb.max_lat > 90.0) {
    throw std::runtime_error(std::string("Failed to parse lon/lat bounding box from '" +
                                         str + "': one or more values out of range"));
  }
  if (bb.min_lon > bb.max_lon || bb.min_lat > bb.max_lat) {
    throw std::runtime_error(std::string("Failed to parse lon/lat bounding box from '" +
                                         str +
                                         "': one or more overlapping min/max values"));
  }
  return bb;
}

}  // namespace shared
