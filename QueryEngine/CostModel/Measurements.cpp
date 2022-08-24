#include "Measurements.h"

namespace costmodel {

std::string templateToString(AnalyticalTemplate templ) {
  switch (templ) {
    case AnalyticalTemplate::Join:
      return "Join";
    case AnalyticalTemplate::GroupBy:
      return "GroupBy";
    case AnalyticalTemplate::Scan:
      return "Scan";
    case AnalyticalTemplate::Reduce:
      return "Reduce";
    default:
      return "Unknown";
  }
}

}  // namespace costmodel
