#pragma once

#include <cstdint>

#include "ForeignTable.h"
#include "Shared/DateTimeParser.h"

namespace foreign_storage {
/**
 * Gets the interval duration in seconds.
 *
 * @param interval - interval string with format of `{interval_count}{interval_type}`
 * (e.g. 5H for "every 5 hours")
 * @return internal duration in seconds
 */
inline int64_t get_interval_duration(const std::string& interval) {
  int interval_count = std::stoi(interval.substr(0, interval.length() - 1));
  auto interval_type = std::tolower(interval[interval.length() - 1]);
  int64_t duration{0};
  if (interval_type == 's') {
    duration = interval_count;
  } else if (interval_type == 'h') {
    duration = interval_count * 60 * 60;
  } else if (interval_type == 'd') {
    duration = interval_count * 60 * 60 * 24;
  } else {
    UNREACHABLE();
  }
  return duration;
}

// TODO: Support complete list of interval types
inline int64_t get_next_refresh_time(
    const std::map<std::string, std::string, std::less<>>& foreign_table_options) {
  int64_t current_time = std::chrono::duration_cast<std::chrono::seconds>(
                             std::chrono::system_clock::now().time_since_epoch())
                             .count();
  auto start_date_entry = foreign_table_options.find(
      foreign_storage::ForeignTable::REFRESH_START_DATE_TIME_KEY);
  CHECK(start_date_entry != foreign_table_options.end());
  auto start_date_time = dateTimeParse<kTIMESTAMP>(start_date_entry->second, 0);

  // If start date time is current or in the future, then that is the next refresh time
  if (start_date_time >= current_time) {
    return start_date_time;
  }
  auto interval_entry =
      foreign_table_options.find(foreign_storage::ForeignTable::REFRESH_INTERVAL_KEY);
  if (interval_entry != foreign_table_options.end()) {
    auto interval_duration = get_interval_duration(interval_entry->second);
    auto num_intervals =
        (current_time - start_date_time + interval_duration - 1) / interval_duration;
    return start_date_time + (num_intervals * interval_duration);
  } else {
    // If this was a one time refresh, then there is no next refresh time
    return foreign_storage::ForeignTable::NULL_REFRESH_TIME;
  }
}
}  // namespace foreign_storage
