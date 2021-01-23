/*
 * Copyright 2020 OmniSci, Inc.
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

#ifndef OMNISCI_QUERYHINT_H
#define OMNISCI_QUERYHINT_H

#include "ThriftHandler/CommandLineOptions.h"

struct QueryHint {
  // for each hint "H", we first define its value as the corresponding system-defined
  // default value "D"
  // After then, if we detect at least one hint is registered (via hint_delivered),
  // we can compare the value btw. "H" and "D" during the query compilation step that H
  // is involved and then use the "H" iff "H" != "D"
  // since that indicates user-given hint is delivered
  // (otherwise, "H" should be the equal to "D")
  // note that we should check if H is valid W.R.T the proper value range
  // i.e., if H is valid in 0.0 ~ 1.0, then we check that at the point
  // when we decide to use H, and use D iff given H does not have a valid value
  QueryHint() {
    hint_delivered = false;
    cpu_mode = false;
    overlaps_bucket_threshold = 0.1;
    overlaps_max_size = g_overlaps_max_table_size_bytes;
  }

  QueryHint& operator=(const QueryHint& other) {
    hint_delivered = other.hint_delivered;
    cpu_mode = other.cpu_mode;
    overlaps_bucket_threshold = other.overlaps_bucket_threshold;
    overlaps_max_size = other.overlaps_max_size;
    return *this;
  }

  QueryHint(const QueryHint& other) {
    hint_delivered = other.hint_delivered;
    cpu_mode = other.cpu_mode;
    overlaps_bucket_threshold = other.overlaps_bucket_threshold;
    overlaps_max_size = other.overlaps_max_size;
  }

  // set true if at least one query hint is delivered
  bool hint_delivered;

  // general query execution
  bool cpu_mode;

  // overlaps hash join
  double overlaps_bucket_threshold;  // defined in "OverlapsJoinHashTable.h"
  size_t overlaps_max_size;

  std::unordered_map<std::string, int> OMNISCI_SUPPORTED_HINT_CLASS = {
      {"cpu_mode", 0},
      {"overlaps_bucket_threshold", 1},
      {"overlaps_max_size", 2}};

  static QueryHint defaults() { return QueryHint(); }
};

#endif  // OMNISCI_QUERYHINT_H
