/*
 * Copyright 2019 OmniSci, Inc.
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

class RenderQueryOptions {
 public:
  enum FlagBits {
    // enableHitTesting field not available in Vega
    // attempt to support it automatically
    // requires physical tables and will enable kInjectRowId for non-insitu renders
    kLegacyHitTestLogic = 1u << 0,
    // automatically inject rowid for projection queries.
    // For example, this should be true when hit-testing is enabled.
    kInjectRowId = 1u << 1,
    // physical tables are required in the results
    // For example, this should be true when hit-testing is enabled.
    kRequiresPhysicalTables = 1u << 2,
  };

  bool shouldAlterRA() const { return flags_ & FlagBits::kInjectRowId; }

  bool useLegacyHitTestLogic() const { return flags_ & FlagBits::kLegacyHitTestLogic; }
  bool injectRowId() const { return flags_ & FlagBits::kInjectRowId; }
  bool requiresPhysicalTables() const {
    return flags_ & FlagBits::kRequiresPhysicalTables;
  }

  void setFlags(FlagBits flags_to_set) { flags_ |= flags_to_set; }
  void clearFlags(FlagBits flags_to_clear) { flags_ &= (~flags_to_clear); }
  void clearAllFlags() { flags_ = 0u; }

  bool operator==(const RenderQueryOptions& other) const {
    return flags_ == other.flags_;
  }
  bool operator!=(const RenderQueryOptions& other) const { return !operator==(other); }

 private:
  uint32_t flags_ = 0u;
};
