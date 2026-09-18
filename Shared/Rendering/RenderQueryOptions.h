/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

class RenderQueryOptions {
 public:
  enum FlagBits {
    // enableHitTesting field not available in Vega
    // attempt to support it automatically
    // requires physical tables and will enable kInjectRowIdForHitTesting for non-insitu
    // renders
    kLegacyHitTestLogic = 1u << 0,
    // automatically inject rowid for projection queries.
    // For example, this should be true when hit-testing is enabled.
    kInjectRowIdForHitTesting = 1u << 1,
    // physical tables are required in the results
    // For example, this should be true when hit-testing is enabled.
    kRequiresPhysicalTables = 1u << 2,
    // automatically inject rowid for ppll poly rendering.
    kInjectRowIdForPPLL = 1u << 3,
  };

  bool shouldAlterRA() const {
    return flags_ & (FlagBits::kInjectRowIdForHitTesting | FlagBits::kInjectRowIdForPPLL);
  }

  bool useLegacyHitTestLogic() const { return flags_ & FlagBits::kLegacyHitTestLogic; }
  bool injectRowIdForHitTesting() const {
    return flags_ & FlagBits::kInjectRowIdForHitTesting;
  }
  bool isHitTestingEnabled() const {
    return useLegacyHitTestLogic() || injectRowIdForHitTesting();
  }
  bool injectRowIdForPPLL() const { return flags_ & FlagBits::kInjectRowIdForPPLL; }
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
