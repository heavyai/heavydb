/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Shared/EnumBitmaskOps.h"

namespace heavyai {

enum class InSituFlags {
  kInSitu = 1u << 0,
  kNonInSitu = 1u << 1,
  kForcedNonInSitu = (kInSitu | kNonInSitu)
};

}  // namespace heavyai

ENABLE_BITMASK_OPS(heavyai::InSituFlags);

namespace heavyai {

// Needs to be defined after the ENABLE_BITMASK_OPS call above
class InSituFlagsOwnerInterface {
 public:
  InSituFlagsOwnerInterface(const InSituFlags insitu_flags)
      : insitu_flags_{insitu_flags} {}

  InSituFlags getInSituFlags() const { return insitu_flags_; }

  bool isForcedNonInSitu() const {
    return (insitu_flags_ & InSituFlags::kForcedNonInSitu) ==
           InSituFlags::kForcedNonInSitu;
  }

  bool isInSitu() const { return insitu_flags_ == InSituFlags::kInSitu; }

  bool isNonInSitu() const {
    return any_bits_set(insitu_flags_ & InSituFlags::kNonInSitu);
  }

  bool couldRunInSitu() const {
    return any_bits_set(insitu_flags_ & InSituFlags::kInSitu);
  }

 protected:
  InSituFlags insitu_flags_;
};

}  // namespace heavyai
