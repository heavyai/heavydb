/*
 * Copyright 2022 HEAVY.AI, Inc.
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
    return any_bits_set(insitu_flags_ & InSituFlags::kForcedNonInSitu);
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
