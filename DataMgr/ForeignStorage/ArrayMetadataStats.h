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

#pragma once

#include "Shared/sqltypes.h"

namespace foreign_storage {

class ArrayMetadataStats {
 public:
  ArrayMetadataStats() : is_min_initialized_(false), is_max_initialized_(false) {}

  void updateStats(const SQLTypeInfo& type_info, const Datum& min, const Datum& max) {
    switch (type_info.get_type()) {
      case kBOOLEAN: {
        updateMin(min_.tinyintval, min.tinyintval);
        updateMax(max_.tinyintval, max.tinyintval);
        break;
      }
      case kTINYINT: {
        updateMin(min_.tinyintval, min.tinyintval);
        updateMax(max_.tinyintval, max.tinyintval);
        break;
      }
      case kSMALLINT: {
        updateMin(min_.smallintval, min.smallintval);
        updateMax(max_.smallintval, max.smallintval);
        break;
      }
      case kINT: {
        updateMin(min_.intval, min.intval);
        updateMax(max_.intval, max.intval);
        break;
      }
      case kBIGINT:
      case kNUMERIC:
      case kDECIMAL: {
        updateMin(min_.bigintval, min.bigintval);
        updateMax(max_.bigintval, max.bigintval);
        break;
      }
      case kTIME:
      case kTIMESTAMP:
      case kDATE: {
        updateMin(min_.bigintval, min.bigintval);
        updateMax(max_.bigintval, max.bigintval);
        break;
      }
      case kFLOAT: {
        updateMin(min_.floatval, min.floatval);
        updateMax(max_.floatval, max.floatval);
        break;
      }
      case kDOUBLE: {
        updateMin(min_.doubleval, min.doubleval);
        updateMax(max_.doubleval, max.doubleval);
        break;
      }
      case kVARCHAR:
      case kCHAR:
      case kTEXT:
        if (type_info.get_compression() == kENCODING_DICT) {
          updateMin(min_.intval, min.intval);
          updateMax(max_.intval, max.intval);
        }
        break;
      case kPOINT:
      case kLINESTRING:
      case kPOLYGON:
      case kMULTIPOLYGON:
        UNREACHABLE();
      default:
        UNREACHABLE();
    }
  }

  template <typename T>
  void updateStats(const SQLTypeInfo& type_info, const T& min, const T& max) {
    switch (type_info.get_type()) {
      case kBOOLEAN: {
        updateMin(min_.tinyintval, min);
        updateMax(max_.tinyintval, max);
        break;
      }
      case kTINYINT: {
        updateMin(min_.tinyintval, min);
        updateMax(max_.tinyintval, max);
        break;
      }
      case kSMALLINT: {
        updateMin(min_.smallintval, min);
        updateMax(max_.smallintval, max);
        break;
      }
      case kINT: {
        updateMin(min_.intval, min);
        updateMax(max_.intval, max);
        break;
      }
      case kBIGINT:
      case kNUMERIC:
      case kDECIMAL: {
        updateMin(min_.bigintval, min);
        updateMax(max_.bigintval, max);
        break;
      }
      case kTIME:
      case kTIMESTAMP:
      case kDATE: {
        updateMin(min_.bigintval, min);
        updateMax(max_.bigintval, max);
        break;
      }
      case kFLOAT: {
        updateMin(min_.floatval, min);
        updateMax(max_.floatval, max);
        break;
      }
      case kDOUBLE: {
        updateMin(min_.doubleval, min);
        updateMax(max_.doubleval, max);
        break;
      }
      case kVARCHAR:
      case kCHAR:
      case kTEXT:
        if (type_info.get_compression() == kENCODING_DICT) {
          updateMin(min_.intval, min);
          updateMax(max_.intval, max);
        }
        break;
      case kPOINT:
      case kLINESTRING:
      case kPOLYGON:
      case kMULTIPOLYGON:
        UNREACHABLE();
      default:
        UNREACHABLE();
    }
  }

  Datum getMin(const SQLTypeInfo& type_info) const {
    if (is_min_initialized_) {
      return min_;
    }
    return getUninitializedMin(type_info);
  }

  Datum getMax(const SQLTypeInfo& type_info) const {
    if (is_max_initialized_) {
      return max_;
    }
    return getUninitializedMax(type_info);
  }

 private:
  Datum getUninitializedMin(const SQLTypeInfo& type_info) const {
    Datum d;
    auto type = type_info.get_type();
    switch (type) {
      case kSMALLINT:
        d.smallintval = std::numeric_limits<int16_t>::max();
        break;
      case kBOOLEAN:
      case kTINYINT:
        d.tinyintval = std::numeric_limits<int8_t>::max();
        break;
      case kFLOAT:
        d.floatval = std::numeric_limits<float>::max();
        break;
      case kDOUBLE:
        d.doubleval = std::numeric_limits<double>::max();
        break;
      case kTEXT:
      case kVARCHAR:
      case kCHAR:
      case kINT:
        d.intval = std::numeric_limits<int32_t>::max();
        break;
      case kTIME:
      case kTIMESTAMP:
      case kDATE:
      case kBIGINT:
      case kNUMERIC:
      case kDECIMAL:
        d.bigintval = std::numeric_limits<int64_t>::max();
        break;
      case kPOINT:
      case kLINESTRING:
      case kPOLYGON:
      case kMULTIPOLYGON:
        UNREACHABLE();
      default:
        UNREACHABLE();
    }
    return d;
  }

  Datum getUninitializedMax(const SQLTypeInfo& type_info) const {
    Datum d;
    auto type = type_info.get_type();
    switch (type) {
      case kSMALLINT:
        d.smallintval = std::numeric_limits<int16_t>::lowest();
        break;
      case kBOOLEAN:
      case kTINYINT:
        d.tinyintval = std::numeric_limits<int8_t>::lowest();
        break;
      case kFLOAT:
        d.floatval = std::numeric_limits<float>::lowest();
        break;
      case kDOUBLE:
        d.doubleval = std::numeric_limits<double>::lowest();
        break;
      case kTEXT:
      case kVARCHAR:
      case kCHAR:
      case kINT:
        d.intval = std::numeric_limits<int32_t>::lowest();
        break;
      case kTIME:
      case kTIMESTAMP:
      case kDATE:
      case kBIGINT:
      case kNUMERIC:
      case kDECIMAL:
        d.bigintval = std::numeric_limits<int64_t>::lowest();
        break;
      case kPOINT:
      case kLINESTRING:
      case kPOLYGON:
      case kMULTIPOLYGON:
        UNREACHABLE();
      default:
        UNREACHABLE();
    }
    return d;
  }

  template <typename V, typename T>
  void updateMin(V& current_min, const T& value) {
    if (!is_min_initialized_) {
      current_min = value;
    } else {
      current_min = std::min<V>(current_min, value);
    }
    is_min_initialized_ = true;
  }

  template <typename V, typename T>
  void updateMax(V& current_max, const T& value) {
    if (!is_max_initialized_) {
      current_max = value;
    } else {
      current_max = std::max<V>(current_max, value);
    }
    is_max_initialized_ = true;
  }

  bool is_min_initialized_;
  bool is_max_initialized_;
  Datum min_, max_;
};

}  // namespace foreign_storage
