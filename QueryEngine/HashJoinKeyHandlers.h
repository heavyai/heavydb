/*
 * Copyright 2018 OmniSci, Inc.
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

#ifndef QUERYENGINE_HASHJOINKEYHANDLERS_H
#define QUERYENGINE_HASHJOINKEYHANDLERS_H

#include "HashJoinRuntime.h"
#include "SqlTypesLayout.h"

#ifdef __CUDACC__
#include "DecodersImpl.h"
#else
#include <glog/logging.h>
#include "../StringDictionary/StringDictionary.h"
#include "../StringDictionary/StringDictionaryProxy.h"
#include "RuntimeFunctions.h"
#endif

#include <cmath>

#include "../Shared/funcannotations.h"

DEVICE inline int64_t get_join_column_element_value(const JoinColumnTypeInfo& type_info,
                                                    const JoinColumn& join_column,
                                                    const size_t i) {
  switch (type_info.column_type) {
    case SmallDate:
      return SUFFIX(fixed_width_small_date_decode_noinline)(
          join_column.col_buff,
          type_info.elem_sz,
          type_info.elem_sz == 4 ? NULL_INT : NULL_SMALLINT,
          type_info.elem_sz == 4 ? NULL_INT : NULL_SMALLINT,
          i);
    case Signed:
      return SUFFIX(fixed_width_int_decode_noinline)(
          join_column.col_buff, type_info.elem_sz, i);
    case Unsigned:
      return SUFFIX(fixed_width_unsigned_decode_noinline)(
          join_column.col_buff, type_info.elem_sz, i);
    default:
#ifndef __CUDACC__
      CHECK(false);
#else
      assert(0);
#endif
      return 0;
  }
}

struct GenericKeyHandler {
  GenericKeyHandler(const size_t key_component_count,
                    const bool should_skip_entries,
                    const JoinColumn* join_column_per_key,
                    const JoinColumnTypeInfo* type_info_per_key
#ifndef __CUDACC__
                    ,
                    const void* const* sd_inner_proxy_per_key,
                    const void* const* sd_outer_proxy_per_key
#endif
                    )
      : key_component_count_(key_component_count)
      , should_skip_entries_(should_skip_entries)
      , join_column_per_key_(join_column_per_key)
      , type_info_per_key_(type_info_per_key) {
#ifndef __CUDACC__
    if (sd_inner_proxy_per_key) {
      CHECK(sd_outer_proxy_per_key);
      sd_inner_proxy_per_key_ = sd_inner_proxy_per_key;
      sd_outer_proxy_per_key_ = sd_outer_proxy_per_key;
    } else
#endif
    {
      sd_inner_proxy_per_key_ = nullptr;
      sd_outer_proxy_per_key_ = nullptr;
    }
  }

  template <typename T, typename KEY_BUFF_HANDLER>
  DEVICE int operator()(const size_t i, T* key_scratch_buff, KEY_BUFF_HANDLER f) const {
    bool skip_entry = false;
    for (size_t key_component_index = 0; key_component_index < key_component_count_;
         ++key_component_index) {
      const auto& join_column = join_column_per_key_[key_component_index];
      const auto& type_info = type_info_per_key_[key_component_index];
      int64_t elem = get_join_column_element_value(type_info, join_column, i);
      if (should_skip_entries_ && elem == type_info.null_val && !type_info.uses_bw_eq) {
        skip_entry = true;
        break;
      }
#ifndef __CUDACC__
      const auto sd_inner_proxy = sd_inner_proxy_per_key_
                                      ? sd_inner_proxy_per_key_[key_component_index]
                                      : nullptr;
      const auto sd_outer_proxy = sd_outer_proxy_per_key_
                                      ? sd_outer_proxy_per_key_[key_component_index]
                                      : nullptr;
      if (sd_inner_proxy && elem != type_info.null_val) {
        CHECK(sd_outer_proxy);
        const auto sd_inner_dict_proxy =
            static_cast<const StringDictionaryProxy*>(sd_inner_proxy);
        const auto sd_outer_dict_proxy =
            static_cast<const StringDictionaryProxy*>(sd_outer_proxy);
        const auto elem_str = sd_inner_dict_proxy->getString(elem);
        const auto outer_id = sd_outer_dict_proxy->getIdOfString(elem_str);
        if (outer_id == StringDictionary::INVALID_STR_ID) {
          skip_entry = true;
          break;
        }
        elem = outer_id;
      }
#endif
      key_scratch_buff[key_component_index] = elem;
    }

    if (!skip_entry) {
      return f(i, key_scratch_buff, key_component_count_);
    }

    return 0;
  }

  const size_t key_component_count_;
  const bool should_skip_entries_;
  const JoinColumn* join_column_per_key_;
  const JoinColumnTypeInfo* type_info_per_key_;
  const void* const* sd_inner_proxy_per_key_;
  const void* const* sd_outer_proxy_per_key_;
};

struct OverlapsKeyHandler {
  OverlapsKeyHandler(const size_t key_dims_count,
                     const JoinColumn* join_column,
                     const double* bucket_sizes_for_dimension)
      : key_dims_count_(key_dims_count)
      , join_column_(join_column)
      , bucket_sizes_for_dimension_(bucket_sizes_for_dimension) {}

  template <typename T, typename KEY_BUFF_HANDLER>
  DEVICE int operator()(const size_t i, T* key_scratch_buff, KEY_BUFF_HANDLER f) const {
    // TODO(adb): hard-coding the 2D case w/ bounds for now. Should support n-dims with a
    // check to ensure we are not exceeding maximum number of dims for coalesced keys
    double bounds[4];
    for (size_t j = 0; j < 2 * key_dims_count_; j++) {
      bounds[j] = SUFFIX(fixed_width_double_decode_noinline)(join_column_[0].col_buff,
                                                             2 * key_dims_count_ * i + j);
    }

    const auto x_bucket_sz = bucket_sizes_for_dimension_[0];
    const auto y_bucket_sz = bucket_sizes_for_dimension_[1];

    for (int64_t x = floor(bounds[0] * x_bucket_sz); x <= floor(bounds[2] * x_bucket_sz);
         x++) {
      for (int64_t y = floor(bounds[1] * y_bucket_sz);
           y <= floor(bounds[3] * y_bucket_sz);
           y++) {
        key_scratch_buff[0] = x;
        key_scratch_buff[1] = y;

        const auto err = f(i, key_scratch_buff, key_dims_count_);
        if (err) {
          return err;
        }
      }
    }
    return 0;
  }

  const size_t key_dims_count_;
  const JoinColumn* join_column_;
  const double* bucket_sizes_for_dimension_;
};

#endif  // QUERYENGINE_HASHJOINKEYHANDLERS_H
