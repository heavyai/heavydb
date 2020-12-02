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

#include "QueryEngine/JoinHashTable/Runtime/HashJoinRuntime.h"
#include "QueryEngine/JoinHashTable/Runtime/JoinColumnIterator.h"
#include "Shared/SqlTypesLayout.h"

#ifdef __CUDACC__
#include "QueryEngine/DecodersImpl.h"
#else
#include "Logger/Logger.h"
#include "QueryEngine/RuntimeFunctions.h"
#include "StringDictionary/StringDictionary.h"
#include "StringDictionary/StringDictionaryProxy.h"
#endif

#include <cmath>

#include "Shared/funcannotations.h"

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
  DEVICE int operator()(JoinColumnIterator* join_column_iterators,
                        T* key_scratch_buff,
                        KEY_BUFF_HANDLER f) const {
    bool skip_entry = false;
    for (size_t key_component_index = 0; key_component_index < key_component_count_;
         ++key_component_index) {
      const auto& join_column_iterator = join_column_iterators[key_component_index];
      int64_t elem = (*join_column_iterator).element;
      if (should_skip_entries_ && elem == join_column_iterator.type_info->null_val &&
          !join_column_iterator.type_info->uses_bw_eq) {
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
      if (sd_inner_proxy && elem != join_column_iterator.type_info->null_val) {
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
      return f(join_column_iterators[0].index, key_scratch_buff, key_component_count_);
    }

    return 0;
  }

  DEVICE size_t get_number_of_columns() const { return key_component_count_; }

  DEVICE size_t get_key_component_count() const { return key_component_count_; }

  DEVICE const JoinColumn* get_join_columns() const { return join_column_per_key_; }

  DEVICE const JoinColumnTypeInfo* get_join_column_type_infos() const {
    return type_info_per_key_;
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
                     const JoinColumn* join_column,  // always 1 column
                     const double* bucket_sizes_for_dimension)
      : key_dims_count_(key_dims_count)
      , join_column_(join_column)
      , bucket_sizes_for_dimension_(bucket_sizes_for_dimension) {}

  template <typename T, typename KEY_BUFF_HANDLER>
  DEVICE int operator()(JoinColumnIterator* join_column_iterators,
                        T* key_scratch_buff,
                        KEY_BUFF_HANDLER f) const {
    // TODO(adb): hard-coding the 2D case w/ bounds for now. Should support n-dims with a
    // check to ensure we are not exceeding maximum number of dims for coalesced keys
    double bounds[4];
    for (size_t j = 0; j < 2 * key_dims_count_; j++) {
      bounds[j] =
          SUFFIX(fixed_width_double_decode_noinline)(join_column_iterators->ptr(), j);
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

        const auto err =
            f(join_column_iterators[0].index, key_scratch_buff, key_dims_count_);
        if (err) {
          return err;
        }
      }
    }
    return 0;
  }

  DEVICE size_t get_number_of_columns() const { return 1; }

  DEVICE size_t get_key_component_count() const { return key_dims_count_; }

  DEVICE const JoinColumn* get_join_columns() const { return join_column_; }

  DEVICE const JoinColumnTypeInfo* get_join_column_type_infos() const { return nullptr; }

  const size_t key_dims_count_;
  const JoinColumn* join_column_;
  const double* bucket_sizes_for_dimension_;
};

#endif  // QUERYENGINE_HASHJOINKEYHANDLERS_H
