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
                    const int32_t* const* sd_inner_to_outer_translation_maps,
                    const int32_t* sd_min_inner_elems
#endif
                    )
      : key_component_count_(key_component_count)
      , should_skip_entries_(should_skip_entries)
      , join_column_per_key_(join_column_per_key)
      , type_info_per_key_(type_info_per_key) {
#ifndef __CUDACC__
    if (sd_inner_to_outer_translation_maps) {
      CHECK(sd_min_inner_elems);
      sd_inner_to_outer_translation_maps_ = sd_inner_to_outer_translation_maps;
      sd_min_inner_elems_ = sd_min_inner_elems;
    } else
#endif
    {
      sd_inner_to_outer_translation_maps_ = nullptr;
      sd_min_inner_elems_ = nullptr;
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
      // Translation map pts will already be set to nullptr if invalid
      if (sd_inner_to_outer_translation_maps_) {
        const auto sd_inner_to_outer_translation_map =
            sd_inner_to_outer_translation_maps_[key_component_index];
        const auto sd_min_inner_elem = sd_min_inner_elems_[key_component_index];
        if (sd_inner_to_outer_translation_map &&
            elem != join_column_iterator.type_info->null_val) {
          const auto outer_id =
              sd_inner_to_outer_translation_map[elem - sd_min_inner_elem];
          if (outer_id == StringDictionary::INVALID_STR_ID) {
            skip_entry = true;
            break;
          }
          elem = outer_id;
        }
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
  const int32_t* const* sd_inner_to_outer_translation_maps_;
  const int32_t* sd_min_inner_elems_;
};

#endif  // QUERYENGINE_HASHJOINKEYHANDLERS_H
