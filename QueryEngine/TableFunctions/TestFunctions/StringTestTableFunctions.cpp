/*
 * Copyright 2021 OmniSci, Inc.
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

#include "TableFunctionsTesting.h"

/*
  This file contains testing string-related compile-time UDTFs.
 */

#ifndef __CUDACC__

EXTENSION_NOINLINE_HOST
int32_t ct_binding_str_length__cpu_(const Column<TextEncodingDict>& input_str,
                                    Column<TextEncodingDict>& out_str,
                                    Column<int64_t>& out_size) {
  const int64_t num_rows = input_str.size();
  set_output_row_size(num_rows);
  for (int64_t i = 0; i < num_rows; i++) {
    out_str[i] = input_str[i];
    const std::string str = input_str.getString(i);
    out_size[i] = str.size();
  }
  return num_rows;
}

EXTENSION_NOINLINE_HOST
int32_t ct_binding_str_equals__cpu_(const ColumnList<TextEncodingDict>& input_strings,
                                    Column<TextEncodingDict>& string_if_equal,
                                    Column<bool>& strings_are_equal) {
  const int64_t num_rows = input_strings.size();
  const int64_t num_cols = input_strings.numCols();
  set_output_row_size(num_rows);
  for (int64_t r = 0; r < num_rows; r++) {
    bool are_equal = true;
    if (num_cols > 0) {
      std::string first_str = input_strings[0].getString(r);
      for (int64_t c = 1; c != num_cols; ++c) {
        if (input_strings[c].getString(r) != first_str) {
          are_equal = false;
          break;
        }
      }
      strings_are_equal[r] = are_equal;
      if (are_equal && num_cols > 0) {
        string_if_equal[r] = input_strings[0][r];
      } else {
        string_if_equal.setNull(r);
      }
    }
  }
  return num_rows;
}

EXTENSION_NOINLINE_HOST
int32_t ct_substr__cpu_(TableFunctionManager& mgr,
                        const Column<TextEncodingDict>& input_str,
                        const Column<int>& pos,
                        const Column<int>& len,
                        Column<TextEncodingDict>& output_substr) {
  const int64_t num_rows = input_str.size();
  mgr.set_output_row_size(num_rows);
  for (int64_t row_idx = 0; row_idx < num_rows; row_idx++) {
    const std::string input_string{input_str.getString(row_idx)};
    const std::string substring = input_string.substr(pos[row_idx], len[row_idx]);
    const TextEncodingDict substr_id = output_substr.getOrAddTransient(substring);
    output_substr[row_idx] = substr_id;
  }
  return num_rows;
}

EXTENSION_NOINLINE_HOST
int32_t ct_string_concat__cpu_(TableFunctionManager& mgr,
                               const ColumnList<TextEncodingDict>& input_strings,
                               const TextEncodingNone& separator,
                               Column<TextEncodingDict>& concatted_string) {
  const int64_t num_rows = input_strings.size();
  const int64_t num_cols = input_strings.numCols();
  const std::string separator_str{separator.getString()};
  mgr.set_output_row_size(num_rows);
  for (int64_t row_idx = 0; row_idx < num_rows; row_idx++) {
    if (num_cols > 0) {
      std::string concatted_output{input_strings[0].getString(row_idx)};
      for (int64_t col_idx = 1; col_idx < num_cols; ++col_idx) {
        concatted_output += separator_str;
        concatted_output += input_strings[col_idx].getString(row_idx);
      }
      const TextEncodingDict concatted_str_id =
          concatted_string.getOrAddTransient(concatted_output);
      concatted_string[row_idx] = concatted_str_id;
    } else {
      concatted_string.setNull(row_idx);
    }
  }
  return num_rows;
}

EXTENSION_NOINLINE_HOST
int32_t ct_synthesize_new_dict__cpu_(TableFunctionManager& mgr,
                                     const int64_t num_strings,
                                     Column<TextEncodingDict>& new_dict_col) {
  mgr.set_output_row_size(num_strings);
  for (int32_t s = 0; s < num_strings; ++s) {
    const std::string new_string = "String_" + std::to_string(s);
    const int32_t string_id = new_dict_col.getOrAddTransient(new_string);
    new_dict_col[s] = string_id;
  }
  return num_strings;
}

#endif  // #ifndef __CUDACC__

EXTENSION_NOINLINE int32_t ct_hamming_distance(const TextEncodingNone& str1,
                                               const TextEncodingNone& str2,
                                               Column<int32_t>& hamming_distance) {
  const int32_t str_len = str1.size() <= str2.size() ? str1.size() : str2.size();

#ifdef __CUDACC__
  const int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  const int32_t step = blockDim.x * gridDim.x;
  int32_t* output_ptr = hamming_distance.ptr_;
#else
  const int32_t start = 0;
  const int32_t step = 1;
#endif

  int32_t num_chars_unequal = 0;
  for (int32_t i = start; i < str_len; i += step) {
    num_chars_unequal += (str1[i] != str2[i]) ? 1 : 0;
  }
#ifdef __CUDACC__
  atomicAdd(output_ptr, num_chars_unequal);
#else
  hamming_distance[0] = num_chars_unequal;
#endif
  return 1;
}

template <typename T>
TEMPLATE_NOINLINE int32_t ct_get_string_chars__template(const Column<T>& indices,
                                                        const TextEncodingNone& str,
                                                        const int32_t multiplier,
                                                        Column<int32_t>& idx,
                                                        Column<int8_t>& char_bytes) {
  const int32_t str_len = str.size();
  // Note: we assume RowMultiplier is 1 for this test, was to make running on
  // GPU easy Todo: Provide Constant RowMultiplier interface
  if (multiplier != 1) {
    return 0;
  }
  const int32_t num_input_rows = indices.size();
  const int32_t num_output_rows = num_input_rows * multiplier;

#ifdef __CUDACC__
  const int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  const int32_t step = blockDim.x * gridDim.x;
#else
  const int32_t start = 0;
  const int32_t step = 1;
#endif

  for (int32_t i = start; i < num_output_rows; i += step) {
    idx[i] = indices[i % num_output_rows];
    char_bytes[i] = str[i % str_len];  // index < str_len ? str[i] : 0;
  }
  return num_output_rows;
}

// forward declarations
template TEMPLATE_NOINLINE int32_t
ct_get_string_chars__template(const Column<int16_t>& indices,
                              const TextEncodingNone& str,
                              const int32_t multiplier,
                              Column<int32_t>& idx,
                              Column<int8_t>& char_bytes);
template TEMPLATE_NOINLINE int32_t
ct_get_string_chars__template(const Column<int32_t>& indices,
                              const TextEncodingNone& str,
                              const int32_t multiplier,
                              Column<int32_t>& idx,
                              Column<int8_t>& char_bytes);

#ifndef __CUDACC__

#include <iostream>
#include <string>

EXTENSION_NOINLINE_HOST int32_t ct_string_to_chars__cpu_(const TextEncodingNone& input,
                                                         Column<int32_t>& char_idx,
                                                         Column<int8_t>& char_bytes) {
  const std::string str{input.getString()};
  const int64_t str_size(str.size());
  set_output_row_size(str_size);
  for (int32_t i = 0; i < str_size; ++i) {
    char_idx[i] = i;
    char_bytes[i] = str[i];
  }
  return str_size;
}

#endif  // #ifndef __CUDACC__