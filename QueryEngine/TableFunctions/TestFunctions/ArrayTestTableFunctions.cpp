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
  This file contains testing array-related compile-time UDTFs.

  NOTE: This file currently contains no GPU UDTFs. If any GPU UDTFs are
  added, it should be added to CUDA_TABLE_FUNCTION_FILES in CMakeLists.txt
 */

#ifndef __CUDACC__

template <typename T>
NEVER_INLINE HOST int32_t sum_along_row__cpu_template(const Column<Array<T>>& input,
                                                      Column<T>& output) {
  int size = input.size();
  for (int i = 0; i < size; i++) {
    const Array<T> arr = input[i];
    if (arr.isNull()) {
      output.setNull(i);
    } else {
      if constexpr (std::is_same<T, TextEncodingDict>::value) {
        auto* mgr = TableFunctionManager::get_singleton();
        std::string acc = "";
        for (size_t j = 0; j < arr.size(); j++) {
          if (!arr.isNull(j)) {
            acc += mgr->getString(input.getDictDbId(), input.getDictId(), arr[j]);
          }
        }
        int32_t out_string_id =
            mgr->getOrAddTransient(output.getDictDbId(), output.getDictId(), acc);
        output[i] = out_string_id;
      } else {
        T acc{0};
        for (size_t j = 0; j < arr.size(); j++) {
          if constexpr (std::is_same_v<T, bool>) {
            // todo: arr.isNull(i) returns arr[i] because bool does not
            // have null value, we should introduce 8-bit boolean type
            // for Arrays
            acc |= arr[j];
          } else {
            if (!arr.isNull(j)) {
              acc += arr[j];
            }
          }
        }
        output[i] = acc;
      }
    }
  }
  return size;
}

// explicit instantiations
template NEVER_INLINE HOST int32_t
sum_along_row__cpu_template(const Column<Array<float>>& input, Column<float>& output);
template NEVER_INLINE HOST int32_t
sum_along_row__cpu_template(const Column<Array<double>>& input, Column<double>& output);
template NEVER_INLINE HOST int32_t
sum_along_row__cpu_template(const Column<Array<int8_t>>& input, Column<int8_t>& output);
template NEVER_INLINE HOST int32_t
sum_along_row__cpu_template(const Column<Array<int16_t>>& input, Column<int16_t>& output);
template NEVER_INLINE HOST int32_t
sum_along_row__cpu_template(const Column<Array<int32_t>>& input, Column<int32_t>& output);
template NEVER_INLINE HOST int32_t
sum_along_row__cpu_template(const Column<Array<int64_t>>& input, Column<int64_t>& output);
template NEVER_INLINE HOST int32_t
sum_along_row__cpu_template(const Column<Array<bool>>& input, Column<bool>& output);
template NEVER_INLINE HOST int32_t
sum_along_row__cpu_template(const Column<Array<TextEncodingDict>>& input,
                            Column<TextEncodingDict>& output);

template <typename T>
NEVER_INLINE HOST int32_t array_copier__cpu_template(TableFunctionManager& mgr,
                                                     const Column<Array<T>>& input,
                                                     Column<Array<T>>& output) {
  int size = input.size();

  // count the number of items in all input arrays:
  int output_values_size = 0;
  for (int i = 0; i < size; i++) {
    output_values_size += input[i].size();
  }

  // set the size and allocate the output columns buffers:
  mgr.set_output_array_values_total_number(
      /*output column index=*/0,
      /*upper bound to the number of items in all output arrays=*/output_values_size);
  mgr.set_output_row_size(size);

  // set the items of output colums:
  for (int i = 0; i < size; i++) {
    output.setItem(i, input[i]);
  }

  return size;
}

// explicit instantiations
template NEVER_INLINE HOST int32_t
array_copier__cpu_template(TableFunctionManager& mgr,
                           const Column<Array<float>>& input,
                           Column<Array<float>>& output);
template NEVER_INLINE HOST int32_t
array_copier__cpu_template(TableFunctionManager& mgr,
                           const Column<Array<double>>& input,
                           Column<Array<double>>& output);
template NEVER_INLINE HOST int32_t
array_copier__cpu_template(TableFunctionManager& mgr,
                           const Column<Array<int8_t>>& input,
                           Column<Array<int8_t>>& output);
template NEVER_INLINE HOST int32_t
array_copier__cpu_template(TableFunctionManager& mgr,
                           const Column<Array<int16_t>>& input,
                           Column<Array<int16_t>>& output);
template NEVER_INLINE HOST int32_t
array_copier__cpu_template(TableFunctionManager& mgr,
                           const Column<Array<int32_t>>& input,
                           Column<Array<int32_t>>& output);
template NEVER_INLINE HOST int32_t
array_copier__cpu_template(TableFunctionManager& mgr,
                           const Column<Array<int64_t>>& input,
                           Column<Array<int64_t>>& output);
template NEVER_INLINE HOST int32_t
array_copier__cpu_template(TableFunctionManager& mgr,
                           const Column<Array<bool>>& input,
                           Column<Array<bool>>& output);
template NEVER_INLINE HOST int32_t
array_copier__cpu_template(TableFunctionManager& mgr,
                           const Column<Array<TextEncodingDict>>& input,
                           Column<Array<TextEncodingDict>>& output);

template <typename T>
NEVER_INLINE HOST int32_t array_concat__cpu_template(TableFunctionManager& mgr,
                                                     const ColumnList<Array<T>>& inputs,
                                                     Column<Array<T>>& output) {
  int size = inputs.size();

  int output_values_size = 0;
  for (int j = 0; j < inputs.numCols(); j++) {
    for (int i = 0; i < size; i++) {
      output_values_size += inputs[j][i].size();
    }
  }
  mgr.set_output_array_values_total_number(
      /*output column index=*/0,
      /*upper bound to the number of items in all output arrays=*/output_values_size);

  mgr.set_output_row_size(size);

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < inputs.numCols(); j++) {
      Column<Array<T>> col = inputs[j];
      output.concatItem(i,
                        col[i]);  // works only if i is the last row set, otherwise throws
    }
  }
  return size;
}

// explicit instantiations
template NEVER_INLINE HOST int32_t
array_concat__cpu_template(TableFunctionManager& mgr,
                           const ColumnList<Array<float>>& inputs,
                           Column<Array<float>>& output);
template NEVER_INLINE HOST int32_t
array_concat__cpu_template(TableFunctionManager& mgr,
                           const ColumnList<Array<double>>& inputs,
                           Column<Array<double>>& output);
template NEVER_INLINE HOST int32_t
array_concat__cpu_template(TableFunctionManager& mgr,
                           const ColumnList<Array<int8_t>>& inputs,
                           Column<Array<int8_t>>& output);
template NEVER_INLINE HOST int32_t
array_concat__cpu_template(TableFunctionManager& mgr,
                           const ColumnList<Array<int16_t>>& inputs,
                           Column<Array<int16_t>>& output);
template NEVER_INLINE HOST int32_t
array_concat__cpu_template(TableFunctionManager& mgr,
                           const ColumnList<Array<int32_t>>& inputs,
                           Column<Array<int32_t>>& output);
template NEVER_INLINE HOST int32_t
array_concat__cpu_template(TableFunctionManager& mgr,
                           const ColumnList<Array<int64_t>>& inputs,
                           Column<Array<int64_t>>& output);
template NEVER_INLINE HOST int32_t
array_concat__cpu_template(TableFunctionManager& mgr,
                           const ColumnList<Array<bool>>& inputs,
                           Column<Array<bool>>& output);
template NEVER_INLINE HOST int32_t
array_concat__cpu_template(TableFunctionManager& mgr,
                           const ColumnList<Array<TextEncodingDict>>& inputs,
                           Column<Array<TextEncodingDict>>& output);

template <typename T>
NEVER_INLINE HOST int32_t array_asarray__cpu_template(TableFunctionManager& mgr,
                                                      const Column<T>& input,
                                                      Column<Array<T>>& output) {
  int size = input.size();
  int output_values_size = 0;
  for (int i = 0; i < size; i++) {
    output_values_size += (input.isNull(i) ? 0 : 1);
  }
  mgr.set_output_array_values_total_number(
      /*output column index=*/0,
      /*upper bound to the number of items in all output arrays=*/output_values_size);
  mgr.set_output_row_size(size);

  if constexpr (std::is_same<T, TextEncodingDict>::value) {
    for (int i = 0; i < size; i++) {
      if (input.isNull(i)) {
        output.setNull(i);
      } else {
        Array<T> arr = output.getItem(i, 1);
        arr[0] = mgr.getOrAddTransient(
            mgr.getNewDictDbId(),
            mgr.getNewDictId(),
            mgr.getString(input.getDictDbId(), input.getDictId(), input[i]));
      }
    }
  } else {
    for (int i = 0; i < size; i++) {
      if (input.isNull(i)) {
        output.setNull(i);
      } else {
        Array<T> arr = output.getItem(i, 1);
        arr[0] = input[i];
      }
    }
  }
  return size;
}

// explicit instantiations

template NEVER_INLINE HOST int32_t
array_asarray__cpu_template(TableFunctionManager& mgr,
                            const Column<int64_t>& input,
                            Column<Array<int64_t>>& output);

template NEVER_INLINE HOST int32_t
array_asarray__cpu_template(TableFunctionManager& mgr,
                            const Column<TextEncodingDict>& input,
                            Column<Array<TextEncodingDict>>& output);

// clang-format off
/*
  UDTF: array_split__cpu_template(TableFunctionManager mgr, Column<Array<T>> input) ->
          Column<Array<T>> | input_id=args<0>, Column<Array<T>> | input_id=args<0>,
          T=[float, double, int8_t, int16_t, int32_t, int64_t, bool, TextEncodingDict]
*/
// clang-format on
template <typename T>
NEVER_INLINE HOST int32_t array_split__cpu_template(TableFunctionManager& mgr,
                                                    const Column<Array<T>>& input,
                                                    Column<Array<T>>& first,
                                                    Column<Array<T>>& second) {
  int size = input.size();
  int first_values_size = 0;
  int second_values_size = 0;
  for (int i = 0; i < size; i++) {
    if (!input.isNull(i)) {
      int64_t sz = input[i].size();
      first_values_size += sz / 2;
      second_values_size += sz - sz / 2;
    }
  }
  mgr.set_output_array_values_total_number(0, first_values_size);
  mgr.set_output_array_values_total_number(1, second_values_size);
  mgr.set_output_row_size(size);

  for (int i = 0; i < size; i++) {
    if (input.isNull(i)) {
      first.setNull(i);
      second.setNull(i);
    } else {
      Array<T> arr = input[i];
      int64_t sz = arr.size();
      Array<T> arr1 = first.getItem(i, sz / 2);
      Array<T> arr2 = second.getItem(i, sz - sz / 2);
      for (int64_t j = 0; j < sz; j++) {
        if (j < sz / 2) {
          arr1[j] = arr[j];
        } else {
          arr2[j - sz / 2] = arr[j];
        }
      }
    }
  }
  return size;
}

// explicit instantiations

template NEVER_INLINE HOST int32_t
array_split__cpu_template(TableFunctionManager& mgr,
                          const Column<Array<float>>& input,
                          Column<Array<float>>& first,
                          Column<Array<float>>& second);

template NEVER_INLINE HOST int32_t
array_split__cpu_template(TableFunctionManager& mgr,
                          const Column<Array<double>>& input,
                          Column<Array<double>>& first,
                          Column<Array<double>>& second);

template NEVER_INLINE HOST int32_t
array_split__cpu_template(TableFunctionManager& mgr,
                          const Column<Array<int8_t>>& input,
                          Column<Array<int8_t>>& first,
                          Column<Array<int8_t>>& second);

template NEVER_INLINE HOST int32_t
array_split__cpu_template(TableFunctionManager& mgr,
                          const Column<Array<int16_t>>& input,
                          Column<Array<int16_t>>& first,
                          Column<Array<int16_t>>& second);

template NEVER_INLINE HOST int32_t
array_split__cpu_template(TableFunctionManager& mgr,
                          const Column<Array<int32_t>>& input,
                          Column<Array<int32_t>>& first,
                          Column<Array<int32_t>>& second);

template NEVER_INLINE HOST int32_t
array_split__cpu_template(TableFunctionManager& mgr,
                          const Column<Array<int64_t>>& input,
                          Column<Array<int64_t>>& first,
                          Column<Array<int64_t>>& second);

template NEVER_INLINE HOST int32_t
array_split__cpu_template(TableFunctionManager& mgr,
                          const Column<Array<bool>>& input,
                          Column<Array<bool>>& first,
                          Column<Array<bool>>& second);

template NEVER_INLINE HOST int32_t
array_split__cpu_template(TableFunctionManager& mgr,
                          const Column<Array<TextEncodingDict>>& input,
                          Column<Array<TextEncodingDict>>& first,
                          Column<Array<TextEncodingDict>>& second);

#endif  // #ifndef __CUDACC__
