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
  This file contains testing compile-time UDTFs. The unit-tests are
  implemented within the RBC package.
 */

#define CPU_DEVICE_CODE 0x637075;  // 'cpu' in hex
#define GPU_DEVICE_CODE 0x677075;  // 'gpu' in hex

EXTENSION_NOINLINE
int32_t ct_device_selection_udtf_any(const Column<int32_t>& input, Column<int64_t>& out) {
#ifdef __CUDACC__
  out[0] = GPU_DEVICE_CODE;
#else
  out[0] = CPU_DEVICE_CODE;
#endif
  return 1;
}

#ifndef __CUDACC__

EXTENSION_NOINLINE_HOST
int32_t ct_device_selection_udtf_cpu__cpu_(const Column<int32_t>& input,
                                           Column<int64_t>& out) {
  out[0] = CPU_DEVICE_CODE;
  return 1;
}

#endif  // #ifndef __CUDACC__

EXTENSION_NOINLINE
int32_t ct_device_selection_udtf_gpu__gpu_(const Column<int32_t>& input,
                                           Column<int64_t>& out) {
  out[0] = GPU_DEVICE_CODE;
  return 1;
}

#ifndef __CUDACC__

EXTENSION_NOINLINE_HOST
int32_t ct_device_selection_udtf_both__cpu_(const Column<int32_t>& input,
                                            Column<int64_t>& out) {
  out[0] = CPU_DEVICE_CODE;
  return 1;
}

#endif  // #ifndef __CUDACC__

EXTENSION_NOINLINE
int32_t ct_device_selection_udtf_both__gpu_(const Column<int32_t>& input,
                                            Column<int64_t>& out) {
  out[0] = GPU_DEVICE_CODE;
  return 1;
}

#undef CPU_DEVICE_CODE
#undef GPU_DEVICE_CODE

#ifndef __CUDACC__

EXTENSION_NOINLINE_HOST int32_t
ct_binding_udtf_constant__cpu_1(const Column<int32_t>& input1, Column<int32_t>& out) {
  out[0] = 1;
  return 1;
}
EXTENSION_NOINLINE_HOST int32_t
ct_binding_udtf_constant__cpu_2(const Column<int32_t>& input1,
                                const Column<int32_t>& input2,
                                Column<int32_t>& out) {
  out[0] = 11;
  return 1;
}
EXTENSION_NOINLINE_HOST int32_t
ct_binding_udtf_constant__cpu_3(const Column<int32_t>& input1,
                                const Column<int32_t>& input2,
                                const Column<int32_t>& input3,
                                Column<int32_t>& out) {
  out[0] = 111;
  return 1;
}
EXTENSION_NOINLINE_HOST int32_t
ct_binding_udtf_constant__cpu_4(const Column<int64_t>& input1,
                                const Column<int32_t>& input2,
                                const Column<int32_t>& input3,
                                Column<int32_t>& out) {
  out[0] = 211;
  return 1;
}
EXTENSION_NOINLINE_HOST int32_t
ct_binding_udtf_constant__cpu_5(const Column<int64_t>& input1,
                                const Column<int64_t>& input2,
                                const Column<int32_t>& input3,
                                Column<int32_t>& out) {
  out[0] = 221;
  return 1;
}
EXTENSION_NOINLINE_HOST int32_t
ct_binding_udtf_constant__cpu_6(const Column<int64_t>& input1,
                                const Column<int32_t>& input2,
                                const Column<int64_t>& input3,
                                Column<int32_t>& out) {
  out[0] = 212;
  return 1;
}
EXTENSION_NOINLINE_HOST int32_t
ct_binding_udtf_constant__cpu_7(const Column<int32_t>& input1,
                                const ColumnList<int32_t>& input2,
                                Column<int32_t>& out) {
  out[0] = 13;
  return 1;
}
EXTENSION_NOINLINE_HOST int32_t
ct_binding_udtf_constant__cpu_8(const ColumnList<int32_t>& input1,
                                const Column<int64_t>& input2,
                                Column<int32_t>& out) {
  out[0] = 32;
  return 1;
}
EXTENSION_NOINLINE_HOST int32_t
ct_binding_udtf_constant__cpu_9(const ColumnList<int32_t>& input1,
                                const ColumnList<int64_t>& input2,
                                Column<int32_t>& out) {
  out[0] = 34;
  return 1;
}
EXTENSION_NOINLINE_HOST int32_t
ct_binding_udtf_constant__cpu_10(const Column<int64_t>& input1,
                                 const ColumnList<int64_t>& input2,
                                 const Column<int64_t>& input3,
                                 Column<int64_t>& out) {
  out[0] = 242;
  return 1;
}

EXTENSION_NOINLINE_HOST int32_t ct_binding_udtf__cpu_11(const Column<int32_t>& input1,
                                                        const int32_t multiplier,
                                                        Column<int32_t>& out) {
  out[0] = 1000 + 19 + multiplier;
  return 1;
}
EXTENSION_NOINLINE_HOST int32_t ct_binding_udtf__cpu_12(const Column<int32_t>& input1,
                                                        const Column<int32_t>& input2,
                                                        const int32_t multiplier,
                                                        Column<int32_t>& out) {
  out[0] = 1000 + 119 + multiplier;
  return 1;
}
EXTENSION_NOINLINE_HOST int32_t ct_binding_udtf__cpu_13(const Column<int32_t>& input1,
                                                        const Column<int32_t>& input2,
                                                        const Column<int32_t>& input3,
                                                        const int32_t multiplier,
                                                        Column<int32_t>& out) {
  out[0] = 1000 + 1119 + multiplier;
  return 1;
}
EXTENSION_NOINLINE_HOST int32_t ct_binding_udtf__cpu_14(const Column<int64_t>& input1,
                                                        const Column<int32_t>& input2,
                                                        const Column<int32_t>& input3,
                                                        const int32_t multiplier,
                                                        Column<int32_t>& out) {
  out[0] = 1000 + 2119 + multiplier;
  return 1;
}
EXTENSION_NOINLINE_HOST int32_t ct_binding_udtf__cpu_15(const Column<int64_t>& input1,
                                                        const Column<int64_t>& input2,
                                                        const Column<int32_t>& input3,
                                                        const int32_t multiplier,
                                                        Column<int32_t>& out) {
  out[0] = 1000 + 2219 + multiplier;
  return 1;
}
EXTENSION_NOINLINE_HOST int32_t ct_binding_udtf__cpu_16(const Column<int64_t>& input1,
                                                        const Column<int32_t>& input2,
                                                        const Column<int64_t>& input3,
                                                        const int32_t multiplier,
                                                        Column<int32_t>& out) {
  out[0] = 1000 + 2129 + multiplier;
  return 1;
}
EXTENSION_NOINLINE_HOST int32_t ct_binding_udtf__cpu_17(const Column<int32_t>& input1,
                                                        const ColumnList<int32_t>& input2,
                                                        const int32_t multiplier,
                                                        Column<int32_t>& out) {
  out[0] = 1000 + 139 + multiplier;
  return 1;
}
EXTENSION_NOINLINE_HOST int32_t ct_binding_udtf__cpu_18(const ColumnList<int32_t>& input1,
                                                        const Column<int64_t>& input2,
                                                        const int32_t multiplier,
                                                        Column<int32_t>& out) {
  out[0] = 1000 + 329 + multiplier;
  return 1;
}
EXTENSION_NOINLINE_HOST int32_t ct_binding_udtf__cpu_19(const ColumnList<int32_t>& input1,
                                                        const ColumnList<int64_t>& input2,
                                                        const int32_t multiplier,
                                                        Column<int32_t>& out) {
  out[0] = 1000 + 349 + multiplier;
  return 1;
}
EXTENSION_NOINLINE_HOST int32_t ct_binding_udtf__cpu_20(const Column<int64_t>& input1,
                                                        const ColumnList<int64_t>& input2,
                                                        const Column<int64_t>& input3,
                                                        const int32_t multiplier,
                                                        Column<int64_t>& out) {
  out[0] = 1000 + 2429 + multiplier;
  return 1;
}

EXTENSION_NOINLINE_HOST int32_t ct_binding_udtf2__cpu_21(const int32_t multiplier,
                                                         const Column<int32_t>& input1,
                                                         Column<int32_t>& out) {
  out[0] = 1000 + 91 + multiplier;
  return 1;
}
EXTENSION_NOINLINE_HOST int32_t ct_binding_udtf6__cpu_22(const Column<int32_t>& input1,
                                                         const int32_t multiplier,
                                                         const int32_t input2,
                                                         Column<int32_t>& out) {
  out[0] = 1000 + 196 + multiplier + 10 * input2;
  return 1;
}
EXTENSION_NOINLINE_HOST int32_t
ct_binding_udtf4__cpu_23(const ColumnList<int32_t>& input1,
                         const int32_t multiplier,
                         const int32_t input2,
                         Column<int32_t>& out) {
  out[0] = 1000 + 396 + multiplier + 10 * input2;
  return 1;
}
EXTENSION_NOINLINE_HOST int32_t
ct_binding_udtf5__cpu_24(const ColumnList<int32_t>& input1,
                         const int32_t input2,
                         const int32_t multiplier,
                         Column<int32_t>& out) {
  out[0] = 1000 + 369 + multiplier + 10 * input2;
  return 1;
}
EXTENSION_NOINLINE_HOST int32_t ct_binding_udtf3__cpu_25(const Column<int32_t>& input1,
                                                         const int32_t input2,
                                                         const int32_t multiplier,
                                                         Column<int32_t>& out) {
  out[0] = 1000 + 169 + multiplier + 10 * input2;
  return 1;
}

#endif  // #ifndef __CUDACC__

/*
 Test functions for default sizer parameter:
*/

#ifndef __CUDACC__

EXTENSION_NOINLINE_HOST int32_t
ct_udtf_default_sizer1a__cpu_1(const Column<int32_t>& input1,
                               const int32_t multiplier,
                               Column<int32_t>& out) {
  out[0] = 1000 + 1 + 10 * multiplier;
  return 1;
}
EXTENSION_NOINLINE_HOST int32_t
ct_udtf_default_sizer1b__cpu_2(const Column<int32_t>& input1,
                               const Column<int32_t>& input2,
                               const int32_t multiplier,
                               Column<int32_t>& out) {
  out[0] = 1000 + 2 + 11 * multiplier;
  return 1;
}
EXTENSION_NOINLINE_HOST int32_t
ct_udtf_default_sizer1c__cpu_3(const Column<int32_t>& input1,
                               const Column<int32_t>& input2,
                               const Column<int32_t>& input3,
                               const int32_t multiplier,
                               const Column<int32_t>& input4,
                               const int32_t x,
                               Column<int32_t>& out) {
  out[0] = 1000 + 101 + 10 * multiplier + x;
  return 1;
}
EXTENSION_NOINLINE_HOST int32_t
ct_udtf_default_sizer1d__cpu_4(const int32_t multiplier,
                               const int32_t x,
                               const Column<int32_t>& input1,
                               Column<int32_t>& out) {
  out[0] = 1000 + 99 + 10 * multiplier + x;
  return 1;
}
EXTENSION_NOINLINE_HOST int32_t
ct_udtf_default_sizer2a__cpu_1(const Column<int32_t>& input1,
                               const int32_t x,
                               const int32_t multiplier,
                               Column<int32_t>& out) {
  out[0] = 1000 + 98 + multiplier + 10 * x;
  return 1;
}
EXTENSION_NOINLINE_HOST int32_t
ct_udtf_default_sizer2b__cpu_2(const Column<int32_t>& input1,
                               const int32_t multiplier,
                               const Column<int32_t>& input2,
                               Column<int32_t>& out) {
  out[0] = 1000 + 2 + multiplier;
  return 1;
}
EXTENSION_NOINLINE_HOST int32_t
ct_udtf_default_sizer2c__cpu_3(const int32_t x,
                               const int32_t multiplier,
                               const Column<int32_t>& input1,
                               Column<int32_t>& out) {
  out[0] = 1000 + 99 + multiplier + 11 * x;
  return 1;
}
EXTENSION_NOINLINE_HOST int32_t
ct_udtf_default_sizer3a__cpu_1(const Column<int32_t>& input1,
                               const int32_t multiplier,
                               const int32_t x,
                               Column<int32_t>& out) {
  out[0] = 1000 + 98 + 100 * multiplier + x;
  return 1;
}
EXTENSION_NOINLINE_HOST int32_t
ct_udtf_default_sizer3b__cpu_2(const Column<int32_t>& input1,
                               const int32_t x,
                               const Column<int32_t>& input2,
                               const int32_t multiplier,
                               Column<int32_t>& out) {
  out[0] = 1000 + 99 + 100 * multiplier + x;
  return 1;
}
EXTENSION_NOINLINE_HOST int32_t
ct_udtf_default_sizer4a__cpu_1(const Column<int32_t>& input1,
                               const int32_t multiplier,
                               const Column<int32_t>& input2,
                               const int32_t x,
                               Column<int32_t>& out) {
  out[0] = 1000 + 99 + 10 * multiplier + x;
  return 1;
}
EXTENSION_NOINLINE_HOST int32_t
ct_udtf_default_sizer4b__cpu_2(const int32_t multiplier,
                               const Column<int32_t>& input,
                               const int32_t x,
                               Column<int32_t>& out) {
  out[0] = 1000 + 99 + 9 * multiplier + x;
  return 1;
}

#endif  // #ifndef __CUDACC__

#ifndef __CUDACC__

EXTENSION_NOINLINE_HOST int32_t
ct_binding_dict_encoded1__cpu_1(const Column<TextEncodingDict>& input,
                                const int32_t multiplier,
                                Column<TextEncodingDict>& out) {
  for (int64_t i = 0; i < input.size(); i++) {
    out[i] = input[i];  // assign string id
  }
  return multiplier * input.size();
}

EXTENSION_NOINLINE_HOST int32_t
ct_binding_dict_encoded2__cpu_1(const Column<TextEncodingDict>& input1,
                                const Column<TextEncodingDict>& input2,
                                Column<TextEncodingDict>& out1,
                                Column<TextEncodingDict>& out2) {
  set_output_row_size(input1.size());
  for (int64_t i = 0; i < input1.size(); i++) {
    out1[i] = input1[i];
    out2[i] = input2[i];
  }
  return input1.size();
}

EXTENSION_NOINLINE_HOST int32_t
ct_binding_dict_encoded3__cpu_1(const Column<TextEncodingDict>& input1,
                                const Column<TextEncodingDict>& input2,
                                Column<TextEncodingDict>& out1,
                                Column<TextEncodingDict>& out2) {
  set_output_row_size(input1.size());
  for (int64_t i = 0; i < input1.size(); i++) {
    out1[i] = input2[i];
    out2[i] = input1[i];
  }
  return input1.size();
}

EXTENSION_NOINLINE_HOST int32_t
ct_binding_dict_encoded4__cpu_1(const ColumnList<TextEncodingDict>& input,
                                Column<TextEncodingDict>& out) {
  int64_t sz = input[0].size();
  set_output_row_size(sz);
  for (int64_t i = 0; i < sz; i++) {
    out[i] = input[0][i];
  }
  return sz;
}

EXTENSION_NOINLINE_HOST int32_t
ct_binding_dict_encoded5__cpu_1(const ColumnList<TextEncodingDict>& input,
                                Column<TextEncodingDict>& out) {
  int64_t sz = input[1].size();
  set_output_row_size(sz);
  for (int64_t i = 0; i < sz; i++) {
    out[i] = input[1][i];
  }
  return sz;
}

EXTENSION_NOINLINE_HOST int32_t
ct_binding_dict_encoded6__cpu_1(const ColumnList<TextEncodingDict>& input,
                                Column<TextEncodingDict>& out0,
                                Column<TextEncodingDict>& out1) {
  int64_t sz = input[0].size();
  set_output_row_size(sz);
  for (int64_t i = 0; i < sz; i++) {
    out0[i] = input[0][i];
    out1[i] = input[1][i];
  }
  return sz;
}

#endif  // #ifndef __CUDACC__

#ifndef __CUDACC__

template <typename T>
NEVER_INLINE HOST int32_t ct_binding_template__cpu_template(const Column<T>& input,
                                                            Column<T>& out) {
  set_output_row_size(input.size());
  for (int64_t i = 0; i < input.size(); i++) {
    out[i] = input[i];
  }
  return input.size();
}

// explicit instantiations
template NEVER_INLINE HOST int32_t
ct_binding_template__cpu_template(const Column<int>& input, Column<int>& out);
template NEVER_INLINE HOST int32_t
ct_binding_template__cpu_template(const Column<TextEncodingDict>& input,
                                  Column<TextEncodingDict>& out);
template NEVER_INLINE HOST int32_t
ct_binding_template__cpu_template(const Column<float>& input, Column<float>& out);

#endif  // #ifndef __CUDACC__

#ifndef __CUDACC__

template <typename T>
NEVER_INLINE HOST int32_t ct_binding_columnlist__cpu_template(const Column<T>& input1,
                                                              const ColumnList<T>& input2,
                                                              Column<int32_t>& out) {
  set_output_row_size(1);
  if constexpr (std::is_same<T, int32_t>::value) {
    out[0] = 1;
  } else if constexpr (std::is_same<T, float>::value) {
    out[0] = 2;
  } else if constexpr (std::is_same<T, TextEncodingDict>::value) {
    out[0] = 3;
  } else {
    out[0] = 4;
  }
  return 1;
}

// explicit instantiations
template NEVER_INLINE HOST int32_t
ct_binding_columnlist__cpu_template(const Column<int32_t>& input1,
                                    const ColumnList<int32_t>& input2,
                                    Column<int32_t>& out);
template NEVER_INLINE HOST int32_t
ct_binding_columnlist__cpu_template(const Column<float>& input1,
                                    const ColumnList<float>& input2,
                                    Column<int32_t>& out);
template NEVER_INLINE HOST int32_t
ct_binding_columnlist__cpu_template(const Column<TextEncodingDict>& input1,
                                    const ColumnList<TextEncodingDict>& input2,
                                    Column<int32_t>& out);
template NEVER_INLINE HOST int32_t
ct_binding_columnlist__cpu_template(const Column<int16_t>& input1,
                                    const ColumnList<int16_t>& input2,
                                    Column<int32_t>& out);

#endif  // #ifndef __CUDACC__

#ifndef __CUDACC__

template <typename T>
NEVER_INLINE HOST int32_t ct_binding_column__cpu_template(const Column<T>& input,
                                                          Column<int32_t>& out) {
  set_output_row_size(1);
  if constexpr (std::is_same<T, int32_t>::value) {
    out[0] = 10;
  } else {
    out[0] = 20;
  }
  return 1;
}

// explicit instantiations
template NEVER_INLINE HOST int32_t
ct_binding_column__cpu_template(const Column<int32_t>& input, Column<int32_t>& out);
template NEVER_INLINE HOST int32_t
ct_binding_column__cpu_template(const Column<float>& input, Column<int32_t>& out);

#endif  // #ifndef __CUDACC__