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

#pragma once

#include "../../QueryEngine/OmniSciTypes.h"

/*
  This file contains tesing compile-time UDTFs. The unit-tests are
  implemented within the RBC package.
 */

#define CPU_DEVICE_CODE 0x637075;  // 'cpu' in hex
#define GPU_DEVICE_CODE 0x677075;  // 'gpu' in hex

// clang-format off
/*
  UDTF: ct_device_selection_udtf_any(Cursor<int32_t>, Constant<1>) -> Column<int32_t>
  UDTF: ct_device_selection_udtf_cpu__cpu_(Cursor<int32_t>, Constant<1>) -> Column<int32_t>
  UDTF: ct_device_selection_udtf_gpu__gpu_(Cursor<int32_t>, Constant<1>) -> Column<int32_t>
  UDTF: ct_device_selection_udtf_both__cpu_(Cursor<int32_t>, Constant<1>) -> Column<int32_t>
  UDTF: ct_device_selection_udtf_both__gpu_(Cursor<int32_t>, Constant<1>) -> Column<int32_t>
*/
// clang-format on

EXTENSION_NOINLINE
int32_t ct_device_selection_udtf_any(const Column<int32_t>& input, Column<int64_t>& out) {
#ifdef __CUDACC__
  out[0] = GPU_DEVICE_CODE;
#else
  out[0] = CPU_DEVICE_CODE;
#endif
  return 1;
}

EXTENSION_NOINLINE
int32_t ct_device_selection_udtf_cpu__cpu_(const Column<int32_t>& input,
                                           Column<int64_t>& out) {
  out[0] = CPU_DEVICE_CODE;
  return 1;
}

EXTENSION_NOINLINE
int32_t ct_device_selection_udtf_gpu__gpu_(const Column<int32_t>& input,
                                           Column<int64_t>& out) {
  out[0] = GPU_DEVICE_CODE;
  return 1;
}

EXTENSION_NOINLINE
int32_t ct_device_selection_udtf_both__cpu_(const Column<int32_t>& input,
                                            Column<int64_t>& out) {
  out[0] = CPU_DEVICE_CODE;
  return 1;
}

EXTENSION_NOINLINE
int32_t ct_device_selection_udtf_both__gpu_(const Column<int32_t>& input,
                                            Column<int64_t>& out) {
  out[0] = GPU_DEVICE_CODE;
  return 1;
}

#undef CPU_DEVICE_CODE
#undef GPU_DEVICE_CODE

// clang-format off
/*
  Test functions for constant sizer parameter:

  UDTF: ct_binding_udtf_constant__cpu_1(Cursor<int32_t>, Constant<1>) -> Column<int32_t>
  UDTF: ct_binding_udtf_constant__cpu_2(Cursor<int32_t, int32_t>, Constant<1>) -> Column<int32_t>
  UDTF: ct_binding_udtf_constant__cpu_3(Cursor<int32_t, int32_t, int32_t>, Constant<1>) -> Column<int32_t>
  UDTF: ct_binding_udtf_constant__cpu_4(Cursor<int64_t, int32_t, int32_t>, Constant<1>) -> Column<int32_t>
  UDTF: ct_binding_udtf_constant__cpu_5(Cursor<int64_t, int64_t, int32_t>, Constant<1>) -> Column<int32_t>
  UDTF: ct_binding_udtf_constant__cpu_6(Cursor<int64_t, int32_t, int64_t>, Constant<1>) -> Column<int32_t>
  UDTF: ct_binding_udtf_constant__cpu_7(Cursor<int32_t, ColumnList<int32_t>>, Constant<1>) -> Column<int32_t>
  UDTF: ct_binding_udtf_constant__cpu_8(Cursor<ColumnList<int32_t>, int64_t>, Constant<1>) -> Column<int32_t>
  UDTF: ct_binding_udtf_constant__cpu_9(Cursor<ColumnList<int32_t>, ColumnList<int64_t>>, Constant<1>) -> Column<int32_t>
  UDTF: ct_binding_udtf_constant__cpu_10(Cursor<int64_t, ColumnList<int64_t>, int64_t>, Constant<1>) -> Column<int32_t>


  Test functions for row multiplier sizer parameter:

  UDTF: ct_binding_udtf__cpu_11(Cursor<int32_t>, RowMultiplier) -> Column<int32_t>
  UDTF: ct_binding_udtf__cpu_12(Cursor<int32_t, int32_t>, RowMultiplier) -> Column<int32_t>
  UDTF: ct_binding_udtf__cpu_13(Cursor<int32_t, int32_t, int32_t>, RowMultiplier) -> Column<int32_t>
  UDTF: ct_binding_udtf__cpu_14(Cursor<int64_t, int32_t, int32_t>, RowMultiplier) -> Column<int32_t>
  UDTF: ct_binding_udtf__cpu_15(Cursor<int64_t, int64_t, int32_t>, RowMultiplier) -> Column<int32_t>
  UDTF: ct_binding_udtf__cpu_16(Cursor<int64_t, int32_t, int64_t>, RowMultiplier) -> Column<int32_t>
  UDTF: ct_binding_udtf__cpu_17(Cursor<int32_t, ColumnList<int32_t>>, RowMultiplier) -> Column<int32_t>
  UDTF: ct_binding_udtf__cpu_18(Cursor<ColumnList<int32_t>, int64_t>, RowMultiplier) -> Column<int32_t>
  UDTF: ct_binding_udtf__cpu_19(Cursor<ColumnList<int32_t>, ColumnList<int64_t>>, RowMultiplier) -> Column<int32_t>
  UDTF: ct_binding_udtf__cpu_20(Cursor<int64_t, ColumnList<int64_t>, int64_t>, RowMultiplier) -> Column<int32_t>

  UDTF: ct_binding_udtf2__cpu_21(RowMultiplier, Cursor<int32_t>) -> Column<int32_t>
  UDTF: ct_binding_udtf6__cpu_22(Cursor<int32_t>, RowMultiplier, int32_t) -> Column<int32_t>
  UDTF: ct_binding_udtf4__cpu_23(Cursor<ColumnList<int32_t>>, RowMultiplier, int32_t) -> Column<int32_t>
  UDTF: ct_binding_udtf5__cpu_24(Cursor<ColumnList<int32_t>>, int32_t, RowMultiplier) -> Column<int32_t>
  UDTF: ct_binding_udtf3__cpu_25(Cursor<Column<int32_t>>, int32_t, RowMultiplier) -> Column<int32_t>
*/
// clang-format on

EXTENSION_NOINLINE int32_t ct_binding_udtf_constant__cpu_1(const Column<int32_t>& input1,
                                                           Column<int32_t>& out) {
  out[0] = 1;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf_constant__cpu_2(const Column<int32_t>& input1,
                                                           const Column<int32_t>& input2,
                                                           Column<int32_t>& out) {
  out[0] = 11;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf_constant__cpu_3(const Column<int32_t>& input1,
                                                           const Column<int32_t>& input2,
                                                           const Column<int32_t>& input3,
                                                           Column<int32_t>& out) {
  out[0] = 111;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf_constant__cpu_4(const Column<int64_t>& input1,
                                                           const Column<int32_t>& input2,
                                                           const Column<int32_t>& input3,
                                                           Column<int32_t>& out) {
  out[0] = 211;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf_constant__cpu_5(const Column<int64_t>& input1,
                                                           const Column<int64_t>& input2,
                                                           const Column<int32_t>& input3,
                                                           Column<int32_t>& out) {
  out[0] = 221;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf_constant__cpu_6(const Column<int64_t>& input1,
                                                           const Column<int32_t>& input2,
                                                           const Column<int64_t>& input3,
                                                           Column<int32_t>& out) {
  out[0] = 212;
  return 1;
}
EXTENSION_NOINLINE int32_t
ct_binding_udtf_constant__cpu_7(const Column<int32_t>& input1,
                                const ColumnList<int32_t>& input2,
                                Column<int32_t>& out) {
  out[0] = 13;
  return 1;
}
EXTENSION_NOINLINE int32_t
ct_binding_udtf_constant__cpu_8(const ColumnList<int32_t>& input1,
                                const Column<int64_t>& input2,
                                Column<int32_t>& out) {
  out[0] = 32;
  return 1;
}
EXTENSION_NOINLINE int32_t
ct_binding_udtf_constant__cpu_9(const ColumnList<int32_t>& input1,
                                const ColumnList<int64_t>& input2,
                                Column<int32_t>& out) {
  out[0] = 34;
  return 1;
}
EXTENSION_NOINLINE int32_t
ct_binding_udtf_constant__cpu_10(const Column<int64_t>& input1,
                                 const ColumnList<int64_t>& input2,
                                 const Column<int64_t>& input3,
                                 Column<int64_t>& out) {
  out[0] = 242;
  return 1;
}

EXTENSION_NOINLINE int32_t ct_binding_udtf__cpu_11(const Column<int32_t>& input1,
                                                   const int32_t multiplier,
                                                   Column<int32_t>& out) {
  out[0] = 1000 + 19 + multiplier;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf__cpu_12(const Column<int32_t>& input1,
                                                   const Column<int32_t>& input2,
                                                   const int32_t multiplier,
                                                   Column<int32_t>& out) {
  out[0] = 1000 + 119 + multiplier;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf__cpu_13(const Column<int32_t>& input1,
                                                   const Column<int32_t>& input2,
                                                   const Column<int32_t>& input3,
                                                   const int32_t multiplier,
                                                   Column<int32_t>& out) {
  out[0] = 1000 + 1119 + multiplier;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf__cpu_14(const Column<int64_t>& input1,
                                                   const Column<int32_t>& input2,
                                                   const Column<int32_t>& input3,
                                                   const int32_t multiplier,
                                                   Column<int32_t>& out) {
  out[0] = 1000 + 2119 + multiplier;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf__cpu_15(const Column<int64_t>& input1,
                                                   const Column<int64_t>& input2,
                                                   const Column<int32_t>& input3,
                                                   const int32_t multiplier,
                                                   Column<int32_t>& out) {
  out[0] = 1000 + 2219 + multiplier;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf__cpu_16(const Column<int64_t>& input1,
                                                   const Column<int32_t>& input2,
                                                   const Column<int64_t>& input3,
                                                   const int32_t multiplier,
                                                   Column<int32_t>& out) {
  out[0] = 1000 + 2129 + multiplier;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf__cpu_17(const Column<int32_t>& input1,
                                                   const ColumnList<int32_t>& input2,
                                                   const int32_t multiplier,
                                                   Column<int32_t>& out) {
  out[0] = 1000 + 139 + multiplier;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf__cpu_18(const ColumnList<int32_t>& input1,
                                                   const Column<int64_t>& input2,
                                                   const int32_t multiplier,
                                                   Column<int32_t>& out) {
  out[0] = 1000 + 329 + multiplier;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf__cpu_19(const ColumnList<int32_t>& input1,
                                                   const ColumnList<int64_t>& input2,
                                                   const int32_t multiplier,
                                                   Column<int32_t>& out) {
  out[0] = 1000 + 349 + multiplier;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf__cpu_20(const Column<int64_t>& input1,
                                                   const ColumnList<int64_t>& input2,
                                                   const Column<int64_t>& input3,
                                                   const int32_t multiplier,
                                                   Column<int64_t>& out) {
  out[0] = 1000 + 2429 + multiplier;
  return 1;
}

EXTENSION_NOINLINE int32_t ct_binding_udtf2__cpu_21(const int32_t multiplier,
                                                    const Column<int32_t>& input1,
                                                    Column<int32_t>& out) {
  out[0] = 1000 + 91 + multiplier;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf6__cpu_22(const Column<int32_t>& input1,
                                                    const int32_t multiplier,
                                                    const int32_t input2,
                                                    Column<int32_t>& out) {
  out[0] = 1000 + 196 + multiplier + 10 * input2;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf4__cpu_23(const ColumnList<int32_t>& input1,
                                                    const int32_t multiplier,
                                                    const int32_t input2,
                                                    Column<int32_t>& out) {
  out[0] = 1000 + 396 + multiplier + 10 * input2;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf5__cpu_24(const ColumnList<int32_t>& input1,
                                                    const int32_t input2,
                                                    const int32_t multiplier,
                                                    Column<int32_t>& out) {
  out[0] = 1000 + 369 + multiplier + 10 * input2;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf3__cpu_25(const Column<int32_t>& input1,
                                                    const int32_t input2,
                                                    const int32_t multiplier,
                                                    Column<int32_t>& out) {
  out[0] = 1000 + 169 + multiplier + 10 * input2;
  return 1;
}

/*
 Test functions for default sizer parameter:
*/

// clang-format off
/*
  UDTF: ct_udtf_default_sizer1a__cpu_1(Cursor<int32_t>, RowMultiplier) -> Column<int32_t>
  UDTF: ct_udtf_default_sizer1b__cpu_2(Cursor<int32_t>, Cursor<int32_t>, RowMultiplier) -> Column<int32_t>
  UDTF: ct_udtf_default_sizer1c__cpu_3(Cursor<int32_t, int32_t, int32_t>, RowMultiplier, Cursor<int32_t>, int32_t) -> Column<int32_t>
  UDTF: ct_udtf_default_sizer1d__cpu_4(RowMultiplier, int32_t, Cursor<int32_t>) -> Column<int32_t>
  UDTF: ct_udtf_default_sizer2a__cpu_1(Cursor<int32_t>, int32_t, RowMultiplier) -> Column<int32_t>
  UDTF: ct_udtf_default_sizer2b__cpu_2(Cursor<int32_t>, RowMultiplier, Cursor<int32_t>) -> Column<int32_t>
  UDTF: ct_udtf_default_sizer2c__cpu_3(int32_t, RowMultiplier, Cursor<int32_t>) -> Column<int32_t>
  UDTF: ct_udtf_default_sizer3a__cpu_1(Cursor<int32_t>, RowMultiplier, int32_t) -> Column<int32_t>
  UDTF: ct_udtf_default_sizer3b__cpu_2(Cursor<int32_t>, int32_t, Cursor<int32_t>, RowMultiplier) -> Column<int32_t>
  UDTF: ct_udtf_default_sizer4a__cpu_1(Cursor<int32_t>, RowMultiplier, Cursor<int32_t>, int32_t) -> Column<int32_t>
  UDTF: ct_udtf_default_sizer4b__cpu_2(RowMultiplier, Cursor<int32_t>, int32_t) -> Column<int32_t>
*/
// clang-format on
EXTENSION_NOINLINE int32_t ct_udtf_default_sizer1a__cpu_1(const Column<int32_t>& input1,
                                                          const int32_t multiplier,
                                                          Column<int32_t>& out) {
  out[0] = 1000 + 1 + 10 * multiplier;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_udtf_default_sizer1b__cpu_2(const Column<int32_t>& input1,
                                                          const Column<int32_t>& input2,
                                                          const int32_t multiplier,
                                                          Column<int32_t>& out) {
  out[0] = 1000 + 2 + 11 * multiplier;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_udtf_default_sizer1c__cpu_3(const Column<int32_t>& input1,
                                                          const Column<int32_t>& input2,
                                                          const Column<int32_t>& input3,
                                                          const int32_t multiplier,
                                                          const Column<int32_t>& input4,
                                                          const int32_t x,
                                                          Column<int32_t>& out) {
  out[0] = 1000 + 101 + 10 * multiplier + x;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_udtf_default_sizer1d__cpu_4(const int32_t multiplier,
                                                          const int32_t x,
                                                          const Column<int32_t>& input1,
                                                          Column<int32_t>& out) {
  out[0] = 1000 + 99 + 10 * multiplier + x;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_udtf_default_sizer2a__cpu_1(const Column<int32_t>& input1,
                                                          const int32_t x,
                                                          const int32_t multiplier,
                                                          Column<int32_t>& out) {
  out[0] = 1000 + 98 + multiplier + 10 * x;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_udtf_default_sizer2b__cpu_2(const Column<int32_t>& input1,
                                                          const int32_t multiplier,
                                                          const Column<int32_t>& input2,
                                                          Column<int32_t>& out) {
  out[0] = 1000 + 2 + multiplier;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_udtf_default_sizer2c__cpu_3(const int32_t x,
                                                          const int32_t multiplier,
                                                          const Column<int32_t>& input1,
                                                          Column<int32_t>& out) {
  out[0] = 1000 + 99 + multiplier + 11 * x;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_udtf_default_sizer3a__cpu_1(const Column<int32_t>& input1,
                                                          const int32_t multiplier,
                                                          const int32_t x,
                                                          Column<int32_t>& out) {
  out[0] = 1000 + 98 + 100 * multiplier + x;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_udtf_default_sizer3b__cpu_2(const Column<int32_t>& input1,
                                                          const int32_t x,
                                                          const Column<int32_t>& input2,
                                                          const int32_t multiplier,
                                                          Column<int32_t>& out) {
  out[0] = 1000 + 99 + 100 * multiplier + x;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_udtf_default_sizer4a__cpu_1(const Column<int32_t>& input1,
                                                          const int32_t multiplier,
                                                          const Column<int32_t>& input2,
                                                          const int32_t x,
                                                          Column<int32_t>& out) {
  out[0] = 1000 + 99 + 10 * multiplier + x;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_udtf_default_sizer4b__cpu_2(const int32_t multiplier,
                                                          const Column<int32_t>& input,
                                                          const int32_t x,
                                                          Column<int32_t>& out) {
  out[0] = 1000 + 99 + 9 * multiplier + x;
  return 1;
}

// clang-format off
/*
  UDTF: ct_binding_dict_encoded1__cpu_1(Cursor<TextEncodingDict>, RowMultiplier) -> Column<TextEncodingDict> | input_id=args<0>
  UDTF: ct_binding_dict_encoded2__cpu_1(Cursor<TextEncodingDict, TextEncodingDict>) -> Column<TextEncodingDict> | input_id=args<0, 0>, Column<TextEncodingDict> | input_id=args<0, 1>
  UDTF: ct_binding_dict_encoded3__cpu_1(Cursor<TextEncodingDict, TextEncodingDict>) -> Column<TextEncodingDict> | input_id=args<0, 1>, Column<TextEncodingDict> | input_id=args<0, 0>
  UDTF: ct_binding_dict_encoded4__cpu_1(Cursor<ColumnList<TextEncodingDict>>) -> Column<TextEncodingDict> | input_id=args<0,0>
  UDTF: ct_binding_dict_encoded5__cpu_1(Cursor<ColumnList<TextEncodingDict>>) -> Column<TextEncodingDict> | input_id=args<0,1>
  UDTF: ct_binding_dict_encoded6__cpu_1(Cursor<ColumnList<TextEncodingDict>>) -> Column<TextEncodingDict> | input_id=args<0,0>, Column<TextEncodingDict> | input_id=args<0,1>
*/
// clang-format on
EXTENSION_NOINLINE int32_t
ct_binding_dict_encoded1__cpu_1(const Column<TextEncodingDict>& input,
                                const int32_t multiplier,
                                Column<TextEncodingDict>& out) {
  for (int64_t i = 0; i < input.size(); i++) {
    out[i] = input[i];  // assign string id
  }
  return multiplier * input.size();
}

EXTENSION_NOINLINE int32_t
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

EXTENSION_NOINLINE int32_t
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

EXTENSION_NOINLINE int32_t
ct_binding_dict_encoded4__cpu_1(const ColumnList<TextEncodingDict>& input,
                                Column<TextEncodingDict>& out) {
  int64_t sz = input[0].size();
  set_output_row_size(sz);
  for (int64_t i = 0; i < sz; i++) {
    out[i] = input[0][i];
  }
  return sz;
}

EXTENSION_NOINLINE int32_t
ct_binding_dict_encoded5__cpu_1(const ColumnList<TextEncodingDict>& input,
                                Column<TextEncodingDict>& out) {
  int64_t sz = input[1].size();
  set_output_row_size(sz);
  for (int64_t i = 0; i < sz; i++) {
    out[i] = input[1][i];
  }
  return sz;
}

EXTENSION_NOINLINE int32_t
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

// clang-format off
/*
  UDTF: ct_binding_template__cpu_template(Cursor<TextEncodingDict>) -> Column<TextEncodingDict> | input_id=args<0>
  UDTF: ct_binding_template__cpu_template(Cursor<int>) -> Column<int>
  UDTF: ct_binding_template__cpu_template(Cursor<float>) -> Column<float>
*/
// clang-format on
template <typename T>
TEMPLATE_NOINLINE int32_t ct_binding_template__cpu_template(const Column<T>& input,
                                                            Column<T>& out) {
  set_output_row_size(input.size());
  for (int64_t i = 0; i < input.size(); i++) {
    out[i] = input[i];
  }
  return input.size();
}

// clang-format off
/*
  UDTF: ct_binding_columnlist__cpu_template(Cursor<int32_t, ColumnList<int32_t>>) -> Column<int32_t>
  UDTF: ct_binding_columnlist__cpu_template(Cursor<float, ColumnList<float>>) -> Column<int32_t>
  UDTF: ct_binding_columnlist__cpu_template(Cursor<TextEncodingDict, ColumnList<TextEncodingDict>>) -> Column<int32_t>
  UDTF: ct_binding_columnlist__cpu_template(Cursor<int16_t, ColumnList<int16_t>>) -> Column<int32_t>
*/
// clang-format on
template <typename T>
TEMPLATE_NOINLINE int32_t ct_binding_columnlist__cpu_template(const Column<T>& input1,
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

// clang-format off
/*
  UDTF: ct_binding_column__cpu_template(Column<int32_t>) -> Column<int32_t>
  UDTF: ct_binding_column__cpu_template(Column<float>) -> Column<int32_t>
*/
// clang-format on
template <typename T>
TEMPLATE_NOINLINE int32_t ct_binding_column__cpu_template(const Column<T>& input,
                                                          Column<int32_t>& out) {
  set_output_row_size(1);
  if constexpr (std::is_same<T, int32_t>::value) {
    out[0] = 10;
  } else {
    out[0] = 20;
  }
  return 1;
}

// clang-format off
/*
  UDTF: ct_binding_scalar_multiply__cpu_template(Cursor<Column<float>>, float) -> Column<float>
  UDTF: ct_binding_scalar_multiply__cpu_template(Cursor<Column<double>>, double) -> Column<double>
  UDTF: ct_binding_scalar_multiply__cpu_template(Cursor<Column<int32_t>>, int32_t) -> Column<int32_t>
  UDTF: ct_binding_scalar_multiply__cpu_template(Cursor<Column<int64_t>>, int64_t) -> Column<int64_t>
*/
// clang-format on
template <typename T>
TEMPLATE_NOINLINE int32_t ct_binding_scalar_multiply__cpu_template(const Column<T>& input,
                                                                   const T multiplier,
                                                                   Column<T>& out) {
  const int64_t num_rows = input.size();
  set_output_row_size(num_rows);
  for (int64_t r = 0; r < num_rows; ++r) {
    if (!input.isNull(r)) {
      out[r] = input[r] * multiplier;
    } else {
      out.setNull(r);
    }
  }
  return num_rows;
}

#ifndef __CUDACC__

#include <algorithm>

template <typename T>
struct SortAsc {
  SortAsc(const bool nulls_last)
      : null_value_(std::numeric_limits<T>::lowest())
      , null_value_mapped_(map_null_value(nulls_last)) {}
  static T map_null_value(const bool nulls_last) {
    return nulls_last ? std::numeric_limits<T>::max() : std::numeric_limits<T>::lowest();
  }
  inline T mapValue(const T& val) {
    return val == null_value_ ? null_value_mapped_ : val;
  }
  bool operator()(const T& a, const T& b) { return mapValue(a) < mapValue(b); }
  const T null_value_;
  const T null_value_mapped_;
};

template <typename T>
struct SortDesc {
  SortDesc(const bool nulls_last)
      : null_value_(std::numeric_limits<T>::lowest())
      , null_value_mapped_(map_null_value(nulls_last)) {}
  static T map_null_value(const bool nulls_last) {
    return nulls_last ? std::numeric_limits<T>::lowest() : std::numeric_limits<T>::max();
  }

  inline T mapValue(const T& val) {
    return val == null_value_ ? null_value_mapped_ : val;
  }

  bool operator()(const T& a, const T& b) { return mapValue(a) > mapValue(b); }
  const T null_value_;
  const T null_value_mapped_;
};

// clang-format off
/*
  UDTF: sort_column_limit__cpu_template(Cursor<Column<int8_t>>, int32_t, bool, bool) -> Column<int8_t>
  UDTF: sort_column_limit__cpu_template(Cursor<Column<int16_t>>, int32_t, bool, bool) -> Column<int16_t>
  UDTF: sort_column_limit__cpu_template(Cursor<Column<int32_t>>, int32_t, bool, bool) -> Column<int32_t>
  UDTF: sort_column_limit__cpu_template(Cursor<Column<int64_t>>, int32_t, bool, bool) -> Column<int64_t>
  UDTF: sort_column_limit__cpu_template(Cursor<Column<float>>, int32_t, bool, bool) -> Column<float>
  UDTF: sort_column_limit__cpu_template(Cursor<Column<double>>, int32_t, bool, bool) -> Column<double>
*/
// clang-format on

template <typename T>
TEMPLATE_NOINLINE int32_t sort_column_limit__cpu_template(const Column<T>& input,
                                                          const int32_t limit,
                                                          const bool sort_ascending,
                                                          const bool nulls_last,
                                                          Column<T>& output) {
  const int64_t num_rows = input.size();
  set_output_row_size(num_rows);
  output = input;
  if (sort_ascending) {
    std::sort(output.ptr_, output.ptr_ + num_rows, SortAsc<T>(nulls_last));
  } else {
    std::sort(output.ptr_, output.ptr_ + num_rows, SortDesc<T>(nulls_last));
  }
  if (limit < 0 || limit > num_rows) {
    return num_rows;
  }
  return limit;
}

#endif

// clang-format off
/*
  UDTF: ct_binding_column2__cpu_template(Column<T>, Column<U>) -> Column<K>, T=[int32_t, double], U=[int32_t, double], K=[int32_t]
  UDTF: ct_binding_column2__cpu_template(Column<T>, Column<T>) -> Column<T> | input_id=args<0>, T=[TextEncodingDict]
*/
// clang-format on
template <typename T, typename U, typename K>
TEMPLATE_NOINLINE int32_t ct_binding_column2__cpu_template(const Column<T>& input1,
                                                           const Column<U>& input2,
                                                           Column<K>& out) {
  if constexpr (std::is_same<T, TextEncodingDict>::value &&
                std::is_same<U, TextEncodingDict>::value) {
    set_output_row_size(input1.size());
    for (int64_t i = 0; i < input1.size(); i++) {
      out[i] = input1[i];
    }
    return input1.size();
  }

  set_output_row_size(1);
  if constexpr (std::is_same<T, int32_t>::value && std::is_same<U, double>::value) {
    out[0] = 10;
  } else if constexpr (std::is_same<T, double>::value && std::is_same<U, double>::value) {
    out[0] = 20;
  } else if constexpr (std::is_same<T, int32_t>::value &&
                       std::is_same<U, int32_t>::value) {
    out[0] = 30;
  } else if constexpr (std::is_same<T, double>::value &&
                       std::is_same<U, int32_t>::value) {
    out[0] = 40;
  }
  return 1;
}

// clang-format off
/*
  UDTF: ct_named_output__cpu_template(Column<T> input) -> Column<T> total, T=[int32_t, double]
  UDTF: ct_named_const_output__template(Column<T> input, Constant<2>) -> Column<T> total, T=[int32_t, double]
  UDTF: ct_named_user_const_output__template(Column<T> input, ConstantParameter c) -> Column<T> total, T=[int32_t, double]
  UDTF: ct_named_rowmul_output__template(Column<T> input, RowMultiplier m) -> Column<T> total, T=[int32_t, double]
*/
// clang-format on

template <typename T>
TEMPLATE_NOINLINE int32_t ct_named_output__cpu_template(const Column<T>& input,
                                                        Column<T>& out) {
  set_output_row_size(1);
  T acc = 0;
  for (int64_t i = 0; i < input.size(); i++) {
    acc += input[i];
  }
  out[0] = acc;
  return 1;
}

template <typename T>
TEMPLATE_NOINLINE int32_t ct_named_const_output__template(const Column<T>& input,
                                                          Column<T>& out) {
  T acc1 = 0, acc2 = 0;
  for (int64_t i = 0; i < input.size(); i++) {
    if (i % 2 == 0) {
      acc1 += input[i];
    } else {
      acc2 += input[i];
    }
  }
  out[0] = acc1;
  out[1] = acc2;
  return 2;
}

template <typename T>
TEMPLATE_NOINLINE int32_t ct_named_user_const_output__template(const Column<T>& input,
                                                               int32_t c,
                                                               Column<T>& out) {
  for (int64_t i = 0; i < c; i++) {
    out[i] = 0;
  }
  for (int64_t i = 0; i < input.size(); i++) {
    out[i % c] += input[i];
  }
  return c;
}

template <typename T>
TEMPLATE_NOINLINE int32_t ct_named_rowmul_output__template(const Column<T>& input,
                                                           int32_t m,
                                                           Column<T>& out) {
  for (int64_t j = 0; j < m; j++) {
    for (int64_t i = 0; i < input.size(); i++) {
      out[j * input.size() + i] += input[i];
    }
  }
  return m * input.size();
}

// clang-format off
/*
  UDTF: ct_no_arg_runtime_sizing__cpu_template() -> Column<T> answer, T=[int32_t]
*/
// clang-format on

template <typename T>
TEMPLATE_NOINLINE int32_t ct_no_arg_runtime_sizing__cpu_template(Column<T>& answer) {
  set_output_row_size(1);
  answer[0] = 42;
  return 1;
}

// clang-format off
/*
  UDTF: ct_no_arg_constant_sizing__cpu_(Constant<42>) -> Column<int32_t> answer 
*/
// clang-format on

EXTENSION_NOINLINE int32_t ct_no_arg_constant_sizing__cpu_(Column<int32_t>& answer) {
#ifdef __CUDACC__
  int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t stop = static_cast<int32_t>(42);
  int32_t step = blockDim.x * gridDim.x;
#else
  auto start = 0;
  auto stop = 42;
  auto step = 1;
#endif
  for (auto i = start; i < stop; i += step) {
    answer[i] = 42 * i;
  }
  return 42;
}

// clang-format off
/*
  UDTF: ct_scalar_1_arg_runtime_sizing__cpu_template(T) -> Column<T> answer, T=[float, double, int32_t, int64_t]
*/
// clang-format on

template <typename T>
TEMPLATE_NOINLINE int32_t
ct_scalar_1_arg_runtime_sizing__cpu_template(const T num, Column<T>& answer) {
  T quotient = num;
  set_output_row_size(30);
  int32_t counter{0};
  while (quotient >= 1) {
    answer[counter++] = quotient;
    quotient /= 10;
  }
  return counter;
}

// clang-format off
/*
  UDTF: ct_scalar_2_args_constant_sizing(int64_t, int64_t, Constant<5>) -> Column<int64_t> answer1, Column<int64_t> answer2
*/
// clang-format on

EXTENSION_NOINLINE int32_t ct_scalar_2_args_constant_sizing(const int64_t num1,
                                                            const int64_t num2,
                                                            Column<int64_t>& answer1,
                                                            Column<int64_t>& answer2) {
#ifdef __CUDACC__
  int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t stop = static_cast<int32_t>(5);
  int32_t step = blockDim.x * gridDim.x;
#else
  auto start = 0;
  auto stop = 5;
  auto step = 1;
#endif
  for (auto i = start; i < stop; i += step) {
    answer1[i] = num1 + i * num2;
    answer2[i] = num1 - i * num2;
  }
  return 5;
}

// clang-format off
/*
  UDTF: ct_no_cursor_user_constant_sizer__cpu_(int32_t, ConstantParameter c) -> Column<int32_t> output
*/
// clang-format on

EXTENSION_NOINLINE int32_t
ct_no_cursor_user_constant_sizer__cpu_(const int32_t input_num,
                                       int32_t c,
                                       Column<int32_t>& output) {
  for (int32_t i = 0; i < c; i++) {
    output[i] = input_num;
  }
  return c;
}

// clang-format off
/*
  UDTF: ct_templated_no_cursor_user_constant_sizer__cpu_template(T, ConstantParameter c) -> Column<T> output, T=[int32_t, float]
*/
// clang-format on

template <typename T>
TEMPLATE_NOINLINE int32_t
ct_templated_no_cursor_user_constant_sizer__cpu_template(const T input_num,
                                                         int32_t c,
                                                         Column<T>& output) {
  for (int32_t i = 0; i < c; i++) {
    output[i] = input_num;
  }
  return c;
}

#ifdef __CUDACC__

// clang-format off
/*
  UDTF: ct_user_constant_sizer__gpu_(int32_t, ConstantParameter c) -> Column<int32_t> output
*/
// clang-format on

EXTENSION_NOINLINE int32_t
ct_no_cursor_user_constant_sizer__gpu_(const int32_t input_num,
                                       int32_t c,
                                       Column<int32_t>& output) {
  int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t step = blockDim.x * gridDim.x;

  for (int32_t i = start; i < c; i += step) {
    output[i] = input_num;
  }
  return c;
}

// clang-format off
/*
  UDTF: ct_templated_no_cursor_user_constant_sizer__gpu_template(T, ConstantParameter c) -> Column<T> output, T=[int32_t, float]
*/
// clang-format on

template <typename T>
TEMPLATE_NOINLINE int32_t
ct_templated_no_cursor_user_constant_sizer__gpu_template(const T input_num,
                                                         int32_t c,
                                                         Column<T>& output) {
  int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t step = blockDim.x * gridDim.x;

  for (int32_t i = start; i < c; i += step) {
    output[i] = input_num;
  }
  return c;
}

#endif  //__CUDACC__

template <typename T>
T safe_addition(T x, T y) {
  if (x >= 0) {
    if (y > (std::numeric_limits<T>::max() - x)) {
      throw std::overflow_error("Addition overflow detected");
    }
  } else {
    if (y < (std::numeric_limits<T>::min() - x)) {
      throw std::underflow_error("Addition underflow detected");
    }
  }
  return x + y;
}

// clang-format off
/*
  UDTF: column_list_safe_row_sum__cpu_template(Cursor<ColumnList<T>>) -> Column<T>, T=[int32_t, int64_t, float, double]
*/
// clang-format on

template <typename T>
int32_t column_list_safe_row_sum__cpu_template(const ColumnList<T>& input,
                                               Column<T>& out) {
  int32_t output_num_rows = input.numCols();
  set_output_row_size(output_num_rows);
  for (int i = 0; i < output_num_rows; i++) {
    auto col = input[i];
    T s = 0;
    for (int j = 0; j < col.size(); j++) {
      try {
        s = safe_addition(s, col[j]);
      } catch (const std::exception& e) {
        return table_function_error(e.what());
      } catch (...) {
        return table_function_error("Unknown error");
      }
    }
    out[i] = s;
  }
  return output_num_rows;
}

// clang-format off
/*
  UDTF: ct_hamming_distance(TextEncodingNone, TextEncodingNone, Constant<1>) -> Column<int32_t> hamming_distance
*/
// clang-format on

EXTENSION_NOINLINE int32_t ct_hamming_distance(const TextEncodingNone& str1,
                                               const TextEncodingNone& str2,
                                               Column<int32_t>& hamming_distance) {
  const int32_t str_len = str1.size() <= str2.size() ? str1.size() : str2.size();

#ifdef __CUDACC__
  const int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  const int32_t step = blockDim.x * gridDim.x;
  if (start == 0) {
    hamming_distance[0] = 0;
  }
  int32_t* output_ptr = hamming_distance.ptr_;
  __syncthreads();
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

// clang-format off
/*
  UDTF: ct_get_string_chars__template(Column<T>, TextEncodingNone, RowMultiplier) -> Column<int32_t> idx, Column<int8_t> char_bytes, T=[int16_t, int32_t]
*/
// clang-format on
template <typename T>
TEMPLATE_NOINLINE int32_t ct_get_string_chars__template(const Column<T>& indices,
                                                        const TextEncodingNone& str,
                                                        const int32_t multiplier,
                                                        Column<int32_t>& idx,
                                                        Column<int8_t>& char_bytes) {
  const int32_t str_len = str.size();
  // Note: we assume RowMultiplier is 1 for this test, was to make running on GPU easy
  // Todo: Provide Constant RowMultiplier interface
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

#ifndef __CUDACC__

#include <iostream>
#include <string>

// clang-format off
/*
  UDTF: ct_string_to_chars__cpu_(TextEncodingNone) -> Column<int32_t> char_idx, Column<int8_t> char_bytes
*/
// clang-format on

EXTENSION_NOINLINE int32_t ct_string_to_chars__cpu_(const TextEncodingNone& input,
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

// clang-format off
/*
  The purpose of ct_sleep1 and ct_sleep2 is to test parallel execution
  of UDTFs (use --num-executors=..). For instance, ct_sleep1 output
  column buffers are managed by a global manager, hence, ct_sleep1 can
  be run only sequentially. However, ct_sleep2 output column buffers
  are managed with a thread-safe manager instance, hence, ct_sleep2
  can be run in parallel.

  UDTF: ct_sleep1(int32_t seconds, int32_t mode) -> Column<int32_t> output
  UDTF: ct_sleep2(TableFunctionManager, int32_t seconds, int32_t mode) -> Column<int32_t> output

  Here mode argument is used to test various approaches of accessing
  the table function manager:

  - mode == 0
    ct_sleep1 uses global set_output_row_size function
    ct_sleep2 uses thread-safe set_output_row_size method

  - mode == 1
    ct_sleep1 retrieves global singleton manager and uses its set_output_row_size method
    ct_sleep2 same as in mode == 1

  - mode == 2
    ct_sleep1 does not call set_output_row_size function, expect error return
    ct_sleep2 does not call set_output_row_size method, expect error return

  - mode == 3
    ct_sleep1 same as mode == 2
    ct_sleep2 calls global set_output_row_size function, expect error return
*/
// clang-format on
EXTENSION_NOINLINE int32_t ct_sleep_worker(int32_t seconds, Column<int32_t>& output) {
  // save entering time
  output[0] = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count() &
              0xffffff;
  // store thread id info
  output[2] = std::hash<std::thread::id>()(std::this_thread::get_id()) & 0xffff;
  // do "computations" for given seconds
  std::this_thread::sleep_for(std::chrono::seconds(seconds));
  // save leaving time
  output[1] = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count() &
              0xffffff;
  return 3;
}

EXTENSION_NOINLINE int32_t ct_sleep1(int32_t seconds,
                                     int32_t mode,
                                     Column<int32_t>& output) {
  switch (mode) {
    case 0: {
      set_output_row_size(3);  // uses global singleton of TableFunctionManager
      break;
    }
    case 1: {
      auto* mgr = TableFunctionManager::get_singleton();
      mgr->set_output_row_size(3);
      break;
    }
    case 2:
    case 3: {
      break;
    }
    default:
      return table_function_error("unexpected mode");
  }
  if (output.size() == 0) {
    return table_function_error("unspecified output columns row size");
  }
  return ct_sleep_worker(seconds, output);
}

EXTENSION_NOINLINE int32_t ct_sleep2(TableFunctionManager& mgr,
                                     int32_t seconds,
                                     int32_t mode,
                                     Column<int32_t>& output) {
  switch (mode) {
    case 0:
    case 1: {
      mgr.set_output_row_size(3);  // uses thread-safe TableFunctionManager instance
      break;
    }
    case 2: {
      break;
    }
    case 3: {
      try {
        auto* mgr0 = TableFunctionManager::get_singleton();  // it may throw "singleton is
                                                             // not initialized"
        mgr0->set_output_row_size(3);
      } catch (std::exception& e) {
        return mgr.error_message(e.what());
      }
      break;
    }
    default:
      return mgr.error_message("unexpected mode");
  }
  if (output.size() == 0) {
    return mgr.error_message("unspecified output columns row size");
  }
  return ct_sleep_worker(seconds, output);
}

// clang-format off
/*
  UDTF: ct_throw_if_gt_100__cpu_template(TableFunctionManager, Column<T>) -> Column<T> val, T=[float, double]
*/
// clang-format on

template <typename T>
TEMPLATE_NOINLINE int32_t ct_throw_if_gt_100__cpu_template(TableFunctionManager& mgr,
                                                           const Column<T>& input,
                                                           Column<T>& output) {
  int64_t num_rows = input.size();
  mgr.set_output_row_size(num_rows);
  for (int64_t r = 0; r < num_rows; ++r) {
    if (input[r] > 100) {
      return mgr.error_message("Values greater than 100 not allowed");
    }
    output[r] = input[r];
  }
  return num_rows;
}

#endif
