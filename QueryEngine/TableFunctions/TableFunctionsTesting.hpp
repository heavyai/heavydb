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

#include "../../QueryEngine/TableFunctions/SystemFunctions/os/Shared/TableFunctionsCommon.hpp"
#include "../../QueryEngine/heavydbTypes.h"

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

// clang-format off
/*
  UDTF: ct_binding_template__cpu_template(Cursor<TextEncodingDict>) -> Column<TextEncodingDict> | input_id=args<0>
  UDTF: ct_binding_template__cpu_template(Cursor<int>) -> Column<int>
  UDTF: ct_binding_template__cpu_template(Cursor<float>) -> Column<float>
*/
// clang-format on

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

#endif  // #ifndef __CUDACC__

// clang-format off
/*
  UDTF: ct_binding_columnlist__cpu_template(Cursor<int32_t, ColumnList<int32_t>>) -> Column<int32_t>
  UDTF: ct_binding_columnlist__cpu_template(Cursor<float, ColumnList<float>>) -> Column<int32_t>
  UDTF: ct_binding_columnlist__cpu_template(Cursor<TextEncodingDict, ColumnList<TextEncodingDict>>) -> Column<int32_t>
  UDTF: ct_binding_columnlist__cpu_template(Cursor<int16_t, ColumnList<int16_t>>) -> Column<int32_t>
*/
// clang-format on

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

#endif  // #ifndef __CUDACC__

// clang-format off
/*
  UDTF: ct_binding_column__cpu_template(Column<int32_t>) -> Column<int32_t>
  UDTF: ct_binding_column__cpu_template(Column<float>) -> Column<int32_t>
*/
// clang-format on

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

#endif  // #ifndef __CUDACC__

// clang-format off
/*
  UDTF: ct_binding_scalar_multiply__cpu_template(Cursor<Column<float>>, float) -> Column<float>
  UDTF: ct_binding_scalar_multiply__cpu_template(Cursor<Column<double>>, double) -> Column<double>
  UDTF: ct_binding_scalar_multiply__cpu_template(Cursor<Column<int32_t>>, int32_t) -> Column<int32_t>
  UDTF: ct_binding_scalar_multiply__cpu_template(Cursor<Column<int64_t>>, int64_t) -> Column<int64_t>
*/
// clang-format on

#ifndef __CUDACC__

template <typename T>
NEVER_INLINE HOST int32_t ct_binding_scalar_multiply__cpu_template(const Column<T>& input,
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

// clang-format off
/*
  UDTF: ct_binding_str_length__cpu_(Cursor<TextEncodingDict>) -> Column<TextEncodingDict> string | input_id=args<0>, Column<int64_t> string_length
*/
// clang-format on
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

// clang-format off
/*
  UDTF: ct_binding_str_equals__cpu_(Cursor<ColumnList<TextEncodingDict>>) -> Column<TextEncodingDict> string_if_equal | input_id=args<0, 0>, Column<bool> strings_are_equal
*/
// clang-format on
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

// clang-format off
/*
  UDTF: ct_substr__cpu_(TableFunctionManager, Cursor<Column<TextEncodingDict> str, Column<int> pos, Column<int> len>) -> Column<TextEncodingDict> substr | input_id=args<0>
*/
// clang-format on

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

// clang-format off
/*
  UDTF: ct_string_concat__cpu_(TableFunctionManager, Cursor<ColumnList<TextEncodingDict>>, TextEncodingNone separator) -> Column<TextEncodingDict> concatted_str | input_id=args<0, 0>
*/
// clang-format on
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

// clang-format off
/*
  UDTF: ct_synthesize_new_dict__cpu_(TableFunctionManager, int32_t num_strings) -> Column<TextEncodingDict> new_dict_col | input_id=args<0>
*/
// clang-format on
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
NEVER_INLINE HOST int32_t sort_column_limit__cpu_template(const Column<T>& input,
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

#ifndef __CUDACC__

template <typename T, typename U, typename K>
NEVER_INLINE HOST int32_t ct_binding_column2__cpu_template(const Column<T>& input1,
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

#endif  // #ifndef __CUDACC__

// clang-format off
/*
  UDTF: ct_named_output__cpu_template(Column<T> input) -> Column<T> total, T=[int32_t, double]
  UDTF: ct_named_const_output__cpu_template(Column<T> input, Constant<2>) -> Column<T> total, T=[int32_t, double]
  UDTF: ct_named_user_const_output__cpu_template(Column<T> input, ConstantParameter c) -> Column<T> total, T=[int32_t, double]
  UDTF: ct_named_rowmul_output__cpu_template(Column<T> input, RowMultiplier m) -> Column<T> total, T=[int32_t, double]
*/
// clang-format on

#ifndef __CUDACC__

template <typename T>
NEVER_INLINE HOST int32_t ct_named_output__cpu_template(const Column<T>& input,
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
NEVER_INLINE HOST int32_t ct_named_const_output__cpu_template(const Column<T>& input,
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
NEVER_INLINE HOST int32_t ct_named_user_const_output__cpu_template(const Column<T>& input,
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
NEVER_INLINE HOST int32_t ct_named_rowmul_output__cpu_template(const Column<T>& input,
                                                               int32_t m,
                                                               Column<T>& out) {
  for (int64_t j = 0; j < m; j++) {
    for (int64_t i = 0; i < input.size(); i++) {
      out[j * input.size() + i] += input[i];
    }
  }
  return m * input.size();
}

#endif  // #ifndef __CUDACC__

// clang-format off
/*
  UDTF: ct_no_arg_runtime_sizing__cpu_template() -> Column<T> answer, T=[int32_t]
*/
// clang-format on

#ifndef __CUDACC__

template <typename T>
NEVER_INLINE HOST int32_t ct_no_arg_runtime_sizing__cpu_template(Column<T>& answer) {
  set_output_row_size(1);
  answer[0] = 42;
  return 1;
}

#endif  // #ifndef __CUDACC__

// clang-format off
/*
  UDTF: ct_no_arg_constant_sizing(Constant<42>) -> Column<int32_t> answer 
*/
// clang-format on

EXTENSION_NOINLINE int32_t ct_no_arg_constant_sizing(Column<int32_t>& answer) {
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
  UDTF: ct_scalar_1_arg_runtime_sizing__cpu_template(T num) -> Column<T> answer, T=[float, double, int32_t, int64_t]
*/
// clang-format on

#ifndef __CUDACC__

template <typename T>
NEVER_INLINE HOST int32_t
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

#endif  // #ifndef __CUDACC__

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

#ifndef __CUDACC__

EXTENSION_NOINLINE_HOST int32_t
ct_no_cursor_user_constant_sizer__cpu_(const int32_t input_num,
                                       int32_t c,
                                       Column<int32_t>& output) {
  for (int32_t i = 0; i < c; i++) {
    output[i] = input_num;
  }
  return c;
}

#endif  // #ifndef __CUDACC__

// clang-format off
/*
  UDTF: ct_templated_no_cursor_user_constant_sizer__cpu_template(T, ConstantParameter c) -> Column<T> output, T=[int32_t, float]
*/
// clang-format on

#ifndef __CUDACC__

template <typename T>
NEVER_INLINE HOST int32_t
ct_templated_no_cursor_user_constant_sizer__cpu_template(const T input_num,
                                                         int32_t c,
                                                         Column<T>& output) {
  for (int32_t i = 0; i < c; i++) {
    output[i] = input_num;
  }
  return c;
}

#endif  // #ifndef __CUDACC__

#ifdef __CUDACC__

// clang-format off
/*
  UDTF: ct_no_cursor_user_constant_sizer__gpu_(int32_t, ConstantParameter c) -> Column<int32_t> output
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

#ifndef __CUDACC__

template <typename T>
NEVER_INLINE HOST int32_t
column_list_safe_row_sum__cpu_template(const ColumnList<T>& input, Column<T>& out) {
  int32_t output_num_rows = input.numCols();
  set_output_row_size(output_num_rows);
  for (int i = 0; i < output_num_rows; i++) {
    auto col = input[i];
    T s = 0;
    for (int j = 0; j < col.size(); j++) {
      try {
        s = safe_addition(s, col[j]);
      } catch (const std::exception& e) {
        return TABLE_FUNCTION_ERROR(e.what());
      } catch (...) {
        return TABLE_FUNCTION_ERROR("Unknown error");
      }
    }
    out[i] = s;
  }
  return output_num_rows;
}

#endif  // #ifndef __CUDACC__

// clang-format off
/*
  UDTF: ct_gpu_default_init__gpu_(Constant<1>) -> Column<int32_t> output_buffer
*/
// clang-format on

EXTENSION_NOINLINE int32_t ct_gpu_default_init__gpu_(Column<int32_t>& output_buffer) {
  // output_buffer[0] should always be 0 due to default initialization for GPU
  return 1;
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

#ifndef __CUDACC__

#include <iostream>
#include <string>

// clang-format off
/*
  UDTF: ct_string_to_chars__cpu_(TextEncodingNone) -> Column<int32_t> char_idx, Column<int8_t> char_bytes
*/
// clang-format on

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

// clang-format off
/*
  The purpose of ct_sleep1 and ct_sleep2 is to test parallel execution
  of UDTFs (use --num-executors=..). For instance, ct_sleep1 output
  column buffers are managed by a global manager, hence, ct_sleep1 can
  be run only sequentially. However, ct_sleep2 output column buffers
  are managed with a thread-safe manager instance, hence, ct_sleep2
  can be run in parallel.

  UDTF: ct_sleep1__cpu_(int32_t seconds, int32_t mode) -> Column<int32_t> output
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

EXTENSION_NOINLINE_HOST int32_t ct_sleep1__cpu_(int32_t seconds,
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
      return TABLE_FUNCTION_ERROR("unexpected mode");
  }
  if (output.size() == 0) {
    return TABLE_FUNCTION_ERROR("unspecified output columns row size");
  }
  return ct_sleep_worker(seconds, output);
}

EXTENSION_NOINLINE_HOST int32_t ct_sleep2(TableFunctionManager& mgr,
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
        return mgr.ERROR_MESSAGE(e.what());
      }
      break;
    }
    default:
      return mgr.ERROR_MESSAGE("unexpected mode");
  }
  if (output.size() == 0) {
    return mgr.ERROR_MESSAGE("unspecified output columns row size");
  }
  return ct_sleep_worker(seconds, output);
}

// clang-format off
/*
  UDTF: ct_throw_if_gt_100__cpu_template(TableFunctionManager, Column<T>) -> Column<T> val, T=[float, double]
*/
// clang-format on

template <typename T>
NEVER_INLINE HOST int32_t ct_throw_if_gt_100__cpu_template(TableFunctionManager& mgr,
                                                           const Column<T>& input,
                                                           Column<T>& output) {
  int64_t num_rows = input.size();
  mgr.set_output_row_size(num_rows);
  for (int64_t r = 0; r < num_rows; ++r) {
    if (input[r] > 100) {
      return mgr.ERROR_MESSAGE("Values greater than 100 not allowed");
    }
    output[r] = input[r];
  }
  return num_rows;
}

// clang-format off
/*
  The following UDTFs are used to test an optimization rule that moves
  filters on UDTF outputs to the inputs when the names of outputs and
  input arguments match in the UDTF signatures. This optimization
  makes sense only if filters and table functions are commutative with
  respect to the corresponding input and output arguments:

    filter(udtf(..., input[j], ...)[i]) == udtf(..., filter(input[j]), ...)[i]

  The UDTFs below invalidate this requirement for the purpose of
  testing the feature: the result will depend on whether the
  optimization is enabled or not.

  UDTF: ct_copy_and_add_size(TableFunctionManager, Cursor<Column<int32_t> x>) | filter_table_function_transpose=on -> Column<int32_t> x
  UDTF: ct_add_size_and_mul_alpha(TableFunctionManager, Cursor<Column<int32_t>, Column<int32_t>> | fields=[x, x2], int32_t alpha) | filter_table_function_transpose=on -> Column<int32_t> x, Column<int32_t> x2

  UDTF: ct_sparse_add(TableFunctionManager, Cursor<Column<int32_t> x, Column<int32_t> d1>, int32_t f1, Cursor<Column<int32_t> x, Column<int32_t> d2>, int32_t f2) | filter_table_function_transpose=on -> Column<int32_t> x, Column<int32_t> d
*/
// clang-format on

EXTENSION_NOINLINE_HOST int32_t ct_copy_and_add_size(TableFunctionManager& mgr,
                                                     const Column<int32_t>& input,
                                                     Column<int32_t>& output) {
  mgr.set_output_row_size(input.size());
  for (int32_t i = 0; i < input.size(); i++) {
    output[i] = input[i] + input.size();
  }
  return output.size();
}

EXTENSION_NOINLINE_HOST int32_t ct_add_size_and_mul_alpha(TableFunctionManager& mgr,
                                                          const Column<int32_t>& input1,
                                                          const Column<int32_t>& input2,
                                                          int32_t alpha,
                                                          Column<int32_t>& output1,
                                                          Column<int32_t>& output2) {
  auto size = input1.size();
  mgr.set_output_row_size(size);
  for (int32_t i = 0; i < size; i++) {
    output1[i] = input1[i] + size;
    output2[i] = input2[i] * alpha;
  }
  return size;
}

/*
  Add two sparse graphs given by pairs of coordinates and the
  corresponding values and multiply with the size of output
  columns. Unspecified points are assumed to have the specified fill
  value.
*/

EXTENSION_NOINLINE_HOST int32_t ct_sparse_add(TableFunctionManager& mgr,
                                              const Column<int32_t>& x1,
                                              const Column<int32_t>& d1,
                                              int32_t f1,
                                              const Column<int32_t>& x2,
                                              const Column<int32_t>& d2,
                                              int32_t f2,
                                              Column<int32_t>& x,
                                              Column<int32_t>& d) {
  // sorted set of common coordinates:
  std::set<int32_t, std::less<int32_t>> x12;
  // inverse map of coordinates and indices, keys are sorted:
  std::map<int32_t, int32_t, std::less<int32_t>> i1, i2;

  for (int32_t i = 0; i < x1.size(); i++) {
    i1[x1[i]] = i;
    x12.insert(x1[i]);
  }
  for (int32_t i = 0; i < x2.size(); i++) {
    i2[x2[i]] = i;
    x12.insert(x2[i]);
  }
  auto size = x12.size();

  mgr.set_output_row_size(size);
  int32_t k = 0;
  for (auto x_ : x12) {
    x[k] = x_;
    auto i1_ = i1.find(x_);
    auto i2_ = i2.find(x_);
    if (i1_ != i1.end()) {
      if (i2_ != i2.end()) {
        d[k] = d1[i1_->second] + d2[i2_->second];
      } else {
        d[k] = d1[i1_->second] + f2;
      }
    } else if (i2_ != i2.end()) {
      d[k] = f1 + d2[i2_->second];
    } else {
      d[k] = f1 + f2;
    }
    d[k] *= size;
    k++;
  }
  return size;
}

#endif  // #ifndef __CUDACC__

enum TFAggType { MIN, MAX };

template <typename T>
TEMPLATE_INLINE T get_min_or_max(const Column<T>& col, const TFAggType min_or_max) {
  const auto input_min_max = get_column_min_max(col);
  if (min_or_max == TFAggType::MIN) {
    return input_min_max.first;
  }
  return input_min_max.second;
}

template <typename T>
TEMPLATE_INLINE T get_min_or_max_union(const Column<T>& col1,
                                       const Column<T>& col2,
                                       const TFAggType min_or_max) {
  const auto input1_min_max = get_column_min_max(col1);
  const auto input2_min_max = get_column_min_max(col2);
  if (min_or_max == TFAggType::MIN) {
    return input1_min_max.first < input2_min_max.first ? input1_min_max.first
                                                       : input2_min_max.first;
  }
  return input1_min_max.second > input2_min_max.second ? input1_min_max.second
                                                       : input2_min_max.second;
}

// clang-format off
/*
  UDTF: ct_pushdown_stats__cpu_template(TableFunctionManager, TextEncodingNone agg_type, Cursor<Column<K> id, Column<T> x, Column<T> y, Column<Z> z>) | filter_table_function_transpose=on -> Column<int32_t> row_count, Column<K> id | input_id=args<0>, Column<T> x, Column<T> y, Column<Z> z, K=[int32_t, int64_t, TextEncodingDict], T=[int32_t, int64_t, float, double], Z=[int32_t, int64_t, float, double]
*/
// clang-format on

#ifndef __CUDACC__

template <typename K, typename T, typename Z>
NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<K>& input_id,
                                const Column<T>& input_x,
                                const Column<T>& input_y,
                                const Column<Z>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<K>& output_id,
                                Column<T>& output_x,
                                Column<T>& output_y,
                                Column<Z>& output_z) {
  const std::string agg_type_str = agg_type.getString();
  const TFAggType min_or_max = agg_type_str == "MIN" ? TFAggType::MIN : TFAggType::MAX;
  mgr.set_output_row_size(1);
  output_row_count[0] = input_id.size();
  output_id[0] = get_min_or_max(input_id, min_or_max);
  output_x[0] = get_min_or_max(input_x, min_or_max);
  output_y[0] = get_min_or_max(input_y, min_or_max);
  output_z[0] = get_min_or_max(input_z, min_or_max);
  return 1;
}

#endif  // #ifndef __CUDACC__

// clang-format off
/*
  UDTF: ct_pushdown_projection__cpu_template(TableFunctionManager, Cursor<Column<K> id, Column<T> x, Column<T> y, Column<Z> z>) | filter_table_function_transpose=on -> Column<K> id | input_id=args<0>, Column<T> x, Column<T> y, Column<Z> z, K=[int32_t, int64_t, TextEncodingDict], T=[int32_t, int64_t, float, double], Z=[int32_t, int64_t, float, double]
*/
// clang-format on

#ifndef __CUDACC__

template <typename K, typename T, typename Z>
NEVER_INLINE HOST int32_t ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                                               const Column<K>& input_id,
                                                               const Column<T>& input_x,
                                                               const Column<T>& input_y,
                                                               const Column<Z>& input_z,
                                                               Column<K>& output_id,
                                                               Column<T>& output_x,
                                                               Column<T>& output_y,
                                                               Column<Z>& output_z) {
  const int64_t input_size = input_id.size();
  mgr.set_output_row_size(input_size);
  for (int64_t input_idx = 0; input_idx < input_size; ++input_idx) {
    output_id[input_idx] = input_id[input_idx];
    output_x[input_idx] = input_x[input_idx];
    output_y[input_idx] = input_y[input_idx];
    output_z[input_idx] = input_z[input_idx];
  }
  return input_size;
}

#endif  // #ifndef __CUDACC__

// clang-format off
/*
  UDTF: ct_union_pushdown_stats__cpu_template(TableFunctionManager, TextEncodingNone agg_type, Cursor<Column<K> id, Column<T> x, Column<T> y, Column<Z> z>, Cursor<Column<K> id, Column<T> x, Column<T> y, Column<Z> z, Column<T> w>) | filter_table_function_transpose=on -> Column<int32_t> row_count, Column<K> id | input_id=args<0, 0>, Column<T> x, Column<T> y, Column<Z> z, Column<T> w, K=[int32_t, int64_t, TextEncodingDict], T=[int32_t, int64_t, float, double], Z=[int32_t, int64_t, float, double]
*/
// clang-format on

#ifndef __CUDACC__

template <typename K, typename T, typename Z>
NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<K>& input1_id,
                                      const Column<T>& input1_x,
                                      const Column<T>& input1_y,
                                      const Column<Z>& input1_z,
                                      const Column<K>& input2_id,
                                      const Column<T>& input2_x,
                                      const Column<T>& input2_y,
                                      const Column<Z>& input2_z,
                                      const Column<T>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<K>& output_id,
                                      Column<T>& output_x,
                                      Column<T>& output_y,
                                      Column<Z>& output_z,
                                      Column<T>& output_w) {
  mgr.set_output_row_size(1);
  const std::string agg_type_str = agg_type.getString();
  const TFAggType min_or_max = agg_type_str == "MIN" ? TFAggType::MIN : TFAggType::MAX;
  output_row_count[0] = input1_id.size() + input2_id.size();
  output_id[0] = get_min_or_max_union(input1_id, input2_id, min_or_max);
  output_x[0] = get_min_or_max_union(input1_x, input2_x, min_or_max);
  output_y[0] = get_min_or_max_union(input1_y, input2_y, min_or_max);
  output_z[0] = get_min_or_max_union(input1_z, input2_z, min_or_max);
  if (input2_w.size() > 0) {
    const auto w_min_max = get_column_min_max(input2_w);
    output_w[0] = agg_type_str == "MIN" ? w_min_max.first : w_min_max.second;
  } else {
    output_w.setNull(0);
  }
  return 1;
}

#endif  // #ifndef __CUDACC__

// clang-format off
/*
  UDTF: ct_union_pushdown_projection__cpu_template(TableFunctionManager, Cursor<Column<K> id, Column<T> x, Column<T> y, Column<Z> z>, Cursor<Column<K> id, Column<T> x, Column<T> y, Column<Z> z, Column<T> w>) | filter_table_function_transpose=on -> Column<K> id | input_id=args<0, 0>, Column<T> x, Column<T> y, Column<Z> z, Column<T> w, K=[int32_t, int64_t, TextEncodingDict], T=[int32_t, int64_t, float, double], Z=[int32_t, int64_t, float, double]
*/
// clang-format on

#ifndef __CUDACC__

template <typename K, typename T, typename Z>
NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<K>& input1_id,
                                           const Column<T>& input1_x,
                                           const Column<T>& input1_y,
                                           const Column<Z>& input1_z,
                                           const Column<K>& input2_id,
                                           const Column<T>& input2_x,
                                           const Column<T>& input2_y,
                                           const Column<Z>& input2_z,
                                           const Column<T>& input2_w,
                                           Column<K>& output_id,
                                           Column<T>& output_x,
                                           Column<T>& output_y,
                                           Column<Z>& output_z,
                                           Column<T>& output_w) {
  const int64_t input1_size = input1_id.size();
  const int64_t input2_size = input2_id.size();
  int64_t output_size = input1_size + input2_size;
  mgr.set_output_row_size(output_size);
  for (int64_t input1_idx = 0; input1_idx < input1_size; ++input1_idx) {
    output_id[input1_idx] = input1_id[input1_idx];
    output_x[input1_idx] = input1_x[input1_idx];
    output_y[input1_idx] = input1_y[input1_idx];
    output_z[input1_idx] = input1_z[input1_idx];
    output_w.setNull(input1_idx);
  }
  for (int64_t input2_idx = 0; input2_idx < input2_size; ++input2_idx) {
    output_id[input1_size + input2_idx] = input2_id[input2_idx];
    output_x[input1_size + input2_idx] = input2_x[input2_idx];
    output_y[input1_size + input2_idx] = input2_y[input2_idx];
    output_z[input1_size + input2_idx] = input2_z[input2_idx];
    output_w[input1_size + input2_idx] = input2_w[input2_idx];
  }
  return output_size;
}

#endif  // #ifndef __CUDACC__

// clang-format off
/*
  UDTF: ct_require__cpu_(Column<int32_t>, int | name=i | require="i > 0") -> Column<int32_t>
  UDTF: ct_require_mgr(TableFunctionManager, Column<int32_t>, int i | require="i > 1" | require="i < 5") -> Column<int32_t>
  UDTF: ct_require_str__cpu_(Column<int32_t>, TextEncodingNone s | require="s == \"hello\"") -> Column<int32_t>
  UDTF: ct_require_templating__cpu_template(Column<T>, int i | require="i > 0") -> Column<K>, T=[int, double], K=[int]
  UDTF: ct_require_and__cpu_(Column<int>, int i | require="i > 0 && i < 5") -> Column<int>
  UDTF: ct_require_or_str__cpu_(Column<int>, TextEncodingNone i | require="i == \"MAX\" || i == \"MIN\"") -> Column<int>
  UDTF: ct_require_str_diff__cpu_(Column<int>, TextEncodingNone i | require="i != \"MAX\"") -> Column<int>
  UDTF: ct_require_text_enc_dict__cpu_(Cursor<Column<TextEncodingDict>>, int64_t x | require="x >= 1") -> Column<int64_t>
*/
// clang-format on

#ifndef __CUDACC__

EXTENSION_NOINLINE_HOST int32_t ct_require__cpu_(const Column<int32_t>& input1,
                                                 const int32_t i,
                                                 Column<int32_t>& out) {
  set_output_row_size(1);
  out[0] = 3;
  return 1;
}

EXTENSION_NOINLINE_HOST int32_t ct_require_str__cpu_(const Column<int32_t>& input1,
                                                     const TextEncodingNone& s,
                                                     Column<int32_t>& out) {
  set_output_row_size(1);
  out[0] = 3;
  return 1;
}

EXTENSION_NOINLINE_HOST int32_t ct_require_mgr(TableFunctionManager& mgr,
                                               const Column<int32_t>& input1,
                                               const int32_t i,
                                               Column<int32_t>& out) {
  set_output_row_size(1);
  out[0] = 4;
  return 1;
}

template <typename T, typename K>
NEVER_INLINE HOST int32_t ct_require_templating__cpu_template(const Column<T>& input1,
                                                              const int32_t i,
                                                              Column<K>& out) {
  set_output_row_size(1);
  if constexpr (std::is_same<T, int32_t>::value) {
    out[0] = 5;
  } else if constexpr (std::is_same<T, double>::value) {
    out[0] = 6.0;
  }
  return 1;
}

EXTENSION_NOINLINE_HOST int32_t ct_require_and__cpu_(const Column<int32_t>& input1,
                                                     const int32_t i,
                                                     Column<int32_t>& out) {
  set_output_row_size(1);
  out[0] = 7;
  return 1;
}

EXTENSION_NOINLINE_HOST int32_t ct_require_or_str__cpu_(const Column<int32_t>& input1,
                                                        const TextEncodingNone& i,
                                                        Column<int32_t>& out) {
  set_output_row_size(1);
  out[0] = 8;
  return 1;
}

EXTENSION_NOINLINE_HOST int32_t ct_require_str_diff__cpu_(const Column<int32_t>& input1,
                                                          const TextEncodingNone& i,
                                                          Column<int32_t>& out) {
  set_output_row_size(1);
  out[0] = 9;
  return 1;
}

EXTENSION_NOINLINE_HOST int32_t
ct_require_text_enc_dict__cpu_(const Column<TextEncodingDict>& input,
                               const int64_t x,
                               Column<int64_t>& out) {
  set_output_row_size(1);
  out[0] = 10;
  return 1;
}

#endif  // #ifndef __CUDACC__

#ifdef __CUDACC__

// clang-format off
/*
  UDTF: ct_require_device_cuda__gpu_(Column<int32_t>, Constant<1>, int | name=i | require="i > 0") -> Column<int32_t>
*/
// clang-format on

EXTENSION_NOINLINE int32_t ct_require_device_cuda__gpu_(const Column<int32_t>& input1,
                                                        const int32_t i,
                                                        Column<int32_t>& out) {
  out[0] = (i > 0 ? 12345 : 54321);
  return 1;
}

// clang-format off
/*
  UDTF: ct_cuda_enumerate_threads__gpu_(ConstantParameter output_size) -> Column<int32_t> local_thread_id, Column<int32_t> block_id, Column<int32_t> global_thread_id
*/
// clang-format on

EXTENSION_NOINLINE int32_t
ct_cuda_enumerate_threads__gpu_(const int32_t output_size,
                                Column<int32_t>& out_local_thread_id,
                                Column<int32_t>& out_block_id,
                                Column<int32_t>& out_global_thread_id) {
  int32_t local_thread_id = threadIdx.x;
  int32_t block_id = blockIdx.x;
  int32_t global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  out_local_thread_id[global_thread_id] = local_thread_id;
  out_block_id[global_thread_id] = block_id;
  out_global_thread_id[global_thread_id] = global_thread_id;
  return output_size;
}

#endif  //__CUDACC__

// clang-format off
/*
  UDTF: ct_test_nullable(Column<int32_t>, RowMultiplier) -> Column<int32_t>
*/
// clang-format on

EXTENSION_NOINLINE int32_t ct_test_nullable(const Column<int32_t>& input,
                                            const int32_t i,
                                            Column<int32_t>& out) {
  for (int i = 0; i < input.size(); i++) {
    if (i % 2 == 0) {
      out.setNull(i);
    } else {
      out[i] = input[i];
    }
  }
  return input.size();
}

// clang-format off
/*
  UDTF: ct_test_preflight_sizer(Column<int32_t> col, int i, int j) -> Column<int32_t> | output_row_size="i + j"
  UDTF: ct_test_preflight_sizer_const(Column<int32_t> col) -> Column<int32_t> | output_row_size=2
*/
// clang-format on

EXTENSION_NOINLINE int32_t ct_test_preflight_sizer(const Column<int32_t>& input,
                                                   const int32_t i,
                                                   const int32_t j,
                                                   Column<int32_t>& out) {
  out[0] = 123;
  out[1] = 456;
  return out.size();
}

EXTENSION_NOINLINE int32_t ct_test_preflight_sizer_const(const Column<int32_t>& input,
                                                         Column<int32_t>& out) {
  out[0] = 789;
  out[1] = 321;
  return out.size();
}

#ifndef __CUDACC__

// clang-format off
/*
  UDTF: ct_test_preflight_singlecursor_qe227__cpu_(TableFunctionManager,
          Cursor<Column<int32_t> col, ColumnList<int32_t> lst>,
          int32_t x, int32_t y | require="x > 0" | require="y > 0") ->
    Column<int32_t> out
 */
// clang-format on

EXTENSION_NOINLINE int32_t
ct_test_preflight_singlecursor_qe227__cpu_(TableFunctionManager& mgr,
                                           const Column<int32_t>& col,
                                           const ColumnList<int32_t>& lst,
                                           const int x,
                                           const int y,
                                           Column<int32_t>& out) {
  mgr.set_output_row_size(lst.numCols() + 1);
  out[0] = col[0];
  for (int i = 0; i < lst.numCols(); i++) {
    out[i + 1] = lst[i][0];
  }
  return out.size();
}

// clang-format off
/*
  UDTF: ct_test_preflight_multicursor_qe227__cpu_(TableFunctionManager,
          Column<int32_t> col, ColumnList<int32_t> lst,
          int32_t x, int32_t y | require="x > 0" | require="y > 0") ->
    Column<int32_t> out
 */
// clang-format on

EXTENSION_NOINLINE int32_t
ct_test_preflight_multicursor_qe227__cpu_(TableFunctionManager& mgr,
                                          const Column<int32_t>& col,
                                          const ColumnList<int32_t>& lst,
                                          const int x,
                                          const int y,
                                          Column<int32_t>& out) {
  mgr.set_output_row_size(lst.numCols() + 1);
  out[0] = col[1];
  for (int i = 0; i < lst.numCols(); i++) {
    out[i + 1] = lst[i][1];
  }
  return out.size();
}

// clang-format off
/*
  UDTF: ct_scalar_named_args__cpu_(TableFunctionManager, int32_t arg1, int32_t arg2) ->
  Column<int32_t> out1, Column<int32_t> out2
*/

EXTENSION_NOINLINE_HOST int32_t
ct_scalar_named_args__cpu_(TableFunctionManager& mgr, const int32_t arg1,
 const int32_t arg2, Column<int32_t>& out1, Column<int32_t>& out2) {
  mgr.set_output_row_size(1);
  out1[0] = arg1;
  out2[0] = arg2;
  return 1;
 }

// clang-format off
/*
  UDTF: ct_cursor_named_args__cpu_(TableFunctionManager, Cursor<Column<int32_t> input_table_arg1, 
  Column<int32_t> input_table_arg2> input_table, int32_t arg1, int32_t arg2) ->
  Column<int32_t> out1, Column<int32_t> out2
*/

EXTENSION_NOINLINE_HOST int32_t
ct_cursor_named_args__cpu_(TableFunctionManager& mgr, const Column<int32_t>& input_arg1,
const Column<int32_t>& input_arg2, const int32_t arg1, const int32_t arg2,
 Column<int32_t>& out1, Column<int32_t>& out2) {
  const int32_t num_rows = input_arg1.size();
  mgr.set_output_row_size(num_rows);
  for (int32_t r = 0; r < num_rows; ++r) {
    out1[r] = input_arg1[r] + arg1;
    out2[r] = input_arg2[r] + arg2;
  }
  return num_rows;
}

#endif

#ifndef __CUDACC__

// clang-format off
/*
  UDTF: ct_timestamp_extract(TableFunctionManager, Column<Timestamp>) -> Column<int64_t> ns, Column<int64_t> us, Column<int64_t> ms, Column<int64_t> s, Column<int64_t> m, Column<int64_t> h, Column<int64_t> d, Column<int64_t> mo, Column<int64_t> y
  UDTF: ct_timestamp_add_offset(TableFunctionManager, Column<Timestamp>, Timestamp) -> Column<Timestamp>
  UDTF: ct_timestamp_test_columns_and_scalars__cpu(Column<Timestamp>, int64_t, RowMultiplier, Column<Timestamp>) -> Column<Timestamp>
  UDTF: ct_timestamp_column_list_input(TableFunctionManager, ColumnList<int64_t>, Column<Timestamp>) -> Column<Timestamp>
  UDTF: ct_timestamp_truncate(TableFunctionManager, Column<Timestamp>) -> Column<Timestamp> y, Column<Timestamp> mo, Column<Timestamp> d, Column<Timestamp> h, Column<Timestamp> m, Column<Timestamp> s, Column<Timestamp> ms, Column<Timestamp> us
  UDTF: ct_timestamp_add_interval__template(TableFunctionManager, Column<Timestamp>, T) -> Column<Timestamp>, T=[YearMonthTimeInterval, DayTimeInterval]
*/
// clang-format on

// Test table functions with Timestamp Column inputs
// and Timestamp type helper functions
EXTENSION_NOINLINE_HOST int32_t ct_timestamp_extract(TableFunctionManager& mgr,
                                                     const Column<Timestamp>& input,
                                                     Column<int64_t>& ns,
                                                     Column<int64_t>& us,
                                                     Column<int64_t>& ms,
                                                     Column<int64_t>& s,
                                                     Column<int64_t>& m,
                                                     Column<int64_t>& h,
                                                     Column<int64_t>& d,
                                                     Column<int64_t>& mo,
                                                     Column<int64_t>& y) {
  int size = input.size();
  mgr.set_output_row_size(size);
  for (int i = 0; i < size; ++i) {
    if (input.isNull(i)) {
      ns.setNull(i);
      us.setNull(i);
      ms.setNull(i);
      s.setNull(i);
      m.setNull(i);
      h.setNull(i);
      d.setNull(i);
      mo.setNull(i);
      y.setNull(i);
    } else {
      ns[i] = input[i].getNanoseconds();
      us[i] = input[i].getMicroseconds();
      ms[i] = input[i].getMilliseconds();
      s[i] = input[i].getSeconds();
      m[i] = input[i].getMinutes();
      h[i] = input[i].getHours();
      d[i] = input[i].getDay();
      mo[i] = input[i].getMonth();
      y[i] = input[i].getYear();
    }
  }
  return size;
}

// Test table functions with scalar Timestamp inputs
EXTENSION_NOINLINE_HOST int32_t ct_timestamp_add_offset(TableFunctionManager& mgr,
                                                        const Column<Timestamp>& input,
                                                        const Timestamp offset,
                                                        Column<Timestamp>& out) {
  int size = input.size();
  mgr.set_output_row_size(size);
  for (int i = 0; i < size; ++i) {
    if (input.isNull(i)) {
      out.setNull(i);
    } else {
      out[i] = input[i] + offset;
    }
  }
  return size;
}

// Test table function with sizer argument, and mix of scalar/column inputs.
EXTENSION_NOINLINE int32_t
ct_timestamp_test_columns_and_scalars__cpu(const Column<Timestamp>& input,
                                           const int64_t dummy,
                                           const int32_t multiplier,
                                           const Column<Timestamp>& input2,
                                           Column<Timestamp>& out) {
  int size = input.size();
  for (int i = 0; i < size; ++i) {
    if (input.isNull(i)) {
      out.setNull(i);
    } else {
      out[i] = input[i] + input2[i] + Timestamp(dummy);
    }
  }
  return size;
}

// Dummy test for ColumnList inputs + Column Timestamp input
EXTENSION_NOINLINE_HOST int32_t
ct_timestamp_column_list_input(TableFunctionManager& mgr,
                               const ColumnList<int64_t>& input,
                               const Column<Timestamp>& input2,
                               Column<int64_t>& out) {
  mgr.set_output_row_size(1);
  out[0] = 1;
  return 1;
}

EXTENSION_NOINLINE_HOST int32_t ct_timestamp_truncate(TableFunctionManager& mgr,
                                                      const Column<Timestamp>& input,
                                                      Column<Timestamp>& y,
                                                      Column<Timestamp>& mo,
                                                      Column<Timestamp>& d,
                                                      Column<Timestamp>& h,
                                                      Column<Timestamp>& m,
                                                      Column<Timestamp>& s,
                                                      Column<Timestamp>& ms,
                                                      Column<Timestamp>& us) {
  int size = input.size();
  mgr.set_output_row_size(size);
  for (int i = 0; i < size; ++i) {
    y[i] = input[i].truncateToYear();
    mo[i] = input[i].truncateToMonth();
    d[i] = input[i].truncateToDay();
    h[i] = input[i].truncateToHours();
    m[i] = input[i].truncateToMinutes();
    s[i] = input[i].truncateToSeconds();
    ms[i] = input[i].truncateToMilliseconds();
    us[i] = input[i].truncateToMicroseconds();
  }

  return size;
}

template <typename T>
NEVER_INLINE HOST int32_t
ct_timestamp_add_interval__template(TableFunctionManager& mgr,
                                    const Column<Timestamp>& input,
                                    const T inter,
                                    Column<Timestamp>& out) {
  int size = input.size();
  mgr.set_output_row_size(size);
  for (int i = 0; i < size; ++i) {
    out[i] = inter + input[i];
  }
  return size;
}

// clang-format off
/*
  UDTF: sum_along_row__cpu_template(Column<Array<T>> input) -> Column<T>,
          T=[float, double, int8_t, int16_t, int32_t, int64_t, bool, TextEncodingDict] | output_row_size="input.size()" | input_id=args<0>
*/
// clang-format on
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
        int32_t input_dict_id = input.getDictId();
        int32_t output_dict_id = output.getDictId();
        std::string acc = "";
        for (auto j = 0; j < arr.getSize(); j++) {
          if (!arr.isNull(j)) {
            acc += mgr->getString(input_dict_id, arr[j]);
          }
        }
        int32_t out_string_id = mgr->getOrAddTransient(output_dict_id, acc);
        output[i] = out_string_id;
      } else {
        T acc{0};
        for (auto j = 0; j < arr.getSize(); j++) {
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

// clang-format off
/*
  UDTF: array_copier__cpu_template(TableFunctionManager mgr, Column<Array<T>> input) -> Column<Array<T>>,
           T=[float, double, int8_t, int16_t, int32_t, int64_t, bool]
  UDTF: array_copier__cpu_template(TableFunctionManager mgr, Column<Array<TextEncodingDict>> input) -> Column<Array<TextEncodingDict>> | input_id=args<0>
*/
// clang-format on
template <typename T>
NEVER_INLINE HOST int32_t array_copier__cpu_template(TableFunctionManager& mgr,
                                                     const Column<Array<T>>& input,
                                                     Column<Array<T>>& output) {
  int size = input.size();

  // count the number of items in all input arrays:
  int output_values_size = 0;
  for (int i = 0; i < size; i++) {
    output_values_size += input[i].getSize();
  }

  // set the size and allocate the output columns buffers:
  mgr.set_output_array_values_total_number(
      /*output column index=*/0,
      /*upper bound to the number of items in all output arrays=*/output_values_size);
  mgr.set_output_row_size(size);

  // set the items of output colums:
  for (int i = 0; i < size; i++) {
    if (input.isNull(i)) {
      output.setNull(i);
    } else {
      output.setItem(i, input[i]);
    }
  }

  return size;
}

// clang-format off
/*
  UDTF: array_concat__cpu_template(TableFunctionManager mgr, ColumnList<Array<T>> input) -> Column<Array<T>> | input_id=args<0>,
          T=[float, double, int8_t, int16_t, int32_t, int64_t, bool, TextEncodingDict]
*/
// clang-format on
template <typename T>
NEVER_INLINE HOST int32_t array_concat__cpu_template(TableFunctionManager& mgr,
                                                     const ColumnList<Array<T>>& inputs,
                                                     Column<Array<T>>& output) {
  int size = inputs.size();

  int output_values_size = 0;
  for (int j = 0; j < inputs.numCols(); j++) {
    for (int i = 0; i < size; i++) {
      output_values_size += inputs[j][i].getSize();
    }
  }
  mgr.set_output_array_values_total_number(
      /*output column index=*/0,
      /*upper bound to the number of items in all output arrays=*/output_values_size);

  mgr.set_output_row_size(size);

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < inputs.numCols(); j++) {
      Column<Array<T>> col = inputs[j];
      Array<T> arr = col[i];
      if constexpr (std::is_same<T, TextEncodingDict>::value) {
        auto input_dict_id = col.getDictId();
        auto output_dict_id = output.getDictId();
        if (input_dict_id == output_dict_id) {
          output.concatItem(i, arr);
        } else {
          throw std::runtime_error(
              "concat of text encoding dict arrays with different dictionary ids is not "
              "implemented");
        }
      } else {
        output.concatItem(i,
                          arr);  // works only if i is the last row set, otherwise throws
      }
    }
  }
  return size;
}

// clang-format off
/*
  UDTF: array_asarray__cpu_template(TableFunctionManager mgr, Column<T> input) -> Column<Array<T>> | input_id=args<>,
          T=[int64_t, TextEncodingDict]
*/
// clang-format on
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
    int32_t input_dict_id = input.getDictId();
    int32_t output_dict_id = mgr.getNewDictId();
    for (int i = 0; i < size; i++) {
      if (input.isNull(i)) {
        output.setNull(i);
      } else {
        Array<T> arr = output.getItem(i, 1);
        arr[0] =
            mgr.getOrAddTransient(output_dict_id, mgr.getString(input_dict_id, input[i]));
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
      int64_t sz = input[i].getSize();
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
      int64_t sz = arr.getSize();
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

// clang-format off
/*
  UDTF: tf_metadata_setter__cpu_template(TableFunctionManager) -> Column<bool> success
*/
// clang-format on

NEVER_INLINE HOST int32_t tf_metadata_setter__cpu_template(TableFunctionManager& mgr,
                                                           Column<bool>& success) {
  // set one of each type
  mgr.set_metadata("test_int8_t", int8_t(1));
  mgr.set_metadata("test_int16_t", int16_t(2));
  mgr.set_metadata("test_int32_t", int32_t(3));
  mgr.set_metadata("test_int64_t", int64_t(4));
  mgr.set_metadata("test_float", 5.0f);
  mgr.set_metadata("test_double", 6.0);
  mgr.set_metadata("test_bool", true);

  mgr.set_output_row_size(1);
  success[0] = true;
  return 1;
}

// clang-format off
/*
  UDTF: tf_metadata_setter_repeated__cpu_template(TableFunctionManager) -> Column<bool> success
*/
// clang-format on

NEVER_INLINE HOST int32_t
tf_metadata_setter_repeated__cpu_template(TableFunctionManager& mgr,
                                          Column<bool>& success) {
  // set the same name twice
  mgr.set_metadata("test_int8_t", int8_t(1));
  mgr.set_metadata("test_int8_t", int8_t(2));

  mgr.set_output_row_size(1);
  success[0] = true;
  return 1;
}

// clang-format off
/*
  UDTF: tf_metadata_setter_size_mismatch__cpu_template(TableFunctionManager) -> Column<bool> success
*/
// clang-format on

NEVER_INLINE HOST int32_t
tf_metadata_setter_size_mismatch__cpu_template(TableFunctionManager& mgr,
                                               Column<bool>& success) {
  // set the same name twice
  mgr.set_metadata("test_int8_t", int8_t(1));
  mgr.set_metadata("test_int8_t", int16_t(2));

  mgr.set_output_row_size(1);
  success[0] = true;
  return 1;
}

// clang-format off
/*
  UDTF: tf_metadata_getter__cpu_template(TableFunctionManager, Column<bool>) -> Column<bool> success
*/
// clang-format on

NEVER_INLINE HOST int32_t tf_metadata_getter__cpu_template(TableFunctionManager& mgr,
                                                           const Column<bool>& input,
                                                           Column<bool>& success) {
  // get them all back and check values
  int8_t i8{};
  int16_t i16{};
  int32_t i32{};
  int64_t i64{};
  float f{};
  double d{};
  bool b{};

  try {
    mgr.get_metadata("test_int8_t", i8);
    mgr.get_metadata("test_int16_t", i16);
    mgr.get_metadata("test_int32_t", i32);
    mgr.get_metadata("test_int64_t", i64);
    mgr.get_metadata("test_float", f);
    mgr.get_metadata("test_double", d);
    mgr.get_metadata("test_bool", b);
  } catch (const std::runtime_error& ex) {
    return mgr.ERROR_MESSAGE(ex.what());
  }

  // return value indicates values were correct
  // types are implicitly correct by this point, or the above would have thrown
  bool result = (i8 == 1) && (i16 == 2) && (i32 == 3) && (i64 == 4) && (f == 5.0f) &&
                (d == 6.0) && b;
  if (!result) {
    return mgr.ERROR_MESSAGE("Metadata return values are incorrect");
  }

  mgr.set_output_row_size(1);
  success[0] = true;
  return 1;
}

// clang-format off
/*
  UDTF: tf_metadata_getter_bad__cpu_template(TableFunctionManager, Column<bool>) -> Column<bool> success
*/
// clang-format on

NEVER_INLINE HOST int32_t tf_metadata_getter_bad__cpu_template(TableFunctionManager& mgr,
                                                               const Column<bool>& input,
                                                               Column<bool>& success) {
  // get one back as the wrong type
  // this should throw
  float f{};
  try {
    mgr.get_metadata("test_double", f);
  } catch (const std::runtime_error& ex) {
    return mgr.ERROR_MESSAGE(ex.what());
  }

  mgr.set_output_row_size(1);
  success[0] = true;
  return 1;
}

// clang-format off
/*
  UDTF: ct_overload_scalar_test__cpu_template(T scalar) -> Column<T>, T=[Timestamp, int64_t]
  UDTF: ct_overload_column_test__cpu_template(Column<T>) -> Column<T>, T=[Timestamp, TextEncodingDict, int64_t]
  UDTF: ct_overload_column_list_test__cpu_template(Cursor<Column<K> first_col, ColumnList<T> col_list, Column<K> last_col>) -> Column<K>, K=[int64_t], T=[int64_t, double]
  UDTF: ct_overload_column_list_test2__cpu_template(Cursor<Column<K> first_col, ColumnList<K> col_list1, ColumnList<T> col_list2, Column<T> last_col>) -> Column<K>, K=[int64_t], T=[int64_t, double]
*/
// clang-format on

// Test table functions overloaded on scalar types
// Calcite should pick the proper overloaded operator for each templated table function
template <typename T>
NEVER_INLINE HOST int32_t ct_overload_scalar_test__cpu_template(const T scalar,
                                                                Column<T>& out) {
  set_output_row_size(1);
  if constexpr (std::is_same<T, int64_t>::value) {
    out[0] = scalar;
  } else if constexpr (std::is_same<T, Timestamp>::value) {
    out[0] = scalar.time;
  }
  return 1;
}

// Test table functions overloaded on column types
// Calcite should pick the proper overloaded operator for each templated table function
template <typename T>
NEVER_INLINE HOST int32_t ct_overload_column_test__cpu_template(const Column<T>& input,
                                                                Column<T>& out) {
  int64_t size = input.size();
  set_output_row_size(size);
  for (int64_t i = 0; i < size; ++i) {
    if (input.isNull(i)) {
      out.setNull(i);
    } else {
      out[i] = input[i];
    }
  }
  return size;
}

// Test Calcite overload resolution for table functions with ColumnList arguments
// Calcite should pick the proper overloaded operator for each templated table function
template <typename T, typename K>
NEVER_INLINE HOST int32_t
ct_overload_column_list_test__cpu_template(const Column<K>& first_col,
                                           const ColumnList<T>& col_list,
                                           const Column<K>& last_col,
                                           Column<K>& out) {
  set_output_row_size(1);
  int64_t num_cols = col_list.numCols();
  T sum = 0;
  for (int64_t i = 0; i < num_cols; ++i) {
    const Column<T>& col = col_list[i];
    for (int64_t j = 0; j < col.size(); ++j) {
      sum += col[j];
    }
  }
  if (sum > 0) {
    out[0] = first_col[0];
  } else {
    out[0] = last_col[0];
  }
  return 1;
}

// Test Calcite overload resolution for table functions with multiple ColumnList arguments
// Calcite should pick the proper overloaded operator for each templated table function
template <typename T, typename K>
NEVER_INLINE HOST int32_t
ct_overload_column_list_test2__cpu_template(const Column<K>& first_col,
                                            const ColumnList<K>& col_list1,
                                            const ColumnList<T>& col_list2,
                                            const Column<T>& last_col,
                                            Column<K>& out) {
  set_output_row_size(1);
  int64_t num_cols = col_list1.numCols();
  K sum = 0;
  for (int64_t i = 0; i < num_cols; ++i) {
    const Column<K>& col = col_list1[i];
    for (int64_t j = 0; j < col.size(); ++j) {
      sum += col[j];
    }
  }

  int64_t num_cols2 = col_list2.numCols();
  T sum2 = 0;
  for (int64_t i = 0; i < num_cols2; ++i) {
    const Column<T>& col = col_list2[i];
    for (int64_t j = 0; j < col.size(); ++j) {
      sum2 += col[j];
    }
  }

  if (sum + sum2 > 0) {
    out[0] = first_col[0];
  } else {
    out[0] = last_col[0];
  }
  return 1;
}

#endif  // ifndef __CUDACC__

// clang-format off
/*
  UDTF: ct_require_range__cpu_(Column<int32_t>, int x | range=[1, 5], RowMultiplier) -> Column<int32_t>
*/
// clang-format on

EXTENSION_NOINLINE int32_t ct_require_range__cpu_(const Column<int32_t>& input1,
                                                  const int32_t x,
                                                  const int32_t multiplier,
                                                  Column<int32_t>& out) {
  out[0] = 0;
  return 1;
}
