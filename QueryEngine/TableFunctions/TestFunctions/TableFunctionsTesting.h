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

#include "../../heavydbTypes.h"

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
int32_t ct_device_selection_udtf_any(const Column<int32_t>& input, Column<int64_t>& out);

#ifndef __CUDACC__

EXTENSION_NOINLINE_HOST
int32_t ct_device_selection_udtf_cpu__cpu_(const Column<int32_t>& input,
                                           Column<int64_t>& out);

#endif  // #ifndef __CUDACC__

EXTENSION_NOINLINE
int32_t ct_device_selection_udtf_gpu__gpu_(const Column<int32_t>& input,
                                           Column<int64_t>& out);

#ifndef __CUDACC__

EXTENSION_NOINLINE_HOST
int32_t ct_device_selection_udtf_both__cpu_(const Column<int32_t>& input,
                                            Column<int64_t>& out);

#endif  // #ifndef __CUDACC__

EXTENSION_NOINLINE
int32_t ct_device_selection_udtf_both__gpu_(const Column<int32_t>& input,
                                            Column<int64_t>& out);

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
ct_binding_udtf_constant__cpu_1(const Column<int32_t>& input1, Column<int32_t>& out);

EXTENSION_NOINLINE_HOST int32_t
ct_binding_udtf_constant__cpu_2(const Column<int32_t>& input1,
                                const Column<int32_t>& input2,
                                Column<int32_t>& out);

EXTENSION_NOINLINE_HOST int32_t
ct_binding_udtf_constant__cpu_3(const Column<int32_t>& input1,
                                const Column<int32_t>& input2,
                                const Column<int32_t>& input3,
                                Column<int32_t>& out);

EXTENSION_NOINLINE_HOST int32_t
ct_binding_udtf_constant__cpu_4(const Column<int64_t>& input1,
                                const Column<int32_t>& input2,
                                const Column<int32_t>& input3,
                                Column<int32_t>& out);

EXTENSION_NOINLINE_HOST int32_t
ct_binding_udtf_constant__cpu_5(const Column<int64_t>& input1,
                                const Column<int64_t>& input2,
                                const Column<int32_t>& input3,
                                Column<int32_t>& out);

EXTENSION_NOINLINE_HOST int32_t
ct_binding_udtf_constant__cpu_6(const Column<int64_t>& input1,
                                const Column<int32_t>& input2,
                                const Column<int64_t>& input3,
                                Column<int32_t>& out);

EXTENSION_NOINLINE_HOST int32_t
ct_binding_udtf_constant__cpu_7(const Column<int32_t>& input1,
                                const ColumnList<int32_t>& input2,
                                Column<int32_t>& out);

EXTENSION_NOINLINE_HOST int32_t
ct_binding_udtf_constant__cpu_8(const ColumnList<int32_t>& input1,
                                const Column<int64_t>& input2,
                                Column<int32_t>& out);

EXTENSION_NOINLINE_HOST int32_t
ct_binding_udtf_constant__cpu_9(const ColumnList<int32_t>& input1,
                                const ColumnList<int64_t>& input2,
                                Column<int32_t>& out);

EXTENSION_NOINLINE_HOST int32_t
ct_binding_udtf_constant__cpu_10(const Column<int64_t>& input1,
                                 const ColumnList<int64_t>& input2,
                                 const Column<int64_t>& input3,
                                 Column<int64_t>& out);

EXTENSION_NOINLINE_HOST int32_t ct_binding_udtf__cpu_11(const Column<int32_t>& input1,
                                                        const int32_t multiplier,
                                                        Column<int32_t>& out);

EXTENSION_NOINLINE_HOST int32_t ct_binding_udtf__cpu_12(const Column<int32_t>& input1,
                                                        const Column<int32_t>& input2,
                                                        const int32_t multiplier,
                                                        Column<int32_t>& out);

EXTENSION_NOINLINE_HOST int32_t ct_binding_udtf__cpu_13(const Column<int32_t>& input1,
                                                        const Column<int32_t>& input2,
                                                        const Column<int32_t>& input3,
                                                        const int32_t multiplier,
                                                        Column<int32_t>& out);

EXTENSION_NOINLINE_HOST int32_t ct_binding_udtf__cpu_14(const Column<int64_t>& input1,
                                                        const Column<int32_t>& input2,
                                                        const Column<int32_t>& input3,
                                                        const int32_t multiplier,
                                                        Column<int32_t>& out);

EXTENSION_NOINLINE_HOST int32_t ct_binding_udtf__cpu_15(const Column<int64_t>& input1,
                                                        const Column<int64_t>& input2,
                                                        const Column<int32_t>& input3,
                                                        const int32_t multiplier,
                                                        Column<int32_t>& out);

EXTENSION_NOINLINE_HOST int32_t ct_binding_udtf__cpu_16(const Column<int64_t>& input1,
                                                        const Column<int32_t>& input2,
                                                        const Column<int64_t>& input3,
                                                        const int32_t multiplier,
                                                        Column<int32_t>& out);

EXTENSION_NOINLINE_HOST int32_t ct_binding_udtf__cpu_17(const Column<int32_t>& input1,
                                                        const ColumnList<int32_t>& input2,
                                                        const int32_t multiplier,
                                                        Column<int32_t>& out);

EXTENSION_NOINLINE_HOST int32_t ct_binding_udtf__cpu_18(const ColumnList<int32_t>& input1,
                                                        const Column<int64_t>& input2,
                                                        const int32_t multiplier,
                                                        Column<int32_t>& out);

EXTENSION_NOINLINE_HOST int32_t ct_binding_udtf__cpu_19(const ColumnList<int32_t>& input1,
                                                        const ColumnList<int64_t>& input2,
                                                        const int32_t multiplier,
                                                        Column<int32_t>& out);

EXTENSION_NOINLINE_HOST int32_t ct_binding_udtf__cpu_20(const Column<int64_t>& input1,
                                                        const ColumnList<int64_t>& input2,
                                                        const Column<int64_t>& input3,
                                                        const int32_t multiplier,
                                                        Column<int64_t>& out);

EXTENSION_NOINLINE_HOST int32_t ct_binding_udtf2__cpu_21(const int32_t multiplier,
                                                         const Column<int32_t>& input1,
                                                         Column<int32_t>& out);

EXTENSION_NOINLINE_HOST int32_t ct_binding_udtf6__cpu_22(const Column<int32_t>& input1,
                                                         const int32_t multiplier,
                                                         const int32_t input2,
                                                         Column<int32_t>& out);

EXTENSION_NOINLINE_HOST int32_t
ct_binding_udtf4__cpu_23(const ColumnList<int32_t>& input1,
                         const int32_t multiplier,
                         const int32_t input2,
                         Column<int32_t>& out);

EXTENSION_NOINLINE_HOST int32_t
ct_binding_udtf5__cpu_24(const ColumnList<int32_t>& input1,
                         const int32_t input2,
                         const int32_t multiplier,
                         Column<int32_t>& out);

EXTENSION_NOINLINE_HOST int32_t ct_binding_udtf3__cpu_25(const Column<int32_t>& input1,
                                                         const int32_t input2,
                                                         const int32_t multiplier,
                                                         Column<int32_t>& out);

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
                               Column<int32_t>& out);

EXTENSION_NOINLINE_HOST int32_t
ct_udtf_default_sizer1b__cpu_2(const Column<int32_t>& input1,
                               const Column<int32_t>& input2,
                               const int32_t multiplier,
                               Column<int32_t>& out);

EXTENSION_NOINLINE_HOST int32_t
ct_udtf_default_sizer1c__cpu_3(const Column<int32_t>& input1,
                               const Column<int32_t>& input2,
                               const Column<int32_t>& input3,
                               const int32_t multiplier,
                               const Column<int32_t>& input4,
                               const int32_t x,
                               Column<int32_t>& out);

EXTENSION_NOINLINE_HOST int32_t
ct_udtf_default_sizer1d__cpu_4(const int32_t multiplier,
                               const int32_t x,
                               const Column<int32_t>& input1,
                               Column<int32_t>& out);

EXTENSION_NOINLINE_HOST int32_t
ct_udtf_default_sizer2a__cpu_1(const Column<int32_t>& input1,
                               const int32_t x,
                               const int32_t multiplier,
                               Column<int32_t>& out);

EXTENSION_NOINLINE_HOST int32_t
ct_udtf_default_sizer2b__cpu_2(const Column<int32_t>& input1,
                               const int32_t multiplier,
                               const Column<int32_t>& input2,
                               Column<int32_t>& out);

EXTENSION_NOINLINE_HOST int32_t
ct_udtf_default_sizer2c__cpu_3(const int32_t x,
                               const int32_t multiplier,
                               const Column<int32_t>& input1,
                               Column<int32_t>& out);

EXTENSION_NOINLINE_HOST int32_t
ct_udtf_default_sizer3a__cpu_1(const Column<int32_t>& input1,
                               const int32_t multiplier,
                               const int32_t x,
                               Column<int32_t>& out);

EXTENSION_NOINLINE_HOST int32_t
ct_udtf_default_sizer3b__cpu_2(const Column<int32_t>& input1,
                               const int32_t x,
                               const Column<int32_t>& input2,
                               const int32_t multiplier,
                               Column<int32_t>& out);

EXTENSION_NOINLINE_HOST int32_t
ct_udtf_default_sizer4a__cpu_1(const Column<int32_t>& input1,
                               const int32_t multiplier,
                               const Column<int32_t>& input2,
                               const int32_t x,
                               Column<int32_t>& out);

EXTENSION_NOINLINE_HOST int32_t
ct_udtf_default_sizer4b__cpu_2(const int32_t multiplier,
                               const Column<int32_t>& input,
                               const int32_t x,
                               Column<int32_t>& out);

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
                                Column<TextEncodingDict>& out);

EXTENSION_NOINLINE_HOST int32_t
ct_binding_dict_encoded2__cpu_1(const Column<TextEncodingDict>& input1,
                                const Column<TextEncodingDict>& input2,
                                Column<TextEncodingDict>& out1,
                                Column<TextEncodingDict>& out2);

EXTENSION_NOINLINE_HOST int32_t
ct_binding_dict_encoded3__cpu_1(const Column<TextEncodingDict>& input1,
                                const Column<TextEncodingDict>& input2,
                                Column<TextEncodingDict>& out1,
                                Column<TextEncodingDict>& out2);

EXTENSION_NOINLINE_HOST int32_t
ct_binding_dict_encoded4__cpu_1(const ColumnList<TextEncodingDict>& input,
                                Column<TextEncodingDict>& out);

EXTENSION_NOINLINE_HOST int32_t
ct_binding_dict_encoded5__cpu_1(const ColumnList<TextEncodingDict>& input,
                                Column<TextEncodingDict>& out);

EXTENSION_NOINLINE_HOST int32_t
ct_binding_dict_encoded6__cpu_1(const ColumnList<TextEncodingDict>& input,
                                Column<TextEncodingDict>& out0,
                                Column<TextEncodingDict>& out1);

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
                                                            Column<T>& out);

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
                                                              Column<int32_t>& out);

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
                                                          Column<int32_t>& out);

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
                                                                   Column<T>& out);

// clang-format off
/*
  UDTF: ct_binding_str_length__cpu_(Cursor<TextEncodingDict>) -> Column<TextEncodingDict> string | input_id=args<0>, Column<int64_t> string_length
*/
// clang-format on
EXTENSION_NOINLINE_HOST
int32_t ct_binding_str_length__cpu_(const Column<TextEncodingDict>& input_str,
                                    Column<TextEncodingDict>& out_str,
                                    Column<int64_t>& out_size);

// clang-format off
/*
  UDTF: ct_binding_str_equals__cpu_(Cursor<ColumnList<TextEncodingDict>>) -> Column<TextEncodingDict> string_if_equal | input_id=args<0, 0>, Column<bool> strings_are_equal
*/
// clang-format on
EXTENSION_NOINLINE_HOST
int32_t ct_binding_str_equals__cpu_(const ColumnList<TextEncodingDict>& input_strings,
                                    Column<TextEncodingDict>& string_if_equal,
                                    Column<bool>& strings_are_equal);

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
                        Column<TextEncodingDict>& output_substr);

// clang-format off
/*
  UDTF: ct_string_concat__cpu_(TableFunctionManager, Cursor<ColumnList<TextEncodingDict>>, TextEncodingNone separator | default = "|") -> Column<TextEncodingDict> concatted_str | input_id=args<0, 0>
*/
// clang-format on
EXTENSION_NOINLINE_HOST
int32_t ct_string_concat__cpu_(TableFunctionManager& mgr,
                               const ColumnList<TextEncodingDict>& input_strings,
                               const TextEncodingNone& separator,
                               Column<TextEncodingDict>& concatted_string);

// clang-format off
/*
  UDTF: ct_synthesize_new_dict__cpu_(TableFunctionManager, int32_t num_strings) -> Column<TextEncodingDict> new_dict_col | input_id=args<0>
*/
// clang-format on
EXTENSION_NOINLINE_HOST
int32_t ct_synthesize_new_dict__cpu_(TableFunctionManager& mgr,
                                     const int64_t num_strings,
                                     Column<TextEncodingDict>& new_dict_col);

#endif  // #ifndef __CUDACC__

#ifndef __CUDACC__

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
                                                          Column<T>& output);

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
                                                           Column<K>& out);

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
                                                        Column<T>& out);

template <typename T>
NEVER_INLINE HOST int32_t ct_named_const_output__cpu_template(const Column<T>& input,
                                                              Column<T>& out);

template <typename T>
NEVER_INLINE HOST int32_t ct_named_user_const_output__cpu_template(const Column<T>& input,
                                                                   int32_t c,
                                                                   Column<T>& out);

template <typename T>
NEVER_INLINE HOST int32_t ct_named_rowmul_output__cpu_template(const Column<T>& input,
                                                               int32_t m,
                                                               Column<T>& out);

#endif  // #ifndef __CUDACC__

// clang-format off
/*
  UDTF: ct_no_arg_runtime_sizing__cpu_template() -> Column<T> answer, T=[int32_t]
*/
// clang-format on

#ifndef __CUDACC__

template <typename T>
NEVER_INLINE HOST int32_t ct_no_arg_runtime_sizing__cpu_template(Column<T>& answer);

#endif  // #ifndef __CUDACC__

// clang-format off
/*
  UDTF: ct_no_arg_constant_sizing(Constant<42>) -> Column<int32_t> answer 
*/
// clang-format on

EXTENSION_NOINLINE int32_t ct_no_arg_constant_sizing(Column<int32_t>& answer);

// clang-format off
/*
  UDTF: ct_scalar_1_arg_runtime_sizing__cpu_template(T num) -> Column<T> answer, T=[float, double, int32_t, int64_t]
*/
// clang-format on

#ifndef __CUDACC__

template <typename T>
NEVER_INLINE HOST int32_t ct_scalar_1_arg_runtime_sizing__cpu_template(const T num,
                                                                       Column<T>& answer);

#endif  // #ifndef __CUDACC__

// clang-format off
/*
  UDTF: ct_scalar_2_args_constant_sizing(int64_t, int64_t, Constant<5>) -> Column<int64_t> answer1, Column<int64_t> answer2
*/
// clang-format on

EXTENSION_NOINLINE int32_t ct_scalar_2_args_constant_sizing(const int64_t num1,
                                                            const int64_t num2,
                                                            Column<int64_t>& answer1,
                                                            Column<int64_t>& answer2);

// clang-format off
/*
  UDTF: ct_no_cursor_user_constant_sizer__cpu_(int32_t, ConstantParameter c) -> Column<int32_t> output
*/
// clang-format on

#ifndef __CUDACC__

EXTENSION_NOINLINE_HOST int32_t
ct_no_cursor_user_constant_sizer__cpu_(const int32_t input_num,
                                       int32_t c,
                                       Column<int32_t>& output);

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
                                                         Column<T>& output);

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
                                       Column<int32_t>& output);

// clang-format off
/*
  UDTF: ct_templated_no_cursor_user_constant_sizer__gpu_template(T, ConstantParameter c) -> Column<T> output, T=[int32_t, float]
*/
// clang-format on

template <typename T>
TEMPLATE_NOINLINE int32_t
ct_templated_no_cursor_user_constant_sizer__gpu_template(const T input_num,
                                                         int32_t c,
                                                         Column<T>& output);

#endif  //__CUDACC__

// clang-format off
/*
  UDTF: column_list_safe_row_sum__cpu_template(Cursor<ColumnList<T>>) -> Column<T>, T=[int32_t, int64_t, float, double]
*/
// clang-format on

#ifndef __CUDACC__

template <typename T>
NEVER_INLINE HOST int32_t
column_list_safe_row_sum__cpu_template(const ColumnList<T>& input, Column<T>& out);

#endif  // #ifndef __CUDACC__

// clang-format off
/*
  UDTF: ct_gpu_default_init__gpu_(Constant<1>) -> Column<int32_t> output_buffer
*/
// clang-format on

EXTENSION_NOINLINE int32_t ct_gpu_default_init__gpu_(Column<int32_t>& output_buffer);

// clang-format off
/*
  UDTF: ct_hamming_distance(TextEncodingNone, TextEncodingNone, Constant<1>) -> Column<int32_t> hamming_distance
*/
// clang-format on

EXTENSION_NOINLINE int32_t ct_hamming_distance(const TextEncodingNone& str1,
                                               const TextEncodingNone& str2,
                                               Column<int32_t>& hamming_distance);

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
                                                        Column<int8_t>& char_bytes);

#ifndef __CUDACC__

// clang-format off
/*
  UDTF: ct_string_to_chars__cpu_(TextEncodingNone) -> Column<int32_t> char_idx, Column<int8_t> char_bytes
*/
// clang-format on

EXTENSION_NOINLINE_HOST int32_t ct_string_to_chars__cpu_(const TextEncodingNone& input,
                                                         Column<int32_t>& char_idx,
                                                         Column<int8_t>& char_bytes);

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
EXTENSION_NOINLINE int32_t ct_sleep_worker(int32_t seconds, Column<int32_t>& output);

EXTENSION_NOINLINE_HOST int32_t ct_sleep1__cpu_(int32_t seconds,
                                                int32_t mode,
                                                Column<int32_t>& output);

EXTENSION_NOINLINE_HOST int32_t ct_sleep2(TableFunctionManager& mgr,
                                          int32_t seconds,
                                          int32_t mode,
                                          Column<int32_t>& output);

// clang-format off
/*
  UDTF: ct_throw_if_gt_100__cpu_template(TableFunctionManager, Column<T>) -> Column<T> val, T=[float, double]
*/
// clang-format on

template <typename T>
NEVER_INLINE HOST int32_t ct_throw_if_gt_100__cpu_template(TableFunctionManager& mgr,
                                                           const Column<T>& input,
                                                           Column<T>& output);

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
                                                     Column<int32_t>& output);

EXTENSION_NOINLINE_HOST int32_t ct_add_size_and_mul_alpha(TableFunctionManager& mgr,
                                                          const Column<int32_t>& input1,
                                                          const Column<int32_t>& input2,
                                                          int32_t alpha,
                                                          Column<int32_t>& output1,
                                                          Column<int32_t>& output2);

EXTENSION_NOINLINE_HOST int32_t ct_sparse_add(TableFunctionManager& mgr,
                                              const Column<int32_t>& x1,
                                              const Column<int32_t>& d1,
                                              int32_t f1,
                                              const Column<int32_t>& x2,
                                              const Column<int32_t>& d2,
                                              int32_t f2,
                                              Column<int32_t>& x,
                                              Column<int32_t>& d);

#endif  // #ifndef __CUDACC__

// clang-format off
/*
  UDTF: ct_pushdown_stats__cpu_template(TableFunctionManager, TextEncodingNone agg_type, Cursor<Column<K> id, Column<T> x, Column<T> y, Column<Z> z>) | filter_table_function_transpose=on -> Column<int32_t> row_count, Column<K> id | input_id=args<0>, Column<T> x, Column<T> y, Column<Z> z, K=[int64_t, TextEncodingDict], T=[int64_t, double], Z=[int64_t, double]
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
                                Column<Z>& output_z);

#endif  // #ifndef __CUDACC__

// clang-format off
/*
  UDTF: ct_pushdown_projection__cpu_template(TableFunctionManager, Cursor<Column<K> id, Column<T> x, Column<T> y, Column<Z> z>) | filter_table_function_transpose=on -> Column<K> id | input_id=args<0>, Column<T> x, Column<T> y, Column<Z> z, K=[int64_t, TextEncodingDict], T=[int64_t, double], Z=[int64_t, double]
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
                                                               Column<Z>& output_z);

#endif  // #ifndef __CUDACC__

// clang-format off
/*
  UDTF: ct_union_pushdown_stats__cpu_template(TableFunctionManager, TextEncodingNone agg_type, Cursor<Column<K> id, Column<T> x, Column<T> y, Column<Z> z>, Cursor<Column<K> id, Column<T> x, Column<T> y, Column<Z> z, Column<T> w>) | filter_table_function_transpose=on -> Column<int32_t> row_count, Column<K> id | input_id=args<0, 0>, Column<T> x, Column<T> y, Column<Z> z, Column<T> w, K=[int64_t, TextEncodingDict], T=[int64_t, double], Z=[int64_t, double]
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
                                      Column<T>& output_w);

#endif  // #ifndef __CUDACC__

// clang-format off
/*
  UDTF: ct_union_pushdown_projection__cpu_template(TableFunctionManager, Cursor<Column<K> id, Column<T> x, Column<T> y, Column<Z> z>, Cursor<Column<K> id, Column<T> x, Column<T> y, Column<Z> z, Column<T> w>) | filter_table_function_transpose=on -> Column<K> id | input_id=args<0, 0>, Column<T> x, Column<T> y, Column<Z> z, Column<T> w, K=[int64_t, TextEncodingDict], T=[int64_t, double], Z=[int64_t, double]
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
                                           Column<T>& output_w);

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
  UDTF: ct_require_text_collist_enc_dict__cpu_(Cursor<ColumnList<TextEncodingDict>>, int64_t x | require="x >= 1") -> Column<int64_t>
*/
// clang-format on

#ifndef __CUDACC__

EXTENSION_NOINLINE_HOST int32_t ct_require__cpu_(const Column<int32_t>& input1,
                                                 const int32_t i,
                                                 Column<int32_t>& out);

EXTENSION_NOINLINE_HOST int32_t ct_require_str__cpu_(const Column<int32_t>& input1,
                                                     const TextEncodingNone& s,
                                                     Column<int32_t>& out);

EXTENSION_NOINLINE_HOST int32_t ct_require_mgr(TableFunctionManager& mgr,
                                               const Column<int32_t>& input1,
                                               const int32_t i,
                                               Column<int32_t>& out);

template <typename T, typename K>
NEVER_INLINE HOST int32_t ct_require_templating__cpu_template(const Column<T>& input1,
                                                              const int32_t i,
                                                              Column<K>& out);

EXTENSION_NOINLINE_HOST int32_t ct_require_and__cpu_(const Column<int32_t>& input1,
                                                     const int32_t i,
                                                     Column<int32_t>& out);

EXTENSION_NOINLINE_HOST int32_t ct_require_or_str__cpu_(const Column<int32_t>& input1,
                                                        const TextEncodingNone& i,
                                                        Column<int32_t>& out);

EXTENSION_NOINLINE_HOST int32_t ct_require_str_diff__cpu_(const Column<int32_t>& input1,
                                                          const TextEncodingNone& i,
                                                          Column<int32_t>& out);

EXTENSION_NOINLINE_HOST int32_t
ct_require_text_enc_dict__cpu_(const Column<TextEncodingDict>& input,
                               const int64_t x,
                               Column<int64_t>& out);

EXTENSION_NOINLINE_HOST int32_t
ct_require_text_collist_enc_dict__cpu_(const ColumnList<TextEncodingDict>& input,
                                       const int64_t x,
                                       Column<int64_t>& out);

/*
  UDTF: ct_test_allocator(TableFunctionManager, Column<int32_t>, TextEncodingNone) ->
  Column<int32_t>
*/
EXTENSION_NOINLINE_HOST int32_t ct_test_allocator(TableFunctionManager& mgr,
                                                  const Column<int32_t>& input,
                                                  const TextEncodingNone& t,
                                                  Column<int32_t>& out);

#endif  // #ifndef __CUDACC__

#ifdef __CUDACC__

// clang-format off
/*
  UDTF: ct_require_device_cuda__gpu_(Column<int32_t>, Constant<1>, int | name=i | require="i > 0") -> Column<int32_t>
*/
// clang-format on

EXTENSION_NOINLINE int32_t ct_require_device_cuda__gpu_(const Column<int32_t>& input1,
                                                        const int32_t i,
                                                        Column<int32_t>& out);

// clang-format off
/*
  UDTF: ct_cuda_enumerate_threads__gpu_(ConstantParameter output_size) -> Column<int32_t> local_thread_id, Column<int32_t> block_id, Column<int32_t> global_thread_id
*/
// clang-format on

EXTENSION_NOINLINE int32_t
ct_cuda_enumerate_threads__gpu_(const int32_t output_size,
                                Column<int32_t>& out_local_thread_id,
                                Column<int32_t>& out_block_id,
                                Column<int32_t>& out_global_thread_id);

#endif  //__CUDACC__

// clang-format off
/*
  UDTF: ct_test_nullable(Column<int32_t>, RowMultiplier) -> Column<int32_t>
*/
// clang-format on

EXTENSION_NOINLINE int32_t ct_test_nullable(const Column<int32_t>& input,
                                            const int32_t i,
                                            Column<int32_t>& out);

// clang-format off
/*
  UDTF: ct_test_preflight_sizer(Column<int32_t> col, int i, int j) -> Column<int32_t> | output_row_size="i + j"
  UDTF: ct_test_preflight_sizer_const(Column<int32_t> col) -> Column<int32_t> | output_row_size=2
*/
// clang-format on

EXTENSION_NOINLINE int32_t ct_test_preflight_sizer(const Column<int32_t>& input,
                                                   const int32_t i,
                                                   const int32_t j,
                                                   Column<int32_t>& out);

EXTENSION_NOINLINE int32_t ct_test_preflight_sizer_const(const Column<int32_t>& input,
                                                         Column<int32_t>& out);

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
                                           Column<int32_t>& out);

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
                                          Column<int32_t>& out);

// clang-format off
/*
  UDTF: ct_scalar_named_args__cpu_(TableFunctionManager, int32_t arg1, int32_t arg2) ->
  Column<int32_t> out1, Column<int32_t> out2
*/

EXTENSION_NOINLINE_HOST int32_t
ct_scalar_named_args__cpu_(TableFunctionManager& mgr, const int32_t arg1,
 const int32_t arg2, Column<int32_t>& out1, Column<int32_t>& out2);

// clang-format off
/*
  UDTF: ct_cursor_named_args__cpu_(TableFunctionManager, Cursor<Column<int32_t> input_table_arg1, 
  Column<int32_t> input_table_arg2> input_table, int32_t arg1, int32_t arg2) ->
  Column<int32_t> out1, Column<int32_t> out2
*/

EXTENSION_NOINLINE_HOST int32_t
ct_cursor_named_args__cpu_(TableFunctionManager& mgr, const Column<int32_t>& input_arg1,
const Column<int32_t>& input_arg2, const int32_t arg1, const int32_t arg2,
 Column<int32_t>& out1, Column<int32_t>& out2);

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
                                                     Column<int64_t>& y);

// Test table functions with scalar Timestamp inputs
EXTENSION_NOINLINE_HOST int32_t ct_timestamp_add_offset(TableFunctionManager& mgr,
                                                        const Column<Timestamp>& input,
                                                        const Timestamp offset,
                                                        Column<Timestamp>& out);

// Test table function with sizer argument, and mix of scalar/column inputs.
EXTENSION_NOINLINE int32_t
ct_timestamp_test_columns_and_scalars__cpu(const Column<Timestamp>& input,
                                           const int64_t dummy,
                                           const int32_t multiplier,
                                           const Column<Timestamp>& input2,
                                           Column<Timestamp>& out);

// Dummy test for ColumnList inputs + Column Timestamp input
EXTENSION_NOINLINE_HOST int32_t
ct_timestamp_column_list_input(TableFunctionManager& mgr,
                               const ColumnList<int64_t>& input,
                               const Column<Timestamp>& input2,
                               Column<int64_t>& out);

EXTENSION_NOINLINE_HOST int32_t ct_timestamp_truncate(TableFunctionManager& mgr,
                                                      const Column<Timestamp>& input,
                                                      Column<Timestamp>& y,
                                                      Column<Timestamp>& mo,
                                                      Column<Timestamp>& d,
                                                      Column<Timestamp>& h,
                                                      Column<Timestamp>& m,
                                                      Column<Timestamp>& s,
                                                      Column<Timestamp>& ms,
                                                      Column<Timestamp>& us);

template <typename T>
NEVER_INLINE HOST int32_t
ct_timestamp_add_interval__template(TableFunctionManager& mgr,
                                    const Column<Timestamp>& input,
                                    const T inter,
                                    Column<Timestamp>& out);

// clang-format off
/*
  UDTF: sum_along_row__cpu_template(Column<Array<T>> input) -> Column<T>,
          T=[float, double, int8_t, int16_t, int32_t, int64_t, bool, TextEncodingDict] | output_row_size="input.size()" | input_id=args<0>
*/
// clang-format on
template <typename T>
NEVER_INLINE HOST int32_t sum_along_row__cpu_template(const Column<Array<T>>& input,
                                                      Column<T>& output);

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
                                                     Column<Array<T>>& output);

// clang-format off
/*
  UDTF: array_concat__cpu_template(TableFunctionManager mgr, ColumnList<Array<T>> input) -> Column<Array<T>> | input_id=args<0>,
          T=[float, double, int8_t, int16_t, int32_t, int64_t, bool, TextEncodingDict]
*/
// clang-format on
template <typename T>
NEVER_INLINE HOST int32_t array_concat__cpu_template(TableFunctionManager& mgr,
                                                     const ColumnList<Array<T>>& inputs,
                                                     Column<Array<T>>& output);

// clang-format off
/*
  UDTF: array_asarray__cpu_template(TableFunctionManager mgr, Column<T> input) -> Column<Array<T>> | input_id=args<>,
          T=[int64_t, TextEncodingDict]
*/
// clang-format on
template <typename T>
NEVER_INLINE HOST int32_t array_asarray__cpu_template(TableFunctionManager& mgr,
                                                      const Column<T>& input,
                                                      Column<Array<T>>& output);

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
                                                    Column<Array<T>>& second);

// clang-format off
/*
  UDTF: tf_metadata_setter__cpu_template(TableFunctionManager) -> Column<bool> success
*/
// clang-format on

NEVER_INLINE HOST int32_t tf_metadata_setter__cpu_template(TableFunctionManager& mgr,
                                                           Column<bool>& success);

// clang-format off
/*
  UDTF: tf_metadata_setter_repeated__cpu_template(TableFunctionManager) -> Column<bool> success
*/
// clang-format on

NEVER_INLINE HOST int32_t
tf_metadata_setter_repeated__cpu_template(TableFunctionManager& mgr,
                                          Column<bool>& success);

// clang-format off
/*
  UDTF: tf_metadata_setter_size_mismatch__cpu_template(TableFunctionManager) -> Column<bool> success
*/
// clang-format on

NEVER_INLINE HOST int32_t
tf_metadata_setter_size_mismatch__cpu_template(TableFunctionManager& mgr,
                                               Column<bool>& success);

// clang-format off
/*
  UDTF: tf_metadata_getter__cpu_template(TableFunctionManager, Column<bool>) -> Column<bool> success
*/
// clang-format on

NEVER_INLINE HOST int32_t tf_metadata_getter__cpu_template(TableFunctionManager& mgr,
                                                           const Column<bool>& input,
                                                           Column<bool>& success);

// clang-format off
/*
  UDTF: tf_metadata_getter_bad__cpu_template(TableFunctionManager, Column<bool>) -> Column<bool> success
*/
// clang-format on

NEVER_INLINE HOST int32_t tf_metadata_getter_bad__cpu_template(TableFunctionManager& mgr,
                                                               const Column<bool>& input,
                                                               Column<bool>& success);

// clang-format off
/*
  UDTF: ct_overload_scalar_test__cpu_template(T scalar) -> Column<T>, T=[Timestamp, int64_t]
  UDTF: ct_overload_column_test__cpu_template(Column<T>) -> Column<T>, T=[Timestamp, TextEncodingDict, int64_t]
  UDTF: ct_overload_column_test__cpu_template(Column<Array<T>>) -> Column<Array<T>>, T=[TextEncodingDict, int64_t]
  UDTF: ct_overload_column_list_test__cpu_template(Cursor<Column<K> first_col, ColumnList<T> col_list, Column<K> last_col>) -> Column<K>, K=[int64_t], T=[int64_t, double]
  UDTF: ct_overload_column_list_test2__cpu_template(Cursor<Column<K> first_col, ColumnList<K> col_list1, ColumnList<T> col_list2, Column<T> last_col>) -> Column<K>, K=[int64_t], T=[int64_t, double]
*/
// clang-format on

// Test table functions overloaded on scalar types
// Calcite should pick the proper overloaded operator for each templated table function
template <typename T>
NEVER_INLINE HOST int32_t ct_overload_scalar_test__cpu_template(const T scalar,
                                                                Column<T>& out);

// Test table functions overloaded on column types
// Calcite should pick the proper overloaded operator for each templated table function
template <typename T>
NEVER_INLINE HOST int32_t ct_overload_column_test__cpu_template(const Column<T>& input,
                                                                Column<T>& out);

// Test Calcite overload resolution for table functions with ColumnList arguments
// Calcite should pick the proper overloaded operator for each templated table function
template <typename T, typename K>
NEVER_INLINE HOST int32_t
ct_overload_column_list_test__cpu_template(const Column<K>& first_col,
                                           const ColumnList<T>& col_list,
                                           const Column<K>& last_col,
                                           Column<K>& out);

// Test Calcite overload resolution for table functions with multiple ColumnList arguments
template <typename T, typename K>
NEVER_INLINE HOST int32_t
ct_overload_column_list_test2__cpu_template(const Column<K>& first_col,
                                            const ColumnList<K>& col_list1,
                                            const ColumnList<T>& col_list2,
                                            const Column<T>& last_col,
                                            Column<K>& out);

#endif

// clang-format off
/*
  UDTF: ct_require_range__cpu_(Column<int32_t>, int x | range=[1, 5], RowMultiplier) -> Column<int32_t>
*/
// clang-format on

EXTENSION_NOINLINE int32_t ct_require_range__cpu_(const Column<int32_t>& input1,
                                                  const int32_t x,
                                                  const int32_t multiplier,
                                                  Column<int32_t>& out);

#ifndef __CUDACC__

// clang-format off
/*
  UDTF: ct_coords__cpu_(TableFunctionManager, Column<GeoPoint> points) -> Column<double> x, Column<double> y
  UDTF: ct_shift__cpu_(TableFunctionManager, Column<GeoPoint> points, double x, double y) -> Column<GeoPoint> shifted
  UDTF: ct_pointn__cpu_template(TableFunctionManager, Column<T> points, int64_t n) -> Column<double> x, Column<double> y, T=[GeoLineString, GeoMultiPoint]
  UDTF: ct_copy__cpu_template(TableFunctionManager mgr, Column<T> inputs) -> Column<T> outputs | input_id=args<0>, T=[GeoMultiPoint, GeoLineString, GeoMultiLineString, GeoPolygon, GeoMultiPolygon]
  UDTF: ct_linestringn__cpu_(TableFunctionManager, Column<GeoPolygon> polygons, int64_t n) -> Column<GeoLineString> linestrings
  UDTF: ct_make_polygon3__cpu_(TableFunctionManager, Cursor<Column<GeoLineString> rings, Column<GeoLineString> holes1, Column<GeoLineString> holes2>) -> Column<GeoPolygon> polygons, Column<int> sizes
  UDTF: ct_make_linestring2__cpu_(TableFunctionManager, Cursor<Column<double> x, Column<double> y>, double dx, double dy) -> Column<GeoLineString> linestrings
  UDTF: ct_make_multipolygon__cpu_(TableFunctionManager, Column<GeoPolygon> polygons) -> Column<GeoMultiPolygon> mpolygons
  UDTF: ct_polygonn__cpu_(TableFunctionManager, Column<GeoMultiPolygon> mpolygons, int64_t n) -> Column<GeoPolygon> polygons
  UDTF: ct_to_multilinestring__cpu_(TableFunctionManager, Column<GeoPolygon> polygons) -> Column<GeoMultiLineString> mlinestrings
  UDTF: ct_to_polygon__cpu_(TableFunctionManager, Column<GeoMultiLineString> mlinestrings) -> Column<GeoPolygon> polygons
*/
// clang-format on

EXTENSION_NOINLINE int32_t ct_coords__cpu_(TableFunctionManager& mgr,
                                           const Column<GeoPoint>& points,
                                           Column<double>& xcoords,
                                           Column<double>& ycoords);

EXTENSION_NOINLINE int32_t ct_shift__cpu_(TableFunctionManager& mgr,
                                          const Column<GeoPoint>& points,
                                          const double x,
                                          const double y,
                                          Column<GeoPoint>& shifted_points);

template <typename T>
NEVER_INLINE HOST int32_t ct_pointn__cpu_template(TableFunctionManager& mgr,
                                                  const Column<T>& points,
                                                  int64_t n,
                                                  Column<double>& xcoords,
                                                  Column<double>& ycoords);

template <typename T>
NEVER_INLINE HOST int32_t ct_copy__cpu_template(TableFunctionManager& mgr,
                                                const Column<T>& inputs,
                                                Column<T>& outputs);

EXTENSION_NOINLINE int32_t ct_linestringn__cpu_(TableFunctionManager& mgr,
                                                const Column<GeoPolygon>& polygons,
                                                int64_t n,
                                                Column<GeoLineString>& linestrings);

EXTENSION_NOINLINE int32_t ct_make_polygon3__cpu_(TableFunctionManager& mgr,
                                                  const Column<GeoLineString>& rings,
                                                  const Column<GeoLineString>& holes1,
                                                  const Column<GeoLineString>& holes2,
                                                  Column<GeoPolygon>& polygons,
                                                  Column<int>& sizes);

EXTENSION_NOINLINE int32_t ct_make_linestring2__cpu_(TableFunctionManager& mgr,
                                                     const Column<double>& x,
                                                     const Column<double>& y,
                                                     double dx,
                                                     double dy,
                                                     Column<GeoLineString>& linestrings);

EXTENSION_NOINLINE int32_t ct_make_multipolygon__cpu_(TableFunctionManager& mgr,
                                                      const Column<GeoPolygon>& polygons,
                                                      Column<GeoMultiPolygon>& mpolygons);

EXTENSION_NOINLINE int32_t ct_polygonn__cpu_(TableFunctionManager& mgr,
                                             const Column<GeoMultiPolygon>& mpolygons,
                                             int64_t n,
                                             Column<GeoPolygon>& polygons);

EXTENSION_NOINLINE int32_t
ct_to_multilinestring__cpu_(TableFunctionManager& mgr,
                            const Column<GeoPolygon>& polygons,
                            Column<GeoMultiLineString>& mlinestrings);

EXTENSION_NOINLINE int32_t
ct_to_polygon__cpu_(TableFunctionManager& mgr,
                    const Column<GeoMultiLineString>& mlinestrings,
                    Column<GeoPolygon>& polygons);

#endif  // ifndef __CUDACC__

// clang-format off
/*
  UDTF: row_copier(Column<double>, RowMultiplier) -> Column<double>
  UDTF: row_copier_text(Column<TextEncodingDict>, RowMultiplier) -> Column<TextEncodingDict> | input_id=args<0>
  UDTF: row_copier_columnlist__cpu__(TableFunctionManager, ColumnList<double> cols) -> Column<double>
  UDTF: row_copier2__cpu__(Column<double>, int) -> Column<double>, Column<double>
*/
// clang-format on
EXTENSION_NOINLINE int32_t row_copier(const Column<double>& input_col,
                                      int copy_multiplier,
                                      Column<double>& output_col);

#ifndef __CUDACC__

EXTENSION_NOINLINE_HOST int32_t row_copier2__cpu__(const Column<double>& input_col,
                                                   int copy_multiplier,
                                                   Column<double>& output_col,
                                                   Column<double>& output_col2);

EXTENSION_NOINLINE_HOST int32_t
row_copier_columnlist__cpu__(TableFunctionManager& mgr,
                             const ColumnList<double>& cols,
                             Column<double>& output_col);

#endif  // #ifndef __CUDACC__

EXTENSION_NOINLINE int32_t row_copier_text(const Column<TextEncodingDict>& input_col,
                                           int copy_multiplier,
                                           Column<TextEncodingDict>& output_col);

/*
  UDTF: row_adder(RowMultiplier<1>, Cursor<ColumnDouble, ColumnDouble>) -> ColumnDouble
*/
EXTENSION_NOINLINE int32_t row_adder(const int copy_multiplier,
                                     const Column<double>& input_col1,
                                     const Column<double>& input_col2,
                                     Column<double>& output_col);

// clang-format off
/*
  UDTF: row_addsub(RowMultiplier, Cursor<double, double>) -> Column<double>, Column<double>
*/
// clang-format on
EXTENSION_NOINLINE int32_t row_addsub(const int copy_multiplier,
                                      const Column<double>& input_col1,
                                      const Column<double>& input_col2,
                                      Column<double>& output_col1,
                                      Column<double>& output_col2);

// clang-format off
/*
  UDTF: get_max_with_row_offset__cpu_(Cursor<int>, Constant<1>) -> Column<int>, Column<int>
*/
// clang-format on

#ifndef __CUDACC__

EXTENSION_NOINLINE_HOST int32_t
get_max_with_row_offset__cpu_(const Column<int>& input_col,
                              Column<int>& output_max_col,
                              Column<int>& output_max_row_col);

#endif  // #ifndef __CUDACC__

// clang-format off
/*
  UDTF: column_list_get__cpu_(ColumnList<double>, int, RowMultiplier) -> Column<double>
*/
// clang-format on

#ifndef __CUDACC__

EXTENSION_NOINLINE_HOST int32_t column_list_get__cpu_(const ColumnList<double>& col_list,
                                                      const int index,
                                                      const int m,
                                                      Column<double>& col);

#endif  // #ifndef __CUDACC__

// clang-format off
/*
  UDTF: column_list_first_last(ColumnList<double>, RowMultiplier) -> Column<double>,
  Column<double>
*/
// clang-format on
EXTENSION_NOINLINE int32_t column_list_first_last(const ColumnList<double>& col_list,
                                                  const int m,
                                                  Column<double>& col1,
                                                  Column<double>& col2);

// clang-format off
/*
  UDTF: column_list_row_sum__cpu_(Cursor<ColumnList<int32_t>>) -> Column<int32_t>
*/
// clang-format on

#ifndef __CUDACC__

EXTENSION_NOINLINE_HOST int32_t
column_list_row_sum__cpu_(const ColumnList<int32_t>& input, Column<int32_t>& out);

#endif  // #ifndef __CUDACC__

// clang-format off
/*
  UDTF: ct_test_int_default_arg__template(Column<T>, T x | default = 10, RowMultiplier) -> Column<T>, T=[int8_t, int16_t, int32_t, int64_t]
  UDTF: ct_test_float_default_arg__template(Column<T>, T x | default = 10.0, RowMultiplier) -> Column<T>, T=[float, double]
*/
// clang-format on

template <typename T>
TEMPLATE_NOINLINE int32_t ct_test_int_default_arg__template(const Column<T>& inp,
                                                            const T x,
                                                            const int32_t multiplier,
                                                            Column<T>& out);

template <typename T>
TEMPLATE_NOINLINE int32_t ct_test_float_default_arg__template(const Column<T>& inp,
                                                              const T x,
                                                              const int32_t multiplier,
                                                              Column<T>& out);

#ifndef __CUDACC__

// clang-format off
/*
  UDTF: ct_test_string_default_arg__cpu_(TableFunctionManager, Column<TextEncodingDict>, TextEncodingNone suffix | default="_suffix") -> Column<TextEncodingDict> | input_id=args<0>
*/
// clang-format on

EXTENSION_NOINLINE int32_t
ct_test_string_default_arg__cpu_(TableFunctionManager& mgr,
                                 const Column<TextEncodingDict>& inp,
                                 const TextEncodingNone& suffix,
                                 Column<TextEncodingDict>& out);

// clang-format off
/*
  UDTF: ct_test_func__cpu_1(Cursor<int32_t>, int32, RowMultiplier) -> Column<int32_t>
  UDTF: ct_test_func__cpu_2(Cursor<int32_t>, Cursor<int32_t>, RowMultiplier) -> Column<int32_t>
*/
// clang-format on
// Tests for QE-646
EXTENSION_NOINLINE int32_t ct_test_func__cpu_1(const Column<int32_t>& input1,
                                               const int32_t x,
                                               const int32_t multiplier,
                                               Column<int32_t>& out);

EXTENSION_NOINLINE int32_t ct_test_func__cpu_2(const Column<int32_t>& input1,
                                               const Column<int32_t>& input2,
                                               const int32_t multiplier,
                                               Column<int32_t>& out);

// clang-format off
/*
  UDTF: ct_test_calcite_casting_char__cpu_(TableFunctionManager, Cursor<TextEncodingDict>) -> Column<TextEncodingDict> | input_id=args<0>
  UDTF: ct_test_calcite_casting_bigint__cpu_(TableFunctionManager, Cursor<int64_t>) -> Column<int64_t>
  UDTF: ct_test_calcite_casting_double__cpu_(TableFunctionManager, Cursor<double>) -> Column<double>
  UDTF: ct_test_calcite_casting_timestamp__cpu_(TableFunctionManager, Cursor<Timestamp>) -> Column<Timestamp>
  UDTF: ct_test_calcite_casting_columnlist__template_cpu_(TableFunctionManager, Cursor<Column<T> first, ColumnList<T> list> data) -> Column<T>, T=[float, double]
*/
// clang-format on
// functions to test calcite auto casting

EXTENSION_NOINLINE int32_t
ct_test_calcite_casting_char__cpu_(TableFunctionManager& mgr,
                                   const Column<TextEncodingDict>& input1,
                                   Column<TextEncodingDict>& out);
EXTENSION_NOINLINE int32_t
ct_test_calcite_casting_bigint__cpu_(TableFunctionManager& mgr,
                                     const Column<int64_t>& input1,
                                     Column<int64_t>& out);
EXTENSION_NOINLINE int32_t
ct_test_calcite_casting_double__cpu_(TableFunctionManager& mgr,
                                     const Column<double>& input1,
                                     Column<double>& out);
EXTENSION_NOINLINE int32_t
ct_test_calcite_casting_timestamp__cpu_(TableFunctionManager& mgr,
                                        const Column<Timestamp>& input1,
                                        Column<Timestamp>& out);

template <typename T>
TEMPLATE_NOINLINE int32_t
ct_test_calcite_casting_columnlist__template_cpu_(TableFunctionManager& mgr,
                                                  const Column<T>& first,
                                                  const ColumnList<T>& list,
                                                  Column<T>& out);
#endif
