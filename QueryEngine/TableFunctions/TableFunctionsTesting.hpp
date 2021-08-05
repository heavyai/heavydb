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
  UDTF: ct_binding_template__template(Cursor<TextEncodingDict>) -> Column<TextEncodingDict> | input_id=args<0>
  UDTF: ct_binding_template__template(Cursor<int>) -> Column<int>
  UDTF: ct_binding_template__template(Cursor<float>) -> Column<float>
*/
// clang-format on
template <typename T>
int32_t ct_binding_template__template(const Column<T>& input, Column<T>& out) {
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
int32_t ct_binding_columnlist__cpu_template(const Column<T>& input1,
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
int32_t ct_binding_column__cpu_template(const Column<T>& input, Column<int32_t>& out) {
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
int32_t ct_binding_scalar_multiply__cpu_template(const Column<T>& input,
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
int32_t sort_column_limit__cpu_template(const Column<T>& input,
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

// clang-format off
/*
  UDTF: ct_binding_column2__cpu_template(Column<T>, Column<U>) -> Column<K>, T=[int32_t, double], U=[int32_t, double], K=[int32_t]
  UDTF: ct_binding_column2__cpu_template(Column<T>, Column<T>) -> Column<T> | input_id=args<0>, T=[TextEncodingDict]
*/
// clang-format on
template <typename T, typename U, typename K>
int32_t ct_binding_column2__cpu_template(const Column<T>& input1,
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
