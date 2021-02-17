/*
  This file contains tesing compile-time UDTFs. The unit-tests are
  implemented within the RBC package.
 */

#define CPU_DEVICE_CODE 0x637075;  // 'cpu' in hex
#define GPU_DEVICE_CODE 0x677075;  // 'gpu' in hex

// clang-format off
/*
  UDTF: ct_device_selection_udtf_any(Cursor<int32_t>) -> Column<int32_t>
  UDTF: ct_device_selection_udtf_cpu__cpu_(Cursor<int32_t>) -> Column<int32_t>
  UDTF: ct_device_selection_udtf_gpu__gpu_(Cursor<int32_t>) -> Column<int32_t>
  UDTF: ct_device_selection_udtf_both__cpu_(Cursor<int32_t>) -> Column<int32_t>
  UDTF: ct_device_selection_udtf_both__gpu_(Cursor<int32_t>) -> Column<int32_t>
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

  UDTF: ct_binding_udtf__cpu_1(Cursor<int32_t>) -> Column<int32_t>
  UDTF: ct_binding_udtf__cpu_2(Cursor<int32_t, int32_t>) -> Column<int32_t>
  UDTF: ct_binding_udtf__cpu_3(Cursor<int32_t, int32_t, int32_t>) -> Column<int32_t>
  UDTF: ct_binding_udtf__cpu_4(Cursor<int64_t, int32_t, int32_t>) -> Column<int32_t>
  UDTF: ct_binding_udtf__cpu_5(Cursor<int64_t, int64_t, int32_t>) -> Column<int32_t>
  UDTF: ct_binding_udtf__cpu_6(Cursor<int64_t, int32_t, int64_t>) -> Column<int32_t>
  UDTF: ct_binding_udtf__cpu_7(Cursor<int32_t, ColumnList<int32_t>>) -> Column<int32_t>
  UDTF: ct_binding_udtf__cpu_8(Cursor<ColumnList<int32_t>, int64_t>) -> Column<int32_t>
  UDTF: ct_binding_udtf__cpu_9(Cursor<ColumnList<int32_t>, ColumnList<int64_t>>) -> Column<int32_t>
  UDTF: ct_binding_udtf__cpu_10(Cursor<int64_t, ColumnList<int64_t>, int64_t>) -> Column<int32_t>


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
  UDTF: ct_binding_udtf__cpu_22(Cursor<int32_t>, RowMultiplier, int32_t) -> Column<int32_t>
  UDTF: ct_binding_udtf__cpu_23(Cursor<ColumnList<int32_t>>, RowMultiplier, int32_t) -> Column<int32_t>
  UDTF: ct_binding_udtf2__cpu_24(Cursor<ColumnList<int32_t>>, int32_t, RowMultiplier) -> Column<int32_t>
  UDTF: ct_binding_udtf3__cpu_25(Cursor<Column<int32_t>>, int32_t, RowMultiplier) -> Column<int32_t>
*/
// clang-format on

EXTENSION_NOINLINE int32_t ct_binding_udtf__cpu_1(const Column<int32_t>& input1,
                                                  Column<int32_t>& out) {
  out[0] = 1;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf__cpu_2(const Column<int32_t>& input1,
                                                  const Column<int32_t>& input2,
                                                  Column<int32_t>& out) {
  out[0] = 11;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf__cpu_3(const Column<int32_t>& input1,
                                                  const Column<int32_t>& input2,
                                                  const Column<int32_t>& input3,
                                                  Column<int32_t>& out) {
  out[0] = 111;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf__cpu_4(const Column<int64_t>& input1,
                                                  const Column<int32_t>& input2,
                                                  const Column<int32_t>& input3,
                                                  Column<int32_t>& out) {
  out[0] = 211;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf__cpu_5(const Column<int64_t>& input1,
                                                  const Column<int64_t>& input2,
                                                  const Column<int32_t>& input3,
                                                  Column<int32_t>& out) {
  out[0] = 221;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf__cpu_6(const Column<int64_t>& input1,
                                                  const Column<int32_t>& input2,
                                                  const Column<int64_t>& input3,
                                                  Column<int32_t>& out) {
  out[0] = 212;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf__cpu_7(const Column<int32_t>& input1,
                                                  const ColumnList<int32_t>& input2,
                                                  Column<int32_t>& out) {
  out[0] = 13;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf__cpu_8(const ColumnList<int32_t>& input1,
                                                  const Column<int64_t>& input2,
                                                  Column<int32_t>& out) {
  out[0] = 32;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf__cpu_9(const ColumnList<int32_t>& input1,
                                                  const ColumnList<int64_t>& input2,
                                                  Column<int32_t>& out) {
  out[0] = 34;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf__cpu_10(const Column<int64_t>& input1,
                                                   const ColumnList<int64_t>& input2,
                                                   const Column<int64_t>& input3,
                                                   Column<int64_t>& out) {
  out[0] = 242;
  return 1;
}

EXTENSION_NOINLINE int32_t ct_binding_udtf__cpu_11(const Column<int32_t>& input1,
                                                   const int32_t multiplier,
                                                   Column<int32_t>& out) {
  out[0] = 1000 + 19;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf__cpu_12(const Column<int32_t>& input1,
                                                   const Column<int32_t>& input2,
                                                   const int32_t multiplier,
                                                   Column<int32_t>& out) {
  out[0] = 1000 + 119;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf__cpu_13(const Column<int32_t>& input1,
                                                   const Column<int32_t>& input2,
                                                   const Column<int32_t>& input3,
                                                   const int32_t multiplier,
                                                   Column<int32_t>& out) {
  out[0] = 1000 + 1119;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf__cpu_14(const Column<int64_t>& input1,
                                                   const Column<int32_t>& input2,
                                                   const Column<int32_t>& input3,
                                                   const int32_t multiplier,
                                                   Column<int32_t>& out) {
  out[0] = 1000 + 2119;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf__cpu_15(const Column<int64_t>& input1,
                                                   const Column<int64_t>& input2,
                                                   const Column<int32_t>& input3,
                                                   const int32_t multiplier,
                                                   Column<int32_t>& out) {
  out[0] = 1000 + 2219;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf__cpu_16(const Column<int64_t>& input1,
                                                   const Column<int32_t>& input2,
                                                   const Column<int64_t>& input3,
                                                   const int32_t multiplier,
                                                   Column<int32_t>& out) {
  out[0] = 1000 + 2129;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf__cpu_17(const Column<int32_t>& input1,
                                                   const ColumnList<int32_t>& input2,
                                                   const int32_t multiplier,
                                                   Column<int32_t>& out) {
  out[0] = 1000 + 139;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf__cpu_18(const ColumnList<int32_t>& input1,
                                                   const Column<int64_t>& input2,
                                                   const int32_t multiplier,
                                                   Column<int32_t>& out) {
  out[0] = 1000 + 329;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf__cpu_19(const ColumnList<int32_t>& input1,
                                                   const ColumnList<int64_t>& input2,
                                                   const int32_t multiplier,
                                                   Column<int32_t>& out) {
  out[0] = 1000 + 349;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf__cpu_20(const Column<int64_t>& input1,
                                                   const ColumnList<int64_t>& input2,
                                                   const Column<int64_t>& input3,
                                                   const int32_t multiplier,
                                                   Column<int64_t>& out) {
  out[0] = 1000 + 2429;
  return 1;
}

EXTENSION_NOINLINE int32_t ct_binding_udtf2__cpu_21(const int32_t multiplier,
                                                    const Column<int32_t>& input1,
                                                    Column<int32_t>& out) {
  out[0] = 1000 + 91;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf__cpu_22(const Column<int32_t>& input1,
                                                   const int32_t multiplier,
                                                   const int32_t input2,
                                                   Column<int32_t>& out) {
  out[0] = 1000 + 196;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf__cpu_23(const ColumnList<int32_t>& input1,
                                                   const int32_t multiplier,
                                                   const int32_t input2,
                                                   Column<int32_t>& out) {
  out[0] = 1000 + 396;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf2__cpu_24(const ColumnList<int32_t>& input1,
                                                    const int32_t input2,
                                                    const int32_t multiplier,
                                                    Column<int32_t>& out) {
  out[0] = 1000 + 369;
  return 1;
}
EXTENSION_NOINLINE int32_t ct_binding_udtf3__cpu_25(const Column<int32_t>& input1,
                                                    const int32_t input2,
                                                    const int32_t multiplier,
                                                    Column<int32_t>& out) {
  out[0] = 1000 + 169;
  return 1;
}
