#define CPU_DEVICE_CODE 0x637075;  // 'cpu' in hex
#define GPU_DEVICE_CODE 0x677075;  // 'gpu' in hex

/*
  UDTF: ct_device_selection_udtf_any(Cursor<int32_t>) -> Column<int32_t>
  UDTF: ct_device_selection_udtf_cpu__cpu_(Cursor<int32_t>) -> Column<int32_t>
  UDTF: ct_device_selection_udtf_gpu__gpu_(Cursor<int32_t>) -> Column<int32_t>
  UDTF: ct_device_selection_udtf_both__cpu_(Cursor<int32_t>) -> Column<int32_t>
  UDTF: ct_device_selection_udtf_both__gpu_(Cursor<int32_t>) -> Column<int32_t>
*/

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
