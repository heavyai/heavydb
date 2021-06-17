/*

  Compile-time UDFs. Used in RBC: rbc/tests/test_omnisci_device_selection.py

 */

#define CPU_DEVICE_CODE 0x637075  // 'cpu' in hex
#define GPU_DEVICE_CODE 0x677075  // 'gpu' in hex

EXTENSION_NOINLINE
int32_t ct_device_selection_udf_any(int32_t input) {
#ifdef __CUDACC__
  return GPU_DEVICE_CODE;
#else
  return CPU_DEVICE_CODE;
#endif
}

EXTENSION_NOINLINE
int32_t ct_device_selection_udf_cpu__cpu_(int32_t input) {
  return CPU_DEVICE_CODE;
}

EXTENSION_NOINLINE
int32_t ct_device_selection_udf_gpu__gpu_(int32_t input) {
  return GPU_DEVICE_CODE;
}

EXTENSION_NOINLINE
int32_t ct_device_selection_udf_both__cpu_(int32_t input) {
  return CPU_DEVICE_CODE;
}

EXTENSION_NOINLINE
int32_t ct_device_selection_udf_both__gpu_(int32_t input) {
  return GPU_DEVICE_CODE;
}

#undef CPU_DEVICE_CODE
#undef GPU_DEVICE_CODE
