/*

  Load-time UDFs. Used in RBC: rbc/tests/test_omnisci_device_selection.py

 */

#include <cstdint>

#define CPU_DEVICE_CODE 0x637075  // 'cpu' in hex
#define GPU_DEVICE_CODE 0x677075  // 'gpu' in hex

#ifdef __CUDACC__
#define DEVICE __device__
#define NEVER_INLINE
#define ALWAYS_INLINE
#define DEVICE_CODE GPU_DEVICE_CODE
#else
#define DEVICE
#define NEVER_INLINE __attribute__((noinline))
#define ALWAYS_INLINE __attribute__((always_inline))
#define DEVICE_CODE CPU_DEVICE_CODE
#endif
#define EXTENSION_NOINLINE extern "C" NEVER_INLINE DEVICE

EXTENSION_NOINLINE
int32_t lt_device_selection_udf_any(int32_t input) {
  return DEVICE_CODE;
}

EXTENSION_NOINLINE
int32_t lt_device_selection_udf_cpu__cpu_(int32_t input) {
  return CPU_DEVICE_CODE;
}

EXTENSION_NOINLINE
int32_t lt_device_selection_udf_gpu__gpu_(int32_t input) {
  return GPU_DEVICE_CODE;
}

EXTENSION_NOINLINE
int32_t lt_device_selection_udf_both__cpu_(int32_t input) {
  return CPU_DEVICE_CODE;
}

EXTENSION_NOINLINE
int32_t lt_device_selection_udf_both__gpu_(int32_t input) {
  return GPU_DEVICE_CODE;
}
