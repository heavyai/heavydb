#include "heavydbTypes.h"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif

EXTENSION_NOINLINE int64_t raw_array_as_scalar_same_type(const int64_t* in_arr,
                                                         const int64_t val) {
  // return sum of array and val as array
  int64_t scalar{};
  for (int64_t i = 0; i < val; i++) {
    scalar += in_arr[i];
  }
  scalar += val;
  return scalar;
}

EXTENSION_NOINLINE int32_t raw_array_as_scalar_diff_type(const int64_t* in_arr,
                                                         const int64_t val) {
  // return sum of array and val as array
  int32_t scalar{};
  for (int64_t i = 0; i < val; i++) {
    scalar += static_cast<int32_t>(in_arr[i]);
  }
  scalar += static_cast<int32_t>(val);
  return scalar;
}

EXTENSION_NOINLINE Array<int64_t> raw_array_as_array_same_type(const int64_t* in_arr,
                                                               const int64_t val) {
  // return array with val appended as array
  auto array = Array<int64_t>(val + 1, false);
  for (int64_t i = 0; i < val; i++) {
    array.ptr[i] = in_arr[i];
  }
  array.ptr[val] = val;
  return array;
}

EXTENSION_NOINLINE Array<int32_t> raw_array_as_array_diff_type(const int64_t* in_arr,
                                                               const int64_t val) {
  // return array with val appended as array
  auto array = Array<int32_t>(val + 1, false);
  for (int64_t i = 0; i < val; i++) {
    array.ptr[i] = static_cast<int32_t>(in_arr[i]);
  }
  array.ptr[val] = static_cast<int32_t>(val);
  return array;
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif
