#include <cassert>

#ifndef __CUDACC__

namespace {

template <typename T>
DEVICE ALWAYS_INLINE Array<T> array_append_impl(const Array<T> in_arr, T val) {
  Array<T> out_arr(in_arr.getSize() + 1);
  for (int64_t i = 0; i < in_arr.getSize(); i++) {
    out_arr[i] = in_arr(i);
  }
  out_arr[in_arr.getSize()] = val;
  return out_arr;
}

}  // namespace

#endif

EXTENSION_NOINLINE Array<int64_t> array_append(const Array<int64_t> in_arr,
                                               const int64_t val) {
#ifndef __CUDACC__
  return array_append_impl(in_arr, val);
#else
  assert(false);
  return Array<int64_t>(0, true);
#endif
}

EXTENSION_NOINLINE Array<int32_t> array_append__(const Array<int32_t> in_arr,
                                                 const int32_t val) {
#ifndef __CUDACC__
  return array_append_impl(in_arr, val);
#else
  assert(false);
  return Array<int32_t>(0, true);
#endif
}
