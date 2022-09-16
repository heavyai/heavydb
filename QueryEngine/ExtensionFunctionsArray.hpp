#include <cassert>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif

#ifndef __CUDACC__

namespace {

template <typename T>
DEVICE ALWAYS_INLINE Array<T> array_append_impl(const Array<T>& in_arr, T val) {
  Array<T> out_arr(in_arr.getSize() + 1);
  for (int64_t i = 0; i < in_arr.getSize(); i++) {
    out_arr[i] = in_arr(i);
  }
  out_arr[in_arr.getSize()] = val;
  return out_arr;
}

// while appending boolean value to bool-type array we need to deal with its
// array storage carefully to correctly represent null sentinel for bool type array
DEVICE ALWAYS_INLINE Array<bool> barray_append_impl(const Array<bool>& in_arr,
                                                    const int8_t val) {
  Array<bool> out_arr(in_arr.getSize() + 1);
  // cast bool array storage to int8_t type to mask null elem correctly
  auto casted_out_arr = (int8_t*)out_arr.ptr;
  for (int64_t i = 0; i < in_arr.getSize(); i++) {
    casted_out_arr[i] = in_arr(i);
  }
  casted_out_arr[in_arr.getSize()] = val;
  return out_arr;
}

}  // namespace

#endif

#ifdef _WIN32
// MSVC doesn't allow extern "C" function using template type
// without explicit instantiation
template struct Array<bool>;
template struct Array<int8_t>;
template struct Array<int16_t>;
template struct Array<int32_t>;
template struct Array<int64_t>;
template struct Array<float>;
template struct Array<double>;
#endif

EXTENSION_NOINLINE Array<int64_t> array_append(const Array<int64_t>& in_arr,
                                               const int64_t val) {
#ifndef __CUDACC__
  return array_append_impl(in_arr, val);
#else
  assert(false);
  return Array<int64_t>(0, true);
#endif
}

EXTENSION_NOINLINE Array<int32_t> array_append__(const Array<int32_t>& in_arr,
                                                 const int32_t val) {
#ifndef __CUDACC__
  return array_append_impl(in_arr, val);
#else
  assert(false);
  return Array<int32_t>(0, true);
#endif
}

EXTENSION_NOINLINE Array<int16_t> array_append__1(const Array<int16_t>& in_arr,
                                                  const int16_t val) {
#ifndef __CUDACC__
  return array_append_impl(in_arr, val);
#else
  assert(false);
  return Array<int16_t>(0, true);
#endif
}

EXTENSION_NOINLINE Array<int8_t> array_append__2(const Array<int8_t>& in_arr,
                                                 const int8_t val) {
#ifndef __CUDACC__
  return array_append_impl(in_arr, val);
#else
  assert(false);
  return Array<int8_t>(0, true);
#endif
}

EXTENSION_NOINLINE Array<double> array_append__3(const Array<double>& in_arr,
                                                 const double val) {
#ifndef __CUDACC__
  return array_append_impl(in_arr, val);
#else
  assert(false);
  return Array<double>(0, true);
#endif
}

EXTENSION_NOINLINE Array<float> array_append__4(const Array<float>& in_arr,
                                                const float val) {
#ifndef __CUDACC__
  return array_append_impl(in_arr, val);
#else
  assert(false);
  return Array<float>(0, true);
#endif
}

/*
  Overloading UDFs works for types in the same SQL family.  BOOLEAN
  does not belong to NUMERIC family, hence we need to use different
  name for boolean UDF.
 */
EXTENSION_NOINLINE Array<bool> barray_append(const Array<bool>& in_arr, const bool val) {
#ifndef __CUDACC__
  // we need to cast 'val' to int8_t type to represent null sentinel correctly
  // i.e., NULL_BOOLEAN = -128
  return barray_append_impl(in_arr, val);
#else
  assert(false);
  return Array<bool>(0, true);
#endif
}

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
