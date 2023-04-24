#include <cassert>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif

#ifndef __CUDACC__

namespace {

template <typename T>
DEVICE ALWAYS_INLINE Array<T> array_append_impl(const Array<T>& in_arr, T val) {
  Array<T> out_arr(in_arr.size() + 1);
  for (size_t i = 0; i < in_arr.size(); i++) {
    out_arr[i] = in_arr(i);
  }
  out_arr[in_arr.size()] = val;
  return out_arr;
}

// while appending boolean value to bool-type array we need to deal with its
// array storage carefully to correctly represent null sentinel for bool type array
DEVICE ALWAYS_INLINE Array<bool> barray_append_impl(const Array<bool>& in_arr,
                                                    const int8_t val) {
  Array<bool> out_arr(in_arr.size() + 1);
  // cast bool array storage to int8_t type to mask null elem correctly
  auto casted_out_arr = (int8_t*)out_arr.data();
  for (size_t i = 0; i < in_arr.size(); i++) {
    casted_out_arr[i] = in_arr(i);
  }
  casted_out_arr[in_arr.size()] = val;
  return out_arr;
}

template <typename T>
DEVICE ALWAYS_INLINE Array<T> array_first_half_impl(const Array<T>& in_arr) {
  auto sz = in_arr.size();
  Array<T> out_arr(sz / 2, in_arr.isNull());
  for (size_t i = 0; i < sz / 2; i++) {
    out_arr[i] = in_arr(i);
  }
  return out_arr;
}

template <typename T>
DEVICE ALWAYS_INLINE Array<T> array_second_half_impl(const Array<T>& in_arr) {
  auto sz = in_arr.size();
  Array<T> out_arr(sz - sz / 2, in_arr.isNull());
  for (size_t i = sz / 2; i < sz; i++) {
    out_arr[i - sz / 2] = in_arr(i);
  }
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
template struct Array<TextEncodingDict>;
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

#ifndef __CUDACC__
EXTENSION_NOINLINE Array<TextEncodingDict> tarray_append(
    RowFunctionManager& mgr,
    const Array<TextEncodingDict>& in_arr,
    const TextEncodingDict val) {
  Array<TextEncodingDict> out_arr(in_arr.size() + 1);
  for (size_t i = 0; i < in_arr.size(); i++) {
    if (in_arr.isNull(i)) {
      out_arr[i] = in_arr[i];
    } else {
      std::string str =
          mgr.getString(GET_DICT_DB_ID(mgr, 0), GET_DICT_ID(mgr, 0), in_arr[i]);
      out_arr[i] = mgr.getOrAddTransient(TRANSIENT_DICT_DB_ID, TRANSIENT_DICT_ID, str);
    }
  }
  if (val.isNull()) {
    out_arr[in_arr.size()] = val;
  } else {
    std::string str = mgr.getString(GET_DICT_DB_ID(mgr, 1), GET_DICT_ID(mgr, 1), val);
    out_arr[in_arr.size()] =
        mgr.getOrAddTransient(TRANSIENT_DICT_DB_ID, TRANSIENT_DICT_ID, str);
  }
  return out_arr;
}
#endif

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

EXTENSION_NOINLINE Array<bool> array_first_half__b8(const Array<bool>& in_arr) {
#ifndef __CUDACC__
  return array_first_half_impl(in_arr);
#else
  assert(false);
  return Array<bool>(0, true);
#endif
}

EXTENSION_NOINLINE Array<bool> array_second_half__b8(const Array<bool>& in_arr) {
#ifndef __CUDACC__
  return array_second_half_impl(in_arr);
#else
  assert(false);
  return Array<bool>(0, true);
#endif
}

EXTENSION_NOINLINE Array<int8_t> array_first_half__i8(const Array<int8_t>& in_arr) {
#ifndef __CUDACC__
  return array_first_half_impl(in_arr);
#else
  assert(false);
  return Array<int8_t>(0, true);
#endif
}

EXTENSION_NOINLINE Array<int8_t> array_second_half__i8(const Array<int8_t>& in_arr) {
#ifndef __CUDACC__
  return array_second_half_impl(in_arr);
#else
  assert(false);
  return Array<int8_t>(0, true);
#endif
}

EXTENSION_NOINLINE Array<int16_t> array_first_half__i16(const Array<int16_t>& in_arr) {
#ifndef __CUDACC__
  return array_first_half_impl(in_arr);
#else
  assert(false);
  return Array<int16_t>(0, true);
#endif
}

EXTENSION_NOINLINE Array<int16_t> array_second_half__i16(const Array<int16_t>& in_arr) {
#ifndef __CUDACC__
  return array_second_half_impl(in_arr);
#else
  assert(false);
  return Array<int16_t>(0, true);
#endif
}

EXTENSION_NOINLINE Array<int32_t> array_first_half__i32(const Array<int32_t>& in_arr) {
#ifndef __CUDACC__
  return array_first_half_impl(in_arr);
#else
  assert(false);
  return Array<int32_t>(0, true);
#endif
}

EXTENSION_NOINLINE Array<int32_t> array_second_half__i32(const Array<int32_t>& in_arr) {
#ifndef __CUDACC__
  return array_second_half_impl(in_arr);
#else
  assert(false);
  return Array<int32_t>(0, true);
#endif
}

EXTENSION_NOINLINE Array<int64_t> array_first_half__i64(const Array<int64_t>& in_arr) {
#ifndef __CUDACC__
  return array_first_half_impl(in_arr);
#else
  assert(false);
  return Array<int64_t>(0, true);
#endif
}

EXTENSION_NOINLINE Array<int64_t> array_second_half__i64(const Array<int64_t>& in_arr) {
#ifndef __CUDACC__
  return array_second_half_impl(in_arr);
#else
  assert(false);
  return Array<int64_t>(0, true);
#endif
}

EXTENSION_NOINLINE Array<float> array_first_half__f32(const Array<float>& in_arr) {
#ifndef __CUDACC__
  return array_first_half_impl(in_arr);
#else
  assert(false);
  return Array<float>(0, true);
#endif
}

EXTENSION_NOINLINE Array<float> array_second_half__f32(const Array<float>& in_arr) {
#ifndef __CUDACC__
  return array_second_half_impl(in_arr);
#else
  assert(false);
  return Array<float>(0, true);
#endif
}

EXTENSION_NOINLINE Array<double> array_first_half__f64(const Array<double>& in_arr) {
#ifndef __CUDACC__
  return array_first_half_impl(in_arr);
#else
  assert(false);
  return Array<double>(0, true);
#endif
}

EXTENSION_NOINLINE Array<double> array_second_half__f64(const Array<double>& in_arr) {
#ifndef __CUDACC__
  return array_second_half_impl(in_arr);
#else
  assert(false);
  return Array<double>(0, true);
#endif
}

#ifdef _WIN32
template struct Array<TextEncodingDict>;
#endif

#ifndef __CUDACC__
EXTENSION_NOINLINE
Array<TextEncodingDict> array_first_half__t32(RowFunctionManager& mgr,
                                              const Array<TextEncodingDict>& in_arr) {
  Array<TextEncodingDict> out_arr = array_first_half_impl(in_arr);
  for (size_t i = 0; i < out_arr.size(); i++) {
    if (!out_arr.isNull(i)) {
      std::string str =
          mgr.getString(GET_DICT_DB_ID(mgr, 0), GET_DICT_ID(mgr, 0), out_arr[i]);
      out_arr[i] = mgr.getOrAddTransient(TRANSIENT_DICT_DB_ID, TRANSIENT_DICT_ID, str);
    }
  }
  return out_arr;
}

EXTENSION_NOINLINE
Array<TextEncodingDict> array_second_half__t32(RowFunctionManager& mgr,
                                               const Array<TextEncodingDict>& in_arr) {
  Array<TextEncodingDict> out_arr = array_second_half_impl(in_arr);
  for (size_t i = 0; i < out_arr.size(); i++) {
    if (!out_arr.isNull(i)) {
      std::string str =
          mgr.getString(GET_DICT_DB_ID(mgr, 0), GET_DICT_ID(mgr, 0), out_arr[i]);
      out_arr[i] = mgr.getOrAddTransient(TRANSIENT_DICT_DB_ID, TRANSIENT_DICT_ID, str);
    }
  }
  return out_arr;
}
#endif

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
