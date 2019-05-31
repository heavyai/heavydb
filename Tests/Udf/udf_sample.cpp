#include <cstdint>
#include <limits>
#include <type_traits>

#if defined(__clang__) && defined(__CUDA__) && defined(__CUDA_ARCH__)
#define DEVICE __device__
#else
#define DEVICE
#endif

#if defined(__clang__) && defined(__CUDA__) && defined(__CUDA_ARCH__)
#define NEVER_INLINE
#else
#define NEVER_INLINE __attribute__((noinline))
#endif

#if defined(__clang__) && defined(__CUDA__) && defined(__CUDA_ARCH__)
#define ALWAYS_INLINE
#else
#define ALWAYS_INLINE __attribute__((always_inline))
#endif

#define EXTENSION_NOINLINE extern "C" NEVER_INLINE DEVICE

#define EXTENSION_INLINE extern "C" ALWAYS_INLINE DEVICE

// use std::size_t;

template <typename T>
struct Array {
  T* ptr;
  std::size_t sz;
  bool is_null;

  DEVICE T operator()(const std::size_t index) {
    if (index < sz)
      return ptr[index];
    else
      return 0;  // see array_at
  }

  DEVICE std::size_t getSize() const { return sz; }

  DEVICE bool isNull() const { return is_null; }

  DEVICE constexpr inline T null_value() {
    return std::is_signed<T>::value ? std::numeric_limits<T>::min()
                                    : std::numeric_limits<T>::max();
  }
};

EXTENSION_NOINLINE
bool array_is_null_double(Array<double> arr) {
  return arr.isNull();
}

EXTENSION_NOINLINE
int32_t array_sz_double(Array<double> arr) {
  return arr.getSize();
}

EXTENSION_NOINLINE
double array_at_double(Array<double> arr, std::size_t idx) {
  return arr(idx);
}

EXTENSION_NOINLINE
bool array_is_null_int32(Array<int32_t> arr) {
  return arr.isNull();
}

EXTENSION_NOINLINE
int32_t array_sz_int32(Array<int32_t> arr) {
  return (int32_t)arr.getSize();
}

EXTENSION_NOINLINE
int32_t array_at_int32(Array<int32_t> arr, std::size_t idx) {
  return arr(idx);
}

EXTENSION_NOINLINE
int8_t array_at_int32_is_null(Array<int32_t> arr, std::size_t idx) {
  return (int8_t)(array_at_int32(arr, idx) == arr.null_value());
}

EXTENSION_NOINLINE
bool array_is_null_int64(Array<int64_t> arr) {
  return arr.isNull();
}

EXTENSION_NOINLINE
int64_t array_sz_int64(Array<int64_t> arr) {
  return arr.getSize();
}

EXTENSION_NOINLINE
int64_t array_at_int64(Array<int64_t> arr, std::size_t idx) {
  return arr(idx);
}

EXTENSION_NOINLINE
int8_t array_at_int64_is_null(Array<int64_t> arr, std::size_t idx) {
  return (int8_t)(array_at_int64(arr, idx) == arr.null_value());
}

EXTENSION_NOINLINE
int32_t udf_diff(const int32_t x, const int32_t y) {
  return x - y;
}

EXTENSION_NOINLINE
double udf_range(const double high_price, const double low_price) {
  return high_price - low_price;
}

EXTENSION_NOINLINE
int64_t udf_range_int(const int64_t high_price, const int64_t low_price) {
  return high_price - low_price;
}

EXTENSION_NOINLINE
double udf_truehigh(const double high_price, const double prev_close_price) {
  return (high_price < prev_close_price) ? prev_close_price : high_price;
}

EXTENSION_NOINLINE
double udf_truelow(const double low_price, const double prev_close_price) {
  return !(prev_close_price < low_price) ? low_price : prev_close_price;
}

EXTENSION_NOINLINE
double udf_truerange(const double high_price,
                     const double low_price,
                     const double prev_close_price) {
  return (udf_truehigh(high_price, prev_close_price) -
          udf_truelow(low_price, prev_close_price));
}
