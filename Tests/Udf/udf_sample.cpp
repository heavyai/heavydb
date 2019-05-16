#include <cstdint>
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
