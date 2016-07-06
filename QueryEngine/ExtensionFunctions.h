#include "../Shared/funcannotations.h"
#ifndef __CUDACC__
#include <cstdint>
#endif

#define EXTENSION_NOINLINE extern "C" NEVER_INLINE DEVICE

/* Example extension function:
 *
 * EXTENSION_NOINLINE
 * int32_t diff(const int32_t x, const int32_t y) {
 *   return x - y;
 * }
 */
