#ifndef QUERYENGINE_MURMURHASH_H
#define QUERYENGINE_MURMURHASH_H

#include "../Shared/funcannotations.h"
#include <stdint.h>

extern "C" NEVER_INLINE DEVICE uint32_t MurmurHash1(const void* key, int len, const uint32_t seed);

extern "C" NEVER_INLINE DEVICE uint64_t MurmurHash64A(const void* key, int len, uint64_t seed);

#endif  // QUERYENGINE_MURMURHASH_H
