#ifndef QUERYENGINE_MURMURHASH1INL_H
#define QUERYENGINE_MURMURHASH1INL_H

#include "../Shared/funcannotations.h"

FORCE_INLINE DEVICE uint32_t MurmurHash1Impl(const void* key, int len, const uint32_t seed) {
  const unsigned int m = 0xc6a4a793;

  const int r = 16;

  unsigned int h = seed ^ (len * m);

  //----------

  const unsigned char* data = (const unsigned char*)key;

  while (len >= 4) {
    unsigned int k = *(unsigned int*)data;

    h += k;
    h *= m;
    h ^= h >> 16;

    data += 4;
    len -= 4;
  }

  //----------

  switch (len) {
    case 3:
      h += data[2] << 16;
    case 2:
      h += data[1] << 8;
    case 1:
      h += data[0];
      h *= m;
      h ^= h >> r;
  };

  //----------

  h *= m;
  h ^= h >> 10;
  h *= m;
  h ^= h >> 17;

  return h;
}

#endif  // QUERYENGINE_MURMURHASH1INL_H
