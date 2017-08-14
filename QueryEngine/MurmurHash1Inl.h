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

FORCE_INLINE DEVICE uint64_t MurmurHash64AImpl(const void* key, int len, uint64_t seed) {
  const uint64_t m = 0xc6a4a7935bd1e995LLU;
  const int r = 47;

  uint64_t h = seed ^ (len * m);

  const uint64_t* data = (const uint64_t*)key;
  const uint64_t* end = data + (len / 8);

  while (data != end) {
    uint64_t k = *data++;

    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;
  }

  const unsigned char* data2 = (const unsigned char*)data;

  switch (len & 7) {
    case 7:
      h ^= ((uint64_t)data2[6]) << 48;
    case 6:
      h ^= ((uint64_t)data2[5]) << 40;
    case 5:
      h ^= ((uint64_t)data2[4]) << 32;
    case 4:
      h ^= ((uint64_t)data2[3]) << 24;
    case 3:
      h ^= ((uint64_t)data2[2]) << 16;
    case 2:
      h ^= ((uint64_t)data2[1]) << 8;
    case 1:
      h ^= ((uint64_t)data2[0]);
      h *= m;
  };

  h ^= h >> r;
  h *= m;
  h ^= h >> r;

  return h;
}

#endif  // QUERYENGINE_MURMURHASH1INL_H
