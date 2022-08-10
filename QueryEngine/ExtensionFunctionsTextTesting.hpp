#include "heavydbTypes.h"

EXTENSION_NOINLINE int32_t text_encoding_none_length(const TextEncodingNone& t) {
  return t.size();
}

EXTENSION_NOINLINE TextEncodingNone text_encoding_none_copy(const TextEncodingNone& t) {
#ifndef __CUDACC__
  return TextEncodingNone(t.getString());
#else
  return TextEncodingNone();
#endif
}

EXTENSION_NOINLINE TextEncodingNone
text_encoding_none_concat(const TextEncodingNone& t1, const TextEncodingNone& t2) {
#ifndef __CUDACC__
  return TextEncodingNone(t1.getString() + ' ' + t2.getString());
#else
  return TextEncodingNone();
#endif
}
