#include <stdint.h>
#include "../Shared/funcannotations.h"

#ifdef EXECUTE_INCLUDE

extern "C" NEVER_INLINE DEVICE int32_t char_length_encoded(const char* str, const int32_t str_len) {  // assumes utf8
  int32_t i = 0, char_count = 0;
  while (i < str_len) {
    const unsigned char ch_masked = str[i] & 0xc0;
    if (ch_masked != 0x80) {
      char_count++;
    }
    i++;
  }
  return char_count;
}

extern "C" NEVER_INLINE DEVICE int32_t char_length_encoded_nullable(const char* str,
                                                                    const int32_t str_len,
                                                                    const int32_t int_null) {  // assumes utf8
  if (!str) {
    return int_null;
  }
  return char_length_encoded(str, str_len);
}

#endif  // EXECUTE_INCLUDE
