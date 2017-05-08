/*
 * Copyright 2017 MapD Technologies, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file		StringLike.cpp
 * @author	Wei Hong <wei@mapd.com>
 * @brief		Functions to support the LIKE and ILIKE operator in SQL.  Only
 * single-byte character set is supported for now.
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/

#include "StringLike.h"

enum LikeStatus {
  kLIKE_TRUE,
  kLIKE_FALSE,
  kLIKE_ABORT,  // means we run out of string characters to match against pattern, can abort early
  kLIKE_ERROR   // error condition
};

DEVICE static int inline lowercase(char c) {
  if ('A' <= c && c <= 'Z')
    return 'a' + (c - 'A');
  return c;
}

extern "C" DEVICE bool string_like_simple(const char* str,
                                          const int32_t str_len,
                                          const char* pattern,
                                          const int32_t pat_len) {
  int i, j;
  int search_len = str_len - pat_len + 1;
  for (i = 0; i < search_len; ++i) {
    for (j = 0; j < pat_len && pattern[j] == str[j + i]; ++j) {
    }
    if (j >= pat_len) {
      return true;
    }
  }
  return false;
}

extern "C" DEVICE bool string_ilike_simple(const char* str,
                                           const int32_t str_len,
                                           const char* pattern,
                                           const int32_t pat_len) {
  int i, j;
  int search_len = str_len - pat_len + 1;
  for (i = 0; i < search_len; ++i) {
    for (j = 0; j < pat_len && pattern[j] == lowercase(str[j + i]); ++j) {
    }
    if (j >= pat_len) {
      return true;
    }
  }
  return false;
}

#define STR_LIKE_SIMPLE_NULLABLE(base_func)                                                                     \
  extern "C" DEVICE int8_t base_func##_nullable(                                                                \
      const char* lhs, const int32_t lhs_len, const char* rhs, const int32_t rhs_len, const int8_t bool_null) { \
    if (!lhs || !rhs) {                                                                                         \
      return bool_null;                                                                                         \
    }                                                                                                           \
    return base_func(lhs, lhs_len, rhs, rhs_len) ? 1 : 0;                                                       \
  }

STR_LIKE_SIMPLE_NULLABLE(string_like_simple)
STR_LIKE_SIMPLE_NULLABLE(string_ilike_simple)

#undef STR_LIKE_SIMPLE_NULLABLE

// internal recursive function for performing LIKE matching.
// when is_ilike is true, pattern is assumed to be already converted to all lowercase
DEVICE static LikeStatus string_like_match(const char* str,
                                           const int32_t str_len,
                                           const char* pattern,
                                           const int32_t pat_len,
                                           const char escape_char,
                                           const bool is_ilike) {
  const char* s = str;
  int slen = str_len;
  const char* p = pattern;
  int plen = pat_len;

  while (slen > 0 && plen > 0) {
    if (*p == escape_char) {
      // next pattern char must match literally, whatever it is
      p++;
      plen--;
      if (plen <= 0)
        return kLIKE_ERROR;
      if ((!is_ilike && *s != *p) || (is_ilike && lowercase(*s) != *p))
        return kLIKE_FALSE;
    } else if (*p == '%') {
      char firstpat;
      p++;
      plen--;
      while (plen > 0) {
        if (*p == '%') {
          p++;
          plen--;
        } else if (*p == '_') {
          if (slen <= 0)
            return kLIKE_ABORT;
          s++;
          slen--;
          p++;
          plen--;
        } else
          break;
      }
      if (plen <= 0)
        return kLIKE_TRUE;
      if (*p == escape_char) {
        if (plen < 2)
          return kLIKE_ERROR;
        firstpat = p[1];
      } else
        firstpat = *p;

      while (slen > 0) {
        bool match = false;
        if (firstpat == '[' && *p != escape_char) {
          const char* pp = p + 1;
          int pplen = plen - 1;
          while (pplen > 0 && *pp != ']') {
            if ((!is_ilike && *s == *pp) || (is_ilike && lowercase(*s) == *pp)) {
              match = true;
              break;
            }
            pp++;
            pplen--;
          }
          if (pplen <= 0)
            return kLIKE_ERROR;  // malformed
        } else if ((!is_ilike && *s == firstpat) || (is_ilike && lowercase(*s) == firstpat)) {
          match = true;
        }
        if (match) {
          LikeStatus status = string_like_match(s, slen, p, plen, escape_char, is_ilike);
          if (status != kLIKE_FALSE)
            return status;
        }
        s++;
        slen--;
      }
      return kLIKE_ABORT;
    } else if (*p == '_') {
      s++;
      slen--;
      p++;
      plen--;
      continue;
    } else if (*p == '[') {
      const char* pp = p + 1;
      int pplen = plen - 1;
      bool match = false;
      while (pplen > 0 && *pp != ']') {
        if ((!is_ilike && *s == *pp) || (is_ilike && lowercase(*s) == *pp)) {
          match = true;
          break;
        }
        pp++;
        pplen--;
      }
      if (match) {
        s++;
        slen--;
        pplen--;
        const char* x;
        for (x = pp + 1; *x != ']' && pplen > 0; x++, pplen--)
          ;
        if (pplen <= 0)
          return kLIKE_ERROR;  // malformed
        plen -= (x - p + 1);
        p = x + 1;
        continue;
      } else
        return kLIKE_FALSE;
    } else if ((!is_ilike && *s != *p) || (is_ilike && lowercase(*s) != *p))
      return kLIKE_FALSE;
    s++;
    slen--;
    p++;
    plen--;
  }
  if (slen > 0)
    return kLIKE_FALSE;
  while (plen > 0 && *p == '%') {
    p++;
    plen--;
  }
  if (plen <= 0)
    return kLIKE_TRUE;
  return kLIKE_ABORT;
}

/*
 * @brief string_like performs the SQL LIKE and ILIKE operation
 * @param str string argument to be matched against pattern.  single-byte
 * character set only for now. null-termination not required.
 * @param str_len length of str
 * @param pattern pattern string for SQL LIKE
 * @param pat_len length of pattern
 * @param escape_char the escape character.  '\\' is expected by default.
 * @param is_ilike true if it is ILIKE, i.e., case-insensitive matching
 * @return true if str matchs pattern, false otherwise.  error condition
 * not handled for now.
 */
extern "C" DEVICE bool string_like(const char* str,
                                   const int32_t str_len,
                                   const char* pattern,
                                   const int32_t pat_len,
                                   const char escape_char) {
  // @TODO(wei/alex) add runtime error handling
  LikeStatus status = string_like_match(str, str_len, pattern, pat_len, escape_char, false);
  return status == kLIKE_TRUE;
}

extern "C" DEVICE bool string_ilike(const char* str,
                                    const int32_t str_len,
                                    const char* pattern,
                                    const int32_t pat_len,
                                    const char escape_char) {
  // @TODO(wei/alex) add runtime error handling
  LikeStatus status = string_like_match(str, str_len, pattern, pat_len, escape_char, true);
  return status == kLIKE_TRUE;
}

extern "C" DEVICE int32_t StringCompare(const char* s1, const int32_t s1_len, const char* s2, const int32_t s2_len) {
  const char* s1_ = s1;
  const char* s2_ = s2;

  while (s1_ < s1 + s1_len && s2_ < s2 + s2_len && *s1_ == *s2_) {
    s1_++;
    s2_++;
  }

  unsigned char c1 = (s1_ < s1 + s1_len) ? (*(unsigned char*)s1_) : 0;
  unsigned char c2 = (s2_ < s2 + s2_len) ? (*(unsigned char*)s2_) : 0;

  return c1 - c2;
}

#define STR_LIKE_NULLABLE(base_func)                                      \
  extern "C" DEVICE int8_t base_func##_nullable(const char* lhs,          \
                                                const int32_t lhs_len,    \
                                                const char* rhs,          \
                                                const int32_t rhs_len,    \
                                                const char escape_char,   \
                                                const int8_t bool_null) { \
    if (!lhs || !rhs) {                                                   \
      return bool_null;                                                   \
    }                                                                     \
    return base_func(lhs, lhs_len, rhs, rhs_len, escape_char) ? 1 : 0;    \
  }

STR_LIKE_NULLABLE(string_like)
STR_LIKE_NULLABLE(string_ilike)

#undef STR_LIKE_NULLABLE

extern "C" DEVICE bool string_lt(const char* lhs, const int32_t lhs_len, const char* rhs, const int32_t rhs_len) {
  return StringCompare(lhs, lhs_len, rhs, rhs_len) < 0;
}

extern "C" DEVICE bool string_le(const char* lhs, const int32_t lhs_len, const char* rhs, const int32_t rhs_len) {
  return StringCompare(lhs, lhs_len, rhs, rhs_len) <= 0;
}

extern "C" DEVICE bool string_gt(const char* lhs, const int32_t lhs_len, const char* rhs, const int32_t rhs_len) {
  return StringCompare(lhs, lhs_len, rhs, rhs_len) > 0;
}

extern "C" DEVICE bool string_ge(const char* lhs, const int32_t lhs_len, const char* rhs, const int32_t rhs_len) {
  return StringCompare(lhs, lhs_len, rhs, rhs_len) >= 0;
}

extern "C" DEVICE bool string_eq(const char* lhs, const int32_t lhs_len, const char* rhs, const int32_t rhs_len) {
  return StringCompare(lhs, lhs_len, rhs, rhs_len) == 0;
}

extern "C" DEVICE bool string_ne(const char* lhs, const int32_t lhs_len, const char* rhs, const int32_t rhs_len) {
  return StringCompare(lhs, lhs_len, rhs, rhs_len) != 0;
}

#define STR_CMP_NULLABLE(base_func)                                                                             \
  extern "C" DEVICE int8_t base_func##_nullable(                                                                \
      const char* lhs, const int32_t lhs_len, const char* rhs, const int32_t rhs_len, const int8_t bool_null) { \
    if (!lhs || !rhs) {                                                                                         \
      return bool_null;                                                                                         \
    }                                                                                                           \
    return base_func(lhs, lhs_len, rhs, rhs_len) ? 1 : 0;                                                       \
  }

STR_CMP_NULLABLE(string_lt)
STR_CMP_NULLABLE(string_le)
STR_CMP_NULLABLE(string_gt)
STR_CMP_NULLABLE(string_ge)
STR_CMP_NULLABLE(string_eq)
STR_CMP_NULLABLE(string_ne)

#undef STR_CMP_NULLABLE
