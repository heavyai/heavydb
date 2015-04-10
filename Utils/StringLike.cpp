/**
 * @file		StringLike.cpp
 * @author	Wei Hong <wei@mapd.com>
 * @brief		Functions to support the LIKE and ILIKE operator in SQL.  Only
 * single-byte character set is supported for now.
 * 
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/

#include "StringLike.h"
#include <stdint.h>

enum LikeStatus {
  kLIKE_TRUE,
  kLIKE_FALSE,
  kLIKE_ABORT, // means we run out of string characters to match against pattern, can abort early
  kLIKE_ERROR // error condition
};

DEVICE static int
lowercase(char c)
{
  if ('A' <= c & c <= 'Z')
    return 'a' + (c - 'A');
  return c;
}

// internal recursive function for performing LIKE matching.
DEVICE static LikeStatus
string_like_match(const char *str, int str_len, const char *pattern, int pat_len, char escape_char, bool is_ilike)
{
  const char *s = str;
  int slen = str_len;
  const char *p = pattern;
  int plen = pat_len;

  while (slen > 0 && plen > 0) {
    if (*p == escape_char) {
      // next pattern char must match literally, whatever it is
      p++; plen--;
      if (plen <= 0)
        return kLIKE_ERROR;
      if ((!is_ilike && *s != *p) || (is_ilike && lowercase(*s) != lowercase(*p)))
        return kLIKE_FALSE;
    } else if (*p == '%') {
      char firstpat;
      p++; plen--;
      while (plen > 0) {
        if (*p == '%') {
          p++; plen--;
        } else if (*p == '_') {
          if (slen <= 0)
            return kLIKE_ABORT;
          s++; slen--;
          p++; plen--;
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
        if (*s == firstpat || (is_ilike && lowercase(*s) == lowercase(firstpat))) {
          LikeStatus status = string_like_match(s, slen, p, plen, escape_char, is_ilike);
          if (status != kLIKE_FALSE)
            return status;
        }
        s++; slen--;
     }
     return kLIKE_ABORT;
    } else if (*p == '_') {
        s++; slen--;
        p++; plen--;
        continue;
    } else if ((!is_ilike && *s != *p) || (is_ilike && lowercase(*s) != lowercase(*p)))
      return kLIKE_FALSE;
    s++; slen--;
    p++; plen--;
  }
  if (slen > 0)
    return kLIKE_FALSE;
  while (plen > 0 && *p == '%') {
    p++; plen--;
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
extern "C" DEVICE
bool string_like(const char *str, int str_len, const char *pattern, int pat_len, char escape_char)
{
  // @TODO(wei/alex) add runtime error handling
  LikeStatus status = string_like_match(str, str_len, pattern, pat_len, escape_char, false);
  return status == kLIKE_TRUE;
}

extern "C" DEVICE
bool string_ilike(const char *str, int str_len, const char *pattern, int pat_len, char escape_char)
{
  // @TODO(wei/alex) add runtime error handling
  LikeStatus status = string_like_match(str, str_len, pattern, pat_len, escape_char, true);
  return status == kLIKE_TRUE;
}

extern "C" DEVICE
int32_t StringCompare(const char* s1, const int32_t s1_len, const char* s2, const int32_t s2_len) {
  const char* s1_ = s1;
  const char* s2_ = s2;

  while (s1_ < s1 + s1_len && s2_ < s2 + s2_len && *s1_ == *s2_) {
    s1_++;
    s2_++;
  }

  unsigned char c1 = (s1_ < s1 + s1_len) ? (*(unsigned char*) s1_) : 0;
  unsigned char c2 = (s2_ < s2 + s2_len) ? (*(unsigned char*) s2_) : 0;

  return c1 - c2;
}

extern "C" DEVICE
bool string_lt(const char* lhs, const int32_t lhs_len, const char* rhs, const int32_t rhs_len) {
  return StringCompare(lhs, lhs_len, rhs, rhs_len) < 0;
}

extern "C" DEVICE
bool string_le(const char* lhs, const int32_t lhs_len, const char* rhs, const int32_t rhs_len) {
  return StringCompare(lhs, lhs_len, rhs, rhs_len) <= 0;
}

extern "C" DEVICE
bool string_gt(const char* lhs, const int32_t lhs_len, const char* rhs, const int32_t rhs_len) {
  return StringCompare(lhs, lhs_len, rhs, rhs_len) > 0;
}

extern "C" DEVICE
bool string_ge(const char* lhs, const int32_t lhs_len, const char* rhs, const int32_t rhs_len) {
  return StringCompare(lhs, lhs_len, rhs, rhs_len) >= 0;
}

extern "C" DEVICE
bool string_eq(const char* lhs, const int32_t lhs_len, const char* rhs, const int32_t rhs_len) {
  return StringCompare(lhs, lhs_len, rhs, rhs_len) == 0;
}

extern "C" DEVICE
bool string_ne(const char* lhs, const int32_t lhs_len, const char* rhs, const int32_t rhs_len) {
  return StringCompare(lhs, lhs_len, rhs, rhs_len) != 0;
}
