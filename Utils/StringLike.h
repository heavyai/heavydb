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
 * @file		StringLike.h
 * @author	Wei Hong <wei@mapd.com>
 * @brief		Functions to support the LIKE and ILIKE operator in SQL.  Only
 * single-byte character set is supported for now.
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/

#ifndef STRING_LIKE_H
#define STRING_LIKE_H

#include "../Shared/funcannotations.h"

#include <stdint.h>

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
extern "C" DEVICE bool string_like(const char* str, int str_len, const char* pattern, int pat_len, char escape_char);

extern "C" DEVICE bool string_ilike(const char* str, int str_len, const char* pattern, int pat_len, char escape_char);

extern "C" DEVICE bool string_like_simple(const char* str,
                                          const int32_t str_len,
                                          const char* pattern,
                                          const int32_t pat_len);

extern "C" DEVICE bool string_ilike_simple(const char* str,
                                           const int32_t str_len,
                                           const char* pattern,
                                           const int32_t pat_len);

extern "C" DEVICE bool string_lt(const char* lhs, const int32_t lhs_len, const char* rhs, const int32_t rhs_len);

extern "C" DEVICE bool string_le(const char* lhs, const int32_t lhs_len, const char* rhs, const int32_t rhs_len);

extern "C" DEVICE bool string_eq(const char* lhs, const int32_t lhs_len, const char* rhs, const int32_t rhs_len);

extern "C" DEVICE bool string_ne(const char* lhs, const int32_t lhs_len, const char* rhs, const int32_t rhs_len);

extern "C" DEVICE bool string_ge(const char* lhs, const int32_t lhs_len, const char* rhs, const int32_t rhs_len);

extern "C" DEVICE bool string_gt(const char* lhs, const int32_t lhs_len, const char* rhs, const int32_t rhs_len);

extern "C" DEVICE int32_t StringCompare(const char* s1, const int32_t s1_len, const char* s2, const int32_t s2_len);

#endif  // STRING_LIKE_H
