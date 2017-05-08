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
 * @file		Regex.cpp
 * @author		Dmitri Shtilman <d@mapd.com>
 * @brief		Support the REGEX operator and REGEX_LIKE function in SQL.
 *
 * Copyright (c) 2016 MapD Technologies, Inc.  All rights reserved.
 **/

#include "Regexp.h"

#ifndef __CUDACC__
#include <boost/regex.hpp>
#endif

/*
 * @brief regexp_like performs the SQL REGEXP operation
 * @param str string argument to be matched against pattern.
 * @param str_len length of str
 * @param pattern regex pattern string for SQL REGEXP
 * @param pat_len length of pattern
 * @param escape_char the escape character.  '\\' is expected by default.
 * @return true if str matches pattern, false otherwise.
 */
extern "C" DEVICE bool regexp_like(const char* str,
                                   const int32_t str_len,
                                   const char* pattern,
                                   const int32_t pat_len,
                                   const char escape_char) {
#ifndef __CUDACC__
  bool result;
  try {
    boost::regex re(pattern, pat_len, boost::regex::extended);
    boost::cmatch what;
    result = boost::regex_match(str, str + str_len, what, re);
  } catch (std::runtime_error& error) {
    // LOG(ERROR) << "Regexp match error: " << error.what();
    result = false;
  }
  return result;
#else
  return false;
#endif
}

extern "C" DEVICE int8_t regexp_like_nullable(const char* str,
                                              const int32_t str_len,
                                              const char* pattern,
                                              const int32_t pat_len,
                                              const char escape_char,
                                              const int8_t bool_null) {
  if (!str || !pattern) {
    return bool_null;
  }

  return regexp_like(str, str_len, pattern, pat_len, escape_char);
}
