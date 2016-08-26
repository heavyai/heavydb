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
    result = boost::regex_match(str, what, re);
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
