/**
 * @file		Regex.h
 * @author		Dmitri Shtilman <d@mapd.com>
 * @brief		Support the REGEX operator and REGEX_LIKE function in SQL.
 *
 * Copyright (c) 2016 MapD Technologies, Inc.  All rights reserved.
 **/

#ifndef REGEX_H
#define REGEX_H

#include "../Shared/funcannotations.h"

#include <stdint.h>

/*
 * @brief regexp_like performs the SQL REGEXP operation
 * @param str string argument to be matched against pattern.
 * @param str_len length of str
 * @param pattern regex pattern string for SQL REGEXP
 * @param pat_len length of pattern
 * @param escape_char the escape character.  '\\' is expected by default.
 * @return true if str matches pattern, false otherwise.
 */

extern "C" DEVICE bool regexp_like(const char* str, int str_len, const char* pattern, int pat_len, char escape_char);

#endif  // REGEX_H
