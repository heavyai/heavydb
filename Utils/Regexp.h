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
