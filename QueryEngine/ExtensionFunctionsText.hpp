/*
 * Copyright 2022 HEAVY.AI, Inc., Inc.
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

#include <cstring>
#include "Shared/toString.h"
#include "heavydbTypes.h"

// To-Do: strtok_to_array with default "delimiters" value

#ifndef __CUDACC__
std::vector<std::string> __strtok_to_array(const std::string& text,
                                           const std::string& delimiters) {
  std::vector<std::string> vec;

  char* str = const_cast<char*>(text.c_str());
  const char* del = delimiters.c_str();

  char* substr = strtok(str, del);
  while (substr != NULL) {
    std::string s(substr);
    vec.emplace_back(s);
    substr = strtok(NULL, del);
  }

  return vec;
}

EXTENSION_NOINLINE
Array<TextEncodingDict> strtok_to_array(RowFunctionManager& mgr,
                                        TextEncodingNone& text,
                                        TextEncodingNone& delimiters) {
  /*
    Rules
    -----
    * If either parameters is NULL => a NULL is returned
    * An empty array is returned if tokenization produces no tokens

    Note
    ----
    <delimiters> argument is optional on snowflake but HeavyDB dont' support
    default values on UDFs at the moment. See:
    https://github.com/heavyai/heavydb-internal/pull/6651

    Examples
    --------
    > select strtok_to_array('a.b.c', '.');
    {a, b, c}

    > select strtok_to_array('user@gmail.com', '.@')
    {user, gmail, com}

    > select strtok_to_array('', '.')
    NULL

    > select strtok_to_array('a.b.c', '')
    NULL
  */

  if (text.isNull() || delimiters.isNull()) {
    return Array<TextEncodingDict>(0, true);
  }

  const auto& vec = __strtok_to_array(text.getString(), delimiters.getString());
  Array<TextEncodingDict> out_arr(vec.size());
  for (size_t i = 0; i < vec.size(); ++i) {
    out_arr[i] = mgr.getOrAddTransient(TRANSIENT_DICT_DB_ID, TRANSIENT_DICT_ID, vec[i]);
  }
  return out_arr;
}

EXTENSION_NOINLINE
Array<TextEncodingDict> strtok_to_array__1(RowFunctionManager& mgr,
                                           TextEncodingDict text,
                                           TextEncodingNone& delimiters) {
  if (text.isNull() || delimiters.isNull()) {
    return Array<TextEncodingDict>(0, true);
  }

  std::string str = mgr.getString(GET_DICT_DB_ID(mgr, 0), GET_DICT_ID(mgr, 0), text);
  const auto& vec = __strtok_to_array(str, delimiters.getString());
  Array<TextEncodingDict> out_arr(vec.size());
  for (size_t i = 0; i < vec.size(); ++i) {
    out_arr[i] = mgr.getOrAddTransient(TRANSIENT_DICT_DB_ID, TRANSIENT_DICT_ID, vec[i]);
  }
  return out_arr;
}
#endif  // #ifndef __CUDACC__
