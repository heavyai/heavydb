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

#include "heavydbTypes.h"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif

EXTENSION_NOINLINE
int32_t text_encoding_none_length(const TextEncodingNone& t) {
  return t.size();
}

#ifndef __CUDACC__

EXTENSION_NOINLINE
TextEncodingNone text_encoding_none_copy(RowFunctionManager& mgr,
                                         const TextEncodingNone& t) {
  return TextEncodingNone(mgr, t.getString());
}

EXTENSION_NOINLINE
TextEncodingNone text_encoding_none_concat(RowFunctionManager& mgr,
                                           const TextEncodingNone& t1,
                                           const TextEncodingNone& t2) {
  return TextEncodingNone(mgr, t1.getString() + ' ' + t2.getString());
}

EXTENSION_NOINLINE
TextEncodingDict text_encoding_dict_concat(RowFunctionManager& mgr,
                                           const TextEncodingDict t_dict,
                                           const TextEncodingNone& t_none) {
  std::string str = mgr.getString(GET_DICT_DB_ID(mgr, 0), GET_DICT_ID(mgr, 0), t_dict);
  return mgr.getOrAddTransient(
      TRANSIENT_DICT_DB_ID, TRANSIENT_DICT_ID, str + t_none.getString());
}

EXTENSION_NOINLINE
TextEncodingDict text_encoding_dict_concat2(RowFunctionManager& mgr,
                                            const TextEncodingNone& t_none,
                                            const TextEncodingDict t_dict) {
  std::string str = mgr.getString(GET_DICT_DB_ID(mgr, 1), GET_DICT_ID(mgr, 1), t_dict);
  return mgr.getOrAddTransient(
      TRANSIENT_DICT_DB_ID, TRANSIENT_DICT_ID, t_none.getString() + str);
}

EXTENSION_NOINLINE
TextEncodingDict text_encoding_dict_concat3(RowFunctionManager& mgr,
                                            const TextEncodingDict t1,
                                            const TextEncodingDict t2) {
  std::string s1 = mgr.getString(GET_DICT_DB_ID(mgr, 0), GET_DICT_ID(mgr, 0), t1);
  std::string s2 = mgr.getString(GET_DICT_DB_ID(mgr, 1), GET_DICT_ID(mgr, 1), t2);
  return mgr.getOrAddTransient(TRANSIENT_DICT_DB_ID, TRANSIENT_DICT_ID, s1 + ' ' + s2);
}

EXTENSION_NOINLINE
TextEncodingDict text_encoding_dict_copy(RowFunctionManager& mgr,
                                         const TextEncodingDict t) {
  std::string str = mgr.getString(GET_DICT_DB_ID(mgr, 0), GET_DICT_ID(mgr, 0), t);
  return mgr.getOrAddTransient(TRANSIENT_DICT_DB_ID, TRANSIENT_DICT_ID, "copy: " + str);
}

EXTENSION_NOINLINE
TextEncodingDict text_encoding_dict_copy_from(RowFunctionManager& mgr,
                                              const TextEncodingDict t1,
                                              const TextEncodingDict t2,
                                              const int32_t select) {
  std::string str;
  if (select == 1) {
    str = mgr.getString(GET_DICT_DB_ID(mgr, 0), GET_DICT_ID(mgr, 0), t1);
  } else {
    str = mgr.getString(GET_DICT_DB_ID(mgr, 1), GET_DICT_ID(mgr, 1), t1);
  }
  return mgr.getOrAddTransient(TRANSIENT_DICT_DB_ID, TRANSIENT_DICT_ID, "copy: " + str);
}

#endif  // #ifndef __CUDACC__

#ifdef __clang__
#pragma clang diagnostic pop
#endif
