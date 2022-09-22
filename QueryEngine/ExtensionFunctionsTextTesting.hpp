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

EXTENSION_NOINLINE
int32_t text_encoding_none_length(const TextEncodingNone& t) {
  return t.size();
}

EXTENSION_NOINLINE
TextEncodingNone text_encoding_none_copy(const TextEncodingNone& t) {
#ifndef __CUDACC__
  return TextEncodingNone(t.getString());
#else
  return TextEncodingNone();
#endif
}

EXTENSION_NOINLINE
TextEncodingNone text_encoding_none_concat(const TextEncodingNone& t1,
                                           const TextEncodingNone& t2) {
#ifndef __CUDACC__
  return TextEncodingNone(t1.getString() + ' ' + t2.getString());
#else
  return TextEncodingNone();
#endif
}

#ifndef __CUDACC__
EXTENSION_NOINLINE
TextEncodingDict text_encoding_dict_concat(RowFunctionManager& mgr,
                                           const TextEncodingDict t_dict,
                                           const TextEncodingNone& t_none) {
  int32_t dict_id = GET_DICT_ID(mgr, 0);
  std::string str = mgr.getString(dict_id, t_dict);
  return mgr.getOrAddTransient(TRANSIENT_DICT_ID, str + t_none.getString());
}

EXTENSION_NOINLINE
TextEncodingDict text_encoding_dict_concat2(RowFunctionManager& mgr,
                                            const TextEncodingNone& t_none,
                                            const TextEncodingDict t_dict) {
  int32_t dict_id = GET_DICT_ID(mgr, 1);
  std::string str = mgr.getString(dict_id, t_dict);
  return mgr.getOrAddTransient(TRANSIENT_DICT_ID, t_none.getString() + str);
}

EXTENSION_NOINLINE
TextEncodingDict text_encoding_dict_copy(RowFunctionManager& mgr,
                                         const TextEncodingDict t) {
  int32_t dict_id = GET_DICT_ID(mgr, 0);
  std::string str = mgr.getString(dict_id, t);
  return mgr.getOrAddTransient(TRANSIENT_DICT_ID, "copy: " + str);
}

EXTENSION_NOINLINE
TextEncodingDict text_encoding_dict_copy_from(RowFunctionManager& mgr,
                                              const TextEncodingDict t1,
                                              const TextEncodingDict t2,
                                              const int32_t select) {
  std::string str;
  if (select == 1) {
    int32_t dict_id = GET_DICT_ID(mgr, 0);
    str = mgr.getString(dict_id, t1);
  } else {
    int32_t dict_id = GET_DICT_ID(mgr, 1);
    str = mgr.getString(dict_id, t1);
  }
  return mgr.getOrAddTransient(TRANSIENT_DICT_ID, "copy: " + str);
}
#endif
