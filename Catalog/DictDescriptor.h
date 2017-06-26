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

#ifndef DICT_DESCRIPTOR_H
#define DICT_DESCRIPTOR_H

#include <cassert>
#include <memory>
#include <string>
#include "../Shared/sqltypes.h"
#include "../StringDictionary/StringDictionary.h"

/**
 * @type DictDescriptor
 * @brief Descriptor for a dictionary for a string columne
 *
 */

struct DictDescriptor {
  int dictId;
  std::string dictName;
  int dictNBits;
  bool dictIsShared;
  std::string dictFolderPath;
  int refcount;
  bool dictIsTemp;
  std::shared_ptr<StringDictionary> stringDict;
  DictDescriptor(int id, const std::string& name, int nbits, bool shared, const int rc, std::string& fname, bool temp)
      : dictId(id),
        dictName(name),
        dictNBits(nbits),
        dictIsShared(shared),
        dictFolderPath(fname),
        refcount(rc),
        dictIsTemp(temp),
        stringDict(nullptr) {}
};

#endif  // DICT_DESCRIPTOR
