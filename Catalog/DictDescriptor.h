/*
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
  DictRef dictRef;
  std::string dictName;
  int dictNBits;
  bool dictIsShared;
  std::string dictFolderPath;
  int refcount;
  bool dictIsTemp;
  std::shared_ptr<StringDictionary> stringDict;
  std::shared_ptr<std::mutex> string_dict_mutex;
  DictDescriptor(DictRef dict_ref,
                 const std::string& name,
                 int nbits,
                 bool shared,
                 const int rc,
                 const std::string& fname,
                 bool temp)
      : dictRef(dict_ref)
      , dictName(name)
      , dictNBits(nbits)
      , dictIsShared(shared)
      , dictFolderPath(fname)
      , refcount(rc)
      , dictIsTemp(temp)
      , stringDict(nullptr)
      , string_dict_mutex(std::make_shared<std::mutex>()) {}

  DictDescriptor(int db_id,
                 int dict_id,
                 const std::string& name,
                 int nbits,
                 bool shared,
                 const int rc,
                 const std::string& fname,
                 bool temp)
      : dictName(name)
      , dictNBits(nbits)
      , dictIsShared(shared)
      , dictFolderPath(fname)
      , refcount(rc)
      , dictIsTemp(temp)
      , stringDict(nullptr)
      , string_dict_mutex(std::make_shared<std::mutex>()) {
    dictRef.dbId = db_id;
    dictRef.dictId = dict_id;
  }
};

#endif  // DICT_DESCRIPTOR
