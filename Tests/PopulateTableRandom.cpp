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
 * @file		PopulateTableRandom.cpp
 * @author	Wei Hong <wei@map-d.com>
 * @brief		Populate a table with random data
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/

#include <boost/functional/hash.hpp>
#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <random>
#include <string>

#include <glog/logging.h>

#include "../Catalog/Catalog.h"
#include "../DataMgr/DataMgr.h"
#include "../Fragmenter/Fragmenter.h"
#include "../Shared/DateConverters.h"
#include "../Shared/measure.h"
#include "../Shared/sqltypes.h"

using namespace Catalog_Namespace;
using namespace Fragmenter_Namespace;

size_t random_fill_int16(int8_t* buf, size_t num_elems) {
  std::default_random_engine gen;
  std::uniform_int_distribution<int16_t> dist(INT16_MIN, INT16_MAX);
  auto p = reinterpret_cast<int16_t*>(buf);
  size_t hash = 0;
  for (size_t i = 0; i < num_elems; i++) {
    p[i] = dist(gen);
    boost::hash_combine(hash, p[i]);
  }
  return hash;
}

size_t random_fill_int32(int8_t* buf, size_t num_elems) {
  std::default_random_engine gen;
  std::uniform_int_distribution<int32_t> dist(INT32_MIN, INT32_MAX);
  auto p = reinterpret_cast<int32_t*>(buf);
  size_t hash = 0;
  for (size_t i = 0; i < num_elems; i++) {
    p[i] = dist(gen);
    boost::hash_combine(hash, p[i]);
  }
  return hash;
}

size_t random_fill_int64(int8_t* buf, size_t num_elems, int64_t min, int64_t max) {
  std::default_random_engine gen;
  std::uniform_int_distribution<int64_t> dist(min, max);
  auto p = reinterpret_cast<int64_t*>(buf);
  size_t hash = 0;
  for (size_t i = 0; i < num_elems; i++) {
    p[i] = dist(gen);
    boost::hash_combine(hash, p[i]);
  }
  return hash;
}

size_t random_fill_int64(int8_t* buf, size_t num_elems) {
  return random_fill_int64(buf, num_elems, INT64_MIN, INT64_MAX);
}

size_t random_fill_float(int8_t* buf, size_t num_elems) {
  std::default_random_engine gen;
  std::uniform_real_distribution<float> dist(FLT_MIN, FLT_MAX);
  auto p = reinterpret_cast<float*>(buf);
  size_t hash = 0;
  for (size_t i = 0; i < num_elems; i++) {
    p[i] = dist(gen);
    boost::hash_combine(hash, p[i]);
  }
  return hash;
}

size_t random_fill_double(int8_t* buf, size_t num_elems) {
  std::default_random_engine gen;
  std::uniform_real_distribution<double> dist(DBL_MIN, DBL_MAX);
  auto p = reinterpret_cast<double*>(buf);
  size_t hash = 0;
  for (size_t i = 0; i < num_elems; i++) {
    p[i] = dist(gen);
    boost::hash_combine(hash, p[i]);
  }
  return hash;
}

size_t random_fill_string(std::vector<std::string>& stringVec,
                          size_t num_elems,
                          int max_len,
                          size_t& data_volumn) {
  std::string chars("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890");
  std::default_random_engine gen;
  std::uniform_int_distribution<> char_dist(0, chars.size() - 1);
  std::uniform_int_distribution<> len_dist(0, max_len);
  size_t hash = 0;
  std::hash<std::string> string_hash;
  for (size_t n = 0; n < num_elems; n++) {
    int len = len_dist(gen);
    std::string s(len, ' ');
    for (int i = 0; i < len; i++) {
      {
        s[i] = chars[char_dist(gen)];
      }
    }
    stringVec[n] = s;
    boost::hash_combine(hash, string_hash(s));
    data_volumn += len;
  }
  return hash;
}

size_t random_fill_int8array(std::vector<std::vector<int8_t>>& stringVec,
                             size_t num_elems,
                             int max_len,
                             size_t& data_volumn) {
  std::default_random_engine gen;
  std::uniform_int_distribution<int8_t> dist(INT8_MIN, INT8_MAX);
  std::uniform_int_distribution<> len_dist(0, max_len);
  size_t hash = 0;
  for (size_t n = 0; n < num_elems; n++) {
    int len = len_dist(gen);
    std::vector<int8_t> s(len);
    for (int i = 0; i < len; i++) {
      s[i] = dist(gen);
      boost::hash_combine(hash, s[i]);
    }
    stringVec[n] = s;
    data_volumn += len * sizeof(int8_t);
  }
  return hash;
}

size_t random_fill_int16array(std::vector<std::vector<int16_t>>& stringVec,
                              size_t num_elems,
                              int max_len,
                              size_t& data_volumn) {
  std::default_random_engine gen;
  std::uniform_int_distribution<int16_t> dist(INT16_MIN, INT16_MAX);
  std::uniform_int_distribution<> len_dist(0, max_len / 2);
  size_t hash = 0;
  for (size_t n = 0; n < num_elems; n++) {
    int len = len_dist(gen);
    std::vector<int16_t> s(len);
    for (int i = 0; i < len; i++) {
      s[i] = dist(gen);
      boost::hash_combine(hash, s[i]);
    }
    stringVec[n] = s;
    data_volumn += len * sizeof(int16_t);
  }
  return hash;
}

size_t random_fill_int32array(std::vector<std::vector<int32_t>>& stringVec,
                              size_t num_elems,
                              int max_len,
                              size_t& data_volumn) {
  std::default_random_engine gen;
  std::uniform_int_distribution<int32_t> dist(INT32_MIN, INT32_MAX);
  std::uniform_int_distribution<> len_dist(0, max_len / 4);
  size_t hash = 0;
  for (size_t n = 0; n < num_elems; n++) {
    int len = len_dist(gen);
    std::vector<int32_t> s(len);
    for (int i = 0; i < len; i++) {
      s[i] = dist(gen);
      boost::hash_combine(hash, s[i]);
    }
    stringVec[n] = s;
    data_volumn += len * sizeof(int32_t);
  }
  return hash;
}

size_t random_fill_dates(int8_t* buf, size_t num_elems) {
  constexpr int64_t kDateMin = -185542587187200;
  constexpr int64_t kDateMax = 185542587100800;
  std::default_random_engine gen;
  std::uniform_int_distribution<int64_t> dist(kDateMin, kDateMax);
  auto p = reinterpret_cast<int64_t*>(buf);
  size_t hash = 0;
  for (size_t i = 0; i < num_elems; i++) {
    p[i] = dist(gen);
    boost::hash_combine(hash, DateConverters::get_epoch_days_from_seconds(p[i]));
  }
  return hash;
}

#define MAX_TEXT_LEN 255

size_t random_fill(const ColumnDescriptor* cd,
                   DataBlockPtr p,
                   size_t num_elems,
                   size_t& data_volumn) {
  size_t hash = 0;
  switch (cd->columnType.get_type()) {
    case kSMALLINT:
      hash = random_fill_int16(p.numbersPtr, num_elems);
      data_volumn += num_elems * sizeof(int16_t);
      break;
    case kINT:
      hash = random_fill_int32(p.numbersPtr, num_elems);
      data_volumn += num_elems * sizeof(int32_t);
      break;
    case kBIGINT:
      hash = random_fill_int64(p.numbersPtr, num_elems, INT64_MIN, INT64_MAX);
      data_volumn += num_elems * sizeof(int64_t);
      break;
    case kNUMERIC:
    case kDECIMAL: {
      int64_t max = std::pow((double)10, cd->columnType.get_precision());
      int64_t min = -max;
      hash = random_fill_int64(p.numbersPtr, num_elems, min, max);
      data_volumn += num_elems * sizeof(int64_t);
    } break;
    case kFLOAT:
      hash = random_fill_float(p.numbersPtr, num_elems);
      data_volumn += num_elems * sizeof(float);
      break;
    case kDOUBLE:
      hash = random_fill_double(p.numbersPtr, num_elems);
      data_volumn += num_elems * sizeof(double);
      break;
    case kVARCHAR:
    case kCHAR:
      if (cd->columnType.get_compression() == kENCODING_NONE) {
        {
          hash = random_fill_string(
              *p.stringsPtr, num_elems, cd->columnType.get_dimension(), data_volumn);
        }
      }
      break;
    case kTEXT:
      if (cd->columnType.get_compression() == kENCODING_NONE) {
        {
          hash = random_fill_string(*p.stringsPtr, num_elems, MAX_TEXT_LEN, data_volumn);
        }
      }
      break;
    case kDATE:
    case kTIME:
    case kTIMESTAMP:
      hash = cd->columnType.get_type() == kDATE
                 ? random_fill_dates(p.numbersPtr, num_elems)
                 : random_fill_int64(p.numbersPtr, num_elems);
      data_volumn += num_elems * sizeof(int64_t);
      break;
    default:
      assert(false);
  }
  return hash;
}

std::vector<size_t> populate_table_random(const std::string& table_name,
                                          const size_t num_rows,
                                          const Catalog& cat) {
  const TableDescriptor* td = cat.getMetadataForTable(table_name);
  const auto cds = cat.getAllColumnMetadataForTable(td->tableId, false, false, false);
  InsertData insert_data;
  insert_data.databaseId = cat.getCurrentDB().dbId;
  insert_data.tableId = td->tableId;
  for (const auto& cd : cds) {
    insert_data.columnIds.push_back(cd->columnId);
  }
  insert_data.numRows = num_rows;
  std::vector<std::vector<int8_t>> numbers_vec;
  std::vector<std::unique_ptr<std::vector<std::string>>> strings_vec;

  DataBlockPtr p{0};
  // now allocate space for insert data
  for (auto cd : cds) {
    if (cd->columnType.is_varlen()) {
      if (cd->columnType.get_compression() == kENCODING_NONE) {
        strings_vec.push_back(std::make_unique<std::vector<std::string>>(num_rows));
        p.stringsPtr = strings_vec.back().get();
      } else {
        CHECK(false);
      }
    } else {
      numbers_vec.emplace_back(num_rows * cd->columnType.get_logical_size());
      p.numbersPtr = numbers_vec.back().data();
    }
    insert_data.data.push_back(p);
  }

  // fill InsertData  with random data
  std::vector<size_t> col_hashs(
      cds.size());  // compute one hash per column for the generated data
  int i = 0;
  size_t data_volumn = 0;
  for (auto cd : cds) {
    col_hashs[i] = random_fill(cd, insert_data.data[i], num_rows, data_volumn);
    i++;
  }

  // now load the data into table
  auto ms = measure<>::execution([&]() { td->fragmenter->insertData(insert_data); });
  std::cout << "Loaded " << num_rows << " rows " << data_volumn << " bytes in " << ms
            << " ms. at " << (double)data_volumn / (ms / 1000.0) / 1e6 << " MB/sec."
            << std::endl;

  return col_hashs;
}
