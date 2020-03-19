/*
 * Copyright 2019 OmniSci, Inc.
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
#include <numeric>

#include "../Catalog/Catalog.h"
#include "SortedOrderFragmenter.h"

namespace Fragmenter_Namespace {

template <typename T>
void shuffleByIndexesImpl(const std::vector<size_t>& indexes, T* buffer) {
  std::vector<T> new_buffer;
  new_buffer.reserve(indexes.size());
  for (const auto i : indexes) {
    new_buffer.push_back(buffer[i]);
  }
  std::memcpy(buffer, new_buffer.data(), indexes.size() * sizeof(T));
}

template <typename T>
void shuffleByIndexesImpl(const std::vector<size_t>& indexes, std::vector<T>& buffer) {
  std::vector<T> new_buffer;
  new_buffer.reserve(indexes.size());
  for (const auto i : indexes) {
    new_buffer.push_back(buffer[i]);
  }
  buffer.swap(new_buffer);
}

void shuffleByIndexes(const ColumnDescriptor* cd,
                      const std::vector<size_t>& indexes,
                      DataBlockPtr& data) {
  const auto& ti = cd->columnType;
  switch (ti.get_type()) {
    case kBOOLEAN:
      shuffleByIndexesImpl(indexes, reinterpret_cast<int8_t*>(data.numbersPtr));
      break;
    case kTINYINT:
      shuffleByIndexesImpl(indexes, reinterpret_cast<int8_t*>(data.numbersPtr));
      break;
    case kSMALLINT:
      shuffleByIndexesImpl(indexes, reinterpret_cast<int16_t*>(data.numbersPtr));
      break;
    case kINT:
      shuffleByIndexesImpl(indexes, reinterpret_cast<int32_t*>(data.numbersPtr));
      break;
    case kBIGINT:
    case kNUMERIC:
    case kDECIMAL:
      shuffleByIndexesImpl(indexes, reinterpret_cast<int64_t*>(data.numbersPtr));
      break;
    case kFLOAT:
      shuffleByIndexesImpl(indexes, reinterpret_cast<float*>(data.numbersPtr));
      break;
    case kDOUBLE:
      shuffleByIndexesImpl(indexes, reinterpret_cast<double*>(data.numbersPtr));
      break;
    case kTEXT:
    case kVARCHAR:
    case kCHAR:
      if (ti.is_varlen()) {
        shuffleByIndexesImpl(indexes, *data.stringsPtr);
      } else {
        switch (ti.get_size()) {
          case 1:
            shuffleByIndexesImpl(indexes, reinterpret_cast<int8_t*>(data.numbersPtr));
            break;
          case 2:
            shuffleByIndexesImpl(indexes, reinterpret_cast<int16_t*>(data.numbersPtr));
            break;
          case 4:
            shuffleByIndexesImpl(indexes, reinterpret_cast<int32_t*>(data.numbersPtr));
            break;
          default:
            CHECK(false);
        }
      }
      break;
    case kDATE:
    case kTIME:
    case kTIMESTAMP:
      shuffleByIndexesImpl(indexes, reinterpret_cast<int64_t*>(data.numbersPtr));
      break;
    case kARRAY:
      shuffleByIndexesImpl(indexes, *data.arraysPtr);
      break;
    case kPOINT:
    case kLINESTRING:
    case kPOLYGON:
    case kMULTIPOLYGON:
      shuffleByIndexesImpl(indexes, *data.stringsPtr);
      break;
    default:
      CHECK(false);
  }
}

template <typename T>
void sortIndexesImpl(std::vector<size_t>& indexes, const T* buffer) {
  CHECK(buffer);
  std::sort(indexes.begin(), indexes.end(), [&](const auto a, const auto b) {
    return buffer[a] < buffer[b];
  });
}

void sortIndexesImpl(std::vector<size_t>& indexes,
                     const std::vector<std::string>& buffer) {
  std::sort(indexes.begin(), indexes.end(), [&](const auto a, const auto b) {
    return buffer[a].size() < buffer[b].size() ||
           (buffer[a].size() == buffer[b].size() && buffer[a] < buffer[b]);
  });
}

void sortIndexesImpl(std::vector<size_t>& indexes,
                     const std::vector<ArrayDatum>& buffer) {
  std::sort(indexes.begin(), indexes.end(), [&](const auto a, const auto b) {
    return buffer[a].is_null || buffer[a].length < buffer[b].length ||
           (!buffer[b].is_null && buffer[a].length == buffer[b].length &&
            memcmp(buffer[a].pointer, buffer[b].pointer, buffer[a].length) < 0);
  });
}

void sortIndexes(const ColumnDescriptor* cd,
                 std::vector<size_t>& indexes,
                 const DataBlockPtr& data) {
  const auto& ti = cd->columnType;
  switch (ti.get_type()) {
    case kBOOLEAN:
      sortIndexesImpl(indexes, reinterpret_cast<int8_t*>(data.numbersPtr));
      break;
    case kTINYINT:
      sortIndexesImpl(indexes, reinterpret_cast<int8_t*>(data.numbersPtr));
      break;
    case kSMALLINT:
      sortIndexesImpl(indexes, reinterpret_cast<int16_t*>(data.numbersPtr));
      break;
    case kINT:
      sortIndexesImpl(indexes, reinterpret_cast<int32_t*>(data.numbersPtr));
      break;
    case kBIGINT:
    case kNUMERIC:
    case kDECIMAL:
      sortIndexesImpl(indexes, reinterpret_cast<int64_t*>(data.numbersPtr));
      break;
    case kFLOAT:
      sortIndexesImpl(indexes, reinterpret_cast<float*>(data.numbersPtr));
      break;
    case kDOUBLE:
      sortIndexesImpl(indexes, reinterpret_cast<double*>(data.numbersPtr));
      break;
    case kTEXT:
    case kVARCHAR:
    case kCHAR:
      if (ti.is_varlen()) {
        sortIndexesImpl(indexes, *data.stringsPtr);
      } else {
        switch (ti.get_size()) {
          case 1:
            sortIndexesImpl(indexes, reinterpret_cast<int8_t*>(data.numbersPtr));
            break;
          case 2:
            sortIndexesImpl(indexes, reinterpret_cast<int16_t*>(data.numbersPtr));
            break;
          case 4:
            sortIndexesImpl(indexes, reinterpret_cast<int32_t*>(data.numbersPtr));
            break;
          default:
            CHECK(false);
        }
      }
      break;
    case kDATE:
    case kTIME:
    case kTIMESTAMP:
      sortIndexesImpl(indexes, reinterpret_cast<int64_t*>(data.numbersPtr));
      break;
    case kARRAY:
      sortIndexesImpl(indexes, *data.arraysPtr);
      break;
    default:
      CHECK(false) << "invalid type '" << ti.get_type() << "' to sort";
  }
}

void SortedOrderFragmenter::sortData(InsertData& insertDataStruct) {
  // coming here table must have defined a sort_column for mini sort
  const auto table_desc = catalog_->getMetadataForTable(physicalTableId_);
  CHECK(table_desc);
  CHECK_GT(table_desc->sortedColumnId, 0);
  const auto logical_cd =
      catalog_->getMetadataForColumn(table_desc->tableId, table_desc->sortedColumnId);
  CHECK(logical_cd);
  const auto physical_cd = catalog_->getMetadataForColumn(
      table_desc->tableId,
      table_desc->sortedColumnId + (logical_cd->columnType.is_geometry() ? 1 : 0));
  const auto it = std::find(insertDataStruct.columnIds.begin(),
                            insertDataStruct.columnIds.end(),
                            physical_cd->columnId);
  CHECK(it != insertDataStruct.columnIds.end());
  // sort row indexes of the sort column
  std::vector<size_t> indexes(insertDataStruct.numRows);
  std::iota(indexes.begin(), indexes.end(), 0);
  const auto dist = std::distance(insertDataStruct.columnIds.begin(), it);
  CHECK_LT(static_cast<size_t>(dist), insertDataStruct.data.size());
  sortIndexes(physical_cd, indexes, insertDataStruct.data[dist]);
  // shuffle rows of all columns
  for (size_t i = 0; i < insertDataStruct.columnIds.size(); ++i) {
    const auto cd = catalog_->getMetadataForColumn(table_desc->tableId,
                                                   insertDataStruct.columnIds[i]);
    shuffleByIndexes(cd, indexes, insertDataStruct.data[i]);
  }
}

}  // namespace Fragmenter_Namespace
