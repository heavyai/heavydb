/*
 * Copyright 2020 OmniSci, Inc.
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

#ifndef DATAFRAME_TABLE_DESCRIPTOR_H
#define DATAFRAME_TABLE_DESCRIPTOR_H

#include <cstdint>
#include <string>

#include "TableDescriptor.h"

/**
 * @type DataframeTableDescriptor
 * @brief specifies the content in-memory of a row in the table metadata table
 *
 */

struct DataframeTableDescriptor : TableDescriptor {
  int64_t skipRows;       // number of skipped rows of data in CSV file
  std::string delimiter;  // delimiter of values in the CSV file
  bool hasHeader;         // does table has a header in CSV file

  DataframeTableDescriptor()
      : TableDescriptor(), skipRows(0), delimiter(","), hasHeader(true) {}

  virtual ~DataframeTableDescriptor() = default;
};

#endif  // DATAFRAME_TABLE_DESCRIPTOR_H
