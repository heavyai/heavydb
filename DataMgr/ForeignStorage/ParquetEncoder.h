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

#pragma once

#include "Catalog/ColumnDescriptor.h"
#include "DataMgr/AbstractBuffer.h"

namespace foreign_storage {

class ParquetEncoder {
 public:
  ParquetEncoder(Data_Namespace::AbstractBuffer* buffer) : buffer_(buffer) {}
  virtual ~ParquetEncoder() = default;

  virtual void appendData(const int16_t* def_levels,
                          const int16_t* rep_levels,
                          const int64_t values_read,
                          const int64_t levels_read,
                          const bool is_last_batch,
                          int8_t* values) = 0;

 protected:
  Data_Namespace::AbstractBuffer* buffer_;
};

class ParquetScalarEncoder : public ParquetEncoder {
 public:
  ParquetScalarEncoder(Data_Namespace::AbstractBuffer* buffer) : ParquetEncoder(buffer) {}

  virtual void setNull(int8_t* omnisci_data_bytes) = 0;
  virtual void copy(const int8_t* omnisci_data_bytes_source,
                    int8_t* omnisci_data_bytes_destination) = 0;
  virtual void encodeAndCopy(const int8_t* parquet_data_bytes,
                             int8_t* omnisci_data_bytes) = 0;

  virtual void encodeAndCopyContiguous(const int8_t* parquet_data_bytes,
                                       int8_t* omnisci_data_bytes,
                                       const size_t num_elements) = 0;
};

}  // namespace foreign_storage
