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

#include "ParquetInPlaceEncoder.h"

namespace foreign_storage {
class ParquetDateInSecondsEncoder : public TypedParquetInPlaceEncoder<int64_t, int32_t>,
                                    public ParquetMetadataValidator {
 public:
  ParquetDateInSecondsEncoder(Data_Namespace::AbstractBuffer* buffer,
                              const ColumnDescriptor* column_desciptor,
                              const parquet::ColumnDescriptor* parquet_column_descriptor)
      : TypedParquetInPlaceEncoder<int64_t, int32_t>(buffer,
                                                     column_desciptor,
                                                     parquet_column_descriptor) {
    CHECK(parquet_column_descriptor->logical_type()->is_date());
  }

  void encodeAndCopy(const int8_t* parquet_data_bytes,
                     int8_t* omnisci_data_bytes) override {
    const auto& parquet_data_value =
        reinterpret_cast<const int32_t*>(parquet_data_bytes)[0];
    auto& omnisci_data_value = reinterpret_cast<int64_t*>(omnisci_data_bytes)[0];
    omnisci_data_value = parquet_data_value * kSecsPerDay;
  }

  void validate(std::shared_ptr<parquet::Statistics> stats,
                const SQLTypeInfo& column_type) const override {
    CHECK(column_type.is_date());
    if (column_type.get_compression() ==
        kENCODING_NONE) {  // do not validate NONE ENCODED dates as it is impossible for
                           // bounds to be exceeded (the conversion done for this case is
                           // from a date in days as a 32-bit integer to a date in seconds
                           // as a 64-bit integer)
      return;
    }
    auto [unencoded_stats_min, unencoded_stats_max] =
        TypedParquetInPlaceEncoder<int64_t, int32_t>::getUnencodedStats(stats);
    DateInSecondsBoundsValidator<int64_t>::validateValue(
        unencoded_stats_max * kSecsPerDay, column_type);
    DateInSecondsBoundsValidator<int64_t>::validateValue(
        unencoded_stats_min * kSecsPerDay, column_type);
  }
};

}  // namespace foreign_storage
