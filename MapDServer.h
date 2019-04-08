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

/*
 * @file    MapDServer.h
 * @author  Alex Suhan <alex@mapd.com>
 *
 * Copyright (c) 2015 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef MAPDSERVER_H
#define MAPDSERVER_H

#include "QueryEngine/AggregatedColRange.h"
#include "QueryEngine/ExtractFromTime.h"
#include "QueryEngine/StringDictionaryGenerations.h"
#include "QueryEngine/TableGenerations.h"
#include "QueryEngine/TargetMetaInfo.h"
#include "Shared/ThriftTypesConvert.h"

inline std::vector<TargetMetaInfo> target_meta_infos_from_thrift(
    const TRowDescriptor& row_desc) {
  std::vector<TargetMetaInfo> target_meta_infos;
  for (const auto& col : row_desc) {
    target_meta_infos.emplace_back(col.col_name, type_info_from_thrift(col.col_type));
  }
  return target_meta_infos;
}

AggregatedColRange column_ranges_from_thrift(
    const std::vector<TColumnRange>& thrift_column_ranges);

StringDictionaryGenerations string_dictionary_generations_from_thrift(
    const std::vector<TDictionaryGeneration>& thrift_string_dictionary_generations);

TableGenerations table_generations_from_thrift(
    const std::vector<TTableGeneration>& table_generations);

#endif  // MAPDSERVER_H
