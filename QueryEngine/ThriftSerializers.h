/*
 * Copyright 2022 HEAVY.AI, Inc.
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
 * @file    ThriftSerializers.h
 * @brief   Serializers for query engine types to/from thrift.
 *
 */

#ifndef QUERYENGINE_THRIFTSERIALIZERS_H
#define QUERYENGINE_THRIFTSERIALIZERS_H

#include "gen-cpp/serialized_result_set_types.h"

#include "Logger/Logger.h"
#include "QueryEngine/AggregatedColRange.h"
#include "QueryEngine/CompilationOptions.h"
#include "QueryEngine/Descriptors/CountDistinctDescriptor.h"
#include "QueryEngine/Descriptors/Types.h"
#include "QueryEngine/ExtensionFunctionsWhitelist.h"
#include "QueryEngine/StringDictionaryGenerations.h"
#include "QueryEngine/TableFunctions/TableFunctionsFactory.h"
#include "QueryEngine/TargetMetaInfo.h"
#include "Shared/ThriftTypesConvert.h"

namespace ThriftSerializers {

#define THRIFT_LAYOUT_CASE(layout)   \
  case QueryDescriptionType::layout: \
    return TResultSetLayout::layout;

inline TResultSetLayout::type layout_to_thrift(const QueryDescriptionType layout) {
  switch (layout) {
    THRIFT_LAYOUT_CASE(GroupByPerfectHash)
    THRIFT_LAYOUT_CASE(GroupByBaselineHash)
    THRIFT_LAYOUT_CASE(Projection)
    THRIFT_LAYOUT_CASE(NonGroupedAggregate)
    default:
      CHECK(false) << static_cast<int>(layout);
  }
  abort();
}

#undef THRIFT_LAYOUT_CASE

#define UNTHRIFT_LAYOUT_CASE(layout) \
  case TResultSetLayout::layout:     \
    return QueryDescriptionType::layout;

inline QueryDescriptionType layout_from_thrift(const TResultSetLayout::type layout) {
  switch (layout) {
    UNTHRIFT_LAYOUT_CASE(GroupByPerfectHash)
    UNTHRIFT_LAYOUT_CASE(GroupByBaselineHash)
    UNTHRIFT_LAYOUT_CASE(Projection)
    UNTHRIFT_LAYOUT_CASE(NonGroupedAggregate)
    default:
      CHECK(false) << static_cast<int>(layout);
  }
  abort();
}

#undef UNTHRIFT_LAYOUT_CASE

#define THRIFT_AGGKIND_CASE(kind) \
  case k##kind:                   \
    return TAggKind::kind;

inline TAggKind::type agg_kind_to_thrift(const SQLAgg agg) {
  switch (agg) {
    THRIFT_AGGKIND_CASE(AVG)
    THRIFT_AGGKIND_CASE(COUNT)
    THRIFT_AGGKIND_CASE(MAX)
    THRIFT_AGGKIND_CASE(MIN)
    THRIFT_AGGKIND_CASE(SUM)
    THRIFT_AGGKIND_CASE(APPROX_COUNT_DISTINCT)
    THRIFT_AGGKIND_CASE(SAMPLE)
    THRIFT_AGGKIND_CASE(SINGLE_VALUE)
    THRIFT_AGGKIND_CASE(COUNT_IF)
    default:
      CHECK(false) << static_cast<int>(agg);
  }
  abort();
}

#undef THRIFT_AGGKIND_CASE

#define UNTHRIFT_AGGKIND_CASE(kind) \
  case TAggKind::kind:              \
    return k##kind;

inline SQLAgg agg_kind_from_thrift(const TAggKind::type agg) {
  switch (agg) {
    UNTHRIFT_AGGKIND_CASE(AVG)
    UNTHRIFT_AGGKIND_CASE(COUNT)
    UNTHRIFT_AGGKIND_CASE(MAX)
    UNTHRIFT_AGGKIND_CASE(MIN)
    UNTHRIFT_AGGKIND_CASE(SUM)
    UNTHRIFT_AGGKIND_CASE(APPROX_COUNT_DISTINCT)
    UNTHRIFT_AGGKIND_CASE(SAMPLE)
    UNTHRIFT_AGGKIND_CASE(SINGLE_VALUE)
    UNTHRIFT_AGGKIND_CASE(COUNT_IF)
    default:
      CHECK(false) << static_cast<int>(agg);
  }
  abort();
}

#undef UNTHRIFT_AGGKIND_CASE

inline AggregatedColRange column_ranges_from_thrift(
    const std::vector<TColumnRange>& thrift_column_ranges) {
  AggregatedColRange column_ranges;
  for (const auto& thrift_column_range : thrift_column_ranges) {
    PhysicalInput phys_input{thrift_column_range.col_id, thrift_column_range.table_id};
    switch (thrift_column_range.type) {
      case TExpressionRangeType::INTEGER:
        column_ranges.setColRange(
            phys_input,
            ExpressionRange::makeIntRange(thrift_column_range.int_min,
                                          thrift_column_range.int_max,
                                          thrift_column_range.bucket,
                                          thrift_column_range.has_nulls));
        break;
      case TExpressionRangeType::FLOAT:
        column_ranges.setColRange(
            phys_input,
            ExpressionRange::makeFloatRange(thrift_column_range.fp_min,
                                            thrift_column_range.fp_max,
                                            thrift_column_range.has_nulls));
        break;
      case TExpressionRangeType::DOUBLE:
        column_ranges.setColRange(
            phys_input,
            ExpressionRange::makeDoubleRange(thrift_column_range.fp_min,
                                             thrift_column_range.fp_max,
                                             thrift_column_range.has_nulls));
        break;
      case TExpressionRangeType::INVALID:
        column_ranges.setColRange(phys_input, ExpressionRange::makeInvalidRange());
        break;
      default:
        CHECK(false);
    }
  }
  return column_ranges;
}

inline StringDictionaryGenerations string_dictionary_generations_from_thrift(
    const std::vector<TDictionaryGeneration>& thrift_string_dictionary_generations) {
  StringDictionaryGenerations string_dictionary_generations;
  for (const auto& thrift_string_dictionary_generation :
       thrift_string_dictionary_generations) {
    string_dictionary_generations.setGeneration(
        thrift_string_dictionary_generation.dict_id,
        thrift_string_dictionary_generation.entry_count);
  }
  return string_dictionary_generations;
}

inline TTypeInfo type_info_to_thrift(const SQLTypeInfo& ti) {
  TTypeInfo thrift_ti;
  thrift_ti.type =
      ti.is_array() ? type_to_thrift(ti.get_elem_type()) : type_to_thrift(ti);
  thrift_ti.encoding = encoding_to_thrift(ti);
  thrift_ti.nullable = !ti.get_notnull();
  thrift_ti.is_array = ti.is_array();
  // TODO: Properly serialize geospatial subtype. For now, the value in precision is the
  // same as the value in scale; overload the precision field with the subtype of the
  // geospatial type (currently kGEOMETRY or kGEOGRAPHY)
  thrift_ti.precision =
      IS_GEO(ti.get_type()) ? static_cast<int32_t>(ti.get_subtype()) : ti.get_precision();
  thrift_ti.scale = ti.get_scale();
  thrift_ti.comp_param = ti.get_comp_param();
  thrift_ti.size = ti.get_size();
  return thrift_ti;
}

inline bool takes_arg(const TargetInfo& target_info) {
  return target_info.is_agg &&
         (target_info.agg_kind != kCOUNT || is_distinct_target(target_info));
}

inline std::vector<TargetMetaInfo> target_meta_infos_from_thrift(
    const TRowDescriptor& row_desc) {
  std::vector<TargetMetaInfo> target_meta_infos;
  for (const auto& col : row_desc) {
    target_meta_infos.emplace_back(col.col_name, type_info_from_thrift(col.col_type));
  }
  return target_meta_infos;
}

inline void fixup_geo_column_descriptor(TColumnType& col_type,
                                        const SQLTypes subtype,
                                        const int output_srid) {
  col_type.col_type.precision = static_cast<int>(subtype);
  col_type.col_type.scale = output_srid;
}

inline TColumnType target_meta_info_to_thrift(const TargetMetaInfo& target,
                                              const size_t idx) {
  TColumnType proj_info;
  proj_info.col_name = target.get_resname();
  if (proj_info.col_name.empty()) {
    proj_info.col_name = "result_" + std::to_string(idx + 1);
  }
  const auto& target_ti = target.get_type_info();
  proj_info.col_type.type = type_to_thrift(target_ti);
  proj_info.col_type.encoding = encoding_to_thrift(target_ti);
  proj_info.col_type.nullable = !target_ti.get_notnull();
  proj_info.col_type.is_array = target_ti.get_type() == kARRAY;
  if (IS_GEO(target_ti.get_type())) {
    fixup_geo_column_descriptor(
        proj_info, target_ti.get_subtype(), target_ti.get_output_srid());
  } else {
    proj_info.col_type.precision = target_ti.get_precision();
    proj_info.col_type.scale = target_ti.get_scale();
  }
  if (target_ti.get_type() == kDATE) {
    proj_info.col_type.size = target_ti.get_size();
  }
  proj_info.col_type.comp_param =
      (target_ti.is_date_in_days() && target_ti.get_comp_param() == 0)
          ? 32
          : target_ti.get_comp_param();
  return proj_info;
}

inline TRowDescriptor target_meta_infos_to_thrift(
    const std::vector<TargetMetaInfo>& targets) {
  TRowDescriptor row_desc;
  size_t i = 0;
  for (const auto& target : targets) {
    row_desc.push_back(target_meta_info_to_thrift(target, i));
    ++i;
  }
  return row_desc;
}

inline TTargetInfo target_info_to_thrift(const TargetInfo& target_info) {
  TTargetInfo thrift_target_info;
  thrift_target_info.is_agg = target_info.is_agg;
  thrift_target_info.kind = agg_kind_to_thrift(target_info.agg_kind);
  thrift_target_info.type = type_info_to_thrift(target_info.sql_type);
  thrift_target_info.arg_type = takes_arg(target_info)
                                    ? type_info_to_thrift(target_info.agg_arg_type)
                                    : thrift_target_info.type;
  thrift_target_info.skip_nulls = target_info.skip_null_val;
  thrift_target_info.is_distinct = target_info.is_distinct;
  return thrift_target_info;
}

inline TargetInfo target_info_from_thrift(const TTargetInfo& thrift_target_info) {
  TargetInfo target_info;
  target_info.is_agg = thrift_target_info.is_agg;
  target_info.agg_kind = agg_kind_from_thrift(thrift_target_info.kind);
  target_info.sql_type = type_info_from_thrift(thrift_target_info.type);
  target_info.is_distinct = thrift_target_info.is_distinct;
  target_info.agg_arg_type = takes_arg(target_info)
                                 ? type_info_from_thrift(thrift_target_info.arg_type)
                                 : SQLTypeInfo(kNULLT, false);
  target_info.skip_null_val = thrift_target_info.skip_nulls;
  return target_info;
}

inline std::vector<TTargetInfo> target_infos_to_thrift(
    const std::vector<TargetInfo>& targets) {
  std::vector<TTargetInfo> thrift_targets;
  for (const auto& target_info : targets) {
    thrift_targets.push_back(target_info_to_thrift(target_info));
  }
  return thrift_targets;
}

inline std::vector<TargetInfo> target_infos_from_thrift(
    const std::vector<TTargetInfo>& thrift_targets) {
  std::vector<TargetInfo> targets;
  for (const auto& thrift_target_info : thrift_targets) {
    targets.push_back(target_info_from_thrift(thrift_target_info));
  }
  return targets;
}

#define THRIFT_COUNTDESCRIPTORIMPL_CASE(kind) \
  case CountDistinctImplType::kind:           \
    return TCountDistinctImplType::kind;

inline TCountDistinctImplType::type count_distinct_impl_type_to_thrift(
    const CountDistinctImplType impl_type) {
  switch (impl_type) {
    THRIFT_COUNTDESCRIPTORIMPL_CASE(Invalid)
    THRIFT_COUNTDESCRIPTORIMPL_CASE(Bitmap)
    THRIFT_COUNTDESCRIPTORIMPL_CASE(UnorderedSet)
    default:
      CHECK(false);
  }
  abort();
}

#undef THRIFT_COUNTDESCRIPTORIMPL_CASE

inline TCountDistinctDescriptor count_distinct_descriptor_to_thrift(
    const CountDistinctDescriptor& count_distinct_descriptor) {
  TCountDistinctDescriptor thrift_count_distinct_descriptor;
  thrift_count_distinct_descriptor.impl_type =
      count_distinct_impl_type_to_thrift(count_distinct_descriptor.impl_type_);
  thrift_count_distinct_descriptor.min_val = count_distinct_descriptor.min_val;
  thrift_count_distinct_descriptor.bitmap_sz_bits =
      count_distinct_descriptor.bitmap_sz_bits;
  thrift_count_distinct_descriptor.approximate = count_distinct_descriptor.approximate;
  thrift_count_distinct_descriptor.device_type =
      count_distinct_descriptor.device_type == ExecutorDeviceType::GPU ? TDeviceType::GPU
                                                                       : TDeviceType::CPU;
  thrift_count_distinct_descriptor.sub_bitmap_count =
      count_distinct_descriptor.sub_bitmap_count;
  return thrift_count_distinct_descriptor;
}

#define UNTHRIFT_COUNTDESCRIPTORIMPL_CASE(kind) \
  case TCountDistinctImplType::kind:            \
    return CountDistinctImplType::kind;

inline CountDistinctImplType count_distinct_impl_type_from_thrift(
    const TCountDistinctImplType::type impl_type) {
  switch (impl_type) {
    UNTHRIFT_COUNTDESCRIPTORIMPL_CASE(Invalid)
    UNTHRIFT_COUNTDESCRIPTORIMPL_CASE(Bitmap)
    UNTHRIFT_COUNTDESCRIPTORIMPL_CASE(UnorderedSet)
    default:
      CHECK(false);
  }
  abort();
}

#undef UNTHRIFT_COUNTDESCRIPTORIMPL_CASE

inline CountDistinctDescriptor count_distinct_descriptor_from_thrift(
    const TCountDistinctDescriptor& thrift_count_distinct_descriptor) {
  CountDistinctDescriptor count_distinct_descriptor;
  count_distinct_descriptor.impl_type_ =
      count_distinct_impl_type_from_thrift(thrift_count_distinct_descriptor.impl_type);
  count_distinct_descriptor.min_val = thrift_count_distinct_descriptor.min_val;
  count_distinct_descriptor.bitmap_sz_bits =
      thrift_count_distinct_descriptor.bitmap_sz_bits;
  count_distinct_descriptor.approximate = thrift_count_distinct_descriptor.approximate;
  count_distinct_descriptor.device_type =
      thrift_count_distinct_descriptor.device_type == TDeviceType::GPU
          ? ExecutorDeviceType::GPU
          : ExecutorDeviceType::CPU;
  count_distinct_descriptor.sub_bitmap_count =
      thrift_count_distinct_descriptor.sub_bitmap_count;
  return count_distinct_descriptor;
}

inline ExtArgumentType from_thrift(const TExtArgumentType::type& t) {
  switch (t) {
    case TExtArgumentType::Int8:
      return ExtArgumentType::Int8;
    case TExtArgumentType::Int16:
      return ExtArgumentType::Int16;
    case TExtArgumentType::Int32:
      return ExtArgumentType::Int32;
    case TExtArgumentType::Int64:
      return ExtArgumentType::Int64;
    case TExtArgumentType::Float:
      return ExtArgumentType::Float;
    case TExtArgumentType::Double:
      return ExtArgumentType::Double;
    case TExtArgumentType::Void:
      return ExtArgumentType::Void;
    case TExtArgumentType::PInt8:
      return ExtArgumentType::PInt8;
    case TExtArgumentType::PInt16:
      return ExtArgumentType::PInt16;
    case TExtArgumentType::PInt32:
      return ExtArgumentType::PInt32;
    case TExtArgumentType::PInt64:
      return ExtArgumentType::PInt64;
    case TExtArgumentType::PFloat:
      return ExtArgumentType::PFloat;
    case TExtArgumentType::PDouble:
      return ExtArgumentType::PDouble;
    case TExtArgumentType::PBool:
      return ExtArgumentType::PBool;
    case TExtArgumentType::Bool:
      return ExtArgumentType::Bool;
    case TExtArgumentType::ArrayInt8:
      return ExtArgumentType::ArrayInt8;
    case TExtArgumentType::ArrayInt16:
      return ExtArgumentType::ArrayInt16;
    case TExtArgumentType::ArrayInt32:
      return ExtArgumentType::ArrayInt32;
    case TExtArgumentType::ArrayInt64:
      return ExtArgumentType::ArrayInt64;
    case TExtArgumentType::ArrayFloat:
      return ExtArgumentType::ArrayFloat;
    case TExtArgumentType::ArrayDouble:
      return ExtArgumentType::ArrayDouble;
    case TExtArgumentType::ArrayBool:
      return ExtArgumentType::ArrayBool;
    case TExtArgumentType::ArrayTextEncodingNone:
      return ExtArgumentType::ArrayTextEncodingNone;
    case TExtArgumentType::ArrayTextEncodingDict:
      return ExtArgumentType::ArrayTextEncodingDict;
    case TExtArgumentType::GeoPoint:
      return ExtArgumentType::GeoPoint;
    case TExtArgumentType::GeoMultiPoint:
      return ExtArgumentType::GeoMultiPoint;
    case TExtArgumentType::GeoLineString:
      return ExtArgumentType::GeoLineString;
    case TExtArgumentType::GeoMultiLineString:
      return ExtArgumentType::GeoMultiLineString;
    case TExtArgumentType::Cursor:
      return ExtArgumentType::Cursor;
    case TExtArgumentType::GeoPolygon:
      return ExtArgumentType::GeoPolygon;
    case TExtArgumentType::GeoMultiPolygon:
      return ExtArgumentType::GeoMultiPolygon;
    case TExtArgumentType::ColumnInt8:
      return ExtArgumentType::ColumnInt8;
    case TExtArgumentType::ColumnInt16:
      return ExtArgumentType::ColumnInt16;
    case TExtArgumentType::ColumnInt32:
      return ExtArgumentType::ColumnInt32;
    case TExtArgumentType::ColumnInt64:
      return ExtArgumentType::ColumnInt64;
    case TExtArgumentType::ColumnFloat:
      return ExtArgumentType::ColumnFloat;
    case TExtArgumentType::ColumnDouble:
      return ExtArgumentType::ColumnDouble;
    case TExtArgumentType::ColumnBool:
      return ExtArgumentType::ColumnBool;
    case TExtArgumentType::ColumnTextEncodingNone:
      return ExtArgumentType::ColumnTextEncodingNone;
    case TExtArgumentType::ColumnTextEncodingDict:
      return ExtArgumentType::ColumnTextEncodingDict;
    case TExtArgumentType::ColumnTimestamp:
      return ExtArgumentType::ColumnTimestamp;
    case TExtArgumentType::TextEncodingNone:
      return ExtArgumentType::TextEncodingNone;
    case TExtArgumentType::TextEncodingDict:
      return ExtArgumentType::TextEncodingDict;
    case TExtArgumentType::Timestamp:
      return ExtArgumentType::Timestamp;
    case TExtArgumentType::ColumnListInt8:
      return ExtArgumentType::ColumnListInt8;
    case TExtArgumentType::ColumnListInt16:
      return ExtArgumentType::ColumnListInt16;
    case TExtArgumentType::ColumnListInt32:
      return ExtArgumentType::ColumnListInt32;
    case TExtArgumentType::ColumnListInt64:
      return ExtArgumentType::ColumnListInt64;
    case TExtArgumentType::ColumnListFloat:
      return ExtArgumentType::ColumnListFloat;
    case TExtArgumentType::ColumnListDouble:
      return ExtArgumentType::ColumnListDouble;
    case TExtArgumentType::ColumnListBool:
      return ExtArgumentType::ColumnListBool;
    case TExtArgumentType::ColumnListTextEncodingNone:
      return ExtArgumentType::ColumnListTextEncodingNone;
    case TExtArgumentType::ColumnListTextEncodingDict:
      return ExtArgumentType::ColumnListTextEncodingDict;
    case TExtArgumentType::ColumnArrayInt8:
      return ExtArgumentType::ColumnArrayInt8;
    case TExtArgumentType::ColumnArrayInt16:
      return ExtArgumentType::ColumnArrayInt16;
    case TExtArgumentType::ColumnArrayInt32:
      return ExtArgumentType::ColumnArrayInt32;
    case TExtArgumentType::ColumnArrayInt64:
      return ExtArgumentType::ColumnArrayInt64;
    case TExtArgumentType::ColumnArrayFloat:
      return ExtArgumentType::ColumnArrayFloat;
    case TExtArgumentType::ColumnArrayDouble:
      return ExtArgumentType::ColumnArrayDouble;
    case TExtArgumentType::ColumnArrayBool:
      return ExtArgumentType::ColumnArrayBool;
    case TExtArgumentType::ColumnArrayTextEncodingNone:
      return ExtArgumentType::ColumnArrayTextEncodingNone;
    case TExtArgumentType::ColumnArrayTextEncodingDict:
      return ExtArgumentType::ColumnArrayTextEncodingDict;
    case TExtArgumentType::ColumnListArrayInt8:
      return ExtArgumentType::ColumnListArrayInt8;
    case TExtArgumentType::ColumnListArrayInt16:
      return ExtArgumentType::ColumnListArrayInt16;
    case TExtArgumentType::ColumnListArrayInt32:
      return ExtArgumentType::ColumnListArrayInt32;
    case TExtArgumentType::ColumnListArrayInt64:
      return ExtArgumentType::ColumnListArrayInt64;
    case TExtArgumentType::ColumnListArrayFloat:
      return ExtArgumentType::ColumnListArrayFloat;
    case TExtArgumentType::ColumnListArrayDouble:
      return ExtArgumentType::ColumnListArrayDouble;
    case TExtArgumentType::ColumnListArrayBool:
      return ExtArgumentType::ColumnListArrayBool;
    case TExtArgumentType::ColumnListArrayTextEncodingNone:
      return ExtArgumentType::ColumnListArrayTextEncodingNone;
    case TExtArgumentType::ColumnListArrayTextEncodingDict:
      return ExtArgumentType::ColumnListArrayTextEncodingDict;
    case TExtArgumentType::DayTimeInterval:
      return ExtArgumentType::DayTimeInterval;
    case TExtArgumentType::YearMonthTimeInterval:
      return ExtArgumentType::YearMonthTimeInterval;
  }
  UNREACHABLE();
  return ExtArgumentType{};
}

inline TExtArgumentType::type to_thrift(const ExtArgumentType& t) {
  switch (t) {
    case ExtArgumentType::Int8:
      return TExtArgumentType::Int8;
    case ExtArgumentType::Int16:
      return TExtArgumentType::Int16;
    case ExtArgumentType::Int32:
      return TExtArgumentType::Int32;
    case ExtArgumentType::Int64:
      return TExtArgumentType::Int64;
    case ExtArgumentType::Float:
      return TExtArgumentType::Float;
    case ExtArgumentType::Double:
      return TExtArgumentType::Double;
    case ExtArgumentType::Void:
      return TExtArgumentType::Void;
    case ExtArgumentType::PInt8:
      return TExtArgumentType::PInt8;
    case ExtArgumentType::PInt16:
      return TExtArgumentType::PInt16;
    case ExtArgumentType::PInt32:
      return TExtArgumentType::PInt32;
    case ExtArgumentType::PInt64:
      return TExtArgumentType::PInt64;
    case ExtArgumentType::PFloat:
      return TExtArgumentType::PFloat;
    case ExtArgumentType::PDouble:
      return TExtArgumentType::PDouble;
    case ExtArgumentType::PBool:
      return TExtArgumentType::PBool;
    case ExtArgumentType::Bool:
      return TExtArgumentType::Bool;
    case ExtArgumentType::ArrayInt8:
      return TExtArgumentType::ArrayInt8;
    case ExtArgumentType::ArrayInt16:
      return TExtArgumentType::ArrayInt16;
    case ExtArgumentType::ArrayInt32:
      return TExtArgumentType::ArrayInt32;
    case ExtArgumentType::ArrayInt64:
      return TExtArgumentType::ArrayInt64;
    case ExtArgumentType::ArrayFloat:
      return TExtArgumentType::ArrayFloat;
    case ExtArgumentType::ArrayDouble:
      return TExtArgumentType::ArrayDouble;
    case ExtArgumentType::ArrayBool:
      return TExtArgumentType::ArrayBool;
    case ExtArgumentType::ArrayTextEncodingNone:
      return TExtArgumentType::ArrayTextEncodingNone;
    case ExtArgumentType::ArrayTextEncodingDict:
      return TExtArgumentType::ArrayTextEncodingDict;
    case ExtArgumentType::GeoPoint:
      return TExtArgumentType::GeoPoint;
    case ExtArgumentType::GeoMultiPoint:
      return TExtArgumentType::GeoMultiPoint;
    case ExtArgumentType::GeoLineString:
      return TExtArgumentType::GeoLineString;
    case ExtArgumentType::GeoMultiLineString:
      return TExtArgumentType::GeoMultiLineString;
    case ExtArgumentType::Cursor:
      return TExtArgumentType::Cursor;
    case ExtArgumentType::GeoPolygon:
      return TExtArgumentType::GeoPolygon;
    case ExtArgumentType::GeoMultiPolygon:
      return TExtArgumentType::GeoMultiPolygon;
    case ExtArgumentType::ColumnInt8:
      return TExtArgumentType::ColumnInt8;
    case ExtArgumentType::ColumnInt16:
      return TExtArgumentType::ColumnInt16;
    case ExtArgumentType::ColumnInt32:
      return TExtArgumentType::ColumnInt32;
    case ExtArgumentType::ColumnInt64:
      return TExtArgumentType::ColumnInt64;
    case ExtArgumentType::ColumnFloat:
      return TExtArgumentType::ColumnFloat;
    case ExtArgumentType::ColumnDouble:
      return TExtArgumentType::ColumnDouble;
    case ExtArgumentType::ColumnBool:
      return TExtArgumentType::ColumnBool;
    case ExtArgumentType::ColumnTextEncodingNone:
      return TExtArgumentType::ColumnTextEncodingNone;
    case ExtArgumentType::ColumnTextEncodingDict:
      return TExtArgumentType::ColumnTextEncodingDict;
    case ExtArgumentType::ColumnTimestamp:
      return TExtArgumentType::ColumnTimestamp;
    case ExtArgumentType::TextEncodingNone:
      return TExtArgumentType::TextEncodingNone;
    case ExtArgumentType::TextEncodingDict:
      return TExtArgumentType::TextEncodingDict;
    case ExtArgumentType::Timestamp:
      return TExtArgumentType::Timestamp;
    case ExtArgumentType::ColumnListInt8:
      return TExtArgumentType::ColumnListInt8;
    case ExtArgumentType::ColumnListInt16:
      return TExtArgumentType::ColumnListInt16;
    case ExtArgumentType::ColumnListInt32:
      return TExtArgumentType::ColumnListInt32;
    case ExtArgumentType::ColumnListInt64:
      return TExtArgumentType::ColumnListInt64;
    case ExtArgumentType::ColumnListFloat:
      return TExtArgumentType::ColumnListFloat;
    case ExtArgumentType::ColumnListDouble:
      return TExtArgumentType::ColumnListDouble;
    case ExtArgumentType::ColumnListBool:
      return TExtArgumentType::ColumnListBool;
    case ExtArgumentType::ColumnListTextEncodingNone:
      return TExtArgumentType::ColumnListTextEncodingNone;
    case ExtArgumentType::ColumnListTextEncodingDict:
      return TExtArgumentType::ColumnListTextEncodingDict;
    case ExtArgumentType::ColumnArrayInt8:
      return TExtArgumentType::ColumnArrayInt8;
    case ExtArgumentType::ColumnArrayInt16:
      return TExtArgumentType::ColumnArrayInt16;
    case ExtArgumentType::ColumnArrayInt32:
      return TExtArgumentType::ColumnArrayInt32;
    case ExtArgumentType::ColumnArrayInt64:
      return TExtArgumentType::ColumnArrayInt64;
    case ExtArgumentType::ColumnArrayFloat:
      return TExtArgumentType::ColumnArrayFloat;
    case ExtArgumentType::ColumnArrayDouble:
      return TExtArgumentType::ColumnArrayDouble;
    case ExtArgumentType::ColumnArrayBool:
      return TExtArgumentType::ColumnArrayBool;
    case ExtArgumentType::ColumnArrayTextEncodingNone:
      return TExtArgumentType::ColumnArrayTextEncodingNone;
    case ExtArgumentType::ColumnArrayTextEncodingDict:
      return TExtArgumentType::ColumnArrayTextEncodingDict;
    case ExtArgumentType::ColumnListArrayInt8:
      return TExtArgumentType::ColumnListArrayInt8;
    case ExtArgumentType::ColumnListArrayInt16:
      return TExtArgumentType::ColumnListArrayInt16;
    case ExtArgumentType::ColumnListArrayInt32:
      return TExtArgumentType::ColumnListArrayInt32;
    case ExtArgumentType::ColumnListArrayInt64:
      return TExtArgumentType::ColumnListArrayInt64;
    case ExtArgumentType::ColumnListArrayFloat:
      return TExtArgumentType::ColumnListArrayFloat;
    case ExtArgumentType::ColumnListArrayDouble:
      return TExtArgumentType::ColumnListArrayDouble;
    case ExtArgumentType::ColumnListArrayBool:
      return TExtArgumentType::ColumnListArrayBool;
    case ExtArgumentType::ColumnListArrayTextEncodingDict:
      return TExtArgumentType::ColumnListArrayTextEncodingDict;
    case ExtArgumentType::ColumnListArrayTextEncodingNone:
      return TExtArgumentType::ColumnListArrayTextEncodingNone;
    case ExtArgumentType::DayTimeInterval:
      return TExtArgumentType::DayTimeInterval;
    case ExtArgumentType::YearMonthTimeInterval:
      return TExtArgumentType::YearMonthTimeInterval;
  }
  UNREACHABLE();
  return TExtArgumentType::type{};
}

inline std::vector<ExtArgumentType> from_thrift(
    const std::vector<TExtArgumentType::type>& v) {
  std::vector<ExtArgumentType> result;
  std::transform(
      v.begin(),
      v.end(),
      std::back_inserter(result),
      [](TExtArgumentType::type c) -> ExtArgumentType { return from_thrift(c); });
  return result;
}

inline std::vector<TExtArgumentType::type> to_thrift(
    const std::vector<ExtArgumentType>& v) {
  std::vector<TExtArgumentType::type> result;
  std::transform(
      v.begin(),
      v.end(),
      std::back_inserter(result),
      [](ExtArgumentType c) -> TExtArgumentType::type { return to_thrift(c); });
  return result;
}

inline table_functions::OutputBufferSizeType from_thrift(
    const TOutputBufferSizeType::type& t) {
  switch (t) {
    case TOutputBufferSizeType::kConstant:
      return table_functions::OutputBufferSizeType::kConstant;
    case TOutputBufferSizeType::kUserSpecifiedConstantParameter:
      return table_functions::OutputBufferSizeType::kUserSpecifiedConstantParameter;
    case TOutputBufferSizeType::kUserSpecifiedRowMultiplier:
      return table_functions::OutputBufferSizeType::kUserSpecifiedRowMultiplier;
    case TOutputBufferSizeType::kTableFunctionSpecifiedParameter:
      return table_functions::OutputBufferSizeType::kTableFunctionSpecifiedParameter;
    case TOutputBufferSizeType::kPreFlightParameter:
      return table_functions::OutputBufferSizeType::kPreFlightParameter;
  }
  UNREACHABLE();
  return table_functions::OutputBufferSizeType{};
}

inline TOutputBufferSizeType::type to_thrift(
    const table_functions::OutputBufferSizeType& t) {
  switch (t) {
    case table_functions::OutputBufferSizeType::kConstant:
      return TOutputBufferSizeType::kConstant;
    case table_functions::OutputBufferSizeType::kUserSpecifiedConstantParameter:
      return TOutputBufferSizeType::kUserSpecifiedConstantParameter;
    case table_functions::OutputBufferSizeType::kUserSpecifiedRowMultiplier:
      return TOutputBufferSizeType::kUserSpecifiedRowMultiplier;
    case table_functions::OutputBufferSizeType::kTableFunctionSpecifiedParameter:
      return TOutputBufferSizeType::kTableFunctionSpecifiedParameter;
    case table_functions::OutputBufferSizeType::kPreFlightParameter:
      return TOutputBufferSizeType::kPreFlightParameter;
  }
  UNREACHABLE();
  return TOutputBufferSizeType::type{};
}

inline TUserDefinedFunction to_thrift(const ExtensionFunction& udf) {
  TUserDefinedFunction tfunc;
  tfunc.name = udf.getName(/* keep_suffix */ true);
  tfunc.argTypes = to_thrift(udf.getInputArgs());
  tfunc.retType = to_thrift(udf.getRet());
  tfunc.usesManager = udf.usesManager();
  return tfunc;
}

inline TUserDefinedTableFunction to_thrift(const table_functions::TableFunction& func) {
  TUserDefinedTableFunction tfunc;
  tfunc.name = func.getName();
  tfunc.sizerType = to_thrift(func.getOutputRowSizeType());
  tfunc.sizerArgPos = func.getOutputRowSizeParameter();
  tfunc.inputArgTypes = to_thrift(func.getInputArgs());
  tfunc.outputArgTypes = to_thrift(func.getOutputArgs());
  tfunc.sqlArgTypes = to_thrift(func.getSqlArgs());
  tfunc.annotations = func.getAnnotations();
  return tfunc;
}

inline std::vector<TUserDefinedTableFunction> to_thrift(
    const std::vector<table_functions::TableFunction>& v) {
  std::vector<TUserDefinedTableFunction> result;
  std::transform(v.begin(),
                 v.end(),
                 std::back_inserter(result),
                 [](table_functions::TableFunction c) -> TUserDefinedTableFunction {
                   return to_thrift(c);
                 });
  return result;
}

}  // namespace ThriftSerializers

#endif  // QUERYENGINE_THRIFTSERIALIZERS_H
