/*
 * Copyright 2018 MapD Technologies, Inc.
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
 * @author  Alex Baden <alex.baden@mapd.com>
 * @brief   Serializers for query engine types to/from thrift.
 */

#ifndef QUERYENGINE_THRIFTSERIALIZERS_H
#define QUERYENGINE_THRIFTSERIALIZERS_H

#include "gen-cpp/serialized_result_set_types.h"

#include "Logger/Logger.h"
#include "QueryEngine/AggregatedColRange.h"
#include "QueryEngine/CompilationOptions.h"
#include "QueryEngine/Descriptors/CountDistinctDescriptor.h"
#include "QueryEngine/Descriptors/Types.h"
#include "QueryEngine/StringDictionaryGenerations.h"
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
      CHECK(false);
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
      CHECK(false);
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
    default:
      CHECK(false);
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
    THRIFT_COUNTDESCRIPTORIMPL_CASE(StdSet)
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
    UNTHRIFT_COUNTDESCRIPTORIMPL_CASE(StdSet)
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

}  // namespace ThriftSerializers

#endif  // QUERYENGINE_THRIFTSERIALIZERS_H
