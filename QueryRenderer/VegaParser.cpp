/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/VegaParser.h"

#include <rapidjson/error/en.h>

#include "GfxDriver/RenderError.h"
#include "GfxDriver/RenderLogger.h"
#include "GfxDriver/ShaderCompiler/SpirvArtifacts.h"
#include "GfxDriver/ShaderCompiler/Types.h"
#include "QueryRenderer/Data/QueryDataTableQueues.h"
#include "QueryRenderer/Data/Types.h"
#include "QueryRenderer/Data/Utils.h"
#include "QueryRenderer/Events/RefEvent.h"
#include "QueryRenderer/JSONRefObject.h"
#include "QueryRenderer/Marks/BaseMark.h"
#include "QueryRenderer/Marks/Utils.h"
#include "QueryRenderer/Projections/Projection.h"
#include "QueryRenderer/Projections/Utils.h"
#include "QueryRenderer/QueryRendererContext.h"
#include "QueryRenderer/Scales/Scale.h"
#include "QueryRenderer/Scales/ScaleAccumState.h"
#include "QueryRenderer/Scales/ScaleFactory.h"
#include "QueryRenderer/VegaElements.h"

namespace QueryRenderer {

namespace JSONSchema_v1 {
constexpr char kMetaProp[] = "usermeta";
constexpr char kShortNameProp[] = "shortName";
constexpr char kShaderArtifactsProp[] = "shaderArtifacts";
constexpr char kClearCachesProp[] = "clearCaches";
constexpr char kDataProp[] = "data";
constexpr char kProjProp[] = "projections";
constexpr char kScaleProp[] = "scales";
constexpr char kMarkProp[] = "marks";
constexpr char kWidthProp[] = "width";
constexpr char kHeightProp[] = "height";
constexpr char kViewRenderOptionsProp[] = "viewRenderOptions";
constexpr char kPremultipliedAlphaProp[] = "premultipliedAlpha";
}  // namespace JSONSchema_v1

VegaParser::VegaParser(QueryRendererContext& ctx) : ctx_(ctx) {}

void VegaParser::parse(const std::string& json_str) {
  RENDER_LOG_SCOPE();
  try {
    parseInternal(json_str);

    // now update the data
    ctx_.updateConfigGpuResources();

    // clean up any update/changed state flags
    ctx_.postJSONUpdate();
  } catch (...) {
    // Context will be partially initialized, so just clear its state completely
    ctx_.clear();
    std::rethrow_exception(std::current_exception());
  }
}

namespace {

/**
 * Internal struct for storing metadata for vega data elements. Metadata includes:
 *  - vega json location info
 *  - data element name
 *  - data element's input format (sql, embedded, etc.)
 *  - data element's output format (i.e. poly/line/other)
 */
struct DataElementInfo {
  explicit DataElementInfo(std::pair<DataInputFormat, DataOutputFormat> io_formats,
                           JSONLocation in_json_location)
      : json_location(std::move(in_json_location))
      , name(getDataTableNameFromJSONObj(json_location))
      , input_format(io_formats.first)
      , output_format(io_formats.second) {}

  const JSONLocation json_location;
  const std::string name;
  DataInputFormat input_format;
  DataOutputFormat output_format;
};

}  // namespace

void VegaParser::parseMetadata(const JSONLocation& root_loc) {
  RENDER_LOG_SCOPE();
  const auto meta_loc =
      root_loc.getMember(JSONSchema_v1::kMetaProp, JSONValueType::kObject, false);

  if (meta_loc.isValid()) {
    auto metadata = std::make_unique<VegaMetaData>();

    const auto shortname_loc =
        meta_loc.getMember(JSONSchema_v1::kShortNameProp, JSONValueType::kString, false);
    if (shortname_loc.isValid()) {
      metadata->short_name = meta_loc[JSONSchema_v1::kShortNameProp].getString();
      LOG(INFO) << "Vega name: " << metadata->short_name;
      RENDER_LOG() << "Vega name: " << metadata->short_name;
    }
    const auto artifacts = meta_loc.getMember(
        JSONSchema_v1::kShaderArtifactsProp, JSONValueType::kArray, false);
    if (artifacts.isValid()) {
      int bits = gfx::ShaderArtifactTypeBits::kNone;
      for (size_t i = 0; i < artifacts.size(); ++i) {
        bits |= gfx::string_to_shader_artifact_type(
            artifacts.getArrayMember(i, JSONValueType::kString).getString());
      }
      metadata->shader_artifacts_to_save = static_cast<gfx::ShaderArtifactTypeBits>(bits);
    }

    const auto clear_caches_json =
        meta_loc.getMember(JSONSchema_v1::kClearCachesProp, JSONValueType::kBool, false);
    if (clear_caches_json.isValid()) {
      metadata->clear_caches = clear_caches_json.getBool();
    }

    if (metadata->clear_caches) {
      ctx_.clear();
    }

    ctx_.vega_elements_->replaceMetaData(std::move(metadata));
  } else {
    ctx_.vega_elements_->clearMetaData();
  }
}

void VegaParser::parseViewRenderOptions(const JSONLocation& root_loc) {
  RENDER_LOG_SCOPE();
  //
  // Render dimensions
  //
  {
    const auto width_loc =
        root_loc.getMember(JSONSchema_v1::kWidthProp, JSONValueType::kUInt, true);
    size_t width = width_loc.getUint();

    const auto height_loc =
        root_loc.getMember(JSONSchema_v1::kHeightProp, JSONValueType::kUInt, true);
    size_t height = height_loc.getUint();

    ctx_.setWidthHeight(width, height);
  }

  //
  // View Render options
  //
  {
    ViewRenderOptions vro;
    const auto view_render_opts_loc = root_loc.getMember(
        JSONSchema_v1::kViewRenderOptionsProp, JSONValueType::kObject, false);

    if (view_render_opts_loc.isValid()) {
      const auto premultiplied_alpha_loc = view_render_opts_loc.getMember(
          JSONSchema_v1::kPremultipliedAlphaProp, JSONValueType::kBool, false);

      if (premultiplied_alpha_loc.isValid()) {
        vro.premultiplied_alpha = premultiplied_alpha_loc.getBool();
      }
    }
    ctx_.setViewRenderOptions(vro);
  }
}

void VegaParser::parseData(const JSONLocation& root_loc) {
  RENDER_LOG_SCOPE();
  auto& data_table_queues = ctx_.getDataTableQueues();

  auto data_loc =
      root_loc.getMember(JSONSchema_v1::kDataProp, JSONValueType::kArray, false);
  if (data_loc.isValid()) {
    std::unordered_set<std::string> visited_names;
    std::unordered_set<std::string> unvisited_names;
    std::unordered_set<std::string> sourced_names;

    auto& data_table_map = ctx_.vega_elements_->getDataTableMap();
    unvisited_names.reserve(data_table_map.size());
    for (auto kv : data_table_map) {
      unvisited_names.insert(kv->getNameRef());
    }

    // Create a new data table or replace an existing one
    auto create_or_update_data_table =
        [this, &unvisited_names, &visited_names, &data_table_queues](
            const DataElementInfo& data_element, const bool force) {
          auto data_table = ctx_.getDataTable(data_element.name);
          auto& data_table_map = ctx_.vega_elements_->getDataTableMap();
          if (!data_table) {
            //
            // Create new DataTable
            //
            data_table =
                createDataTable(data_element.json_location, ctx_, data_element.name);
            auto data_table_json =
                std::dynamic_pointer_cast<BaseQueryDataTableSQLJSON>(data_table);
            CHECK(data_table_json);
            // Update Sql and internals, and run query if required after
            data_table_json->initFromJSONObjAndQueueQuery(data_element.json_location);
            data_table_map.push_back(data_table_json);
          } else {
            //
            // Existing DataTable is changing
            //
            // TODO(croot): Need to validate any previously
            // existing references. One way to do this is store a map of all objects
            // changing in-place in order to validate.
            if (data_table->getOutputFormat() != data_element.output_format ||
                data_table->getInputFormat() != data_element.input_format) {
              //
              // IOFormat changed: replace DataTable
              //
              CHECK(ctx_.hasDataTable(data_element.name));
              data_table =
                  createDataTable(data_element.json_location, ctx_, data_element.name);
              auto data_table_json =
                  std::dynamic_pointer_cast<BaseQueryDataTableSQLJSON>(data_table);
              CHECK(data_table_json);

              // Update Sql and internals, and run query if required after
              // SourceDataTable does not queue a query but will create its XForm operator
              // Creating XFormOp will add to SourceQueue
              data_table_json->initFromJSONObjAndQueueQuery(data_element.json_location);

              // Replace existing map element and broadcast notification
              data_table_map.replace(data_element.name, data_table_json);
              data_table_queues.addToNotifyQueue(data_table_json, RefEventType::kReplace);
            } else {
              //
              // Update: Data table changed, but input / output formats unchanged
              //
              auto data_table_json =
                  std::dynamic_pointer_cast<BaseQueryDataTableSQLJSON>(data_table);
              CHECK(data_table_json);
              if (data_table_json->updateFromJSONObjAndQueueQuery(
                      data_element.json_location, force)) {
                // NOTE: All Marks pass this through to kReplace
                data_table_queues.addToNotifyQueue(data_table_json,
                                                   RefEventType::kUpdate);
              }
            }
          }

          unvisited_names.erase(data_element.name);
          visited_names.insert(data_element.name);
        };

    // NOTE: force_multi_layer_data_update is a bandaid fix for BE-5760. It will force
    // the re-execution of all sql queries if there is any difference with the data
    // section of sql-based data elements between successive renders. It is considered
    // a temporary bandaid in favor of correctness. See the jira issue for more
    // details.
    //
    // NOTE: this works under the assumption that the query is executed upon vega
    // parse. If that were to change to a lazy system, then this logic would likely
    // need changing, or could possibly be removed altogether.
    bool force_multi_layer_data_update = false;

    // need to iterate all the data elements in the vega to check for differences.
    // We'll cache the metadata for those elements here for use later when
    // creating/updating nodes.
    std::vector<DataElementInfo> all_data_elements;
    std::vector<size_t> data_table_indices;
    std::vector<size_t> source_data_table_indices;
    for (size_t i = 0; i < data_loc.size(); ++i) {
      auto const& data_element = all_data_elements.emplace_back(
          getDataIOFormatsFromJSONObj(data_loc[i]), data_loc[i]);
      if (!force_multi_layer_data_update && data_loc.size() > 1 &&
          data_element.input_format == DataInputFormat::kSQL) {
        auto const curr_table_idx = ctx_.getDataIndex(data_element.name);
        if (!curr_table_idx ||
            !ctx_.isJSONCacheUpToDate(
                ctx_.vega_elements_->getDataTableMap()[*curr_table_idx]->getJsonPathRef(),
                data_element.json_location)) {
          force_multi_layer_data_update = true;
        }
      }
      if (data_element.input_format == DataInputFormat::kSourced) {
        source_data_table_indices.push_back(i);
      } else {
        data_table_indices.push_back(i);
      }
    }

    //
    // Iterate through regular data table elements
    //
    for (auto i : data_table_indices) {
      bool force = false;
      const auto data_item_loc = data_loc[i];
      auto const& data_element = all_data_elements[i];
      auto const& this_table_name = data_element.name;

      RUNTIME_EX_ASSERT(
          visited_names.find(this_table_name) == visited_names.end(),
          RapidJSONUtils::createJsonParseError(
              data_item_loc,
              "a data table with the name \"" + this_table_name + "\" already exists."));

      create_or_update_data_table(data_element, force_multi_layer_data_update || force);
    }

    // Now loop over the source data tables, check if the referenced table is valid
    // then create the SourceDataTable itself
    for (auto i : source_data_table_indices) {
      const auto data_item_loc = data_loc[i];
      auto const& data_element = all_data_elements[i];

      auto referenced_table_name = getSourcedDataTableNameFromJSONObj(data_item_loc);
      auto itr =
          std::find_if(data_table_indices.begin(),
                       data_table_indices.end(),
                       [&](size_t index) -> bool {
                         if (all_data_elements[index].name == referenced_table_name) {
                           return true;
                         }
                         return false;
                       });
      RUNTIME_EX_ASSERT(itr != data_table_indices.end(),
                        "Failed to find DataTable \'" + referenced_table_name +
                            "\' referenced by SourceDataTable \'" + data_element.name +
                            "\'");

      // if the dependency data table was updated/replaced, we need to force
      // update the sourced data table so that the vega transforms are rebuilt
      // even tho they might not have changed in the vega. We'll iterate through
      // the to-be-fired data events looking for the dependency. If a
      // replace/update event for the dependency is found, we'll force update the
      // sourced table.
      bool force_update = force_multi_layer_data_update ||
                          data_table_queues.isTableInNotifyQueue(referenced_table_name);
      create_or_update_data_table(data_element, force_update);
    }

    // now remove any unused tables that may be lingering around
    for (const auto& unvisited_name : unvisited_names) {
      auto data_table = ctx_.getDataTable(unvisited_name);
      auto data_table_json =
          std::dynamic_pointer_cast<BaseQueryDataTableSQLJSON>(data_table);
      CHECK(data_table_json);
      data_table_queues.addToNotifyQueue(data_table_json, RefEventType::kRemove);
      ctx_.vega_elements_->getDataTableMap().erase(unvisited_name);
    }
  } else {
    //
    // No data elements in Vega, clear existing DataTables
    //
    auto& data_table_map = ctx_.vega_elements_->getDataTableMap();
    for (auto kv : data_table_map) {
      auto data_table = ctx_.getDataTable(kv->getNameRef());
      auto data_table_json =
          std::dynamic_pointer_cast<BaseQueryDataTableSQLJSON>(data_table);
      CHECK(data_table_json);
      data_table_queues.addToNotifyQueue(data_table_json, RefEventType::kRemove);
    }
    data_table_map.clear();
  }

  //
  // Run queries and update SourceDataTables
  //
  data_table_queues.processQueryQueue();
  data_table_queues.processSourceTableQueue();
}

VegaParser::ProjectionEvents VegaParser::parseProjections(const JSONLocation& root_loc) {
  RENDER_LOG_SCOPE();
  ProjectionEvents projection_events(
      [this](const RefEventType event_type, const RefObjShPtr& event_obj) {
        ctx_.notifyRefEvent(event_type, event_obj);
      });
  auto proj_loc =
      root_loc.getMember(JSONSchema_v1::kProjProp, JSONValueType::kArray, false);
  if (proj_loc.isValid()) {
    ProjectionShPtr projection;
    std::unordered_set<std::string> visited_names;
    std::unordered_set<std::string> unvisited_names;
    auto& projection_map = ctx_.vega_elements_->getProjectionMap();
    unvisited_names.reserve(projection_map.size());
    for (const auto& kv : projection_map) {
      unvisited_names.insert(kv->getNameRef());
    }

    for (size_t i = 0; i < proj_loc.size(); ++i) {
      const auto proj_item_loc = proj_loc[i];
      std::string projection_name = getProjectionNameFromJSONObj(proj_item_loc);
      RUNTIME_EX_ASSERT(
          visited_names.find(projection_name) == visited_names.end(),
          RapidJSONUtils::createJsonParseError(
              proj_item_loc,
              "a projection with the name \"" + projection_name + "\" already exists."));

      projection = ctx_.getProjection(projection_name);
      if (!projection) {
        projection = createProjection(proj_item_loc, ctx_, projection_name);
        projection_map.push_back(projection);
      } else {
        // update from JSON obj
        // TODO(adb): if we template the projection based on data type, we may need to
        // match the scale ptr type checking below
        if (projection->updateFromJSONObj(proj_item_loc)) {
          projection_events.push_back(projection, RefEventType::kUpdate);
        }
      }

      unvisited_names.erase(projection_name);
      visited_names.insert(std::move(projection_name));
    }
    for (const auto& unvisited_name : unvisited_names) {
      projection = ctx_.getProjection(unvisited_name);
      projection_map.erase(unvisited_name);
      projection_events.push_back(projection, RefEventType::kRemove);
    }

  } else {
    auto& projection_map = ctx_.vega_elements_->getProjectionMap();
    for (const auto& kv : projection_map) {
      const auto projection = ctx_.getProjection(kv->getNameRef());
      projection_events.push_back(projection, RefEventType::kRemove);
    }
    projection_map.clear();
  }
  return projection_events;
}

void VegaParser::parseMarks(const JSONLocation& root_loc) {
  RENDER_LOG_SCOPE();
  auto mark_loc =
      root_loc.getMember(JSONSchema_v1::kMarkProp, JSONValueType::kArray, false);
  auto& mark_vector = ctx_.vega_elements_->getMarkVector();
  if (mark_loc.isValid()) {
    size_t i;
    for (i = 0; i < mark_loc.size(); ++i) {
      const auto mark_item_loc = mark_loc[i];
      if (i == mark_vector.size()) {
        mark_vector.emplace_back(createMark(mark_item_loc, ctx_));
      } else {
        // do an update
        if (mark_vector[i]->getType() != getMarkTypeFromJSONObj(mark_item_loc)) {
          mark_vector[i] = createMark(mark_item_loc, ctx_);
        } else {
          mark_vector[i]->updateFromJSONObj(mark_item_loc);
        }
      }
    }

    if (i < mark_vector.size()) {
      mark_vector.resize(i);
    }
  } else {
    mark_vector.clear();
  }
}

VegaParser::ScaleEvents VegaParser::parseScales(const JSONLocation& root_loc) {
  RENDER_LOG_SCOPE();
  VegaParser::ScaleEvents scale_events(
      [this](const RefEventType event_type, const RefObjShPtr& event_obj) {
        ctx_.notifyRefEvent(event_type, event_obj);
      });
  auto scale_loc =
      root_loc.getMember(JSONSchema_v1::kScaleProp, JSONValueType::kArray, false);
  auto& scale_map = ctx_.vega_elements_->getScaleMap();
  if (scale_loc.isValid()) {
    ScaleShPtr scale;
    std::unordered_set<std::string> visited_names;
    std::unordered_set<std::string> unvisited_names;
    unvisited_names.reserve(scale_map.size());
    for (auto& kv : scale_map) {
      unvisited_names.insert(kv->getNameRef());
    }

    for (size_t i = 0; i < scale_loc.size(); ++i) {
      const auto scale_item_loc = scale_loc[i];
      std::string scale_name = getScaleNameFromJSONObj(scale_item_loc);

      RUNTIME_EX_ASSERT(
          visited_names.find(scale_name) == visited_names.end(),
          RapidJSONUtils::createJsonParseError(
              scale_item_loc,
              "a scale with the name \"" + scale_name + "\" already exists."));

      scale = ctx_.getScale(scale_name);

      if (!scale) {
        scale = createScale(scale_item_loc, ctx_, scale_name);
        CHECK(scale);
        scale_map.push_back(scale);

        // TODO(croot): add an Add event type?
      } else {
        // TODO(croot): scale config is changing. Need to validate any previously
        // existing references. One way to do this is store a map of all objects
        // changing in-place in order to validate.
        auto curr_scale_type = getScaleTypeFromJSONObj(scale_item_loc);
        auto* curr_accum_state = scale->getAccumState();
        auto curr_accum_type =
            curr_accum_state ? curr_accum_state->getType() : AccumulatorType::kUndefined;
        QueryDataType range_data_type;
        if (scale->getType() != curr_scale_type ||
            curr_accum_type != getScaleAccumulatorTypeFromJSONObj(scale_item_loc) ||
            scale->getDomainDataType() != getScaleDomainDataTypeFromJSONObj(
                                              scale_item_loc, ctx_, curr_scale_type) ||
            (range_data_type = scale->getRangeDataType()) !=
                getScaleRangeDataTypeFromJSONObj(scale_item_loc, ctx_, curr_scale_type) ||
            (range_data_type == QueryDataType::COLOR &&
             getScaleRangeColorTypeFromJSONObj(scale_item_loc).first !=
                 scale->getRangeColorType())) {
          // completely new scale type, so destroy previous one and
          // build a new one from scratch.
          auto prev_scale = ctx_.getScale(scale_name);
          scale = createScale(scale_item_loc, ctx_, scale_name);
          scale_map.replace(scale_name, scale);
          prev_scale->markForDeletion();

          scale_events.push_back(scale, RefEventType::kReplace);
        } else {
          if (scale->updateFromJSONObj(scale_item_loc)) {
            scale_events.push_back(scale, RefEventType::kUpdate);
          }
        }
      }

      unvisited_names.erase(scale_name);
      visited_names.insert(std::move(scale_name));
    }

    // now remove any unused scales that may be lingering around
    for (const auto& unvisited_name : unvisited_names) {
      scale = ctx_.getScale(unvisited_name);
      scale->markForDeletion();
      scale_map.erase(unvisited_name);
      scale_events.push_back(scale, RefEventType::kRemove);
    }

  } else {
    for (auto& kv : scale_map) {
      const auto scale = ctx_.getScale(kv->getNameRef());
      scale->markForDeletion();
      scale_events.push_back(scale, RefEventType::kRemove);
    }
    scale_map.clear();
  }
  return scale_events;
}

void VegaParser::parseInternal(const std::string& json_str) {
  RENDER_LOG_SCOPE();
  auto json_document = std::make_unique<rapidjson::Document>();

  json_document->Parse(json_str.c_str());

  // TODO(croot): this can be removed if the executor will handle the initial parse.
  RUNTIME_EX_ASSERT(
      !json_document->HasParseError(),
      RapidJSONUtils::createJsonParseError(
          JSONLocation(ctx_.getRenderSessionKey(), nullptr, rapidjson::Pointer()),
          "json offset: " + std::to_string(json_document->GetErrorOffset()) +
              ", error: " + rapidjson::GetParseError_En(json_document->GetParseError())));

  JSONLocation root_loc(
      ctx_.getRenderSessionKey(), json_document.get(), rapidjson::Pointer());

  // Reset data table update state flags
  ctx_.resetDataTableStates();

  // Check if the json cache is either empty or not equivalent to json_document
  // Also check if a data table member invalidates the json (jsonNeedsUpdating)
  if (ctx_.jsonNeedsUpdating(root_loc)) {
    RUNTIME_EX_ASSERT(root_loc.isObject(),
                      RapidJSONUtils::createJsonParseError(
                          root_loc, "Root object is not a JSON object."));

    // Parse metadata first in case we need to clear caches
    parseMetadata(root_loc);

    // ViewRenderOptions and render dimensions
    parseViewRenderOptions(root_loc);

    // Data table (DataRef in context)
    parseData(root_loc);

    // Projections
    auto projection_events = parseProjections(root_loc);

    // Scales
    auto scale_events = parseScales(root_loc);

    // Marks
    parseMarks(root_loc);

    // setting ownership of the json document to the ctx.
    // NOTE: json_document should not be used after this point
    ctx_.json_cache_ = std::move(json_document);

    // broadcast notification events so dependencies are cleaned and validated
    ctx_.getDataTableQueues().processNotifyQueue();
    projection_events.notify();
    scale_events.notify();

    // validate the render order after all updates
    std::unordered_set<std::string> visited_accumulators;
    std::string active_accumulator, curr_accumulator;

    auto& mark_vector = ctx_.vega_elements_->getMarkVector();
    for (size_t i = 0; i < mark_vector.size(); ++i) {
      if (mark_vector[i]->hasAccumulator()) {
        curr_accumulator = mark_vector[i]->getAccumulatorScaleName();
        if (active_accumulator != curr_accumulator) {
          active_accumulator = curr_accumulator;
          auto rtn_pair = visited_accumulators.insert(curr_accumulator);
          RUNTIME_EX_ASSERT(
              rtn_pair.second,
              "Invalid render order. All geometry layers that use accumulator scales "
              "must be rendered "
              "one after the other. There are at least 2 layers using the accumulator "
              "scale \"" +
                  curr_accumulator +
                  "\" that are separated by layers not using this accumulator.");
        }
      } else {
        active_accumulator = "";
      }
    }
  }
}

}  // namespace QueryRenderer
