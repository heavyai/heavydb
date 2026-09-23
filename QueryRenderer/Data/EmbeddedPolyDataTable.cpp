/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/EmbeddedPolyDataTable.h"

#include <boost/algorithm/string.hpp>

#include "GfxDriver/Colors/ColorRGBA.h"
#include "GfxDriver/Resources/ShaderBlockLayout.h"
#include "QueryRenderer/Data/EmbeddedDataUtils.h"
#include "QueryRenderer/Data/PolyData2d.h"
#include "QueryRenderer/Data/PolyDataTableGpuResources.h"
#include "QueryRenderer/Data/Utils.h"
#include "QueryRenderer/QueryRendererContext.h"

using ::gfx::BufferAttrType;
using ::gfx::BufferLayoutShPtr;
using ::gfx::ColorRGBA;
using ::gfx::IndirectDrawIndexData;
using ::gfx::IndirectDrawVertexData;
using ::gfx::InterleavedBufferLayout;
using ::gfx::SequentialBufferLayout;
using ::gfx::ShaderBlockLayout;
using ::gfx::ShaderBlockLayoutShPtr;
using ::gfx::ShaderBlockType;

namespace QueryRenderer {

static JSONLocation validateCoordinate(const JSONLocation& json_loc,
                                       const std::string& coord_name) {
  const auto coord_loc = json_loc.getMember(coord_name);
  RUNTIME_EX_ASSERT(
      coord_loc.isValid(),
      RapidJSONUtils::createJsonParseError(
          json_loc, "Polygonal data is missing the \"" + coord_name + "\" coordinate."));

  RUNTIME_EX_ASSERT(
      coord_loc.isArray(),
      RapidJSONUtils::createJsonParseError(
          coord_loc,
          "Polygonal data for coordinate \"" + coord_name + "\" must be an array."));

  RUNTIME_EX_ASSERT(
      coord_loc.size() > 0,
      RapidJSONUtils::createJsonParseError(
          coord_loc, "Data for polygonal coordinate \"" + coord_name + "\" is empty."));

  const auto first_item_loc = coord_loc[0];
  bool is_array{false};
  RUNTIME_EX_ASSERT((is_array = first_item_loc.isArray()) || first_item_loc.isNumber(),
                    RapidJSONUtils::createJsonParseError(
                        first_item_loc,
                        "Unsupported type for coordinate \"" + coord_name +
                            "\". Coordinates must be arrays of numbers, or arrays of "
                            "arrays of numbers."));

  auto array_loc = (is_array ? &first_item_loc : &coord_loc);

  RUNTIME_EX_ASSERT(array_loc->size() >= 3,
                    RapidJSONUtils::createJsonParseError(
                        *array_loc,
                        "Coordinate \"" + coord_name +
                            "\" needs to have at least 3 values to create a polygon."));

  const auto first_coord_loc = (*array_loc)[0];
  RUNTIME_EX_ASSERT(
      first_coord_loc.isNumber(),
      RapidJSONUtils::createJsonParseError(
          first_coord_loc, "Coordinate \"" + coord_name + "\" must be a number."));

  return coord_loc;
}

template <typename T>
void buildPolygonFromJSONObj(const JSONLocation& x_arr_loc,
                             const JSONLocation& y_arr_loc,
                             T& poly_data) {
  typename T::datatype x, y;
  double x_d, y_d;

  poly_data.beginRing();

  // NOTE: xarray & yarray have already been validated (both are arrays and are of the
  // same size)
  for (size_t i = 0; i < x_arr_loc.size(); i++) {
    const auto x_item_loc = x_arr_loc[i];
    RUNTIME_EX_ASSERT(
        x_item_loc.isNumber(),
        RapidJSONUtils::createJsonParseError(
            x_item_loc,
            "Found a non-number at index " + std::to_string(i) +
                " in the \"x\" coord of the polygon. All coords must be numbers."));

    const auto y_item_loc = y_arr_loc[i];
    RUNTIME_EX_ASSERT(
        y_item_loc.isNumber(),
        RapidJSONUtils::createJsonParseError(
            y_item_loc,
            "Found a non-number at index " + std::to_string(i) +
                " in the \"y\" coord of the polygon. All coords must be numbers."));

    x = RapidJSONUtils::getNumValFromJSONObj<typename T::datatype>(x_item_loc);
    y = RapidJSONUtils::getNumValFromJSONObj<typename T::datatype>(y_item_loc);

    // TODO(croot): provide some transform functions?
    // x_d = lon2x_d(x);
    // y_d = lat2y_d(y);

    x_d = static_cast<double>(x);
    y_d = static_cast<double>(y);

    poly_data.addPoint(x_d, y_d);
  }

  poly_data.endRing();
}

template <typename T, bool INTERLEAVED>
int populatePolyData(std::unique_ptr<char[]>& byte_data,
                     PolyDrawBatchInfoUqPtr& poly_draw_batch_info,
                     TDataColumn<T>* poly_data_col) {
  // data size
  int data_size = sizeof(typename T::datatype);

  // separator value
  const auto SEPARATOR_VAL = std::numeric_limits<typename T::datatype>::lowest();

  // the PolyData
  std::vector<T>* poly_data_vec = poly_data_col->getColumnData().get();
  CHECK(poly_data_vec);

  // first pass
  // count verts and determine expansion for repeats and separators
  int num_total_verts = 0;
  int num_total_polys = 0;
  for (const auto& pd : *poly_data_vec) {
    // how many raw verts
    int num_raw_verts = pd.numVerts();
    CHECK_GT(num_raw_verts, 0);
    // how many rings
    int num_rings = pd.ring_sizes.size();
    CHECK_GT(num_rings, 0);
    // now we know how many repeats and separators will be inserted
    int num_repeat_verts = num_rings * 3;
    int num_separator_verts = num_rings - 1;
    // total verts
    num_total_verts += (num_raw_verts + num_repeat_verts + num_separator_verts);
    // total polys
    num_total_polys += pd.ring_sizes.size();
  }

  // batch info (all in one batch)
  std::vector<uint32_t> num_rows_per_batch(1, poly_data_vec->size());
  std::vector<uint32_t> num_polys_per_batch(1, num_total_polys);
  poly_draw_batch_info = std::make_unique<PolyDrawBatchInfo>(
      std::move(num_rows_per_batch), std::move(num_polys_per_batch));

  // how many total bytes
  int num_total_bytes = num_total_verts * data_size * 2;

  // now we can make the output buffer
  byte_data = std::make_unique<char[]>(num_total_bytes);
  CHECK(byte_data.get());
  memset(byte_data.get(), 0x0, num_total_bytes);

  // second pass
  // copy in the verts, with repeats and separators
  if (INTERLEAVED) {
    // interleaved version
    int start_bytes = 0;
    for (const auto& pd : *poly_data_vec) {
      size_t num_rings = pd.ring_sizes.size();
      CHECK_GE(num_rings, 0u);
      size_t last_ring = num_rings - 1;
      int start_vert = 0;
      for (size_t ring = 0; ring < num_rings; ring++) {
        int rs = pd.ring_sizes[ring];
        CHECK_GE(rs, 3);
        // copy raw verts
        for (int i = 0; i < rs; ++i) {
          CHECK_LT(start_bytes, num_total_bytes);
          memcpy(byte_data.get() + start_bytes, &pd.x_coords[start_vert + i], data_size);
          start_bytes += data_size;
          CHECK_LT(start_bytes, num_total_bytes);
          memcpy(byte_data.get() + start_bytes, &pd.y_coords[start_vert + i], data_size);
          start_bytes += data_size;
        }
        // repeat first three verts
        for (int i = 0; i < 3; ++i) {
          CHECK_LT(start_bytes, num_total_bytes);
          memcpy(byte_data.get() + start_bytes, &pd.x_coords[start_vert + i], data_size);
          start_bytes += data_size;
          CHECK_LT(start_bytes, num_total_bytes);
          memcpy(byte_data.get() + start_bytes, &pd.y_coords[start_vert + i], data_size);
          start_bytes += data_size;
        }
        // separator between rings only
        if (ring < last_ring) {
          // separator
          CHECK_LT(start_bytes, num_total_bytes);
          memcpy(byte_data.get() + start_bytes, &SEPARATOR_VAL, data_size);
          start_bytes += data_size;
          CHECK_LT(start_bytes, num_total_bytes);
          memcpy(byte_data.get() + start_bytes, &SEPARATOR_VAL, data_size);
          start_bytes += data_size;
        }
        // next ring
        start_vert += rs;
      }
      CHECK_LT(start_bytes, num_total_bytes);
    }
    CHECK_EQ(start_bytes, num_total_bytes);
  } else {
    // sequential version
    int start_bytes_x = 0;
    int start_bytes_y = num_total_verts * data_size;
    const size_t num_bytes3 = 3 * data_size;
    for (const auto& pd : *poly_data_vec) {
      // x and y must be the same size
      CHECK(pd.x_coords.size() == pd.y_coords.size());
      size_t num_rings = pd.ring_sizes.size();
      size_t last_ring = num_rings - 1;
      int start_vert = 0;
      for (size_t ring = 0; ring < num_rings; ring++) {
        int rs = pd.ring_sizes[ring];
        CHECK_GE(rs, 3);
        // copy x and y raw verts
        const size_t numBytesRaw = rs * data_size;
        CHECK_LT(start_bytes_x, num_total_bytes);
        CHECK_LT(start_bytes_y, num_total_bytes);
        memcpy(byte_data.get() + start_bytes_x, &pd.x_coords[start_vert], numBytesRaw);
        memcpy(byte_data.get() + start_bytes_y, &pd.y_coords[start_vert], numBytesRaw);
        // step
        start_bytes_x += numBytesRaw;
        start_bytes_y += numBytesRaw;
        // repeat and and y first three verts
        CHECK_LT(start_bytes_x, num_total_bytes);
        CHECK_LT(start_bytes_y, num_total_bytes);
        memcpy(byte_data.get() + start_bytes_x, &pd.x_coords[start_vert], num_bytes3);
        memcpy(byte_data.get() + start_bytes_y, &pd.y_coords[start_vert], num_bytes3);
        // step
        start_bytes_x += num_bytes3;
        start_bytes_y += num_bytes3;
        // separator between rings only
        if (ring < last_ring) {
          // separator
          CHECK_LT(start_bytes_x, num_total_bytes);
          CHECK_LT(start_bytes_y, num_total_bytes);
          memcpy(byte_data.get() + start_bytes_x, &SEPARATOR_VAL, data_size);
          memcpy(byte_data.get() + start_bytes_y, &SEPARATOR_VAL, data_size);
          // step
          start_bytes_x += data_size;
          start_bytes_y += data_size;
        }
        // next ring
        start_vert += rs;
      }
      CHECK_LT(start_bytes_x, num_total_bytes);
      CHECK_LT(start_bytes_y, num_total_bytes);
    }
    CHECK_EQ(start_bytes_x, num_total_verts * data_size);
    CHECK_EQ(start_bytes_y, num_total_bytes);
  }

  // done
  return num_total_bytes;
}

JSONLocation getCoordStartItem(const JSONLocation& coord_loc) {
  // NOTE: validation of mitr will have taken place before getting in here
  // via the validateCoordinate() function
  const auto first_item_loc = coord_loc[0];
  if (first_item_loc.isArray()) {
    return first_item_loc[0];
  }
  return first_item_loc;
}

DataColumnUqPtr createPolyDataColumnFromRowMajorObj(const std::string& column_name,
                                                    const JSONLocation& x_coord_loc,
                                                    const JSONLocation& y_coord_loc,
                                                    const JSONLocation& arr_loc) {
  const auto x_item_loc = getCoordStartItem(x_coord_loc);
  const auto y_item_loc = getCoordStartItem(y_coord_loc);
  DataColumnUqPtr poly_data_column_ptr;
  if (x_item_loc.isNumber()) {
    RUNTIME_EX_ASSERT(
        y_item_loc.isNumber(),
        RapidJSONUtils::createJsonParseError(
            arr_loc, "x and y coordinates of the array are different types."));
    // TODO(croot): How do we properly handle floats?
    poly_data_column_ptr = std::make_unique<TDataColumn<PolyData2dDS>>(
        column_name, arr_loc, DataColumn::InitType::kRowMajor);
    auto poly_data_column =
        dynamic_cast<TDataColumn<PolyData2dDS>*>(poly_data_column_ptr.get());
    CHECK(poly_data_column != nullptr);
    return poly_data_column_ptr;
  } else {
    THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
        x_item_loc,
        "Cannot create poly data. The JSON data type for the coordinates is not "
        "supported."));
  }
  return nullptr;
}

template <>
void TDataColumn<PolyData2dDS>::push_back(const std::string& val) {}

template <>
void TDataColumn<PolyData2dDS>::initFromRowMajorJSONObj(const JSONLocation& json_loc) {
  RUNTIME_EX_ASSERT(json_loc.isArray(),
                    RapidJSONUtils::createJsonParseError(
                        json_loc, "Row-major data object is not an array."));

  for (size_t i = 0; i < json_loc.size(); ++i) {
    const auto item_loc = json_loc[i];
    RUNTIME_EX_ASSERT(
        item_loc.isObject(),
        RapidJSONUtils::createJsonParseError(
            item_loc,
            "Item " + std::to_string(i) +
                " in data array must be an object for row-major-defined data."));

    const auto x_loc = validateCoordinate(item_loc, EmbeddedPolyDataTable::x_coord_name);
    const auto y_loc = validateCoordinate(item_loc, EmbeddedPolyDataTable::y_coord_name);

    // We know after validation that the objects at mitrx & mitry are arrays > 0
    RUNTIME_EX_ASSERT(x_loc.size() == y_loc.size(),
                      RapidJSONUtils::createJsonParseError(
                          item_loc,
                          "Item " + std::to_string(i) +
                              " in data array has mismatched sizes in the \"x\" & \"y\" "
                              "coords. They must be the same length."));

    column_data_->emplace_back();
    if (x_loc[0].isArray()) {
      column_data_->back().beginPoly();
      for (size_t i = 0; i < x_loc.size(); i++) {
        const auto x_item_loc = x_loc[i];
        const auto y_item_loc = y_loc[i];
        RUNTIME_EX_ASSERT(
            x_item_loc.isArray() && y_item_loc.isArray() &&
                x_item_loc.size() == y_item_loc.size(),
            RapidJSONUtils::createJsonParseError(
                item_loc,
                "Item " + std::to_string(i) +
                    " in data array has mismatched types/sizes in the \"x\" & \"y\" "
                    "coords. They must be the same type & same length."));

        buildPolygonFromJSONObj<PolyData2dDS>(
            x_item_loc, y_item_loc, column_data_->back());
      }
      column_data_->back().endPoly();
    } else {
      column_data_->back().beginPoly();
      buildPolygonFromJSONObj<PolyData2dDS>(x_loc, y_loc, column_data_->back());
      column_data_->back().endPoly();
    }
  }
}

template <>
QueryDataType TDataColumn<PolyData2dDS>::getColumnType() {
  return QueryDataType::POLYGON_DOUBLE;
}

std::string EmbeddedPolyDataTable::kDefaultPolyDataColumnName = "__polydata__";

EmbeddedPolyDataTable::EmbeddedPolyDataTable(QueryRendererContext& ctx,
                                             const std::string& name,
                                             const JSONLocation& json_loc,
                                             DataInputFormat input_format,
                                             bool build_id_column,
                                             EmbeddedDataVboType vbo_type)
    : BasePolyDataTable(input_format)
    , BaseEmbeddedDataTable(ctx, name, json_loc, RenderQuerySpecialtyType::kPolys)
    , vbo_type_{vbo_type}
    , num_rows_{0} {
  buildPolyDataFromJSONObj(json_loc, build_id_column);
  update();
}

EmbeddedPolyDataTable::~EmbeddedPolyDataTable() {}

bool EmbeddedPolyDataTable::hasData() const {
  // only need to check for vbo data here. If that doesn't exist, there would be no
  // geom to render
  return gpu_resources_->hasVerticesForLayout(nullptr);
}

bool EmbeddedPolyDataTable::hasAttribute(const std::string& attr_name) const {
  if (attr_name == x_coord_name || attr_name == y_coord_name) {
    return true;
  } else {
    const auto& nameLookup = columns_.get<DataColumn::ColumnName>();
    return (nameLookup.find(attr_name) != nameLookup.end());
  }
}

std::set<std::string> EmbeddedPolyDataTable::getAllAttrNames() const {
  std::set<std::string> rtn = {x_coord_name, y_coord_name};
  for (auto& column : columns_) {
    rtn.insert(column->column_name);
  }
  return rtn;
}

QueryLayoutBufferWkPtr EmbeddedPolyDataTable::getAttributeDataBuffer(
    const GpuId gpu_id,
    const std::string& attr_name) {
  auto& gpu_data = gpu_resources_->getGpuDataMap().getData(gpu_id);

  initBuffers(gpu_data);

  if (gpu_data.vbo->hasAttribute(attr_name)) {
    return gpu_data.vbo;
  } else if (gpu_data.ssbo->hasAttribute(attr_name)) {
    return gpu_data.ssbo;
  } else {
    THROW_RUNTIME_EX(createJSONRefError("Cannot get buffer for \"" + attr_name +
                                        "\". Attribute does not exist."));
  }

  return QueryLayoutBufferWkPtr();
}

std::map<GpuId, QueryLayoutBufferWkPtr> EmbeddedPolyDataTable::getAttributeDataBuffers(
    const std::string& attr_name) {
  std::map<GpuId, QueryLayoutBufferWkPtr> rtn;
  std::map<GpuId, QueryLayoutBufferWkPtr>::iterator inserted_itr;
  std::pair<BufferLayoutShPtr, std::pair<std::unique_ptr<char[]>, size_t>> vbo_data;

  gpu_resources_->getGpuDataMap().visitData(
      [&](GpuId gpu_id, PolyDataTablePerGpuData& gpu_data) {
        initBuffers(gpu_data);

        if (gpu_data.vbo->hasAttribute(attr_name)) {
          inserted_itr = rtn.emplace(gpu_id, gpu_data.vbo).first;
        } else if (gpu_data.ssbo->hasAttribute(attr_name)) {
          inserted_itr = rtn.emplace(gpu_id, gpu_data.ssbo).first;
        } else {
          THROW_RUNTIME_EX(createJSONRefError("Cannot get buffers for \"" + attr_name +
                                              "\". Attribute does not exist."));
        }

        CHECK(!rtn.begin()->second.expired() && !inserted_itr->second.expired() &&
              rtn.begin()->second.lock()->getQueryBufferType() ==
                  inserted_itr->second.lock()->getQueryBufferType());
        return true;
      });

  return rtn;
}

SQLTypeInfo EmbeddedPolyDataTable::getAttributeTypeInfo(
    const std::string& attr_name) const {
  return render_type_to_sql_type(getAttributeBufferType(attr_name));
}

QueryDataType EmbeddedPolyDataTable::getAttributeType(
    const std::string& attr_name) const {
  return convertToQueryDataType(getAttributeBufferType(attr_name));
}

BufferAttrType EmbeddedPolyDataTable::getAttributeBufferType(
    const std::string& attr_name) const {
  // all buffers should have the same set of attributes, so only need to check the first
  // one.
  auto* gpu_data = gpu_resources_->getGpuDataMap().getFirstData();
  CHECK(gpu_data);

  initBuffers(*gpu_data);

  BufferAttrType attr_type;
  if (gpu_data->vbo->hasAttribute(attr_name)) {
    attr_type = gpu_data->vbo->getAttributeType(attr_name);
  } else if (gpu_data->ssbo->hasAttribute(attr_name)) {
    attr_type = gpu_data->ssbo->getAttributeType(attr_name);
  } else {
    THROW_RUNTIME_EX(createJSONRefError("Cannot get type for \"" + attr_name +
                                        "\". Attribute does not exist."));
  }

  return attr_type;
}

BufferLayoutShPtr EmbeddedPolyDataTable::getAttributeBufferLayout(
    const std::string& attr_name) {
  // all buffers should have the same set of attributes, so only need to check the first
  // one.
  auto* gpu_data = gpu_resources_->getGpuDataMap().getFirstData();
  CHECK(gpu_data);

  initBuffers(*gpu_data);

  BufferLayoutShPtr rtn_layout;
  if (gpu_data->vbo->hasAttribute(attr_name)) {
    rtn_layout = gpu_data->vbo->getLayoutManager().getBufferLayoutAtIndex(0);
  } else if (gpu_data->ssbo->hasAttribute(attr_name)) {
    rtn_layout = gpu_data->ssbo->getLayoutManager().getBufferLayoutAtIndex(0);
  } else {
    THROW_RUNTIME_EX(createJSONRefError("Cannot get layout for \"" + attr_name +
                                        "\". Attribute does not exist."));
  }

  return rtn_layout;
}

DataColumnShPtr EmbeddedPolyDataTable::getColumn(const std::string& column_name) {
  RUNTIME_EX_ASSERT(
      column_name != x_coord_name && column_name != y_coord_name,
      createJSONRefError("\"" + column_name +
                         "\" is embedded in a special column specific to the polygon "
                         "and is not supported for external use yet."));

  ColumnMap_by_name& name_lookup = columns_.get<DataColumn::ColumnName>();

  ColumnMap_by_name::iterator itr;
  RUNTIME_EX_ASSERT((itr = name_lookup.find(column_name)) != name_lookup.end(),
                    createJSONRefError("Cannot get data for \"" + column_name +
                                       "\". The column does not exist."));

  return *itr;
}

void EmbeddedPolyDataTable::readDataFromFile(const JSONLocation& data_loc) {
  CHECK(data_loc.isString());
  const auto filename = data_loc.getString();
  boost::filesystem::path p(filename);  // avoid repeated path construction below

  RUNTIME_EX_ASSERT(boost::filesystem::exists(p),
                    RapidJSONUtils::createJsonParseError(
                        data_loc, ": File " + filename + " does not exist."));

  RUNTIME_EX_ASSERT(boost::filesystem::is_regular_file(p),
                    RapidJSONUtils::createJsonParseError(
                        data_loc,
                        "File " + filename +
                            " is not a regular file. Cannot read contents "
                            "to build a poly data table."));

  RUNTIME_EX_ASSERT(
      p.has_extension(),
      RapidJSONUtils::createJsonParseError(data_loc,
                                           "File " + filename +
                                               " does not have an extension. Cannot read "
                                               "contents to build a poly data table."));

  std::string ext = p.extension().string();
  boost::to_lower(ext);

  // TODO(croot): import from shapefile/geojson here

  THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
      data_loc,
      "File " + filename + " with extension \"" + ext +
          "\" is not a supported poly data file."));
}

void EmbeddedPolyDataTable::buildPolyRowsFromJSONObj(const JSONLocation& json_loc) {
  // NOTE: obj has already been verified to be an array and have at least 1 item
  const auto item_loc = json_loc[0];
  RUNTIME_EX_ASSERT(
      item_loc.isObject(),
      RapidJSONUtils::createJsonParseError(
          item_loc, "Every row of JSON polygon data must be defined as an object."));

  const auto x_loc = validateCoordinate(item_loc, x_coord_name);
  const auto y_loc = validateCoordinate(item_loc, y_coord_name);

  columns_.push_back(createPolyDataColumnFromRowMajorObj(
      kDefaultPolyDataColumnName, x_loc, y_loc, json_loc));

  // init all non-coord columns
  for (const auto& col_name : item_loc.getMemberNames()) {
    if (col_name == x_coord_name || col_name == y_coord_name) {
      continue;
    }

    const auto col_loc = item_loc[col_name];
    // TODO: Support strings? bools? Anything else?
    if (col_loc.isNumber()) {
      columns_.push_back(
          create_data_column_from_row_major_obj(col_name, col_loc, json_loc));
    } else if (col_loc.isString()) {
      RUNTIME_EX_ASSERT(ColorRGBA::isColorString(col_loc.getString()),
                        RapidJSONUtils::createJsonParseError(
                            col_loc,
                            "Unsupported string for polygonal data column \"" + col_name +
                                "\". Only color strings are currently supported."));
      columns_.push_back(
          create_color_data_column_from_row_major_obj(col_name, col_loc, json_loc));
    } else {
      THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
          col_loc,
          "Unsupported type for polygonal data column \"" + std::string(col_name) +
              "\""));
    }
  }
}

void EmbeddedPolyDataTable::buildPolyDataFromJSONObj(const JSONLocation& json_loc,
                                                     bool build_id_column) {
  RUNTIME_EX_ASSERT(
      json_loc.isObject(),
      RapidJSONUtils::createJsonParseError(
          json_loc, "Data must be an object. Cannot build data table from JSON."));

  const auto format_loc = json_loc.getMember(JSONSchema_v1::Data::kFormatProp);
  RUNTIME_EX_ASSERT(
      format_loc.isValid() && format_loc.isString() &&
          std::string(format_loc.getString()) == "polys",
      RapidJSONUtils::createJsonParseError(
          format_loc.isValid() ? format_loc : json_loc,
          "Polygon data must have a \"" + std::string(JSONSchema_v1::Data::kFormatProp) +
              "\" string property that is set to \"polys\""));

  auto data_loc = json_loc.getMember(JSONSchema_v1::Data::kValuesProp);
  if (data_loc.isValid()) {
    RUNTIME_EX_ASSERT(data_loc.isArray(),
                      RapidJSONUtils::createJsonParseError(
                          data_loc,
                          "\"" + std::string(JSONSchema_v1::Data::kValuesProp) +
                              "\" property in the json must be an array."));

    // row format in an array

    // TODO(croot) - should we just log a warning if no data is supplied instead?
    RUNTIME_EX_ASSERT(data_loc.size() > 0,
                      RapidJSONUtils::createJsonParseError(
                          data_loc, "There is no polygon data defined."));

    buildPolyRowsFromJSONObj(data_loc);
  } else if ((data_loc = json_loc.getMember(JSONSchema_v1::Data::kUrlProp)).isValid()) {
    RUNTIME_EX_ASSERT(data_loc.isString(),
                      RapidJSONUtils::createJsonParseError(
                          data_loc,
                          "\"" + std::string(JSONSchema_v1::Data::kUrlProp) +
                              "\" property must be a string."));

    readDataFromFile(data_loc);
  } else {
    THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
        json_loc,
        "JSON data object must contain either a \"" +
            std::string(JSONSchema_v1::Data::kValuesProp) + "\" or \"" +
            std::string(JSONSchema_v1::Data::kUrlProp) + "\" property."));
  }

  // TODO(croot) - throw a warning instead if no data?
  RUNTIME_EX_ASSERT(columns_.size(),
                    RapidJSONUtils::createJsonParseError(
                        json_loc, "There are no columns in the poly data table."));
  num_rows_ = (*columns_.begin())->size();

  if (build_id_column) {
    TDataColumn<unsigned int>* id_column =
        new TDataColumn<unsigned int>(std::string(kDefaultIdColumnName), num_rows_);

    for (int i = 0; i < num_rows_; ++i) {
      (*id_column)[i] = i;
    }

    columns_.push_back(DataColumnUqPtr(id_column));
  }
}

DataColumnShPtr EmbeddedPolyDataTable::getPolyDataColumn() const {
  const auto& nameLookup = columns_.get<DataColumn::ColumnName>();
  auto polyitr = nameLookup.find(kDefaultPolyDataColumnName);
  CHECK(polyitr != nameLookup.end());
  return *polyitr;
}

void EmbeddedPolyDataTable::initBuffers(PolyDataTablePerGpuData& gpu_data) const {
  if (gpu_data.vbo == nullptr) {
    const auto& qrm_gpu_data = gpu_data.getRootPerGpuData();
    auto& query_buffer_mgr = qrm_gpu_data.getQueryBufferManager();

    auto vbo_data = createVBOData(gpu_data.poly_draw_batch_info);
    gpu_data.vbo = std::make_shared<QueryVertexBuffer>(query_buffer_mgr,
                                                       std::get<1>(vbo_data).get(),
                                                       std::get<2>(vbo_data),
                                                       std::get<0>(vbo_data));

    auto ssbo_data = createSSBOData();
    gpu_data.ssbo =
        std::make_shared<QueryShaderStorageBuffer>(query_buffer_mgr,
                                                   std::get<1>(ssbo_data).get(),
                                                   std::get<2>(ssbo_data),
                                                   std::get<0>(ssbo_data));

    auto line_data = createLineDrawData();
    gpu_data.ivbo_lines = std::make_shared<QueryIndirectVbo>(query_buffer_mgr, line_data);

    auto poly_data = createPolyDrawData();
    gpu_data.ivbo_polys = std::make_shared<QueryIndirectVbo>(query_buffer_mgr, poly_data);

    auto poly_rowids_data = createPolyRowIDsData();
    gpu_data.poly_rowids = std::make_shared<QueryShaderStorageBuffer>(
        query_buffer_mgr,
        poly_rowids_data.second.data(),
        poly_rowids_data.second.size() * sizeof(uint32_t),
        poly_rowids_data.first);
  }
}

std::tuple<BufferLayoutShPtr, std::unique_ptr<char[]>, size_t>
EmbeddedPolyDataTable::createVBOData(PolyDrawBatchInfoUqPtr& poly_draw_batch_info) const {
  BufferLayoutShPtr rtn_layout;
  QueryVertexBufferShPtr vbo;

  auto poly_data_column = getPolyDataColumn();

  switch (vbo_type_) {
    case EmbeddedDataVboType::kSequential: {
      rtn_layout = std::make_shared<SequentialBufferLayout>();
      auto vbo_layout = dynamic_cast<SequentialBufferLayout*>(rtn_layout.get());

      // build up the layout of the vertex buffer

      // TODO(croot): We need to add per-vertex attributes here if we
      // support them ever, like fill/stroke color defined per vertex,
      // or varying stroke widths

      int num_bytes = 0;

      std::unique_ptr<char[]> byte_data;

      switch (poly_data_column->getColumnType()) {
        case QueryDataType::POLYGON_DOUBLE: {
          vbo_layout->addAttribute(x_coord_name, BufferAttrType::kDouble);
          vbo_layout->addAttribute(y_coord_name, BufferAttrType::kDouble);

          auto t_poly_data_column =
              dynamic_cast<TDataColumn<PolyData2dDS>*>(poly_data_column.get());

          num_bytes = populatePolyData<PolyData2dDS, false>(
              byte_data, poly_draw_batch_info, t_poly_data_column);

          break;
        }
        default:
          THROW_RUNTIME_EX(
              createJSONRefError("Column type for polygonal data is not supported. "
                                 "Cannot build vertex buffer."));
          break;
      }

      return std::make_tuple(rtn_layout, std::move(byte_data), num_bytes);

      break;
    }

    case EmbeddedDataVboType::kInterleaved: {
      rtn_layout = std::make_shared<InterleavedBufferLayout>();
      auto vbo_layout = dynamic_cast<InterleavedBufferLayout*>(rtn_layout.get());

      // build up the layout of the vertex buffer

      // TODO(croot): We need to add per-vertex attributes here if we
      // support them ever, like fill/stroke color defined per vertex,
      // or varying stroke widths

      int num_bytes = 0;

      std::unique_ptr<char[]> byte_data;

      switch (poly_data_column->getColumnType()) {
        case QueryDataType::POLYGON_DOUBLE: {
          vbo_layout->addAttribute(x_coord_name, BufferAttrType::kDouble);
          vbo_layout->addAttribute(y_coord_name, BufferAttrType::kDouble);

          auto t_poly_data_column =
              dynamic_cast<TDataColumn<PolyData2dDS>*>(poly_data_column.get());
          num_bytes = populatePolyData<PolyData2dDS, true>(
              byte_data, poly_draw_batch_info, t_poly_data_column);

          break;
        }
        default:
          THROW_RUNTIME_EX(
              createJSONRefError("Column type for polygonal data is not supported. "
                                 "Cannot build vertex buffer."));
          break;
      }

      return std::make_tuple(rtn_layout, std::move(byte_data), num_bytes);

      break;
    }
  }

  return std::make_tuple(nullptr, nullptr, 0);
}

std::tuple<ShaderBlockLayoutShPtr, std::unique_ptr<char[]>, size_t>
EmbeddedPolyDataTable::createSSBOData() const {
  ShaderBlockLayoutShPtr block_layout =
      std::make_shared<ShaderBlockLayout>(ShaderBlockType::kStorageBuffer);

  ColumnMap::iterator itr;

  // don't include the polydata column
  std::vector<std::pair<TypelessColumnData, int>> column_data(columns_.size() - 1);

  int idx = 0;
  block_layout->beginAddingAttrs();
  for (itr = columns_.begin(); itr != columns_.end(); ++itr) {
    if ((*itr)->column_name == kDefaultPolyDataColumnName) {
      continue;
    }

    switch ((*itr)->getColumnType()) {
      case QueryDataType::UINT:
        block_layout->addAttribute<unsigned int>((*itr)->column_name);
        break;
      case QueryDataType::INT:
        block_layout->addAttribute<int>((*itr)->column_name);
        break;
      case QueryDataType::FLOAT:
        block_layout->addAttribute<float>((*itr)->column_name);
        break;
      case QueryDataType::DOUBLE:
        block_layout->addAttribute<double>((*itr)->column_name);
        break;
      case QueryDataType::COLOR:
        block_layout->addAttribute<float, 4>((*itr)->column_name);
        break;
      default:
        THROW_RUNTIME_EX(createJSONRefError(
            "Column type for column \"" + (*itr)->column_name + "\" in data table \"" +
            name_ + "\" is not supported. Cannot build vertex buffer."));
        break;
    }

    column_data[idx] =
        std::make_pair((*itr)->getTypelessColumnData(),
                       block_layout->getAttributeByteOffset((*itr)->column_name));

    idx++;
  }
  block_layout->endAddingAttrs();

  // now cpy the column data into one big buffer, interleaving the data, and
  // buffer it all to the gpu via the VBO.
  // char byteData[numBytes];
  size_t bytes_in_block = block_layout->getNumBytesInBlock();
  size_t num_bytes = bytes_in_block * num_rows_;

  auto byte_data = std::make_unique<char[]>(num_bytes);
  memset(byte_data.get(), 0x0, num_bytes);

  int start_idx = 0;
  int offset = 0;
  int bytes_per_item;
  for (int i = 0; i < num_rows_; ++i) {
    for (size_t j = 0; j < column_data.size(); ++j) {
      bytes_per_item = column_data[j].first.num_bytes_per_item;
      offset = column_data[j].second;
      memcpy(&byte_data[start_idx + offset],
             static_cast<char*>(column_data[j].first.data) + (i * bytes_per_item),
             bytes_per_item);
    }
    start_idx += bytes_in_block;
  }

  return std::make_tuple(block_layout, std::move(byte_data), num_bytes);
}

std::vector<IndirectDrawVertexData> EmbeddedPolyDataTable::createLineDrawData() const {
  std::vector<IndirectDrawVertexData> data;
  auto poly_data_column = getPolyDataColumn();

  switch (poly_data_column->getColumnType()) {
    case QueryDataType::POLYGON_DOUBLE: {
      auto t_poly_data_column =
          dynamic_cast<TDataColumn<PolyData2dDS>*>(poly_data_column.get());
      auto* poly_data_vec = t_poly_data_column->getColumnData().get();

      unsigned int first_index = 0U;
      for (const auto& pd : *poly_data_vec) {
        // all lines for row together
        unsigned int count = 0U;
        for (auto rs : pd.ring_sizes) {
          // ring plus three repeated points plus separator
          count += rs + 4;
        }
        // no separator after last ring of row
        count--;
        // store
        data.emplace_back(count, first_index);
        // next row
        first_index += count;
      }

      break;
    }
    default:
      THROW_RUNTIME_EX(
          createJSONRefError("Column type for polygonal data is not "
                             "supported. Cannot build vertex buffer."));
      break;
  }

  return data;
}

std::vector<IndirectDrawVertexData> EmbeddedPolyDataTable::createPolyDrawData() const {
  std::vector<IndirectDrawVertexData> data;
  auto poly_data_column = getPolyDataColumn();

  switch (poly_data_column->getColumnType()) {
    case QueryDataType::POLYGON_DOUBLE: {
      auto t_poly_data_column =
          dynamic_cast<TDataColumn<PolyData2dDS>*>(poly_data_column.get());
      auto* poly_data_vec = t_poly_data_column->getColumnData().get();

      unsigned int first_index = 0U;
      for (const auto& pd : *poly_data_vec) {
        // all polys for row individually
        size_t num_rings = pd.ring_sizes.size();
        size_t last_ring = num_rings - 1;
        for (size_t ring = 0; ring < num_rings; ring++) {
          // ring only
          unsigned int count = pd.ring_sizes[ring];
          data.emplace_back(count, first_index);
          // skip repeated points
          first_index += count + 3;
          // skip separator?
          if (ring < last_ring) {
            first_index++;
          }
        }
      }

      break;
    }
    default:
      THROW_RUNTIME_EX(
          createJSONRefError("Column type for polygonal data is not "
                             "supported. Cannot build vertex buffer."));
      break;
  }

  return data;
}

std::pair<ShaderBlockLayoutShPtr, std::vector<uint32_t>>
EmbeddedPolyDataTable::createPolyRowIDsData() const {
  // layout with a single uint32_t
  ShaderBlockLayoutShPtr layout =
      std::make_shared<ShaderBlockLayout>(ShaderBlockType::kStorageBuffer);
  layout->beginAddingAttrs();
  layout->addAttribute<unsigned int>("rowid");
  layout->endAddingAttrs();

  // map from poly index to rowid (output index)
  std::vector<uint32_t> data;
  auto poly_data_column = getPolyDataColumn();
  switch (poly_data_column->getColumnType()) {
    case QueryDataType::POLYGON_DOUBLE: {
      auto t_poly_data_column =
          dynamic_cast<TDataColumn<PolyData2dDS>*>(poly_data_column.get());
      auto* poly_data_vec = t_poly_data_column->getColumnData().get();
      uint32_t index = 0u;
      for (const auto& pd : *poly_data_vec) {
        for (uint32_t i = 0; i < pd.ring_sizes.size(); i++) {
          data.push_back(index);
        }
        index++;
      }
      break;
    }
    default:
      THROW_RUNTIME_EX(
          createJSONRefError("Column type for polygonal data is not "
                             "supported. Cannot build vertex buffer."));
      break;
  }

  return {layout, std::move(data)};
}

bool EmbeddedPolyDataTable::update() {
  gpu_resources_->initGpuResourcesFromBuffers(ctx_.getGlobalContext(), "");
  return false;  // the actual data was not updated here, so returning false
}
}  // namespace QueryRenderer
