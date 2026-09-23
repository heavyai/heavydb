/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/EmbeddedLineDataTable.h"

#include "GfxDriver/Colors/ColorRGBA.h"
#include "GfxDriver/Resources/ShaderBlockLayout.h"
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

template <typename T>
struct ArrayData2d {
  std::vector<T> x_coords;
  std::vector<T> y_coords;

  ArrayData2d(unsigned int start_vert = 0) : start_vert_{start_vert} {}
  ~ArrayData2d() {}

  size_t numVerts() const { return x_coords.size(); }
  size_t numLineSegments() const { return x_coords.size() - 1; }
  unsigned int startVert() const { return start_vert_; }

  void reserve(size_t sz) {
    x_coords.reserve(sz);
    y_coords.reserve(sz);
  }

  std::vector<unsigned int> getLineIndices() {
    std::vector<unsigned int> indices;
    indices.reserve(x_coords.size() + 2);

    indices.push_back(0);

    for (size_t i = 0; i < x_coords.size(); i++) {
      indices.push_back(i);
    }

    indices.push_back(x_coords.size() - 1);
    return indices;
  }

 private:
  unsigned int start_vert_;
};

static JSONLocation validateCoordinate(const JSONLocation& json_loc,
                                       const std::string& coord_name) {
  auto coord_loc = json_loc.getMember(coord_name);
  RUNTIME_EX_ASSERT(
      coord_loc.isValid(),
      RapidJSONUtils::createJsonParseError(
          json_loc, "Line data is missing the \"" + coord_name + "\" coordinate."));

  RUNTIME_EX_ASSERT(
      coord_loc.isArray(),
      RapidJSONUtils::createJsonParseError(
          coord_loc,
          "Line data for coordinate \"" + coord_name + "\" must be an array."));

  RUNTIME_EX_ASSERT(
      coord_loc.size() > 0,
      RapidJSONUtils::createJsonParseError(
          coord_loc, "Data for line coordinate \"" + coord_name + "\" is empty."));

  const auto first_item_loc = coord_loc[0];
  bool is_array;
  RUNTIME_EX_ASSERT((is_array = first_item_loc.isArray()) || first_item_loc.isNumber(),
                    RapidJSONUtils::createJsonParseError(
                        coord_loc,
                        "Unsupported type for coordinate \"" + coord_name +
                            "\". Coordinates must be arrays of numbers, or arrays of "
                            "arrays of numbers."));

  auto array_loc = (is_array ? &first_item_loc : &coord_loc);
  RUNTIME_EX_ASSERT(array_loc->size() >= 2,
                    RapidJSONUtils::createJsonParseError(
                        *array_loc,
                        "Coordinate \"" + coord_name +
                            "\" needs to have at least 2 values to create a line."));

  const auto first_coord_item_loc = (*array_loc)[0];
  RUNTIME_EX_ASSERT(
      first_coord_item_loc.isNumber(),
      RapidJSONUtils::createJsonParseError(
          first_coord_item_loc, "Coordinate \"" + coord_name + "\" must be a number."));

  return coord_loc;
}

template <typename T>
static void buildLineFromJSONObj(const JSONLocation& x_arr_loc,
                                 const JSONLocation& y_arr_loc,
                                 ArrayData2d<T>& line_data) {
  T x, y;

  // NOTE: xarray & yarray have already been validated (both are arrays and are of the
  // same size)
  line_data.reserve(x_arr_loc.size());

  for (size_t i = 0; i < x_arr_loc.size(); i++) {
    const auto x_item_loc = x_arr_loc[i];
    RUNTIME_EX_ASSERT(
        x_item_loc.isNumber(),
        RapidJSONUtils::createJsonParseError(
            x_item_loc,
            "Found a non-number at index " + std::to_string(i) +
                " in the \"x\" coord of the line. All coords must be numbers."));

    const auto y_item_loc = y_arr_loc[i];
    RUNTIME_EX_ASSERT(
        y_item_loc.isNumber(),
        RapidJSONUtils::createJsonParseError(
            y_item_loc,
            "Found a non-number at index " + std::to_string(i) +
                " in the \"y\" coord of the line. All coords must be numbers."));

    x = RapidJSONUtils::getNumValFromJSONObj<T>(x_item_loc);
    y = RapidJSONUtils::getNumValFromJSONObj<T>(y_item_loc);

    line_data.x_coords.push_back(x);
    line_data.y_coords.push_back(y);
  }
}

template <>
void TDataColumn<ArrayData2d<double>>::push_back(const std::string& val) {}

template <>
void TDataColumn<ArrayData2d<double>>::initFromRowMajorJSONObj(
    const JSONLocation& json_loc) {
  RUNTIME_EX_ASSERT(json_loc.isArray(),
                    RapidJSONUtils::createJsonParseError(
                        json_loc, "Row-major data object is not an array."));

  for (size_t i = 0; i < json_loc.size(); ++i) {
    const auto array_item_loc = json_loc[i];
    RUNTIME_EX_ASSERT(
        array_item_loc.isObject(),
        RapidJSONUtils::createJsonParseError(
            array_item_loc,
            "Item " + std::to_string(i) +
                "in data array must be an object for row-major-defined data."));

    const auto x_loc =
        validateCoordinate(array_item_loc, EmbeddedLineDataTable::x_coord_name);
    const auto y_loc =
        validateCoordinate(array_item_loc, EmbeddedLineDataTable::y_coord_name);

    // We know after validation that the objects at mitrx & mitry are arrays > 0
    RUNTIME_EX_ASSERT(x_loc.size() == y_loc.size(),
                      RapidJSONUtils::createJsonParseError(
                          array_item_loc,
                          "Item " + std::to_string(i) +
                              " in data array has mismatched sizes in the \"x\" & \"y\" "
                              "coords. They must be the same length."));

    if (column_data_->size()) {
      column_data_->emplace_back(column_data_->back().startVert() +
                                 column_data_->back().numVerts());
    } else {
      column_data_->emplace_back();
    }
    auto item_loc = x_loc[0];
    if (item_loc.isArray()) {
      for (size_t i = 0; i < x_loc.size(); ++i) {
        const auto x_item_loc = x_loc[i];
        const auto y_item_loc = y_loc[i];
        RUNTIME_EX_ASSERT(
            x_item_loc.isArray() && y_item_loc.isArray() &&
                x_item_loc.size() == y_item_loc.size(),
            RapidJSONUtils::createJsonParseError(
                array_item_loc,
                "Item " + std::to_string(i) +
                    " in data array has mismatched types/sizes in the \"x\" & \"y\" "
                    "coords. They must be the same type & same length."));
        buildLineFromJSONObj<double>(x_item_loc, y_item_loc, column_data_->back());
      }
    } else {
      buildLineFromJSONObj<double>(x_loc, y_loc, column_data_->back());
    }
  }
}

template <>
QueryDataType TDataColumn<ArrayData2d<double>>::getColumnType() {
  return QueryDataType::LINE_DOUBLE;
}

template <typename T>
static int getNumVertsInLineColumn(TDataColumn<ArrayData2d<T>>* line_data_col) {
  int num_verts = 0;
  auto* line_data_vec = line_data_col->getColumnData().get();
  for (auto& line_data : (*line_data_vec)) {
    num_verts += line_data.numVerts();
  }
  return num_verts;
}

template <typename T>
static int setSequentialLineData(std::unique_ptr<char[]>& byte_data,
                                 TDataColumn<ArrayData2d<T>>* line_data_col,
                                 int num_total_verts) {
  int data_size = sizeof(T);

  // get x & y data
  int num_bytes = num_total_verts * data_size * 2;
  int num_bytes_per_item = data_size * 2;

  // now cpy the column data into one big buffer, sequentially, and
  // buffer it all to the gpu via the VBO.
  byte_data.reset(new char[num_bytes]);
  memset(byte_data.get(), 0x0, num_bytes);

  int start_idx = 0;
  int buf_size;

  std::vector<ArrayData2d<T>>* line_data_vec = line_data_col->getColumnData().get();
  // get x coords first
  for (auto& line_data : (*line_data_vec)) {
    buf_size = line_data.x_coords.size() * data_size;
    memcpy(&byte_data[start_idx], &line_data.x_coords[0], buf_size);
    start_idx += buf_size;
  }

  // now get y coords
  for (auto& line_data : (*line_data_vec)) {
    buf_size = line_data.y_coords.size() * data_size;
    memcpy(&byte_data[start_idx], &line_data.y_coords[0], buf_size);
    start_idx += buf_size;
  }

  return num_bytes_per_item;
}

template <typename T>
static int setInterleavedLineData(std::unique_ptr<char[]>& byte_data,
                                  TDataColumn<ArrayData2d<T>>* line_data_col,
                                  int num_total_verts) {
  int data_size = sizeof(T);

  // get x & y data
  int num_bytes = num_total_verts * data_size * 2;
  int num_bytes_per_item = data_size * 2;

  // now cpy the column data into one big buffer, sequentially, and
  // buffer it all to the gpu via the VBO.
  byte_data.reset(new char[num_bytes]);
  memset(byte_data.get(), 0x0, num_bytes);

  int start_idx = 0;

  std::vector<ArrayData2d<T>>* line_data_vec = line_data_col->getColumnData().get();

  // get x/y coords in interleaved fashion
  for (auto& line_data : (*line_data_vec)) {
    for (size_t i = 0; i < line_data.numVerts(); ++i) {
      memcpy(&byte_data[start_idx], &line_data.x_coords[i], data_size);
      start_idx += data_size;
      memcpy(&byte_data[start_idx], &line_data.y_coords[i], data_size);
      start_idx += data_size;
    }
  }

  return num_bytes_per_item;
}

template <typename T>
static int getNumLineSegmentsInLineColumn(TDataColumn<ArrayData2d<T>>* line_data_col) {
  int num_line_segments = 0;

  std::vector<ArrayData2d<T>>* line_data_vec = line_data_col->getColumnData().get();

  for (auto& line_data : (*line_data_vec)) {
    num_line_segments += line_data.numLineSegments();
  }

  return num_line_segments;
}

static JSONLocation getCoordStartItem(const JSONLocation& coord_loc) {
  // NOTE: validation of mitr will have taken place before getting in here
  // via the validateCoordinate() function
  const auto first_item_loc = coord_loc[0];
  if (first_item_loc.isArray()) {
    return first_item_loc[0];
  }
  return first_item_loc;
}

std::tuple<DataColumnUqPtr, int, int> createLineDataColumnFromRowMajorObj(
    const std::string& column_name,
    const JSONLocation& x_coord_loc,
    const JSONLocation& y_coord_loc,
    const JSONLocation& array_loc) {
  const auto x_item_loc = getCoordStartItem(x_coord_loc);
  const auto y_item_loc = getCoordStartItem(y_coord_loc);

  DataColumnUqPtr line_data_col_ptr;
  if (x_item_loc.isNumber()) {
    RUNTIME_EX_ASSERT(
        y_item_loc.isNumber(),
        RapidJSONUtils::createJsonParseError(
            array_loc, "x and y coordinates of the array are different types."));

    line_data_col_ptr.reset(new TDataColumn<ArrayData2d<double>>(
        column_name, array_loc, DataColumn::InitType::kRowMajor));
    auto line_data_column =
        dynamic_cast<TDataColumn<ArrayData2d<double>>*>(line_data_col_ptr.get());
    CHECK(line_data_column != nullptr);

    int num_verts = getNumVertsInLineColumn(line_data_column);
    int num_line_segments = getNumLineSegmentsInLineColumn(line_data_column);

    return std::make_tuple(std::move(line_data_col_ptr), num_verts, num_line_segments);
  } else {
    THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
        x_item_loc,
        "Cannot create line data. The JSON data type for the coordinates is not "
        "supported."));
  }
  return std::make_tuple(nullptr, -1, -1);
}

std::string EmbeddedLineDataTable::kDefaultLineDataColumnName = "__linedata__";

EmbeddedLineDataTable::EmbeddedLineDataTable(QueryRendererContext& ctx,
                                             const std::string& name,
                                             const JSONLocation& json_loc,
                                             DataInputFormat input_format,
                                             bool build_id_column,
                                             EmbeddedDataVboType vbo_type)
    : BaseLineDataTable(input_format)
    , BaseEmbeddedDataTable(ctx, name, json_loc, RenderQuerySpecialtyType::kLines)
    , vbo_type_{vbo_type}
    , num_rows_{0}
    , num_verts_{0}
    , num_line_segments_{0} {
  buildLineDataFromJSONObj(json_loc, build_id_column);
  update();
}

EmbeddedLineDataTable::~EmbeddedLineDataTable() {}

bool EmbeddedLineDataTable::hasData() const {
  // NOTE: only checking the vbo here because that's needed to
  // draw anything. No need to check the ssbo here.
  return gpu_resources_->hasVerticesForLayout(nullptr);
}

std::vector<GpuId> EmbeddedLineDataTable::getUsedGpuIds() const {
  return gpu_resources_->getGpuDataMap().getGpuIds();
}

bool EmbeddedLineDataTable::hasAttribute(const std::string& attr_name) const {
  if (attr_name == x_coord_name || attr_name == y_coord_name) {
    return true;
  } else {
    const auto& nameLookup = columns_.get<DataColumn::ColumnName>();
    return (nameLookup.find(attr_name) != nameLookup.end());
  }
}

std::set<std::string> EmbeddedLineDataTable::getAllAttrNames() const {
  std::set<std::string> rtn = {x_coord_name, y_coord_name};
  for (auto& column : columns_) {
    rtn.insert(column->column_name);
  }
  return rtn;
}

QueryLayoutBufferWkPtr EmbeddedLineDataTable::getAttributeDataBuffer(
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
                                        "\". The attribute does not exist."));
  }

  return QueryLayoutBufferWkPtr();
}

std::map<GpuId, QueryLayoutBufferWkPtr> EmbeddedLineDataTable::getAttributeDataBuffers(
    const std::string& attr_name) {
  std::map<GpuId, QueryLayoutBufferWkPtr> rtn;
  std::map<GpuId, QueryLayoutBufferWkPtr>::iterator inserted_itr;
  gpu_resources_->getGpuDataMap().visitData(
      [&](GpuId gpu_id, LineDataTablePerGpuData& gpu_data) {
        initBuffers(gpu_data);

        if (gpu_data.vbo->hasAttribute(attr_name)) {
          inserted_itr = rtn.emplace(gpu_id, gpu_data.vbo).first;
        } else if (gpu_data.ssbo->hasAttribute(attr_name)) {
          inserted_itr = rtn.emplace(gpu_id, gpu_data.ssbo).first;
        } else {
          THROW_RUNTIME_EX(createJSONRefError("Cannot get buffer for \"" + attr_name +
                                              "\". The attribute does not exist."));
        }

        CHECK(!rtn.begin()->second.expired() && !inserted_itr->second.expired() &&
              rtn.begin()->second.lock()->getQueryBufferType() ==
                  inserted_itr->second.lock()->getQueryBufferType());
        return true;
      });

  return rtn;
}

SQLTypeInfo EmbeddedLineDataTable::getAttributeTypeInfo(
    const std::string& attr_name) const {
  return render_type_to_sql_type(getAttributeBufferType(attr_name));
}

QueryDataType EmbeddedLineDataTable::getAttributeType(
    const std::string& attr_name) const {
  return convertToQueryDataType(getAttributeBufferType(attr_name));
}

BufferAttrType EmbeddedLineDataTable::getAttributeBufferType(
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
                                        "\". The attribute does not exist."));
  }

  return attr_type;
}

BufferLayoutShPtr EmbeddedLineDataTable::getAttributeBufferLayout(
    const std::string& attr_name) {
  // all buffers should have the same set of attributes, so only need to check the first
  // one.
  auto* gpu_data = gpu_resources_->getGpuDataMap().getFirstData();
  CHECK(gpu_data);

  initBuffers(*gpu_data);

  BufferLayoutShPtr layout;
  if (gpu_data->vbo->hasAttribute(attr_name)) {
    layout = gpu_data->vbo->getLayoutManager().getBufferLayoutAtIndex(0);
  } else if (gpu_data->ssbo->hasAttribute(attr_name)) {
    layout = gpu_data->ssbo->getLayoutManager().getBufferLayoutAtIndex(0);
  } else {
    THROW_RUNTIME_EX(
        createJSONRefError("Attribute \"" + attr_name + "\" does not exist in table."));
  }

  return layout;
}

DataColumnShPtr EmbeddedLineDataTable::getColumn(const std::string& column_name) {
  RUNTIME_EX_ASSERT(
      column_name != x_coord_name && column_name != y_coord_name,
      createJSONRefError("\"" + column_name +
                         "\" is embedded in a special column specific to the line "
                         "and is not supported for external use yet."));

  ColumnMap_by_name& nameLookup = columns_.get<DataColumn::ColumnName>();

  ColumnMap_by_name::iterator itr;
  RUNTIME_EX_ASSERT((itr = nameLookup.find(column_name)) != nameLookup.end(),
                    createJSONRefError("Cannot get column data for \"" + column_name +
                                       "\". The column does not exist."));

  return *itr;
}

void EmbeddedLineDataTable::buildLineRowsFromJSONObj(const JSONLocation& json_loc) {
  // NOTE: obj has already been verified to be an array and have at least 1 item

  const auto item_loc = json_loc[0];
  RUNTIME_EX_ASSERT(
      item_loc.isObject(),
      RapidJSONUtils::createJsonParseError(
          item_loc, "Every row of JSON line data must be defined as an object."));

  const auto x_loc = validateCoordinate(item_loc, x_coord_name);
  const auto y_loc = validateCoordinate(item_loc, y_coord_name);

  auto data_tuple = createLineDataColumnFromRowMajorObj(
      kDefaultLineDataColumnName, x_loc, y_loc, json_loc);
  columns_.push_back(std::move(std::get<0>(data_tuple)));
  num_verts_ = std::get<1>(data_tuple);
  num_line_segments_ = std::get<2>(data_tuple);

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
                            "Unsupported string for line data column \"" + col_name +
                                "\". Only color strings are currently supported."));
      columns_.push_back(
          create_color_data_column_from_row_major_obj(col_name, col_loc, json_loc));
    } else {
      THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
          col_loc,
          "Unsupported type for line data column \"" + std::string(col_name) + "\""));
    }
  }
}

void EmbeddedLineDataTable::buildLineDataFromJSONObj(const JSONLocation& json_loc,
                                                     bool build_id_column) {
  RUNTIME_EX_ASSERT(
      json_loc.isObject(),
      RapidJSONUtils::createJsonParseError(
          json_loc, "Data must be an object. Cannot build data table from JSON."));

  const auto format_loc = json_loc.getMember(JSONSchema_v1::Data::kFormatProp);
  RUNTIME_EX_ASSERT(
      format_loc.isValid() && format_loc.isObject(),
      RapidJSONUtils::createJsonParseError(
          (format_loc.isValid() ? format_loc : json_loc),
          "Line data must have a \"" + std::string(JSONSchema_v1::Data::kFormatProp) +
              "\" object property"));

  const auto type_loc = format_loc.getMember(JSONSchema_v1::Data::kTypeProp);
  RUNTIME_EX_ASSERT(type_loc.isValid() && type_loc.isString() &&
                        type_loc.getString() == std::string("lines"),
                    RapidJSONUtils::createJsonParseError(
                        type_loc.isValid() ? type_loc : json_loc,
                        "Line data \"" + std::string(JSONSchema_v1::Data::kFormatProp) +
                            "\" object property must contain a \"" +
                            std::string(JSONSchema_v1::Data::kTypeProp) +
                            "\" string field with value \"lines\""));

  const auto values_loc = json_loc.getMember(JSONSchema_v1::Data::kValuesProp);
  if (values_loc.isValid()) {
    RUNTIME_EX_ASSERT(values_loc.isArray(),
                      RapidJSONUtils::createJsonParseError(
                          values_loc,
                          "\"" + std::string(JSONSchema_v1::Data::kValuesProp) +
                              "\" property in the json must be an array."));

    // row format in an array

    // TODO(croot) - should we just log a warning if no data is supplied instead?
    RUNTIME_EX_ASSERT(values_loc.size() > 0,
                      RapidJSONUtils::createJsonParseError(
                          values_loc, "There is no line data defined."));

    buildLineRowsFromJSONObj(values_loc);

  } else {
    THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
        json_loc, "JSON data object must contain a \"values\" property."));
  }

  // TODO(croot) - throw a warning instead if no data?
  RUNTIME_EX_ASSERT(columns_.size(),
                    RapidJSONUtils::createJsonParseError(
                        json_loc, "There are no columns in the line data table."));
  num_rows_ = (*columns_.begin())->size();

  if (build_id_column) {
    auto id_column = std::make_unique<TDataColumn<unsigned int>>(
        std::string(kDefaultIdColumnName), num_rows_);

    for (int i = 0; i < num_rows_; ++i) {
      (*id_column)[i] = i;
    }

    columns_.push_back(std::move(id_column));
  }
}

DataColumnShPtr EmbeddedLineDataTable::getLineDataColumn() const {
  const auto& name_lookup = columns_.get<DataColumn::ColumnName>();
  auto lineitr = name_lookup.find(kDefaultLineDataColumnName);
  CHECK(lineitr != name_lookup.end());
  return *lineitr;
}

void EmbeddedLineDataTable::initBuffers(LineDataTablePerGpuData& gpu_data) const {
  if (gpu_data.vbo == nullptr) {
    const auto& qrm_gpu_data = gpu_data.getRootPerGpuData();
    auto& query_buffer_mgr = qrm_gpu_data.getQueryBufferManager();

    // TODO(adb): flag whether or not to use index buffer
    auto vbo_data = createVBOData();
    gpu_data.vbo = std::make_shared<QueryVertexBuffer>(query_buffer_mgr,
                                                       std::get<1>(vbo_data).get(),
                                                       num_verts_ * std::get<2>(vbo_data),
                                                       std::get<0>(vbo_data));

    auto ssbo_data = createSSBOData();
    gpu_data.ssbo =
        std::make_shared<QueryShaderStorageBuffer>(query_buffer_mgr,
                                                   std::get<1>(ssbo_data).get(),
                                                   num_rows_ * std::get<2>(ssbo_data),
                                                   std::get<0>(ssbo_data));

    auto ibo_data = createIBOData();
    gpu_data.ibo = std::make_shared<QueryIndexBuffer>(query_buffer_mgr, ibo_data);

    auto line_vertex_data = createLineVertexData();
    gpu_data.ind_vbo =
        std::make_shared<QueryIndirectVbo>(query_buffer_mgr, line_vertex_data);

    auto line_index_data = createLineIndexData();
    gpu_data.ind_ibo =
        std::make_shared<QueryIndirectIbo>(query_buffer_mgr, line_index_data);
  }
}

std::tuple<BufferLayoutShPtr, std::unique_ptr<char[]>, size_t>
EmbeddedLineDataTable::createVBOData() const {
  BufferLayoutShPtr rtn_layout;
  QueryVertexBufferShPtr vbo;

  auto line_data_column = getLineDataColumn();

  switch (vbo_type_) {
    case EmbeddedDataVboType::kSequential: {
      rtn_layout = std::make_shared<SequentialBufferLayout>();
      auto vbo_layout = dynamic_cast<SequentialBufferLayout*>(rtn_layout.get());

      // build up the layout of the vertex buffer

      // TODO(croot): We need to add per-vertex attributes here if we
      // support them ever, like fill/stroke color defined per vertex,
      // or varying stroke widths

      int num_bytes_per_item = 0;

      std::unique_ptr<char[]> byte_data;

      switch (line_data_column->getColumnType()) {
        case QueryDataType::LINE_DOUBLE: {
          vbo_layout->addAttribute(x_coord_name, BufferAttrType::kDouble);
          vbo_layout->addAttribute(y_coord_name, BufferAttrType::kDouble);

          auto t_line_data_column =
              dynamic_cast<TDataColumn<ArrayData2d<double>>*>(line_data_column.get());
          num_bytes_per_item =
              setSequentialLineData<double>(byte_data, t_line_data_column, num_verts_);
          break;
        }
        default:
          THROW_RUNTIME_EX(createJSONRefError(
              "Column type " + to_string(line_data_column->getColumnType()) +
              " is not supported. Cannot build vertex buffer."));
          break;
      }

      return std::make_tuple(rtn_layout, std::move(byte_data), num_bytes_per_item);

      break;
    }

    case EmbeddedDataVboType::kInterleaved: {
      rtn_layout = std::make_shared<InterleavedBufferLayout>();
      auto vbo_layout = dynamic_cast<InterleavedBufferLayout*>(rtn_layout.get());

      // build up the layout of the vertex buffer

      // TODO(croot): We need to add per-vertex attributes here if we
      // support them ever, like fill/stroke color defined per vertex,
      // or varying stroke widths

      int num_bytes_per_item = 0;

      std::unique_ptr<char[]> byte_data;

      switch (line_data_column->getColumnType()) {
        case QueryDataType::LINE_DOUBLE: {
          vbo_layout->addAttribute(x_coord_name, BufferAttrType::kDouble);
          vbo_layout->addAttribute(y_coord_name, BufferAttrType::kDouble);

          auto t_line_data_column =
              dynamic_cast<TDataColumn<ArrayData2d<double>>*>(line_data_column.get());
          num_bytes_per_item =
              setInterleavedLineData<double>(byte_data, t_line_data_column, num_verts_);
          break;
        }
        default:
          THROW_RUNTIME_EX(createJSONRefError(
              "Column type " + to_string(line_data_column->getColumnType()) +
              " is not supported. Cannot build vertex buffer."));
          break;
      }

      return std::make_tuple(rtn_layout, std::move(byte_data), num_bytes_per_item);

      break;
    }
  }

  return std::make_tuple(nullptr, nullptr, 0);
}

std::tuple<ShaderBlockLayoutShPtr, std::unique_ptr<char[]>, size_t>
EmbeddedLineDataTable::createSSBOData() const {
  ShaderBlockLayoutShPtr block_layout =
      std::make_shared<ShaderBlockLayout>(ShaderBlockType::kStorageBuffer);

  ColumnMap::iterator itr;

  // don't include the linedata column
  std::vector<std::pair<TypelessColumnData, int>> column_data(columns_.size() - 1);

  int idx = 0;
  block_layout->beginAddingAttrs();
  for (itr = columns_.begin(); itr != columns_.end(); ++itr) {
    if ((*itr)->column_name == kDefaultLineDataColumnName) {
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
            "Column type " + to_string((*itr)->getColumnType()) + " for \"" +
            (*itr)->column_name + "\" is not supported. Cannot build vertex buffer."));
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

  return std::make_tuple(block_layout, std::move(byte_data), bytes_in_block);
}

std::vector<unsigned int> EmbeddedLineDataTable::createIBOData() const {
  std::vector<unsigned int> combined_idxs;
  auto line_data_column = getLineDataColumn();
  switch (line_data_column->getColumnType()) {
    case QueryDataType::LINE_DOUBLE: {
      auto t_line_data_column =
          dynamic_cast<TDataColumn<ArrayData2d<double>>*>(line_data_column.get());

      CHECK(num_line_segments_);
      combined_idxs.reserve(num_verts_ + 2 * num_rows_);

      auto* line_data_vec = t_line_data_column->getColumnData().get();

      for (auto& line_data : (*line_data_vec)) {
        const auto indices = line_data.getLineIndices();
        for (size_t i = 0; i < indices.size(); ++i) {
          combined_idxs.push_back(indices[i]);
        }
      }
      break;
    }
    default:
      THROW_RUNTIME_EX(createJSONRefError(
          "Column type " + to_string(line_data_column->getColumnType()) +
          " is not supported. Cannot build index buffer."));
      break;
  }

  return combined_idxs;
}

std::vector<IndirectDrawVertexData> EmbeddedLineDataTable::createLineVertexData() const {
  std::vector<IndirectDrawVertexData> vertex_data;
  auto line_data_column = getLineDataColumn();

  switch (line_data_column->getColumnType()) {
    case QueryDataType::LINE_DOUBLE: {
      auto t_line_data_column =
          dynamic_cast<TDataColumn<ArrayData2d<double>>*>(line_data_column.get());

      vertex_data.reserve(num_rows_);
      auto* line_data_vec = t_line_data_column->getColumnData().get();

      // unsigned int vertex_count = 0;
      for (auto& line_data : (*line_data_vec)) {
        vertex_data.emplace_back(line_data.numVerts());
        // vertex_count += line_data.numVerts();
      }
      break;
    }
    default:
      THROW_RUNTIME_EX(createJSONRefError(
          "Column type " + to_string(line_data_column->getColumnType()) +
          " is not supported. Cannot build vertex buffer."));
  }

  return vertex_data;
}

std::vector<IndirectDrawIndexData> EmbeddedLineDataTable::createLineIndexData() const {
  std::vector<IndirectDrawIndexData> index_data;
  auto line_data_column = getLineDataColumn();

  switch (line_data_column->getColumnType()) {
    case QueryDataType::LINE_DOUBLE: {
      auto t_line_data_column =
          dynamic_cast<TDataColumn<ArrayData2d<double>>*>(line_data_column.get());

      index_data.reserve(num_rows_);
      auto* line_data_vec = t_line_data_column->getColumnData().get();

      unsigned int vertex_count = 0;
      unsigned int index_count = 0;
      for (auto& line_data : (*line_data_vec)) {
        index_data.emplace_back(line_data.numVerts() + 2, index_count, vertex_count);
        vertex_count += line_data.numVerts();
        index_count += line_data.numVerts() + 2;
      }
      break;
    }
    default:
      THROW_RUNTIME_EX(createJSONRefError(
          "Column type " + to_string(line_data_column->getColumnType()) +
          " is not supported. Cannot build index buffer."));
      break;
  }

  return index_data;
}

bool EmbeddedLineDataTable::update() {
  gpu_resources_->initGpuResourcesFromBuffers(ctx_.getGlobalContext(),
                                              /*use_index_buffer=*/true);
  return false;  // data was not updated here, so returning false
}

}  // namespace QueryRenderer
