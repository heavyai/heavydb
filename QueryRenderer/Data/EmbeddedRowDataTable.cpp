/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/EmbeddedRowDataTable.h"

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include "GfxDriver/Colors/Utils.h"
#include "GfxDriver/Resources/BufferLayout.h"
#include "QueryRenderer/Data/Types.h"
#include "QueryRenderer/Data/Utils.h"
#include "QueryRenderer/QueryRendererContext.h"
#include "QueryRenderer/Utils/TypeUtils.h"

using ::gfx::BufferAttrType;
using ::gfx::BufferLayoutShPtr;
using ::gfx::InterleavedBufferLayout;
using ::gfx::SequentialBufferLayout;

namespace QueryRenderer {

EmbeddedRowDataTable::EmbeddedRowDataTable(QueryRendererContext& ctx,
                                           const std::string& name,
                                           const JSONLocation& json_loc,
                                           DataInputFormat input_format,
                                           bool build_id_column,
                                           EmbeddedDataVboType vbo_type)
    : BaseRowDataTable(input_format)
    , BaseEmbeddedDataTable(ctx, name, json_loc, RenderQuerySpecialtyType::kNone)
    , vbo_type_{vbo_type}
    , num_rows_{0}
    , gpu_resources_{std::make_unique<RowDataTableGpuResources>(input_format)} {
  update();
  buildColumnsFromJSONObj(json_loc, build_id_column);
}

void EmbeddedRowDataTable::readFromCsvFile(const std::string& file_name) {
  // // typedef boost::escaped_list_separator<char> char_separator;
  // // typedef boost::tokenizer<char_separator> tokenizer;

  // typedef std::regex_token_iterator<std::string::iterator> tokenizer;

  // // static const std::regex sep("\\b\\s*,*\\s*\\b");
  // static const std::regex sep("\\b[\\s,]+");
  // // static const std::regex sep("\\s+");

  // std::string line;
  // std::ifstream inFile(filename.c_str());

  // // TODO: check for errors and throw exceptions on bad reads, eofs, etc.

  // // get the first line. There needs to be header info in the first line:
  // std::getline(inFile, line);

  // tokenizer tok_itr, tok_end;

  // // TODO: use a set in order to error on same column name
  // std::vector<std::string> colNames;

  // for (tok_itr = tokenizer(line.begin(), line.end(), sep, -1); tok_itr != tok_end;
  // ++tok_itr) {
  //   colNames.push_back(*tok_itr);
  // }

  // // Now iterate through the first line of data to determine types
  // std::getline(inFile, line);

  // int idx = 0;
  // for (idx = 0, tok_itr = tokenizer(line.begin(), line.end(), sep, -1); tok_itr !=
  // tok_end; ++tok_itr, ++idx) {
  //   // TODO: what if there are not enough or too many tokens in this line?
  //   columns_.push_back(createDataColumnFromString(colNames[idx], *tok_itr));
  // }

  // // now get the rest of the data
  // int linecnt = 2;
  // while (std::getline(inFile, line)) {
  //   for (idx = 0, tok_itr = tokenizer(line.begin(), line.end(), sep, -1); tok_itr !=
  //   tok_end; ++tok_itr, ++idx) {
  //     // TODO: what if there are not enough or too many tokens in this line?
  //     columns_[idx]->push_back(*tok_itr);
  //   }
  //   ++linecnt;

  //   if (linecnt % 5000 == 0) {
  //     std::cout << "line cnt update: " << linecnt << std::endl;
  //   }
  // }

  // inFile.close();
}

void EmbeddedRowDataTable::readDataFromFile(const std::string& file_name) {
  boost::filesystem::path p(file_name);  // avoid repeated path construction below

  RUNTIME_EX_ASSERT(boost::filesystem::exists(p),
                    createJSONRefError("File " + file_name + " does not exist."));

  RUNTIME_EX_ASSERT(
      boost::filesystem::is_regular_file(p),
      createJSONRefError(
          "File " + file_name +
          " is not a regular file. Cannot read contents to build a data table."));

  RUNTIME_EX_ASSERT(
      p.has_extension(),
      createJSONRefError(
          "File " + file_name +
          " does not have an extension. Cannot read contents to build a data table."));

  std::string ext = p.extension().string();
  boost::to_lower(ext);

  if (ext == ".csv") {
    readFromCsvFile(file_name);
  } else {
    THROW_RUNTIME_EX(createJSONRefError("File " + file_name + " with extension \"" + ext +
                                        "\" is not a supported data file."));
  }
}

void EmbeddedRowDataTable::buildColumnsFromJSONObj(const JSONLocation& json_loc,
                                                   bool build_id_column) {
  RUNTIME_EX_ASSERT(
      json_loc.isObject(),
      RapidJSONUtils::createJsonParseError(
          json_loc, "Data must be an object. Cannot build data table from JSON."));

  bool is_object = false;
  auto data_loc = json_loc.getMember(JSONSchema_v1::Data::kValuesProp);
  if (data_loc.isValid()) {
    RUNTIME_EX_ASSERT((is_object = data_loc.isObject()) || data_loc.isArray(),
                      RapidJSONUtils::createJsonParseError(
                          data_loc,
                          "\"" + std::string(JSONSchema_v1::Data::kValuesProp) +
                              "\" property in the json must be an object or an array."));

    if (is_object) {
      // in column format
      // TODO: fill out
      THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
          data_loc, "Column format not supported yet."));
    } else {
      // in row format in an array

      // TODO(croot) - should we just log a warning if no data is supplied instead?
      RUNTIME_EX_ASSERT(
          data_loc.size() > 0,
          RapidJSONUtils::createJsonParseError(data_loc, "There is no data defined."));

      const auto values_item_loc = data_loc[0];
      RUNTIME_EX_ASSERT(
          values_item_loc.isObject(),
          RapidJSONUtils::createJsonParseError(
              values_item_loc, "Every row of JSON data must be defined as an object."));

      for (const auto& member_name : values_item_loc.getMemberNames()) {
        const auto prop_loc = values_item_loc[member_name];
        // TODO: Support strings? bools? Anything else?
        if (prop_loc.isNumber()) {
          columns_.push_back(
              create_data_column_from_row_major_obj(member_name, prop_loc, data_loc));
        } else if (prop_loc.isString()) {
          const std::string val = prop_loc.getString();
          if (gfx::isColorString(val)) {
            columns_.push_back(create_color_data_column_from_row_major_obj(
                member_name, prop_loc, data_loc));
          } else {
            THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
                prop_loc,
                "Currently only color strings are supported in embedded data tables. \"" +
                    val + "\" is not a valid color string."));
          }
        } else if (prop_loc.isBool()) {
          columns_.push_back(
              create_data_column_from_row_major_obj(member_name, prop_loc, data_loc));
        }
      }
    }
  } else if ((data_loc = json_loc.getMember(JSONSchema_v1::Data::kUrlProp)).isValid()) {
    RUNTIME_EX_ASSERT(data_loc.isString(),
                      RapidJSONUtils::createJsonParseError(
                          data_loc,
                          "\"" + std::string(JSONSchema_v1::Data::kUrlProp) +
                              "\" property must be a string."));

    readDataFromFile(data_loc.getString());
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
                        data_loc, "There are no columns in the data table."));
  num_rows_ = (*columns_.begin())->size();

  if (build_id_column) {
    TDataColumn<unsigned int>* id_column =
        new TDataColumn<unsigned int>(kDefaultIdColumnName, num_rows_);

    for (int i = 0; i < num_rows_; ++i) {
      (*id_column)[i] = i;
    }

    columns_.push_back(DataColumnUqPtr(id_column));
  }
}

bool EmbeddedRowDataTable::hasData() const {
  return gpu_resources_->hasVerticesForLayout(nullptr);
}

std::set<std::string> EmbeddedRowDataTable::getAllAttrNames() const {
  std::set<std::string> rtn;
  for (auto& item : columns_) {
    rtn.insert(item->column_name);
  }
  return rtn;
}

SQLTypeInfo EmbeddedRowDataTable::getAttributeTypeInfo(
    const std::string& attr_name) const {
  return render_type_to_sql_type(getAttributeBufferType(attr_name));
}

QueryDataType EmbeddedRowDataTable::getAttributeType(const std::string& attr_name) const {
  auto const& name_lookup = columns_.get<DataColumn::ColumnName>();

  ColumnMap_by_name::iterator itr;
  RUNTIME_EX_ASSERT((itr = name_lookup.find(attr_name)) != name_lookup.end(),
                    createJSONRefError("Cannot get attribute type for column \"" +
                                       attr_name + "\". Column does not exist."));

  return (*itr)->getColumnType();
}

BufferAttrType EmbeddedRowDataTable::getAttributeBufferType(
    const std::string& attr_name) const {
  // all vbos should have the same set of columns, so only need to check the first one.
  auto* gpu_data = gpu_resources_->getGpuDataMap().getFirstData();
  CHECK(gpu_data);
  initBuffers(*gpu_data);
  return gpu_data->vbo->getAttributeType(attr_name);
}

BufferLayoutShPtr EmbeddedRowDataTable::getAttributeBufferLayout(
    const std::string& attr_name) {
  auto* gpu_data = gpu_resources_->getGpuDataMap().getFirstData();
  CHECK(gpu_data);
  initBuffers(*gpu_data);
  auto layout = gpu_data->vbo->getLayoutManager().getBufferLayoutAtIndex(0);
  RUNTIME_EX_ASSERT(
      layout,
      createJSONRefError("Cannot get the layout for attribute \"" + attr_name +
                         "\". The vega data table has no data."));
  RUNTIME_EX_ASSERT(layout->hasAttribute(attr_name),
                    createJSONRefError("Cannot find a layout for the attribute \"" +
                                       attr_name + "\" in the vega data table."));
  return layout;
}

DataColumnShPtr EmbeddedRowDataTable::getColumn(const std::string& column_name) {
  ColumnMap_by_name& name_lookup = columns_.get<DataColumn::ColumnName>();

  ColumnMap_by_name::iterator itr;
  RUNTIME_EX_ASSERT((itr = name_lookup.find(column_name)) != name_lookup.end(),
                    createJSONRefError("Column \"" + column_name +
                                       "\" does not exist. Cannot get column."));

  return *itr;
}

void EmbeddedRowDataTable::initBuffers(RowDataTablePerGpuData& per_gpu_data) const {
  if (per_gpu_data.vbo == nullptr) {
    auto const& root_gpu_data = per_gpu_data.getRootPerGpuData();

    std::pair<BufferLayoutShPtr, std::pair<std::unique_ptr<char[]>, size_t>> vbo_data =
        createVBOData();

    per_gpu_data.vbo =
        std::make_shared<QueryVertexBuffer>(root_gpu_data.getQueryBufferManager(),
                                            vbo_data.second.first.get(),
                                            num_rows_ * vbo_data.second.second,
                                            vbo_data.first);
  }
}

std::pair<BufferLayoutShPtr, std::pair<std::unique_ptr<char[]>, size_t>>
EmbeddedRowDataTable::createVBOData() const {
  BufferLayoutShPtr rtn_layout;
  QueryVertexBufferShPtr vbo;

  switch (vbo_type_) {
    case EmbeddedDataVboType::kSequential: {
      auto* vbo_layout = new SequentialBufferLayout();

      ColumnMap::iterator itr;
      int num_bytes = 0;
      int num_bytes_per_item = 0;

      // build up the layout of the vertex buffer
      for (itr = columns_.begin(); itr != columns_.end(); ++itr) {
        switch ((*itr)->getColumnType()) {
          case QueryDataType::UINT:
            vbo_layout->addAttribute((*itr)->column_name, BufferAttrType::kUint);
            break;
          case QueryDataType::INT:
            vbo_layout->addAttribute((*itr)->column_name, BufferAttrType::kInt);
            break;
          case QueryDataType::FLOAT:
            vbo_layout->addAttribute((*itr)->column_name, BufferAttrType::kFloat);
            break;
          case QueryDataType::DOUBLE:
            vbo_layout->addAttribute((*itr)->column_name, BufferAttrType::kDouble);
            break;
          case QueryDataType::COLOR:
            vbo_layout->addAttribute((*itr)->column_name, BufferAttrType::kVec4f);
            break;
          default:
            THROW_RUNTIME_EX(
                createJSONRefError("Column type for column \"" + (*itr)->column_name +
                                   "\" in data table \"" + name_ +
                                   "\" is not supported. Cannot build vertex buffer."));
            break;
        }

        TypelessColumnData data = (*itr)->getTypelessColumnData();
        num_bytes += data.num_items * data.num_bytes_per_item;
        num_bytes_per_item += data.num_bytes_per_item;
      }

      // now cpy the column data into one big buffer, sequentially, and
      // buffer it all to the gpu via the VBO.
      auto byte_data = std::make_unique<char[]>(num_bytes);
      memset(byte_data.get(), 0x0, num_bytes);

      int start_idx = 0;
      for (itr = columns_.begin(); itr != columns_.end(); ++itr) {
        TypelessColumnData data = (*itr)->getTypelessColumnData();
        memcpy(
            &byte_data[start_idx], data.data, data.num_items * data.num_bytes_per_item);

        start_idx += data.num_items * data.num_bytes_per_item;
      }

      rtn_layout.reset(vbo_layout);

      return std::make_pair(rtn_layout,
                            std::make_pair(std::move(byte_data), num_bytes_per_item));

      break;
    }

    case EmbeddedDataVboType::kInterleaved: {
      InterleavedBufferLayout* vbo_layout = new InterleavedBufferLayout();

      ColumnMap::iterator itr;
      int num_bytes = 0;

      std::vector<TypelessColumnData> column_data(columns_.size());

      // build up the layout of the vertex buffer
      for (itr = columns_.begin(); itr != columns_.end(); ++itr) {
        switch ((*itr)->getColumnType()) {
          case QueryDataType::UINT:
            vbo_layout->addAttribute((*itr)->column_name, BufferAttrType::kUint);
            break;
          case QueryDataType::INT:
            vbo_layout->addAttribute((*itr)->column_name, BufferAttrType::kInt);
            break;
          case QueryDataType::FLOAT:
            vbo_layout->addAttribute((*itr)->column_name, BufferAttrType::kFloat);
            break;
          case QueryDataType::DOUBLE:
            vbo_layout->addAttribute((*itr)->column_name, BufferAttrType::kDouble);
            break;
          case QueryDataType::COLOR:
            vbo_layout->addAttribute((*itr)->column_name, BufferAttrType::kVec4f);
            break;
          default:
            THROW_RUNTIME_EX(
                createJSONRefError("Column type for column \"" + (*itr)->column_name +
                                   "\" in data table \"" + name_ +
                                   "\" is not supported. Cannot build vertex buffer."));
            break;
        }

        int idx = itr - columns_.begin();
        column_data[idx] = (*itr)->getTypelessColumnData();
        num_bytes += column_data[idx].num_items * column_data[idx].num_bytes_per_item;
      }

      // now cpy the column data into one big buffer, interleaving the data, and
      // buffer it all to the gpu via the VBO.
      auto byte_data = std::make_unique<char[]>(num_bytes);
      memset(byte_data.get(), 0x0, num_bytes);

      int start_idx = 0;
      for (int i = 0; i < num_rows_; ++i) {
        for (size_t j = 0; j < column_data.size(); ++j) {
          int bytes_per_item = column_data[j].num_bytes_per_item;
          memcpy(&byte_data[start_idx],
                 static_cast<char*>(column_data[j].data) + (i * bytes_per_item),
                 bytes_per_item);
          start_idx += bytes_per_item;
        }
      }

      rtn_layout.reset(vbo_layout);

      return std::make_pair(
          rtn_layout,
          std::make_pair(std::move(byte_data), vbo_layout->getNumBytesPerItem()));

      break;
    }
  }

  return std::make_pair(nullptr, std::make_pair(nullptr, 0));
}

bool EmbeddedRowDataTable::update() {
  gpu_resources_->initGpuResourcesFromBuffers(ctx_.getGlobalContext(), nullptr);
  return false;  // the data has not been updated at this point, so returning false
}

QueryLayoutBufferWkPtr EmbeddedRowDataTable::getAttributeDataBuffer(
    const GpuId gpu_id,
    const std::string& attr_name) {
  RUNTIME_EX_ASSERT(
      gpu_resources_->getGpuDataMap().hasData(gpu_id),
      createJSONRefError("Cannot get column data for gpu " + std::to_string(gpu_id)));

  auto& gpu_data = gpu_resources_->getGpuDataMap().getData(gpu_id);

  initBuffers(gpu_data);
  CHECK(gpu_data.vbo);

  RUNTIME_EX_ASSERT(gpu_data.vbo->hasAttribute(attr_name),
                    createJSONRefError("Cannot get buffer for attribute \"" + attr_name +
                                       "\". Attribute does not exist."));

  return gpu_data.vbo;
}

std::map<GpuId, QueryLayoutBufferWkPtr> EmbeddedRowDataTable::getAttributeDataBuffers(
    const std::string& attr_name) {
  std::map<GpuId, QueryLayoutBufferWkPtr> rtn;

  gpu_resources_->getGpuDataMap().visitData(
      [&](GpuId gpu_id, RowDataTablePerGpuData& gpu_data) {
        initBuffers(gpu_data);
        CHECK(gpu_data.vbo);
        rtn.emplace(gpu_id, gpu_data.vbo);
        return true;
      });

  return rtn;
}

std::vector<GpuId> EmbeddedRowDataTable::getUsedGpuIds() const {
  return gpu_resources_->getGpuDataMap().getGpuIds();
}

}  // namespace QueryRenderer
