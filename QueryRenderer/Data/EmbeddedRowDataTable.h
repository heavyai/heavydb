/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/BaseEmbeddedDataTable.h"
#include "QueryRenderer/Data/BaseRowDataTable.h"

#include "QueryRenderer/Data/EmbeddedDataUtils.h"
#include "QueryRenderer/Data/RowDataTableGpuResources.h"

namespace QueryRenderer {
//
// class EmbeddedRowDataTable
//
// Generic inline row based data table
//
class EmbeddedRowDataTable : public BaseRowDataTable, public BaseEmbeddedDataTable {
 public:
  EmbeddedRowDataTable(QueryRendererContext& ctx,
                       const std::string& name,
                       const JSONLocation& json_loc,
                       DataInputFormat input_format,
                       bool build_id_column = false,
                       EmbeddedDataVboType vbo_type = EmbeddedDataVboType::kSequential);
  ~EmbeddedRowDataTable() override {}

  // from BaseDataTable
  bool hasData() const final;

  bool hasAttribute(const std::string& attr_name) const final {
    auto const& name_lookup = columns_.get<DataColumn::ColumnName>();
    return (name_lookup.find(attr_name) != name_lookup.end());
  }

  std::set<std::string> getAllAttrNames() const final;
  SQLTypeInfo getAttributeTypeInfo(const std::string& attr_name) const final;
  QueryDataType getAttributeType(const std::string& attr_name) const final;
  gfx::BufferAttrType getAttributeBufferType(const std::string& attr_name) const final;
  gfx::BufferLayoutShPtr getAttributeBufferLayout(const std::string& attr_name) final;
  DataColumnShPtr getColumn(const std::string& column_name);

  QueryLayoutBufferWkPtr getAttributeDataBuffer(const GpuId gpu_id,
                                                const std::string& attr_name) final;
  std::map<GpuId, QueryLayoutBufferWkPtr> getAttributeDataBuffers(
      const std::string& attr_name) final;

  std::vector<GpuId> getUsedGpuIds() const final;

  // local methods
  template <typename C1, typename C2>
  std::pair<C1, C2> getExtrema(const std::string& column);

 private:
  EmbeddedDataVboType vbo_type_;
  int num_rows_;
  std::unique_ptr<RowDataTableGpuResources> gpu_resources_;

  using ColumnMap = boost::multi_index_container<
      DataColumnShPtr,
      boost::multi_index::indexed_by<
          boost::multi_index::random_access<>,

          // hashed on name
          boost::multi_index::hashed_unique<
              boost::multi_index::tag<DataColumn::ColumnName>,
              boost::multi_index::
                  member<DataColumn, std::string, &DataColumn::column_name>>>>;

  using ColumnMap_by_name = ColumnMap::index<DataColumn::ColumnName>::type;

  ColumnMap columns_;

  // from BaseDataTable
  bool update() final;

  // local methods
  void buildColumnsFromJSONObj(const JSONLocation& json_loc, bool build_id_column);
  void populateColumnsFromJSONObj(const JSONLocation& json_loc);
  void readDataFromFile(const std::string& file_name);
  void readFromCsvFile(const std::string& file_name);

  void initBuffers(RowDataTablePerGpuData& per_gpu_data) const;
  std::pair<gfx::BufferLayoutShPtr, std::pair<std::unique_ptr<char[]>, size_t>>
  createVBOData() const;
};

}  // namespace QueryRenderer
