/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/BaseEmbeddedDataTable.h"
#include "QueryRenderer/Data/BaseLineDataTable.h"

#include "QueryRenderer/Data/EmbeddedDataUtils.h"

namespace QueryRenderer {

//
// class EmbeddedLineDataTable
//
// Data table class for embedded line data tables
//
class EmbeddedLineDataTable : public BaseLineDataTable, public BaseEmbeddedDataTable {
 public:
  static std::string kDefaultLineDataColumnName;

  EmbeddedLineDataTable(QueryRendererContext& ctx,
                        const std::string& name,
                        const JSONLocation& json_loc,
                        DataInputFormat input_format,
                        bool build_id_column = false,
                        EmbeddedDataVboType vbo_type = EmbeddedDataVboType::kSequential);
  ~EmbeddedLineDataTable() override;

  // from BaseDataTable
  bool hasData() const final;
  bool hasAttribute(const std::string& attr_name) const final;
  std::set<std::string> getAllAttrNames() const final;
  QueryLayoutBufferWkPtr getAttributeDataBuffer(const GpuId gpu_id,
                                                const std::string& attr_name) final;
  std::map<GpuId, QueryLayoutBufferWkPtr> getAttributeDataBuffers(
      const std::string& attr_name) final;
  SQLTypeInfo getAttributeTypeInfo(const std::string& attr_name) const final;
  QueryDataType getAttributeType(const std::string& attr_name) const final;
  gfx::BufferAttrType getAttributeBufferType(const std::string& attr_name) const final;
  gfx::BufferLayoutShPtr getAttributeBufferLayout(const std::string& attr_name) final;

  std::vector<GpuId> getUsedGpuIds() const override;

  // local methods
  DataColumnShPtr getColumn(const std::string& column_name);

 private:
  EmbeddedDataVboType vbo_type_;
  int num_rows_;
  int num_verts_;
  int num_line_segments_;

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
  void buildLineRowsFromJSONObj(const JSONLocation& json_loc);
  void buildLineDataFromJSONObj(const JSONLocation& json_loc, bool build_id_column);

  DataColumnShPtr getLineDataColumn() const;

  void initBuffers(LineDataTablePerGpuData& per_gpu_data) const;
  std::tuple<gfx::BufferLayoutShPtr, std::unique_ptr<char[]>, size_t> createVBOData()
      const;
  std::tuple<gfx::ShaderBlockLayoutShPtr, std::unique_ptr<char[]>, size_t>
  createSSBOData() const;
  std::vector<unsigned int> createIBOData() const;
  std::vector<gfx::IndirectDrawVertexData> createLineVertexData() const;
  std::vector<gfx::IndirectDrawIndexData> createLineIndexData() const;
};

}  // namespace QueryRenderer
