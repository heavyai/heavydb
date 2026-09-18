/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/BaseEmbeddedDataTable.h"
#include "QueryRenderer/Data/BasePolyDataTable.h"

#include "QueryRenderer/Data/EmbeddedDataUtils.h"

namespace QueryRenderer {

//
// class EmbeddedPolyDataTable
//
// Data table class for embedded poly data
//
class EmbeddedPolyDataTable : public BasePolyDataTable, public BaseEmbeddedDataTable {
 public:
  static std::string kDefaultPolyDataColumnName;

  EmbeddedPolyDataTable(QueryRendererContext& ctx,
                        const std::string& name,
                        const JSONLocation& json_loc,
                        DataInputFormat input_format,
                        bool build_id_column = false,
                        EmbeddedDataVboType vbo_type = EmbeddedDataVboType::kSequential);
  ~EmbeddedPolyDataTable() override;

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

  // local methods
  DataColumnShPtr getColumn(const std::string& column_name);

 private:
  EmbeddedDataVboType vbo_type_;
  int num_rows_;

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
  void buildPolyRowsFromJSONObj(const JSONLocation& json_loc);
  void buildPolyDataFromJSONObj(const JSONLocation& json_loc, bool build_id_column);

  void readDataFromFile(const JSONLocation& data_loc);

  DataColumnShPtr getPolyDataColumn() const;

  void initBuffers(PolyDataTablePerGpuData& per_gpu_data) const;
  std::tuple<gfx::BufferLayoutShPtr, std::unique_ptr<char[]>, size_t> createVBOData(
      PolyDrawBatchInfoUqPtr& poly_draw_batch_info) const;
  std::tuple<gfx::ShaderBlockLayoutShPtr, std::unique_ptr<char[]>, size_t>
  createSSBOData() const;
  std::vector<gfx::IndirectDrawVertexData> createLineDrawData() const;
  std::vector<gfx::IndirectDrawVertexData> createPolyDrawData() const;
  std::pair<gfx::ShaderBlockLayoutShPtr, std::vector<uint32_t>> createPolyRowIDsData()
      const;
};

}  // namespace QueryRenderer
