/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

#include <rapidjson/document.h>
#include <rapidjson/pointer.h>

#include "GfxDriver/Resources/Types.h"
#include "QueryRenderer/Data/Types.h"
#include "QueryRenderer/Interop/Types.h"
#include "QueryRenderer/Types.h"

namespace QueryRenderer {

//
// class BaseDataTable
//
// Base class for all DataTable types
// Stores gpu data for the table and provides access to attributes
// and associated buffers
// Both Sql and Embedded DataTables must derive from this class
//
class BaseDataTable {
 public:
  BaseDataTable(DataInputFormat input_format, DataOutputFormat output_format)
      : input_format_(input_format), output_format_(output_format) {}
  virtual ~BaseDataTable() = default;

  // Data
  virtual bool hasData() const = 0;

  // Attributes
  virtual bool hasAttribute(const std::string& attr_name) const = 0;
  virtual std::set<std::string> getAllAttrNames() const = 0;
  virtual QueryLayoutBufferWkPtr getAttributeDataBuffer(const GpuId gpu_id,
                                                        const std::string& attr_name) = 0;
  virtual std::map<GpuId, QueryLayoutBufferWkPtr> getAttributeDataBuffers(
      const std::string& attr_name) = 0;
  virtual SQLTypeInfo getAttributeTypeInfo(const std::string& attr_name) const = 0;
  virtual QueryDataType getAttributeType(const std::string& attr_name) const = 0;
  virtual gfx::BufferAttrType getAttributeBufferType(
      const std::string& attr_name) const = 0;
  virtual gfx::BufferLayoutShPtr getAttributeBufferLayout(
      const std::string& attr_name) = 0;

  // Data format
  DataInputFormat getInputFormat() const { return input_format_; }
  DataOutputFormat getOutputFormat() const { return output_format_; }

  // GpuIds of gpus with valid data in the GpuDataMap
  virtual std::vector<GpuId> getUsedGpuIds() const = 0;

 protected:
  DataInputFormat input_format_;
  DataOutputFormat output_format_;

 private:
  // update()
  //
  // Called on every render by VegaParser after parsing is complete, via
  // QueryRendererContext::updateConfigGpuResources
  //
  // SQLJSON data tables (e.g. SqlQueryPolyDataTableJSON)
  //  - Runs query if out-of-date (calling runQueryAndInitResources)
  //  - Creates and populates PerGpuData elements in GpuDataMap (buffer pointers)
  // Embedded data tables (e.g. PolyDataTable)
  //  - Creates PerGpuData elements in GpuDataMap
  //  - Derived class is expected to populate the buffers
  //  - update called in constructor
  //
  // NOTE: During vega parsing SQLJSON implementations also call runQueryAndInitResources,
  // so the first time a vega is rendered we need protect against double query execution
  //
  virtual bool update() = 0;

  friend class QueryRendererContext;
  friend class VegaParser;
};

}  // namespace QueryRenderer
