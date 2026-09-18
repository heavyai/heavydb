/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include <rapidjson/document.h>
#include <rapidjson/pointer.h>

#include "QueryRenderer/Data/Types.h"
#include "QueryRenderer/Events/RefEvent.h"
#include "QueryRenderer/Events/Types.h"
#include "QueryRenderer/JSONRefObject.h"
#include "QueryRenderer/Scales/Types.h"

namespace QueryRenderer {

class BaseScaleDomainRangeData {
 public:
  BaseScaleDomainRangeData(QueryRendererContext& ctx,
                           const bool is_domain,
                           const std::string& name,
                           const QueryDataType data_type,
                           const bool use_string = false)
      : prnt_scale_{nullptr}
      , ctx_{ctx}
      , is_domain_{is_domain}
      , name_{name}
      , data_type_{data_type}
      , use_string_{use_string}
      , data_ref_subscription_id_{0}
      , just_updated_{false} {}
  virtual ~BaseScaleDomainRangeData() {}

  //
  // Required
  //
  virtual uint32_t size() const = 0;  // TODO(scb): 32-bits enough?
  QueryDataType getType() const { return data_type_; }
  virtual const gfx::TypeGLSLShPtr& getTypeGLSL() = 0;
  virtual const std::type_info& getTypeInfo() const = 0;

  virtual rapidjson::Value toJSON(
      rapidjson::Document::AllocatorType& allocator) const = 0;

  //
  // Basic properties
  //
  bool isDomain() const { return is_domain_; }
  std::string getName() const { return name_; }
  bool useString() const { return use_string_; }

  //
  // Parsing and updates
  //
  JSONLocation getJSONLocation();
  void updateJSONPath(const rapidjson::Pointer& obj_path, const bool updating);
  void postJSONUpdate();
  void markForDeletion();

  //
  // DataRef access
  //
  bool hasDataRef() const { return data_ref_ != nullptr; }
  BaseDataTableShPtr getDataRef() const { return data_ref_; }

  // Scale binding
  void setParentScale(BaseScale* parent) { prnt_scale_ = parent; }

  // Logging
  virtual operator std::string() const = 0;

 protected:
  BaseScale* prnt_scale_;
  QueryRendererContext& ctx_;
  const bool is_domain_;
  const std::string name_;
  const QueryDataType data_type_;
  const bool use_string_;
  rapidjson::Pointer json_path_;

  void subscribeToDataEvent(const BaseDataTableShPtr& data_ref);
  void unsubscribeFromDataEvent();

  static std::string validateArrayColsFromDataRef(
      const std::vector<std::string>& column_names,
      const QuerySourceDataTable* source_data);

 private:
  BaseDataTableShPtr data_ref_;
  RefCallbackId data_ref_subscription_id_;
  bool just_updated_;

  void dataRefUpdateCB(RefEventType ref_event_type, const RefObjShPtr& ref_obj);

  virtual std::pair<bool, bool> updateDataFromDataRef(
      const ScaleType type,
      const JSONLocation& data_loc,
      const BaseDataTableShPtr& data_ref) {
    CHECK(false);
    return std::make_pair(false, false);
  }
};

}  // namespace QueryRenderer
