/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Scales/BaseScaleDomainRangeData.h"

#include "QueryRenderer/Data/QuerySourceDataTable.h"
#include "QueryRenderer/Scales/BaseScale.h"
#include "QueryRenderer/Utils/StringUtils.h"

namespace QueryRenderer {

std::string BaseScaleDomainRangeData::validateArrayColsFromDataRef(
    const std::vector<std::string>& column_names,
    const QuerySourceDataTable* source_data) {
  std::vector<std::string> array_cols;
  for (const auto& column_name : column_names) {
    if (source_data->isVectorAttribute(column_name)) {
      array_cols.push_back(column_name);
    }
  }
  const auto array_cnt = array_cols.size();
  if (array_cnt > 1 || (array_cnt == 1 && column_names.size() != 1)) {
    throw std::runtime_error(
        (array_cnt == 1
             ? "Data column " + array_cols[0] + " is an array column, but there are " +
                   std::to_string(column_names.size()) +
                   " columns attempted to be retrieved from the " +
                   source_data->getName() + " data table."
             : to_string(array_cols) + " are all array columns.") +
        " When an array column is being referenced, it must be the only column.");
  }

  return (array_cnt ? array_cols[0] : "");
}

void BaseScaleDomainRangeData::subscribeToDataEvent(const BaseDataTableShPtr& data_ref) {
  // setup callbacks for data updates
  CHECK(prnt_scale_);
  CHECK_EQ(data_ref_subscription_id_, RefCallbackId(0));
  auto data_json = std::dynamic_pointer_cast<BaseQueryDataTableSQLJSON>(data_ref);
  CHECK(data_json);
  auto cb = [this](RefEventType ref_event_type, const RefObjShPtr& ref_obj) {
    dataRefUpdateCB(ref_event_type, ref_obj);
  };
  data_ref_subscription_id_ = ctx_.subscribeToRefEvent(RefEventType::kAll, data_json, cb);
  data_ref_ = data_ref;
}

void BaseScaleDomainRangeData::unsubscribeFromDataEvent() {
  if (data_ref_ && data_ref_subscription_id_) {
    CHECK(prnt_scale_);
    auto data_json = std::dynamic_pointer_cast<BaseQueryDataTableSQLJSON>(data_ref_);
    CHECK(data_json);
    ctx_.unsubscribeFromRefEvent(
        RefEventType::kAll, data_json, data_ref_subscription_id_);
    data_ref_subscription_id_ = 0;
  }
}

void BaseScaleDomainRangeData::dataRefUpdateCB(RefEventType ref_event_type,
                                               const RefObjShPtr& ref_obj) {
  CHECK(prnt_scale_);
  auto data = std::dynamic_pointer_cast<BaseDataTable>(ref_obj);
  CHECK(data);

  switch (ref_event_type) {
    case RefEventType::kUpdate:
      CHECK(data == data_ref_);
    // pass thru to the REPLACE code
    case RefEventType::kReplace: {
      if (just_updated_) {
        return;
      }
      const auto data_obj_loc = ctx_.getJSONObj(json_path_);
      CHECK(data_obj_loc.isValid())
          << RapidJSONUtils::getPointerPath(data_obj_loc.getPathRef());
      auto const [size_changed, vals_changed] =
          updateDataFromDataRef(prnt_scale_->getType(), data_obj_loc, data);
      ScaleDRChangedFlags changed_flags(ScaleDRChangedFlags::kNone);

      if (size_changed) {
        if (isDomain()) {
          changed_flags |= ScaleDRChangedFlags::kDomainSize;
        } else {
          changed_flags |= ScaleDRChangedFlags::kRangeSize;
        }
      }

      if (vals_changed) {
        if (isDomain()) {
          changed_flags |= ScaleDRChangedFlags::kDomainVals;
        } else {
          changed_flags |= ScaleDRChangedFlags::kRangeVals;
        }

        if (changed_flags != ScaleDRChangedFlags::kNone) {
          prnt_scale_->setDRChangedFlags(static_cast<ScaleDRChangedFlags>(changed_flags));
        }

        // NOTE: not doing a vals changed validation here as that is currently being done
        // in the above _updateDataFromDataRef() call
      }
      ctx_.notifyRefEvent(RefEventType::kUpdate, prnt_scale_);
      break;
    }
    case RefEventType::kRemove:
      THROW_RUNTIME_EX(
          std::string(*this) + ": Error, data table " + ref_obj->getName() +
          " has been removed but is still being referenced by this scale domain/range.")
      break;
    case RefEventType::kAll:
      CHECK(false);
      break;
  }
}

//
// Parsing and updates
//
JSONLocation BaseScaleDomainRangeData::getJSONLocation() {
  return JSONLocation(ctx_.getRenderSessionKey(), nullptr, json_path_);
}

void BaseScaleDomainRangeData::updateJSONPath(const rapidjson::Pointer& obj_path,
                                              const bool updating) {
  // json_path_ should only be updated via this function
  json_path_ = obj_path.Append(name_.c_str(), name_.length());
  just_updated_ = updating;
}
void BaseScaleDomainRangeData::postJSONUpdate() {
  just_updated_ = false;
}
void BaseScaleDomainRangeData::markForDeletion() {
  unsubscribeFromDataEvent();
}

}  // namespace QueryRenderer
