/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Transforms/Aggregate/AggXformOp.h"

#include <boost/algorithm/string.hpp>

#include "QueryRenderer/Data/Transforms/BaseXform.h"
#include "QueryRenderer/Data/Utils.h"
#include "QueryRenderer/QueryDataLayout.h"
#include "QueryRenderer/QueryRendererContext.h"

namespace QueryRenderer {

using ::gfx::BufferAttrType;
using ::gfx::BufferLayoutShPtr;
using ::gfx::BufferLayoutType;

namespace {

void validate_data_layout_for_thrust(const AggOp* curr_op,
                                     const std::string& attr,
                                     const gfx::BufferLayoutShPtr& data_layout) {
  CHECK(curr_op && data_layout && data_layout->hasAttribute(attr));

  switch (data_layout->getLayoutType()) {
    case BufferLayoutType::kInterleaved: {
      auto& attr_info = data_layout->getAttributeInfo(attr);
      auto stride = static_cast<int>(data_layout->getNumBytesPerItem());
      auto byte_sz = attr_info.type_info->numBytes();

      if (attr_info.offset % byte_sz != 0) {
        throw std::runtime_error(
            "An invalid alignment of the data buffer containing the attribute \"" + attr +
            "\" exists. The attribute's buffer type is: " +
            attr_info.type_info->declString() + " which has a byte size of " +
            std::to_string(byte_sz) + " which does not align with its offset " +
            std::to_string(attr_info.offset) +
            " in the buffer. The offset must be divisible by the attribute's byte size.");
      }

      if (stride % byte_sz != 0) {
        throw std::runtime_error(
            "An invalid alignment of the data buffer containing the attribute \"" + attr +
            "\" exists. The attribute's buffer type is: " +
            attr_info.type_info->declString() + " which has a byte size of " +
            std::to_string(byte_sz) +
            " which does not align with the stride: " + std::to_string(stride) +
            ". The stride must be divisible by the attribute's byte size.");
      }

      // NOTE: attrs in interleaved layouts are always retrieved in offset ascending order
      // when retrieved by ascending index
      int start_offset = 0;
      for (int i = 0; i < data_layout->numAttributes(); ++i) {
        auto info = (*data_layout)[i];
        auto num_bytes = info.type_info->numBytes();
        auto byte_diff = info.offset + info.type_info->numComponentBytes() - start_offset;
        if (byte_diff > byte_sz) {
          if (byte_diff % byte_sz != 0) {
            throw std::runtime_error(
                "An invalid alignment of the data buffer containing the attribute \"" +
                attr + "\" exists. The attribute's buffer type is: " +
                attr_info.type_info->declString() + " which has a byte size of " +
                std::to_string(byte_sz) + " but it does not align with the attribute \"" +
                info.name + "\" which has an offset of " + std::to_string(info.offset) +
                " and a component byte size of " +
                std::to_string(info.type_info->numComponentBytes()) +
                " in the buffer. The offset must be divisible by the attribute's byte "
                "size.");
          }
        } else if (byte_diff < byte_sz) {
          CHECK_EQ((byte_sz < num_bytes ? num_bytes % byte_sz : byte_sz % num_bytes), 0u)
              << "Invalid layout: " << attr_info.name << ": "
              << attr_info.type_info->declString() << ", " << info.name << ": "
              << info.type_info->declString();
        }

        start_offset += ((info.offset + num_bytes - start_offset) / byte_sz) * byte_sz;
      }

      if (start_offset != stride) {
        std::vector<std::string> attrs;
        for (int i = 0; i < data_layout->numAttributes(); ++i) {
          auto info = (*data_layout)[i];
          attrs.push_back(info.name);
        }
        CHECK(false) << "Invalid layout: " << attr_info.name << ": "
                     << attr_info.type_info->declString() << ", "
                     << ::boost::algorithm::join(attrs, ", ");
      }
      break;
    }
    case BufferLayoutType::kSequential:
      throw std::runtime_error("Data layouts of type " +
                               to_string(data_layout->getLayoutType()) +
                               " are not currently supported for aggregation operators.");
      break;
  }
}

void validate_input_dependency(const XformOp* op, const XformOpShPtr& dep_op) {
  CHECK(dep_op);
  auto dep_info = op->getRequiredDependencyInfo();
  auto const dep_type = XformOp::serializeOperatorProps(*dep_op);
  auto const inputs = dep_op->getInputAttrNames();
  std::for_each(
      inputs.begin(), inputs.end(), [&dep_info, &dep_type](auto const& input_attr) {
        auto itr = dep_info.find(input_attr);
        CHECK(itr != dep_info.end());
        auto dep_itr = itr->second.find(dep_type);
        CHECK(dep_itr != itr->second.end());
      });
}

}  // namespace

SQLTypeInfo AggOp::getOutputType() const {
  auto const parent_xform = parent_xform_.lock();
  CHECK(parent_xform);
  auto in_data = parent_xform->getInputDataTable();
  CHECK(in_data);
  CHECK_EQ(inputs_.size(), 1u);
  auto const itr = inputs_.begin();
  auto layout = getDataLayoutForAttribute(in_data, itr->attr_name);
  if (layout) {
    return layout->getAttrSQLTypeInfoRef(itr->attr_name);
  }
  return render_type_to_sql_type(in_data->getAttributeBufferType(itr->attr_name));
}

BufferAttrType AggOp::getOutputBufferAttrType() const {
  auto const parent_xform = parent_xform_.lock();
  CHECK(parent_xform);
  auto in_data = parent_xform->getInputDataTable();
  CHECK(in_data);
  CHECK_EQ(inputs_.size(), 1u);

  auto const itr = inputs_.begin();
  if (!in_data->hasAttribute(itr->attr_name)) {
    auto layout = getDataLayoutForAttribute(in_data, itr->attr_name);
    CHECK(layout) << "attr: \"" << itr->attr_name << "\", op: " << to_string(getOpType());
    return layout->getBufferLayout()->getAttributeType(itr->attr_name);
  }
  return in_data->getAttributeBufferType(itr->attr_name);
}

void AggOp::validateInputBuffers() {
  if (!visited_inputs_) {
    visited_inputs_ = std::make_shared<std::unordered_set<std::string>>();
  }
  for (auto& input : inputs_) {
    if (visited_inputs_->find(input.attr_name) == visited_inputs_->end()) {
      validate_data_layout_for_thrust(this, input.attr_name, input.buffer_layout);
      visited_inputs_->insert(input.attr_name);
    }
  }
}

/*********************** Operator Dependency Info ***********************/
XformOp::DependencyOpTypeMap AggDepOp::generateInputDependencyInfo(
    const XformOp* op,
    const XformOp::DependencyOutputsMap& dep_outputs) {
  XformOp::OpTypeContainer ops;
  for (auto& dep_output : dep_outputs) {
    ops.insert(dep_output.first);
  }
  auto const inputs = op->getInputAttrNames();
  XformOp::DependencyOpTypeMap rtn;
  for (auto const& input : inputs) {
    rtn.insert({input, ops});
  }
  return rtn;
}

void AggDepOp::setInputDependency(XformOp* this_op,
                                  const XformOp::DependencyOutputsMap& dep_outputs,
                                  XformOp::DependencyOpMap& dependent_ops_map,
                                  const XformOpShPtr& dep_op) {
  validate_input_dependency(this_op, dep_op);
  auto itr = dep_outputs.find(XformOp::serializeOpType(dep_op->getOpType()));
  RUNTIME_EX_ASSERT(itr != dep_outputs.end(),
                    "The operator " + std::string(*this_op) +
                        " does not accept a dependency operator of type " +
                        to_string(dep_op->getOpType()));
  CHECK(dependent_ops_map.insert({itr->second, dep_op}).second);
}

}  // namespace QueryRenderer
