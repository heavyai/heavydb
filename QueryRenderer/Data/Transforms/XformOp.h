/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <unordered_set>
#include <vector>

#include "GfxDriver/Resources/Types.h"
#include "QueryRenderer/Data/Transforms/Enums.h"
#include "QueryRenderer/Data/Transforms/Types.h"
#include "QueryRenderer/Interface/AggDataTypes.h"
#include "QueryRenderer/Interface/DataMgr_ForwardDeclarations.h"
#include "QueryRenderer/Interop/InteropBufferMgr.h"
#include "QueryRenderer/Interop/Types.h"
#include "QueryRenderer/Types.h"
#include "QueryRenderer/Utils/AnyDataType.h"
#include "QueryRenderer/Utils/RapidJSONUtils.h"
#include "gen-cpp/heavy_types.h"

namespace QueryRenderer {

// This needs a specialization for every bottom-level derived op that inherits from
// XformOp
template <OpType opType>
struct OpSelector {
  using type = void;
};

class XformOp {
 public:
  virtual ~XformOp() = default;

  using OpTypeContainer = std::unordered_set<std::string>;
  using DependencyOpTypeMap = std::unordered_map<std::string, OpTypeContainer>;
  using DependencyOpResultsMap = std::unordered_map<std::string, AggDataList>;
  using DependencyOpMap = std::unordered_map<std::string, XformOpShPtr>;
  using DependencyOutputsMap = std::unordered_map<std::string, std::string>;

  struct OpResult {
    bool is_execution_complete;
    AggDataList op_results;

    inline const AggDataList& getDataResults() const { return op_results; }

    void reset() {
      op_results = {};
      is_execution_complete = false;
    }
  };

  template <typename T>
  struct TypedOpResult {
    const bool is_execution_complete;
    const T value;
  };

  template <typename T>
  struct TypedOpArrayResult {
    const bool is_execution_complete;
    const std::vector<T> vector;
  };

  const LayoutAttrInfoSet& getInputs() const { return inputs_; }
  std::unordered_set<std::string> getInputAttrNames() const;

  virtual OpType getOpType() const = 0;
  virtual std::string getOpTypeAsStr() const { return to_string(getOpType()); }
  bool isVectorOp() const { return is_vector_; }
  virtual SQLTypeInfo getOutputType() const = 0;
  virtual gfx::BufferAttrType getOutputBufferAttrType() const = 0;
  virtual bool isDependentOp() const { return false; }
  virtual DependencyOpTypeMap getRequiredDependencyInfo() const { return {{}}; }

  virtual void setDependency(const XformOpShPtr& op) {
    CHECK(false) << "Operator: " + std::string(*this) + " doesn't support dependencies";
  }

  XformOpShPtr getDependency(const std::string& dep_input_attr,
                             const std::string& dep_op_type) const;

  void setDependent(const XformOpWkPtr& op);

  virtual operator std::string() const;

  template <typename T, typename std::enable_if_t<std::is_arithmetic_v<T>>* = nullptr>
  TypedOpResult<T> evaluateValue(const std::string& evaluator_name,
                                 InteropBufferMgr* mapped_buffers = nullptr) {
    RUNTIME_EX_ASSERT(
        !is_vector_,
        std::string(*this) +
            ": Cannot evaluate to a singular value. This op results in an vector.");

    auto eval_result = evaluate(evaluator_name, mapped_buffers);
    if (!eval_result.is_execution_complete) {
      return {false, T(0)};
    }
    auto const& data_results = eval_result.getDataResults();
    CHECK_EQ(data_results.size(), 1u);
    CHECK(data_results[0]);
    CHECK(!data_results[0]->isVector());
    return {eval_result.is_execution_complete, data_results[0]->getVal<T>()};
  }

  template <
      typename T,
      typename std::enable_if_t<std::bool_constant<
          std::is_arithmetic_v<T> ||
          std::is_same_v<std::string, typename std::remove_cv_t<T>>>::value>* = nullptr>
  TypedOpArrayResult<T> evaluateVector(const std::string& evaluator_name,
                                       InteropBufferMgr* mapped_buffers = nullptr) {
    RUNTIME_EX_ASSERT(
        is_vector_,
        std::string(*this) +
            ": Cannot evaluate to an vector. This op results in singular value.");
    auto eval_result = evaluate(evaluator_name, mapped_buffers);
    if (!eval_result.is_execution_complete) {
      return {false, std::vector<T>()};
    }
    auto const& data_results = eval_result.getDataResults();
    CHECK_EQ(data_results.size(), 1u);
    CHECK(data_results[0]);
    CHECK(data_results[0]->isVector());
    return {eval_result.is_execution_complete, data_results[0]->getVectorVal<T>()};
  }

  const OpResult evaluate(const std::string& evaluator_name,
                          InteropBufferMgr* mapped_buffers = nullptr);

  void setDirty();
  void clearCacheWithoutPropagatingDirtyFlag();

  static void validateNumericAttrType(const BaseDataTableShPtr& in_data,
                                      const LayoutAttrInfoSet& inputs,
                                      const XformOp* current_op);

  static inline std::string serializeOpType(const OpType op) {
    return std::to_string(static_cast<int>(op));
  }
  static std::string serializeOperatorProps(const XformOp* op);
  static std::string serializeOperatorProps(const XformOp& op);
  static std::pair<OpType, std::string> getOpTypeAndSerializeOperatorFromJSONObj(
      const JSONLocation& json_loc);
  static std::pair<OpType, std::vector<AnyDataType>> deserializeOperatorAndProps(
      const std::string& serialized_str);

 protected:
  XformOp(const XformShPtr& parent_xform,
          const LayoutAttrInfoSet& inputs,
          const bool is_array_op);

  XformOp(const XformShPtr& parent_xform,
          const LayoutAttrInfo& input_info,
          const bool is_array_op);

  XformOp(const XformShPtr& parent_xform,
          const std::vector<LayoutAttrInfo>& all_input_info,
          const bool is_array_op);

  const XformWkPtr parent_xform_;
  LayoutAttrInfoSet inputs_;
  OpResult cached_result_;
  std::unordered_map<XformOp*, XformOpWkPtr> dependents_;

  Data_Namespace::DataMgr& getDataMgr();
  const CudaMgr_Namespace::CudaMgr* getCudaMgr() const;

  const QueryRendererContext& getRenderContext() const;
  QueryRendererContext& getRenderContextNonConst();

  const OpResult evaluateInternal(const std::string& evaluator_name,
                                  InteropBufferMgr* mapped_buffers);

  virtual const OpResult executeOp(const std::string& evaluator_name,
                                   InteropBufferMgr* mapped_buffers,
                                   const DependencyOpResultsMap& dependency_results) = 0;

  DependencyOpMap* getDependencyOps(const std::string* op_type = nullptr) {
    return const_cast<DependencyOpMap*>(
        static_cast<const XformOp&>(*this).getDependencyOps(op_type));
  }
  virtual const DependencyOpMap* getDependencyOps(
      const std::string* op_type = nullptr) const {
    return nullptr;
  }
  virtual const DependencyOutputsMap* getDependencyOutputsMap() const { return nullptr; }
  virtual bool isValidDepInput(const std::string& dep_input_attr) const {
    return inputs_.find(dep_input_attr) != inputs_.end();
  }

  virtual void validateInputs() const;

 private:
  bool dirty_;
  const bool is_vector_;

  bool isDirty() const;
  void cleanupDependents();

  void serialize(std::stringstream& ss) const;
  virtual void serializeProps(std::stringstream& ss) const {}
  static std::vector<AnyDataType> deserializeProps(std::istringstream& ss) { return {}; }
  static void serializePropsFromJSONObj(std::stringstream& ss,
                                        const JSONLocation& json_loc) {
    // noop
  }
};

/***** specializations *****/
template <>
XformOp::TypedOpArrayResult<std::string> XformOp::evaluateVector<std::string>(
    const std::string& evaluator_name,
    InteropBufferMgr* mapped_buffers);
/*** specializations done **/

class XformDepOp : public XformOp {
 public:
  ~XformDepOp() override {}
  bool isDependentOp() const final { return true; }

 protected:
  XformDepOp(const XformShPtr& parent_xform,
             const LayoutAttrInfoSet& inputs,
             const bool is_array_op)
      : XformOp(parent_xform, inputs, is_array_op) {}

  XformDepOp(const XformShPtr& parent_xform,
             const LayoutAttrInfo& input_info,
             const bool is_array_op)
      : XformOp(parent_xform, input_info, is_array_op) {}

  XformDepOp(const XformShPtr& parent_xform,
             const std::vector<LayoutAttrInfo>& all_input_info,
             const bool is_array_op)
      : XformOp(parent_xform, all_input_info, is_array_op) {}

  mutable DependencyOpMap dependent_ops_;
};

}  // namespace QueryRenderer
