#ifndef QUERYENGINE_GROUPBYANDAGGREGATE_H
#define QUERYENGINE_GROUPBYANDAGGREGATE_H

#include "../Fragmenter/Fragmenter.h"
#include "../Planner/Planner.h"
#include "../QueryEngine/Execute.h"
#include "../Shared/sqltypes.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/Value.h>

#include <vector>


class GroupByAndAggregate {
public:
  GroupByAndAggregate(
    Executor* executor,
    const Planner::Plan* plan,
    const Fragmenter_Namespace::QueryInfo& query_info);

  GroupByBufferDescriptor getGroupByBufferDescriptor();

  void codegen(
    llvm::Value* filter_result,
    const ExecutorDeviceType,
    const bool hoist_literals);

  void allocateBuffers(const ExecutorDeviceType);

private:
  struct ColRangeInfo {
    const GroupByColRangeType hash_type_;
    const int64_t min;
    const int64_t max;
  };

  llvm::Value* codegenGroupBy(const bool hoist_literals);

  GroupByAndAggregate::ColRangeInfo getColRangeInfo(
    const Planner::Plan*,
    const std::vector<Fragmenter_Namespace::FragmentInfo>&);

  void codegenAggCalls(
    llvm::Value* agg_out_start_ptr,
    const std::vector<llvm::Value*>& agg_out_vec,
    const ExecutorDeviceType,
    const bool hoist_literals);

  llvm::Value* codegenAggArg(
    const Analyzer::Expr* target_expr,
    const bool hoist_literals);

  llvm::Value* toDoublePrecision(llvm::Value* val);

  llvm::Value* emitCall(const std::string& fname, const std::vector<llvm::Value*>& args);

  llvm::Function* getFunction(const std::string& name) const;

  Executor* executor_;
  const Planner::Plan* plan_;
  const Fragmenter_Namespace::QueryInfo& query_info_;
};

namespace {

inline size_t get_bit_width(const SQLTypes type) {
  switch (type) {
    case kSMALLINT:
      return 16;
    case kINT:
      return 32;
    case kBIGINT:
      return 64;
    case kFLOAT:
      return 32;
    case kDOUBLE:
      return 64;
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
      return sizeof(time_t) * 8;
    case kTEXT:
    case kVARCHAR:
    case kCHAR:
      return 32;
    default:
      CHECK(false);
  }
}

inline std::vector<Analyzer::Expr*> get_agg_target_exprs(const Planner::Plan* plan) {
  const auto& target_list = plan->get_targetlist();
  std::vector<Analyzer::Expr*> result;
  for (auto target : target_list) {
    auto target_expr = target->get_expr();
    CHECK(target_expr);
    result.push_back(target_expr);
  }
  return result;
}

}  // namespace

#endif // QUERYENGINE_GROUPBYANDAGGREGATE_H
