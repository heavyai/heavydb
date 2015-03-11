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
    llvm::Value* filter_result,
    const Planner::Plan* plan,
    const Fragmenter_Namespace::QueryInfo& query_info);

  void codegen(const ExecutorDeviceType, const bool hoist_literals);

  llvm::Value* codegenGroupBy(const std::list<Analyzer::Expr*>& groupby_list,
                              const bool hoist_literals);

  void allocateBuffers(const ExecutorDeviceType);

  enum class ColRangeType {
    OneColConsecutiveKeys,  // statically known and consecutive keys, used for dictionary encoded columns
    OneColKnownRange,       // statically known range, only possible for column expressions
    OneColGuessedRange,     // best guess: small hash for the guess plus overflow for outliers
    MultiCol,
    Scan,                   // the plan is not a group by plan
  };

  // Private: each thread has its own memory, no atomic operations required
  // Shared: threads in the same block share memory, atomic operations required
  enum class Sharing {
    Private,
    Shared
  };

private:
  struct ColRangeInfo {
    GroupByAndAggregate::ColRangeType hash_type_;
    int64_t min;
    int64_t max;
  };

  GroupByAndAggregate::ColRangeInfo getColRangeInfo(
    const Planner::AggPlan*,
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
  llvm::Value* filter_result_;
  const Planner::Plan* plan_;
  const Fragmenter_Namespace::QueryInfo& query_info_;
};

#endif // QUERYENGINE_GROUPBYANDAGGREGATE_H
