#ifndef QUERYENGINE_GROUPBYANDAGGREGATE_H
#define QUERYENGINE_GROUPBYANDAGGREGATE_H

#include "../Fragmenter/Fragmenter.h"
#include "../Planner/Planner.h"
#include "../QueryEngine/Execute.h"
#include "../Shared/sqltypes.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/Value.h>

#include <vector>


enum class GroupByColRangeType {
  OneColConsecutiveKeys,  // statically known and consecutive keys, used for dictionary encoded columns
  OneColKnownRange,       // statically known range, only possible for column expressions
  OneColGuessedRange,     // best guess: small hash for the guess plus overflow for outliers
  MultiCol,
  Scan,                   // the plan is not a group by plan
};

// Private: each thread has its own memory, no atomic operations required
// Shared: threads in the same block share memory, atomic operations required
enum class GroupByMemSharing {
  Private,
  Shared
};

struct GroupByBufferDescriptor {
  GroupByColRangeType hash_type;
  std::vector<int8_t> group_col_widths;
  std::vector<int8_t> agg_col_widths;
  size_t entry_count;                    // the number of entries in the main buffer
  bool use_shared_memory;                // use shared memory for the main buffer?
  size_t entry_count_small;              // the number of entries in the small buffer
  bool use_shared_memory_small;          // use shared memory for the small buffer?
  int64_t min_val;                       // meaningful for OneCol{KnownRange, ConsecutiveKeys} only
  GroupByMemSharing sharing;             // meaningful for GPU only
};

class GroupByAndAggregate {
public:
  GroupByAndAggregate(
    Executor* executor,
    llvm::Value* filter_result,
    const Planner::Plan* plan,
    const Fragmenter_Namespace::QueryInfo& query_info);

  void codegen(const ExecutorDeviceType, const bool hoist_literals);

  GroupByBufferDescriptor getGroupByBufferDescriptor();

  llvm::Value* codegenGroupBy(const bool hoist_literals);

  void allocateBuffers(const ExecutorDeviceType);

private:
  struct ColRangeInfo {
    GroupByColRangeType hash_type_;
    int64_t min;
    int64_t max;
  };

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
  llvm::Value* filter_result_;
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

}  // namespace

#endif // QUERYENGINE_GROUPBYANDAGGREGATE_H
