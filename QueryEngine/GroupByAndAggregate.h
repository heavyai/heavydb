#ifndef QUERYENGINE_GROUPBYANDAGGREGATE_H
#define QUERYENGINE_GROUPBYANDAGGREGATE_H

#include "../Fragmenter/Fragmenter.h"
#include "../Planner/Planner.h"
#include "../Shared/sqltypes.h"

#include <boost/noncopyable.hpp>
#include <boost/variant.hpp>
#include <glog/logging.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Value.h>

#include <vector>


enum class ExecutorDeviceType {
  CPU,
  GPU
};

class Executor;

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

struct GroupByBufferDescriptor;

typedef boost::variant<int64_t, double, std::string> AggResult;

inline bool approx_eq(const double v, const double target, const double eps = 0.01) {
  return target - eps < v && v < target + eps;
}

class ResultRow {
public:
  ResultRow(const Executor* executor) : executor_(executor) {}

  AggResult agg_result(const size_t idx, const bool translate_strings = true) const;

  SQLTypeInfo agg_type(const size_t idx) const;

  size_t size() const {
    return agg_results_idx_.size();
  }

  std::vector<int64_t> value_tuple() const {
    return value_tuple_;
  }

  bool operator==(const ResultRow& r) const {
    if (size() != r.size()) {
      return false;
    }
    for (size_t idx = 0; idx < size(); ++idx) {
      const auto lhs_val = agg_result(idx);
      const auto rhs_val = r.agg_result(idx);
      {
        const auto lhs_pd = boost::get<double>(&lhs_val);
        if (lhs_pd) {
          const auto rhs_pd = boost::get<double>(&rhs_val);
          if (!rhs_pd) {
            return false;
          }
          if (!approx_eq(*lhs_pd, *rhs_pd)) {
            return false;
          }
        } else {
          if (lhs_val < rhs_val || rhs_val < lhs_val) {
            return false;
          }
        }
      }
    }
    return true;
  }

private:
  // TODO(alex): support for strings
  std::vector<int64_t> value_tuple_;
  std::vector<int64_t> agg_results_;
  std::vector<size_t> agg_results_idx_;
  std::vector<SQLAgg> agg_kinds_;
  std::vector<SQLTypeInfo> agg_types_;
  const Executor* executor_;

  friend class Executor;
};

class GroupByMemory : boost::noncopyable {
public:
  // TODO(alex): move init_agg_vals to GroupByBufferDescriptor, remove device_type
  GroupByMemory(const GroupByBufferDescriptor&,
                const std::vector<int64_t>& init_agg_vals,
                const Executor* executor,
                const ExecutorDeviceType device_type);
  ~GroupByMemory();

  // TOOD(alex): get rid of targets parameter
  std::vector<ResultRow> getRowSet(const std::vector<Analyzer::Expr*>& targets) const;

private:
  const GroupByBufferDescriptor& group_buff_desc_;
  const Executor* executor_;
  const ExecutorDeviceType device_type_;
  const size_t num_buffers_;

  std::vector<int64_t*> group_by_buffers_;
  std::vector<int64_t*> small_group_by_buffers_;

  friend class Executor;
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

  std::unique_ptr<GroupByMemory> allocateGroupByMem(
    const std::vector<int64_t>& init_agg_vals,
    const Executor* executor,
    const ExecutorDeviceType device_type) const;

  size_t getBufferSize() const;
  size_t getSmallBufferSize() const;

  // TODO(alex): remove
  bool usesGetGroupValueFast() const;
};

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
