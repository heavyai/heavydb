#include "Execute.h"

#include "CartesianProduct.h"
#include "Codec.h"
#include "ExpressionRewrite.h"
#include "GpuMemUtils.h"
#include "InPlaceSort.h"
#include "GroupByAndAggregate.h"
#include "AggregateUtils.h"
#include "NvidiaKernel.h"
#include "QueryTemplateGenerator.h"
#include "QueryRewrite.h"
#include "RuntimeFunctions.h"
#include "JsonAccessors.h"
#include "DataMgr/BufferMgr/BufferMgr.h"
#include "CudaMgr/CudaMgr.h"
#include "Parser/ParserNode.h"
#include "Shared/mapdpath.h"
#include "Shared/checked_alloc.h"

#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/InstIterator.h>
#include "llvm/IR/Intrinsics.h"
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/FormattedStream.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Transforms/Instrumentation.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Bitcode/ReaderWriter.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>

#include <cuda.h>

#include <algorithm>
#include <numeric>
#include <thread>
#include <unistd.h>
#include <map>
#include <set>

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

bool g_enable_watchdog{false};

Executor::Executor(const int db_id,
                   const size_t block_size_x,
                   const size_t grid_size_x,
                   const std::string& debug_dir,
                   const std::string& debug_file,
                   ::QueryRenderer::QueryRenderManager* render_manager)
    : cgen_state_(new CgenState()),
      is_nested_(false),
      render_manager_(render_manager),
      block_size_x_(block_size_x),
      grid_size_x_(grid_size_x),
      debug_dir_(debug_dir),
      debug_file_(debug_file),
      db_id_(db_id),
      catalog_(nullptr),
      temporary_tables_(nullptr) {
}

std::shared_ptr<Executor> Executor::getExecutor(const int db_id,
                                                const std::string& debug_dir,
                                                const std::string& debug_file,
                                                const size_t block_size_x,
                                                const size_t grid_size_x,
                                                ::QueryRenderer::QueryRenderManager* render_manager) {
  const auto executor_key = std::make_pair(db_id, render_manager);
  {
    mapd_shared_lock<mapd_shared_mutex> read_lock(executors_cache_mutex_);
    auto it = executors_.find(executor_key);
    if (it != executors_.end()) {
      return it->second;
    }
  }
  {
    mapd_unique_lock<mapd_shared_mutex> write_lock(executors_cache_mutex_);
    auto it = executors_.find(executor_key);
    if (it != executors_.end()) {
      return it->second;
    }
    auto executor = std::make_shared<Executor>(db_id, block_size_x, grid_size_x, debug_dir, debug_file, render_manager);
    auto it_ok = executors_.insert(std::make_pair(executor_key, executor));
    CHECK(it_ok.second);
    return executor;
  }
}

namespace {

int64_t get_scan_limit(const Planner::Plan* plan, const int64_t limit) {
  return (dynamic_cast<const Planner::Scan*>(plan) || dynamic_cast<const Planner::Join*>(plan)) ? limit : 0;
}

const Planner::Scan* get_scan_child(const Planner::Plan* plan) {
  const auto agg_plan = dynamic_cast<const Planner::AggPlan*>(plan);
  return agg_plan ? dynamic_cast<const Planner::Scan*>(plan->get_child_plan())
                  : dynamic_cast<const Planner::Scan*>(plan);
}

const Planner::Join* get_join_child(const Planner::Plan* plan) {
  const auto join_plan = dynamic_cast<const Planner::Join*>(plan);
  return join_plan ? join_plan : dynamic_cast<const Planner::Join*>(plan->get_child_plan());
}

void collect_input_col_descs(std::list<InputColDescriptor>& input_col_descs,
                             const Planner::Scan* scan_plan,
                             const Catalog_Namespace::Catalog& cat,
                             const bool is_join,
                             const size_t scan_idx) {
  CHECK(scan_idx == 0 || is_join);
  CHECK(scan_plan);
  const int table_id = scan_plan->get_table_id();
  for (const int scan_col_id : scan_plan->get_col_list()) {
    auto cd = get_column_descriptor(scan_col_id, table_id, cat);
    if (cd->isVirtualCol) {
      CHECK_EQ("rowid", cd->columnName);
    } else {
      input_col_descs.emplace_back(scan_col_id, table_id, scan_idx);
    }
  }
}

void collect_input_descs(std::vector<InputDescriptor>& input_descs,
                         std::list<InputColDescriptor>& input_col_descs,
                         const Planner::Plan* plan,
                         const Catalog_Namespace::Catalog& cat) {
  const auto scan_plan = get_scan_child(plan);
  const auto join_plan = get_join_child(plan);
  const Planner::Scan* outer_plan{nullptr};
  const Planner::Scan* inner_plan{nullptr};
  if (join_plan) {
    outer_plan = get_scan_child(join_plan->get_outerplan());
    CHECK(outer_plan);
    inner_plan = get_scan_child(join_plan->get_innerplan());
    CHECK(inner_plan);
    input_descs.emplace_back(outer_plan->get_table_id(), 0);
    input_descs.emplace_back(inner_plan->get_table_id(), 1);
    collect_input_col_descs(input_col_descs, outer_plan, cat, true, 0);
    collect_input_col_descs(input_col_descs, inner_plan, cat, true, 1);
  } else {
    CHECK(scan_plan);
    input_descs.emplace_back(scan_plan->get_table_id(), 0);
    collect_input_col_descs(input_col_descs, scan_plan, cat, false, 0);
  }
}

void collect_simple_quals(std::list<std::shared_ptr<Analyzer::Expr>>& simple_quals, const Planner::Scan* scan_plan) {
  CHECK(scan_plan);
  const auto& more_simple_quals = scan_plan->get_simple_quals();
  simple_quals.insert(simple_quals.end(), more_simple_quals.begin(), more_simple_quals.end());
}

void collect_quals(std::list<std::shared_ptr<Analyzer::Expr>>& quals, const Planner::Scan* scan_plan) {
  CHECK(scan_plan);
  const auto& more_quals = scan_plan->get_quals();
  quals.insert(quals.end(), more_quals.begin(), more_quals.end());
}

void collect_quals_from_join(std::list<std::shared_ptr<Analyzer::Expr>>& simple_quals,
                             std::list<std::shared_ptr<Analyzer::Expr>>& quals,
                             const Planner::Join* join_plan) {
  const auto outer_plan = get_scan_child(join_plan->get_outerplan());
  CHECK(outer_plan);
  const auto inner_plan = get_scan_child(join_plan->get_innerplan());
  CHECK(inner_plan);
  collect_simple_quals(simple_quals, outer_plan);
  collect_simple_quals(simple_quals, inner_plan);
  collect_quals(quals, outer_plan);
  collect_quals(quals, inner_plan);
}

bool check_plan_sanity(const Planner::Plan* plan) {
  const auto join_plan = get_join_child(plan);
  const auto scan_plan = get_scan_child(plan);
  return static_cast<bool>(scan_plan) != static_cast<bool>(join_plan);
}

bool is_unnest(const Analyzer::Expr* expr) {
  return dynamic_cast<const Analyzer::UOper*>(expr) &&
         static_cast<const Analyzer::UOper*>(expr)->get_optype() == kUNNEST;
}

}  // namespace

ResultRows Executor::executeSelectPlan(const Planner::Plan* plan,
                                       const int64_t limit,
                                       const int64_t offset,
                                       const bool hoist_literals,
                                       const ExecutorDeviceType device_type,
                                       const ExecutorOptLevel opt_level,
                                       const Catalog_Namespace::Catalog& cat,
                                       size_t& max_groups_buffer_entry_guess,
                                       int32_t* error_code,
                                       const Planner::Sort* sort_plan_in,
                                       const bool allow_multifrag,
                                       const bool just_explain,
                                       const bool allow_loop_joins,
                                       RenderAllocatorMap* render_allocator_map) {
  if (dynamic_cast<const Planner::Scan*>(plan) || dynamic_cast<const Planner::AggPlan*>(plan) ||
      dynamic_cast<const Planner::Join*>(plan)) {
    row_set_mem_owner_ = std::make_shared<RowSetMemoryOwner>();
    lit_str_dict_ = nullptr;
    const auto target_exprs = get_agg_target_exprs(plan);
    const auto scan_plan = get_scan_child(plan);
    auto simple_quals = scan_plan ? scan_plan->get_simple_quals() : std::list<std::shared_ptr<Analyzer::Expr>>{};
    auto quals = scan_plan ? scan_plan->get_quals() : std::list<std::shared_ptr<Analyzer::Expr>>{};
    const auto agg_plan = dynamic_cast<const Planner::AggPlan*>(plan);
    auto groupby_exprs = agg_plan ? agg_plan->get_groupby_list() : std::list<std::shared_ptr<Analyzer::Expr>>{nullptr};
    std::vector<InputDescriptor> input_descs;
    std::list<InputColDescriptor> input_col_descs;
    collect_input_descs(input_descs, input_col_descs, plan, cat);
    const auto join_plan = get_join_child(plan);
    if (join_plan) {
      collect_quals_from_join(simple_quals, quals, join_plan);
    }
    const auto join_quals = join_plan ? join_plan->get_quals() : std::list<std::shared_ptr<Analyzer::Expr>>{};
    CHECK(check_plan_sanity(plan));
    const bool is_agg = dynamic_cast<const Planner::AggPlan*>(plan);
    const auto order_entries = sort_plan_in ? sort_plan_in->get_order_entries() : std::list<Analyzer::OrderEntry>{};
    const auto query_infos = get_table_infos(input_descs, cat, TemporaryTables{});
    const size_t scan_limit = get_scan_limit(plan, limit);
    const size_t scan_total_limit = scan_limit ? get_scan_limit(plan, scan_limit + offset) : 0;
    const auto ra_exe_unit_in = RelAlgExecutionUnit{input_descs,
                                                    input_col_descs,
                                                    simple_quals,
                                                    quals,
                                                    JoinType::INVALID,
                                                    join_quals,
                                                    {},
                                                    groupby_exprs,
                                                    target_exprs,
                                                    order_entries,
                                                    scan_total_limit};
    QueryRewriter query_rewriter(ra_exe_unit_in, query_infos, this, agg_plan);
    const auto ra_exe_unit = query_rewriter.rewrite();
    if (limit || offset) {
      size_t max_groups_buffer_entry_guess_limit{scan_total_limit ? scan_total_limit : max_groups_buffer_entry_guess};
      auto rows = executeWorkUnit(error_code,
                                  max_groups_buffer_entry_guess_limit,
                                  is_agg,
                                  query_infos,
                                  ra_exe_unit,
                                  {device_type, hoist_literals, opt_level},
                                  {false, allow_multifrag, just_explain, allow_loop_joins, g_enable_watchdog},
                                  cat,
                                  row_set_mem_owner_,
                                  render_allocator_map);
      max_groups_buffer_entry_guess = max_groups_buffer_entry_guess_limit;
      rows.dropFirstN(offset);
      if (limit) {
        rows.keepFirstN(limit);
      }
      return rows;
    }
    return executeWorkUnit(error_code,
                           max_groups_buffer_entry_guess,
                           is_agg,
                           query_infos,
                           ra_exe_unit,
                           {device_type, hoist_literals, opt_level},
                           {false, allow_multifrag, just_explain, allow_loop_joins, g_enable_watchdog},
                           cat,
                           row_set_mem_owner_,
                           render_allocator_map);
  }
  const auto result_plan = dynamic_cast<const Planner::Result*>(plan);
  if (result_plan) {
    if (limit || offset) {
      auto rows = executeResultPlan(result_plan,
                                    hoist_literals,
                                    device_type,
                                    opt_level,
                                    cat,
                                    max_groups_buffer_entry_guess,
                                    error_code,
                                    sort_plan_in,
                                    allow_multifrag,
                                    just_explain,
                                    allow_loop_joins);
      rows.dropFirstN(offset);
      if (limit) {
        rows.keepFirstN(limit);
      }
      return rows;
    }
    return executeResultPlan(result_plan,
                             hoist_literals,
                             device_type,
                             opt_level,
                             cat,
                             max_groups_buffer_entry_guess,
                             error_code,
                             sort_plan_in,
                             allow_multifrag,
                             just_explain,
                             allow_loop_joins);
  }
  const auto sort_plan = dynamic_cast<const Planner::Sort*>(plan);
  if (sort_plan) {
    return executeSortPlan(sort_plan,
                           limit,
                           offset,
                           hoist_literals,
                           device_type,
                           opt_level,
                           cat,
                           max_groups_buffer_entry_guess,
                           error_code,
                           allow_multifrag,
                           just_explain,
                           allow_loop_joins);
  }
  CHECK(false);
}

/*
 * x64 benchmark: "SELECT COUNT(*) FROM test WHERE x > 41;"
 *                x = 42, 64-bit column, 1-byte encoding
 *                3B rows in 1.2s on a i7-4870HQ core
 *
 * TODO(alex): check we haven't introduced a regression with the new translator.
 */

ResultRows Executor::execute(const Planner::RootPlan* root_plan,
                             const Catalog_Namespace::SessionInfo& session,
                             const int render_widget_id,
                             const bool hoist_literals,
                             const ExecutorDeviceType device_type,
                             const ExecutorOptLevel opt_level,
                             const bool allow_multifrag,
                             const bool allow_loop_joins) {
  catalog_ = &root_plan->get_catalog();
  const auto stmt_type = root_plan->get_stmt_type();
  // capture the lock acquistion time
  auto clock_begin = timer_start();
  std::lock_guard<std::mutex> lock(execute_mutex_);
  int64_t queue_time_ms = timer_stop(clock_begin);
  RowSetHolder row_set_holder(this);
  switch (stmt_type) {
    case kSELECT: {
      int32_t error_code{0};
      size_t max_groups_buffer_entry_guess{16384};

      std::unique_ptr<RenderAllocatorMap> render_allocator_map;
      if (root_plan->get_plan_dest() == Planner::RootPlan::kRENDER) {
        if (device_type != ExecutorDeviceType::GPU) {
          throw std::runtime_error("Backend rendering is only supported on GPU");
        }

        if (!render_manager_) {
          throw std::runtime_error("This build doesn't support backend rendering");
        }

        render_allocator_map.reset(
            new RenderAllocatorMap(catalog_->get_dataMgr().cudaMgr_, render_manager_, blockSize(), gridSize()));
      }
      auto rows = executeSelectPlan(root_plan->get_plan(),
                                    root_plan->get_limit(),
                                    root_plan->get_offset(),
                                    hoist_literals,
                                    device_type,
                                    opt_level,
                                    root_plan->get_catalog(),
                                    max_groups_buffer_entry_guess,
                                    &error_code,
                                    nullptr,
                                    allow_multifrag,
                                    root_plan->get_plan_dest() == Planner::RootPlan::kEXPLAIN,
                                    allow_loop_joins,
                                    render_allocator_map.get());
#ifdef ENABLE_COMPACTION
      if (error_code == ERR_OVERFLOW_OR_UNDERFLOW) {
        throw std::runtime_error("Overflow or underflow");
      }
#endif
      if (error_code == ERR_DIV_BY_ZERO) {
        throw std::runtime_error("Division by zero");
      }
      if (error_code == ERR_UNSUPPORTED_SELF_JOIN) {
        throw std::runtime_error("Self joins not supported yet");
      }
      if (error_code == ERR_OUT_OF_CPU_MEM) {
        throw std::runtime_error("Not enough host memory to execute the query");
      }
      if (error_code == ERR_OUT_OF_GPU_MEM) {
        rows = executeSelectPlan(root_plan->get_plan(),
                                 root_plan->get_limit(),
                                 root_plan->get_offset(),
                                 hoist_literals,
                                 device_type,
                                 opt_level,
                                 root_plan->get_catalog(),
                                 max_groups_buffer_entry_guess,
                                 &error_code,
                                 nullptr,
                                 false,
                                 false,
                                 allow_loop_joins,
                                 nullptr);
      }
      if (error_code) {
        max_groups_buffer_entry_guess = 0;
        while (true) {
          rows = executeSelectPlan(root_plan->get_plan(),
                                   root_plan->get_limit(),
                                   root_plan->get_offset(),
                                   hoist_literals,
                                   ExecutorDeviceType::CPU,
                                   opt_level,
                                   root_plan->get_catalog(),
                                   max_groups_buffer_entry_guess,
                                   &error_code,
                                   nullptr,
                                   false,
                                   false,
                                   allow_loop_joins,
                                   nullptr);
          if (!error_code) {
            rows.setQueueTime(queue_time_ms);
            return rows;
          }
          // Even the conservative guess failed; it should only happen when we group
          // by a huge cardinality array. Maybe we should throw an exception instead?
          // Such a heavy query is entirely capable of exhausting all the host memory.
          CHECK(max_groups_buffer_entry_guess);
          max_groups_buffer_entry_guess *= 2;
        }
      }
      rows.setQueueTime(queue_time_ms);
      return rows;
    }
    case kINSERT: {
      if (root_plan->get_plan_dest() == Planner::RootPlan::kEXPLAIN) {
        return ResultRows("No explanation available.", queue_time_ms);
      }
      executeSimpleInsert(root_plan);
      return ResultRows({}, {}, nullptr, nullptr, {}, ExecutorDeviceType::CPU, nullptr, 0, 0, 0, queue_time_ms);
    }
    default:
      CHECK(false);
  }
  CHECK(false);
  return ResultRows({}, {}, nullptr, nullptr, {}, ExecutorDeviceType::CPU, nullptr, 0, 0, 0, queue_time_ms);
}


StringDictionary* Executor::getStringDictionary(const int dict_id_in,
                                                std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner) const {
  const int dict_id{dict_id_in < 0 ? REGULAR_DICT(dict_id_in) : dict_id_in};
  CHECK(catalog_);
  const auto dd = catalog_->getMetadataForDict(dict_id);
  std::lock_guard<std::mutex> lock(str_dict_mutex_);
  if (dd) {
    if (row_set_mem_owner) {
      CHECK(dd->stringDict);
      row_set_mem_owner->addStringDict(dd->stringDict.get());
    }
    CHECK_EQ(32, dd->dictNBits);
    return dd->stringDict.get();
  }
  CHECK_EQ(0, dict_id);
  if (!lit_str_dict_) {
    lit_str_dict_.reset(new StringDictionary(""));
  }
  return lit_str_dict_.get();
}

bool Executor::isCPUOnly() const {
  CHECK(catalog_);
  return !catalog_->get_dataMgr().cudaMgr_;
}

const ColumnDescriptor* Executor::getColumnDescriptor(const Analyzer::ColumnVar* col_var) const {
  return get_column_descriptor_maybe(col_var->get_column_id(), col_var->get_table_id(), *catalog_);
}

std::vector<int8_t> Executor::serializeLiterals(const std::unordered_map<int, Executor::LiteralValues>& literals,
                                                const int device_id) {
  if (literals.empty()) {
    return {};
  }
  const auto dev_literals_it = literals.find(device_id);
  CHECK(dev_literals_it != literals.end());
  const auto& dev_literals = dev_literals_it->second;
  size_t lit_buf_size{0};
  std::vector<std::string> real_strings;
  for (const auto& lit : dev_literals) {
    lit_buf_size = addAligned(lit_buf_size, Executor::literalBytes(lit));
    if (lit.which() == 7) {
      const auto p = boost::get<std::string>(&lit);
      CHECK(p);
      real_strings.push_back(*p);
    }
  }
  if (lit_buf_size > static_cast<size_t>(std::numeric_limits<int16_t>::max())) {
    throw TooManyLiterals();
  }
  int16_t crt_real_str_off = lit_buf_size;
  for (const auto& real_str : real_strings) {
    CHECK_LE(real_str.size(), static_cast<size_t>(std::numeric_limits<int16_t>::max()));
    lit_buf_size += real_str.size();
  }
  unsigned crt_real_str_idx = 0;
  std::vector<int8_t> serialized(lit_buf_size);
  size_t off{0};
  for (const auto& lit : dev_literals) {
    const auto lit_bytes = Executor::literalBytes(lit);
    off = addAligned(off, lit_bytes);
    switch (lit.which()) {
      case 0: {
        const auto p = boost::get<int8_t>(&lit);
        CHECK(p);
        serialized[off - lit_bytes] = *p;
        break;
      }
      case 1: {
        const auto p = boost::get<int16_t>(&lit);
        CHECK(p);
        memcpy(&serialized[off - lit_bytes], p, lit_bytes);
        break;
      }
      case 2: {
        const auto p = boost::get<int32_t>(&lit);
        CHECK(p);
        memcpy(&serialized[off - lit_bytes], p, lit_bytes);
        break;
      }
      case 3: {
        const auto p = boost::get<int64_t>(&lit);
        CHECK(p);
        memcpy(&serialized[off - lit_bytes], p, lit_bytes);
        break;
      }
      case 4: {
        const auto p = boost::get<float>(&lit);
        CHECK(p);
        memcpy(&serialized[off - lit_bytes], p, lit_bytes);
        break;
      }
      case 5: {
        const auto p = boost::get<double>(&lit);
        CHECK(p);
        memcpy(&serialized[off - lit_bytes], p, lit_bytes);
        break;
      }
      case 6: {
        const auto p = boost::get<std::pair<std::string, int>>(&lit);
        CHECK(p);
        const auto str_id = getStringDictionary(p->second, row_set_mem_owner_)->get(p->first);
        memcpy(&serialized[off - lit_bytes], &str_id, lit_bytes);
        break;
      }
      case 7: {
        const auto p = boost::get<std::string>(&lit);
        CHECK(p);
        int32_t off_and_len = crt_real_str_off << 16;
        const auto& crt_real_str = real_strings[crt_real_str_idx];
        off_and_len |= static_cast<int16_t>(crt_real_str.size());
        memcpy(&serialized[off - lit_bytes], &off_and_len, lit_bytes);
        memcpy(&serialized[crt_real_str_off], crt_real_str.data(), crt_real_str.size());
        ++crt_real_str_idx;
        crt_real_str_off += crt_real_str.size();
        break;
      }
      default:
        CHECK(false);
    }
  }
  return serialized;
}

std::vector<llvm::Value*> Executor::codegen(const Analyzer::Expr* expr,
                                            const bool fetch_columns,
                                            const CompilationOptions& co) {
  if (!expr) {
    return {posArg(expr)};
  }
  auto bin_oper = dynamic_cast<const Analyzer::BinOper*>(expr);
  if (bin_oper) {
    return {codegen(bin_oper, co)};
  }
  auto u_oper = dynamic_cast<const Analyzer::UOper*>(expr);
  if (u_oper) {
    return {codegen(u_oper, co)};
  }
  auto col_var = dynamic_cast<const Analyzer::ColumnVar*>(expr);
  if (col_var) {
    return codegen(col_var, fetch_columns, co.hoist_literals_);
  }
  auto constant = dynamic_cast<const Analyzer::Constant*>(expr);
  if (constant) {
    if (constant->get_type_info().get_compression() == kENCODING_DICT) {
      CHECK(constant->get_is_null());
      return {inlineIntNull(constant->get_type_info())};
    }
    // The dictionary encoding case should be handled by the parent expression
    // (cast, for now), here is too late to know the dictionary id
    return {codegen(constant, constant->get_type_info().get_compression(), 0, co)};
  }
  auto case_expr = dynamic_cast<const Analyzer::CaseExpr*>(expr);
  if (case_expr) {
    return {codegen(case_expr, co)};
  }
  auto extract_expr = dynamic_cast<const Analyzer::ExtractExpr*>(expr);
  if (extract_expr) {
    return {codegen(extract_expr, co)};
  }
  auto datetrunc_expr = dynamic_cast<const Analyzer::DatetruncExpr*>(expr);
  if (datetrunc_expr) {
    return {codegen(datetrunc_expr, co)};
  }
  auto charlength_expr = dynamic_cast<const Analyzer::CharLengthExpr*>(expr);
  if (charlength_expr) {
    return {codegen(charlength_expr, co)};
  }

  auto like_expr = dynamic_cast<const Analyzer::LikeExpr*>(expr);
  if (like_expr) {
    return {codegen(like_expr, co)};
  }

  auto in_expr = dynamic_cast<const Analyzer::InValues*>(expr);
  if (in_expr) {
    return {codegen(in_expr, co)};
  }
#ifdef HAVE_CALCITE
  CHECK(false);
#else
  throw std::runtime_error("Invalid scalar expression");
#endif
}

extern "C" uint64_t string_decode(int8_t* chunk_iter_, int64_t pos) {
  auto chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);
  VarlenDatum vd;
  bool is_end;
  ChunkIter_get_nth(chunk_iter, pos, false, &vd, &is_end);
  CHECK(!is_end);
  return vd.is_null ? 0 : (reinterpret_cast<uint64_t>(vd.pointer) & 0xffffffffffff) |
                              (static_cast<uint64_t>(vd.length) << 48);
}

extern "C" uint64_t string_decompress(const int32_t string_id, const int64_t string_dict_handle) {
  if (string_id == NULL_INT) {
    return 0;
  }
  auto string_dict = reinterpret_cast<const StringDictionary*>(string_dict_handle);
  auto string_bytes = string_dict->getStringBytes(string_id);
  CHECK(string_bytes.first);
  return (reinterpret_cast<uint64_t>(string_bytes.first) & 0xffffffffffff) |
         (static_cast<uint64_t>(string_bytes.second) << 48);
}

extern "C" int32_t string_compress(const int64_t ptr_and_len, const int64_t string_dict_handle) {
  std::string raw_str(reinterpret_cast<char*>(extract_str_ptr_noinline(ptr_and_len)),
                      extract_str_len_noinline(ptr_and_len));
  auto string_dict = reinterpret_cast<const StringDictionary*>(string_dict_handle);
  return string_dict->get(raw_str);
}

llvm::Value* Executor::codegen(const Analyzer::CharLengthExpr* expr, const CompilationOptions& co) {
  auto str_lv = codegen(expr->get_arg(), true, co);
  if (str_lv.size() != 3) {
    CHECK_EQ(size_t(1), str_lv.size());
    str_lv.push_back(cgen_state_->emitCall("extract_str_ptr", {str_lv.front()}));
    str_lv.push_back(cgen_state_->emitCall("extract_str_len", {str_lv.front()}));
    cgen_state_->must_run_on_cpu_ = true;
  }
  std::vector<llvm::Value*> charlength_args{str_lv[1], str_lv[2]};
  std::string fn_name("char_length");
  if (expr->get_calc_encoded_length())
    fn_name += "_encoded";
  const bool is_nullable{!expr->get_arg()->get_type_info().get_notnull()};
  if (is_nullable) {
    fn_name += "_nullable";
    charlength_args.push_back(inlineIntNull(expr->get_type_info()));
  }
  return expr->get_calc_encoded_length()
             ? cgen_state_->emitExternalCall(fn_name, get_int_type(32, cgen_state_->context_), charlength_args)
             : cgen_state_->emitCall(fn_name, charlength_args);
}

llvm::Value* Executor::codegen(const Analyzer::LikeExpr* expr, const CompilationOptions& co) {
  if (is_unnest(extract_cast_arg(expr->get_arg()))) {
    throw std::runtime_error("LIKE not supported for unnested expressions");
  }
  char escape_char{'\\'};
  if (expr->get_escape_expr()) {
    auto escape_char_expr = dynamic_cast<const Analyzer::Constant*>(expr->get_escape_expr());
    CHECK(escape_char_expr);
    CHECK(escape_char_expr->get_type_info().is_string());
    CHECK_EQ(size_t(1), escape_char_expr->get_constval().stringval->size());
    escape_char = (*escape_char_expr->get_constval().stringval)[0];
  }
  auto pattern = dynamic_cast<const Analyzer::Constant*>(expr->get_like_expr());
  CHECK(pattern);
  auto fast_dict_like_lv =
      codegenDictLike(expr->get_own_arg(), pattern, expr->get_is_ilike(), expr->get_is_simple(), escape_char, co);
  if (fast_dict_like_lv) {
    return fast_dict_like_lv;
  }
  auto str_lv = codegen(expr->get_arg(), true, co);
  if (str_lv.size() != 3) {
    CHECK_EQ(size_t(1), str_lv.size());
    str_lv.push_back(cgen_state_->emitCall("extract_str_ptr", {str_lv.front()}));
    str_lv.push_back(cgen_state_->emitCall("extract_str_len", {str_lv.front()}));
    cgen_state_->must_run_on_cpu_ = true;
  }
  auto like_expr_arg_lvs = codegen(expr->get_like_expr(), true, co);
  CHECK_EQ(size_t(3), like_expr_arg_lvs.size());
  const bool is_nullable{!expr->get_arg()->get_type_info().get_notnull()};
  std::vector<llvm::Value*> str_like_args{str_lv[1], str_lv[2], like_expr_arg_lvs[1], like_expr_arg_lvs[2]};
  std::string fn_name{expr->get_is_ilike() ? "string_ilike" : "string_like"};
  if (expr->get_is_simple()) {
    fn_name += "_simple";
  } else {
    str_like_args.push_back(ll_int(int8_t(escape_char)));
  }
  if (is_nullable) {
    fn_name += "_nullable";
    str_like_args.push_back(inlineIntNull(expr->get_type_info()));
  }
  return cgen_state_->emitCall(fn_name, str_like_args);
}

llvm::Value* Executor::codegenDictLike(const std::shared_ptr<Analyzer::Expr> like_arg,
                                       const Analyzer::Constant* pattern,
                                       const bool ilike,
                                       const bool is_simple,
                                       const char escape_char,
                                       const CompilationOptions& co) {
  const auto cast_oper = std::dynamic_pointer_cast<Analyzer::UOper>(like_arg);
  if (!cast_oper) {
    return nullptr;
  }
  CHECK(cast_oper);
  CHECK_EQ(kCAST, cast_oper->get_optype());
  const auto dict_like_arg = cast_oper->get_own_operand();
  const auto& dict_like_arg_ti = dict_like_arg->get_type_info();
  CHECK(dict_like_arg_ti.is_string());
  CHECK_EQ(kENCODING_DICT, dict_like_arg_ti.get_compression());
  const auto sd = getStringDictionary(dict_like_arg_ti.get_comp_param(), row_set_mem_owner_);
  if (sd->size() > 10000000) {
    return nullptr;
  }
  const auto& pattern_ti = pattern->get_type_info();
  CHECK(pattern_ti.is_string());
  CHECK_EQ(kENCODING_NONE, pattern_ti.get_compression());
  const auto& pattern_datum = pattern->get_constval();
  const auto& pattern_str = *pattern_datum.stringval;
  const auto matching_strings = sd->getLike(pattern_str, ilike, is_simple, escape_char);
  std::list<std::shared_ptr<Analyzer::Expr>> matching_str_exprs;
  for (const auto& matching_string : matching_strings) {
    auto const_val = Parser::StringLiteral::analyzeValue(matching_string);
    matching_str_exprs.push_back(const_val->add_cast(dict_like_arg_ti));
  }
  const auto in_values = makeExpr<Analyzer::InValues>(dict_like_arg, matching_str_exprs);
  return codegen(in_values.get(), co);
}

llvm::Value* Executor::codegen(const Analyzer::InValues* expr, const CompilationOptions& co) {
  const auto in_arg = expr->get_arg();
  if (is_unnest(in_arg)) {
    throw std::runtime_error("IN not supported for unnested expressions");
  }
  const auto lhs_lvs = codegen(in_arg, true, co);
  if (co.hoist_literals_) {  // TODO(alex): remove this constraint
    const auto in_vals_bitmap = createInValuesBitmap(expr, co);
    if (in_vals_bitmap) {
      cgen_state_->addInValuesBitmap(in_vals_bitmap);
      CHECK_EQ(size_t(1), lhs_lvs.size());
      return in_vals_bitmap->codegen(lhs_lvs.front(), this);
    }
  }
  const auto& arg_ti = in_arg->get_type_info();
  llvm::Value* result = llvm::ConstantInt::get(llvm::IntegerType::getInt1Ty(cgen_state_->context_), false);
  if (arg_ti.get_notnull()) {
    for (auto in_val : expr->get_value_list()) {
      result = cgen_state_->ir_builder_.CreateOr(
          result, toBool(codegenCmp(kEQ, kONE, lhs_lvs, in_arg->get_type_info(), in_val.get(), co)));
    }
  } else {
    result = castToTypeIn(result, 8);
    const auto& expr_ti = expr->get_type_info();
    CHECK(expr_ti.is_boolean());
    for (auto in_val : expr->get_value_list()) {
      const auto crt = codegenCmp(kEQ, kONE, lhs_lvs, in_arg->get_type_info(), in_val.get(), co);
      result = cgen_state_->emitCall("logical_or", {result, crt, inlineIntNull(expr_ti)});
    }
  }
  return result;
}

InValuesBitmap* Executor::createInValuesBitmap(const Analyzer::InValues* in_values, const CompilationOptions& co) {
  const auto& value_list = in_values->get_value_list();
  std::vector<int64_t> values;
  const auto& ti = in_values->get_arg()->get_type_info();
  if (!(ti.is_integer() || (ti.is_string() && ti.get_compression() == kENCODING_DICT))) {
    return nullptr;
  }
  const auto sd = ti.is_string() ? getStringDictionary(ti.get_comp_param(), row_set_mem_owner_) : nullptr;
  if (value_list.size() > 3) {
    for (const auto in_val : value_list) {
      const auto in_val_const = dynamic_cast<const Analyzer::Constant*>(extract_cast_arg(in_val.get()));
      if (!in_val_const) {
        return nullptr;
      }
      const auto& in_val_ti = in_val->get_type_info();
      CHECK(in_val_ti == ti);
      if (ti.is_string()) {
        CHECK(sd);
        const auto string_id = sd->get(*in_val_const->get_constval().stringval);
        if (string_id >= 0) {
          values.push_back(string_id);
        }
      } else {
        values.push_back(codegenIntConst(in_val_const)->getSExtValue());
      }
    }
    try {
      return new InValuesBitmap(
          values,
          inline_int_null_val(ti),
          co.device_type_ == ExecutorDeviceType::GPU ? Data_Namespace::GPU_LEVEL : Data_Namespace::CPU_LEVEL,
          deviceCount(co.device_type_),
          &catalog_->get_dataMgr());
    } catch (...) {
      return nullptr;
    }
  }
  return nullptr;
}

llvm::Value* Executor::codegen(const Analyzer::BinOper* bin_oper, const CompilationOptions& co) {
  const auto optype = bin_oper->get_optype();
  if (IS_ARITHMETIC(optype)) {
    return codegenArith(bin_oper, co);
  }
  if (IS_COMPARISON(optype)) {
    return codegenCmp(bin_oper, co);
  }
  if (IS_LOGIC(optype)) {
    return codegenLogical(bin_oper, co);
  }
  if (optype == kARRAY_AT) {
    return codegenArrayAt(bin_oper, co);
  }
  CHECK(false);
}

llvm::Value* Executor::codegen(const Analyzer::UOper* u_oper, const CompilationOptions& co) {
  const auto optype = u_oper->get_optype();
  switch (optype) {
    case kNOT:
      return codegenLogical(u_oper, co);
    case kCAST:
      return codegenCast(u_oper, co);
    case kUMINUS:
      return codegenUMinus(u_oper, co);
    case kISNULL:
      return codegenIsNull(u_oper, co);
    case kUNNEST:
      return codegenUnnest(u_oper, co);
    default:
      CHECK(false);
  }
}

namespace {

std::shared_ptr<Decoder> get_col_decoder(const Analyzer::ColumnVar* col_var) {
  const auto enc_type = col_var->get_compression();
  const auto& ti = col_var->get_type_info();
  switch (enc_type) {
    case kENCODING_NONE: {
      const auto int_type = ti.is_decimal() ? decimal_to_int_type(ti) : ti.get_type();
      switch (int_type) {
        case kBOOLEAN:
          return std::make_shared<FixedWidthInt>(1);
        case kSMALLINT:
          return std::make_shared<FixedWidthInt>(2);
        case kINT:
          return std::make_shared<FixedWidthInt>(4);
        case kBIGINT:
          return std::make_shared<FixedWidthInt>(8);
        case kFLOAT:
          return std::make_shared<FixedWidthReal>(false);
        case kDOUBLE:
          return std::make_shared<FixedWidthReal>(true);
        case kTIME:
        case kTIMESTAMP:
        case kDATE:
          return std::make_shared<FixedWidthInt>(sizeof(time_t));
        default:
          // TODO(alex): make columnar results write the correct encoding
          if (ti.is_string()) {
            return std::make_shared<FixedWidthInt>(4);
          }
          CHECK(false);
      }
    }
    case kENCODING_DICT:
      CHECK(ti.is_string());
      return std::make_shared<FixedWidthInt>(4);
    case kENCODING_FIXED: {
      const auto bit_width = col_var->get_comp_param();
      CHECK_EQ(0, bit_width % 8);
      return std::make_shared<FixedWidthInt>(bit_width / 8);
    }
    default:
      CHECK(false);
  }
}

size_t get_col_bit_width(const Analyzer::ColumnVar* col_var) {
  const auto& type_info = col_var->get_type_info();
  return get_bit_width(type_info);
}

}  // namespace

std::vector<llvm::Value*> Executor::codegen(const Analyzer::ColumnVar* col_var,
                                            const bool fetch_column,
                                            const bool hoist_literals) {
  const auto col_var_lvs = codegenColVar(col_var, fetch_column, hoist_literals);
  if (!cgen_state_->outer_join_cond_lv_ || col_var->get_rte_idx() == 0) {
    return col_var_lvs;
  }
  return codegenOuterJoinNullPlaceholder(col_var_lvs, col_var);
}

std::vector<llvm::Value*> Executor::codegenColVar(const Analyzer::ColumnVar* col_var,
                                                  const bool fetch_column,
                                                  const bool hoist_literals) {
  auto col_id = col_var->get_column_id();
  if (col_var->get_table_id() > 0) {
    auto cd = get_column_descriptor(col_id, col_var->get_table_id(), *catalog_);
    if (cd->isVirtualCol) {
      CHECK(cd->columnName == "rowid");
      if (col_var->get_rte_idx() > 0) {
        // rowid for inner scan, the fragment offset from the outer scan
        // is meaningless, the relative position in the scan is the rowid
        return {posArg(col_var)};
      }
      return {cgen_state_->ir_builder_.CreateAdd(posArg(col_var), fragRowOff())};
    }
  }
  if (col_var->get_rte_idx() >= 0 && !is_nested_) {
    CHECK(col_id > 0 || temporary_tables_);
  } else {
    CHECK((col_id == 0) || (col_var->get_rte_idx() >= 0 && col_var->get_table_id() > 0));
    const auto var = dynamic_cast<const Analyzer::Var*>(col_var);
    CHECK(var);
    col_id = var->get_varno();
    CHECK_GE(col_id, 1);
    if (var->get_which_row() == Analyzer::Var::kGROUPBY) {
      CHECK_LE(static_cast<size_t>(col_id), cgen_state_->group_by_expr_cache_.size());
      return {cgen_state_->group_by_expr_cache_[col_id - 1]};
    }
  }
  const int local_col_id = getLocalColumnId(col_var, fetch_column);
  // only generate the decoding code once; if a column has been previously
  // fetched in the generated IR, we'll reuse it
  auto it = cgen_state_->fetch_cache_.find(local_col_id);
  if (it != cgen_state_->fetch_cache_.end()) {
    return {it->second};
  }
  const auto hash_join_lhs = hashJoinLhs(col_var);
  if (hash_join_lhs && hash_join_lhs->get_rte_idx() == 0) {
    return codegen(hash_join_lhs, fetch_column, hoist_literals);
  }
  auto pos_arg = posArg(col_var);
  auto col_byte_stream = colByteStream(col_var, fetch_column, hoist_literals);
  if (plan_state_->isLazyFetchColumn(col_var)) {
    plan_state_->columns_to_not_fetch_.insert(col_id);
    return {pos_arg};
  }
  if (col_var->get_type_info().is_string() && col_var->get_type_info().get_compression() == kENCODING_NONE) {
    // real (not dictionary-encoded) strings; store the pointer to the payload
    auto ptr_and_len = cgen_state_->emitExternalCall(
        "string_decode", get_int_type(64, cgen_state_->context_), {col_byte_stream, pos_arg});
    // Unpack the pointer + length, see string_decode function.
    auto str_lv = cgen_state_->emitCall("extract_str_ptr", {ptr_and_len});
    auto len_lv = cgen_state_->emitCall("extract_str_len", {ptr_and_len});
    auto it_ok = cgen_state_->fetch_cache_.insert(
        std::make_pair(local_col_id, std::vector<llvm::Value*>{ptr_and_len, str_lv, len_lv}));
    CHECK(it_ok.second);
    return {ptr_and_len, str_lv, len_lv};
  }
  if (col_var->get_type_info().is_array()) {
    return {col_byte_stream};
  }
  const auto decoder = get_col_decoder(col_var);
  auto dec_val = decoder->codegenDecode(col_byte_stream, pos_arg, cgen_state_->module_);
  cgen_state_->ir_builder_.Insert(dec_val);
  auto dec_type = dec_val->getType();
  llvm::Value* dec_val_cast{nullptr};
  if (dec_type->isIntegerTy()) {
    auto dec_width = static_cast<llvm::IntegerType*>(dec_type)->getBitWidth();
    auto col_width = get_col_bit_width(col_var);
    dec_val_cast = cgen_state_->ir_builder_.CreateCast(static_cast<size_t>(col_width) > dec_width
                                                           ? llvm::Instruction::CastOps::SExt
                                                           : llvm::Instruction::CastOps::Trunc,
                                                       dec_val,
                                                       get_int_type(col_width, cgen_state_->context_));
  } else {
    CHECK(dec_type->isFloatTy() || dec_type->isDoubleTy());
    if (dec_type->isDoubleTy()) {
      CHECK(col_var->get_type_info().get_type() == kDOUBLE);
    } else if (dec_type->isFloatTy()) {
      CHECK(col_var->get_type_info().get_type() == kFLOAT);
    }
    dec_val_cast = dec_val;
  }
  CHECK(dec_val_cast);
  auto it_ok = cgen_state_->fetch_cache_.insert(std::make_pair(local_col_id, std::vector<llvm::Value*>{dec_val_cast}));
  CHECK(it_ok.second);
  return {it_ok.first->second};
}

std::vector<llvm::Value*> Executor::codegenOuterJoinNullPlaceholder(const std::vector<llvm::Value*>& orig_lvs,
                                                                    const Analyzer::Expr* orig_expr) {
  const auto bb = cgen_state_->ir_builder_.GetInsertBlock();
  const auto outer_join_args_bb =
      llvm::BasicBlock::Create(cgen_state_->context_, "outer_join_args", cgen_state_->row_func_);
  const auto outer_join_nulls_bb =
      llvm::BasicBlock::Create(cgen_state_->context_, "outer_join_nulls", cgen_state_->row_func_);
  const auto phi_bb = llvm::BasicBlock::Create(cgen_state_->context_, "outer_join_phi", cgen_state_->row_func_);
  cgen_state_->ir_builder_.SetInsertPoint(bb);
  cgen_state_->ir_builder_.CreateCondBr(cgen_state_->outer_join_cond_lv_, outer_join_args_bb, outer_join_nulls_bb);
  const auto back_from_outer_join_bb =
      llvm::BasicBlock::Create(cgen_state_->context_, "back_from_outer_join", cgen_state_->row_func_);
  cgen_state_->ir_builder_.SetInsertPoint(outer_join_args_bb);
  cgen_state_->ir_builder_.CreateBr(phi_bb);
  cgen_state_->ir_builder_.SetInsertPoint(outer_join_nulls_bb);
  const auto& null_ti = orig_expr->get_type_info();
  const auto null_constant = makeExpr<Analyzer::Constant>(null_ti, true, Datum{0});
  const auto null_target_lvs = codegen(
      null_constant.get(), false, CompilationOptions{ExecutorDeviceType::CPU, false, ExecutorOptLevel::Default});
  cgen_state_->ir_builder_.CreateBr(phi_bb);
  CHECK_EQ(orig_lvs.size(), null_target_lvs.size());
  cgen_state_->ir_builder_.SetInsertPoint(phi_bb);
  std::vector<llvm::Value*> target_lvs;
  for (size_t i = 0; i < orig_lvs.size(); ++i) {
    const auto target_type = orig_lvs[i]->getType();
    CHECK_EQ(target_type, null_target_lvs[i]->getType());
    auto target_phi = cgen_state_->ir_builder_.CreatePHI(target_type, 2);
    target_phi->addIncoming(orig_lvs[i], outer_join_args_bb);
    target_phi->addIncoming(null_target_lvs[i], outer_join_nulls_bb);
    target_lvs.push_back(target_phi);
  }
  cgen_state_->ir_builder_.CreateBr(back_from_outer_join_bb);
  cgen_state_->ir_builder_.SetInsertPoint(back_from_outer_join_bb);
  return target_lvs;
}

// returns the byte stream argument and the position for the given column
llvm::Value* Executor::colByteStream(const Analyzer::ColumnVar* col_var,
                                     const bool fetch_column,
                                     const bool hoist_literals) {
  auto& in_arg_list = cgen_state_->row_func_->getArgumentList();
  CHECK_GE(in_arg_list.size(), size_t(3));
  size_t arg_idx = 0;
  size_t pos_idx = 0;
  llvm::Value* pos_arg{nullptr};
  const int local_col_id = getLocalColumnId(col_var, fetch_column);
  for (auto& arg : in_arg_list) {
    if (arg.getType()->isIntegerTy() && !pos_arg) {
      pos_arg = &arg;
      pos_idx = arg_idx;
    } else if (pos_arg && arg_idx == pos_idx + 3 + static_cast<size_t>(local_col_id) + (hoist_literals ? 1 : 0)) {
      return &arg;
    }
    ++arg_idx;
  }
  CHECK(false);
}

llvm::Value* Executor::posArg(const Analyzer::Expr* expr) const {
  if (dynamic_cast<const Analyzer::ColumnVar*>(expr)) {
    const auto col_var = static_cast<const Analyzer::ColumnVar*>(expr);
    const auto hash_pos_it = cgen_state_->scan_idx_to_hash_pos_.find(col_var->get_rte_idx());
    if (hash_pos_it != cgen_state_->scan_idx_to_hash_pos_.end()) {
      return hash_pos_it->second;
    }
    const auto inner_it =
        cgen_state_->scan_to_iterator_.find(InputDescriptor(col_var->get_table_id(), col_var->get_rte_idx()));
    if (inner_it != cgen_state_->scan_to_iterator_.end()) {
      CHECK(inner_it->second.first);
      CHECK(inner_it->second.first->getType()->isIntegerTy(64));
      return inner_it->second.first;
    }
  }
  auto& in_arg_list = cgen_state_->row_func_->getArgumentList();
  for (auto& arg : in_arg_list) {
    if (arg.getType()->isIntegerTy()) {
      CHECK(arg.getType()->isIntegerTy(64));
      return &arg;
    }
  }
  CHECK(false);
}

const Analyzer::ColumnVar* Executor::hashJoinLhs(const Analyzer::ColumnVar* rhs) const {
  for (const auto tautological_eq : plan_state_->join_info_.equi_join_tautologies_) {
    CHECK(tautological_eq->get_optype() == kEQ);
    if (*tautological_eq->get_right_operand() == *rhs) {
      auto lhs_col = dynamic_cast<const Analyzer::ColumnVar*>(tautological_eq->get_left_operand());
      CHECK(lhs_col);
      return lhs_col;
    }
  }
  return nullptr;
}

llvm::Value* Executor::fragRowOff() const {
  for (auto arg_it = cgen_state_->row_func_->arg_begin(); arg_it != cgen_state_->row_func_->arg_end(); ++arg_it) {
    if (arg_it->getType()->isIntegerTy()) {
      ++arg_it;
      return arg_it;
    }
  }
  CHECK(false);
}

llvm::Value* Executor::rowsPerScan() const {
  for (auto arg_it = cgen_state_->row_func_->arg_begin(); arg_it != cgen_state_->row_func_->arg_end(); ++arg_it) {
    if (arg_it->getType()->isIntegerTy()) {
      ++arg_it;
      ++arg_it;
      return arg_it;
    }
  }
  CHECK(false);
}

namespace {

llvm::Value* getLiteralBuffArg(llvm::Function* row_func) {
  auto arg_it = row_func->arg_begin();
  while (arg_it != row_func->arg_end()) {
    if (arg_it->getType()->isIntegerTy()) {
      ++arg_it;
      ++arg_it;
      ++arg_it;
      break;
    }
    ++arg_it;
  }
  CHECK(arg_it != row_func->arg_end());
  return arg_it;
}

}  // namespace

std::vector<llvm::Value*> Executor::codegen(const Analyzer::Constant* constant,
                                            const EncodingType enc_type,
                                            const int dict_id,
                                            const CompilationOptions& co) {
  if (co.hoist_literals_) {
    std::vector<const Analyzer::Constant*> constants(deviceCount(co.device_type_), constant);
    return codegenHoistedConstants(constants, enc_type, dict_id);
  }
  const auto& type_info = constant->get_type_info();
  const auto type = type_info.is_decimal() ? decimal_to_int_type(type_info) : type_info.get_type();
  switch (type) {
    case kBOOLEAN:
      return {llvm::ConstantInt::get(get_int_type(8, cgen_state_->context_), constant->get_constval().boolval)};
    case kSMALLINT:
    case kINT:
    case kBIGINT:
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
      return {codegenIntConst(constant)};
    case kFLOAT:
      return {llvm::ConstantFP::get(llvm::Type::getFloatTy(cgen_state_->context_), constant->get_constval().floatval)};
    case kDOUBLE:
      return {
          llvm::ConstantFP::get(llvm::Type::getDoubleTy(cgen_state_->context_), constant->get_constval().doubleval)};
    case kVARCHAR:
    case kCHAR:
    case kTEXT: {
      CHECK(constant->get_constval().stringval || constant->get_is_null());
      if (constant->get_is_null()) {
        return {ll_int(int64_t(0)),
                llvm::Constant::getNullValue(llvm::PointerType::get(get_int_type(8, cgen_state_->context_), 0)),
                ll_int(int32_t(0))};
      }
      const auto& str_const = *constant->get_constval().stringval;
      if (enc_type == kENCODING_DICT) {
        return {ll_int(getStringDictionary(dict_id, row_set_mem_owner_)->get(str_const))};
      }
      return {ll_int(int64_t(0)),
              cgen_state_->addStringConstant(str_const),
              ll_int(static_cast<int32_t>(str_const.size()))};
    }
    default:
      CHECK(false);
  }
  CHECK(false);
}

std::vector<llvm::Value*> Executor::codegenHoistedConstants(const std::vector<const Analyzer::Constant*>& constants,
                                                            const EncodingType enc_type,
                                                            const int dict_id) {
  CHECK(!constants.empty());
  const auto& type_info = constants.front()->get_type_info();
  auto lit_buff_lv = getLiteralBuffArg(cgen_state_->row_func_);
  int16_t lit_off{-1};
  for (size_t device_id = 0; device_id < constants.size(); ++device_id) {
    const auto constant = constants[device_id];
    const auto& crt_type_info = constant->get_type_info();
    CHECK(type_info == crt_type_info);
    const int16_t dev_lit_off = cgen_state_->getOrAddLiteral(constant, enc_type, dict_id, device_id);
    if (device_id) {
      CHECK_EQ(lit_off, dev_lit_off);
    } else {
      lit_off = dev_lit_off;
    }
  }
  CHECK_GE(lit_off, int16_t(0));
  const auto lit_buf_start = cgen_state_->ir_builder_.CreateGEP(lit_buff_lv, ll_int(lit_off));
  if (type_info.is_string() && enc_type != kENCODING_DICT) {
    CHECK_EQ(kENCODING_NONE, type_info.get_compression());
    CHECK_EQ(size_t(4), literalBytes(LiteralValue(std::string(""))));
    auto off_and_len_ptr = cgen_state_->ir_builder_.CreateBitCast(
        lit_buf_start, llvm::PointerType::get(get_int_type(32, cgen_state_->context_), 0));
    // packed offset + length, 16 bits each
    auto off_and_len = cgen_state_->ir_builder_.CreateLoad(off_and_len_ptr);
    auto off_lv = cgen_state_->ir_builder_.CreateLShr(
        cgen_state_->ir_builder_.CreateAnd(off_and_len, ll_int(int32_t(0xffff0000))), ll_int(int32_t(16)));
    auto len_lv = cgen_state_->ir_builder_.CreateAnd(off_and_len, ll_int(int32_t(0x0000ffff)));
    return {ll_int(int64_t(0)), cgen_state_->ir_builder_.CreateGEP(lit_buff_lv, off_lv), len_lv};
  }
  llvm::Type* val_ptr_type{nullptr};
  const auto val_bits = get_bit_width(type_info);
  CHECK_EQ(size_t(0), val_bits % 8);
  if (type_info.is_integer() || type_info.is_decimal() || type_info.is_time() || type_info.is_string() ||
      type_info.is_boolean()) {
    val_ptr_type = llvm::PointerType::get(llvm::IntegerType::get(cgen_state_->context_, val_bits), 0);
  } else {
    CHECK(type_info.get_type() == kFLOAT || type_info.get_type() == kDOUBLE);
    val_ptr_type = (type_info.get_type() == kFLOAT) ? llvm::Type::getFloatPtrTy(cgen_state_->context_)
                                                    : llvm::Type::getDoublePtrTy(cgen_state_->context_);
  }
  auto lit_lv =
      cgen_state_->ir_builder_.CreateLoad(cgen_state_->ir_builder_.CreateBitCast(lit_buf_start, val_ptr_type));
  return {lit_lv};
}

int Executor::deviceCount(const ExecutorDeviceType device_type) const {
  return device_type == ExecutorDeviceType::GPU ? catalog_->get_dataMgr().cudaMgr_->getDeviceCount() : 1;
}

std::vector<llvm::Value*> Executor::codegen(const Analyzer::CaseExpr* case_expr, const CompilationOptions& co) {
  const auto case_ti = case_expr->get_type_info();
  llvm::Type* case_llvm_type = nullptr;
  bool is_real_str = false;
  if (case_ti.is_integer() || case_ti.is_time() || case_ti.is_decimal()) {
    case_llvm_type = get_int_type(get_bit_width(case_ti), cgen_state_->context_);
  } else if (case_ti.is_fp()) {
    case_llvm_type = case_ti.get_type() == kFLOAT ? llvm::Type::getFloatTy(cgen_state_->context_)
                                                  : llvm::Type::getDoubleTy(cgen_state_->context_);
  } else if (case_ti.is_string()) {
    if (case_ti.get_compression() == kENCODING_DICT) {
      case_llvm_type = get_int_type(8 * case_ti.get_size(), cgen_state_->context_);
    } else {
      is_real_str = true;
      case_llvm_type = get_int_type(64, cgen_state_->context_);
    }
  }
  CHECK(case_llvm_type);
  CHECK(case_expr->get_else_expr()->get_type_info() == case_ti);
  llvm::Value* case_val = codegenCase(case_expr, case_llvm_type, is_real_str, co);
  std::vector<llvm::Value*> ret_vals{case_val};
  if (is_real_str) {
    ret_vals.push_back(cgen_state_->emitCall("extract_str_ptr", {case_val}));
    ret_vals.push_back(cgen_state_->emitCall("extract_str_len", {case_val}));
  }
  return ret_vals;
}

llvm::Value* Executor::codegenCase(const Analyzer::CaseExpr* case_expr,
                                   llvm::Type* case_llvm_type,
                                   const bool is_real_str,
                                   const CompilationOptions& co) {
  // Here the linear control flow will diverge and expressions cached during the
  // code branch code generation (currently just column decoding) are not going
  // to be available once we're done generating the case. Take a snapshot of
  // the cache with FetchCacheAnchor and restore it once we're done with CASE.
  FetchCacheAnchor anchor(cgen_state_.get());
  const auto& expr_pair_list = case_expr->get_expr_pair_list();
  std::vector<llvm::Value*> then_lvs;
  std::vector<llvm::BasicBlock*> then_bbs;
  const auto end_bb = llvm::BasicBlock::Create(cgen_state_->context_, "end_case", cgen_state_->row_func_);
  for (const auto& expr_pair : expr_pair_list) {
    FetchCacheAnchor branch_anchor(cgen_state_.get());
    const auto when_lv = toBool(codegen(expr_pair.first.get(), true, co).front());
    const auto cmp_bb = cgen_state_->ir_builder_.GetInsertBlock();
    const auto then_bb = llvm::BasicBlock::Create(cgen_state_->context_, "then_case", cgen_state_->row_func_);
    cgen_state_->ir_builder_.SetInsertPoint(then_bb);
    auto then_bb_lvs = codegen(expr_pair.second.get(), true, co);
    if (is_real_str) {
      if (then_bb_lvs.size() == 3) {
        then_lvs.push_back(cgen_state_->emitCall("string_pack", {then_bb_lvs[1], then_bb_lvs[2]}));
      } else {
        then_lvs.push_back(then_bb_lvs.front());
      }
    } else {
      CHECK_EQ(size_t(1), then_bb_lvs.size());
      then_lvs.push_back(then_bb_lvs.front());
    }
    then_bbs.push_back(cgen_state_->ir_builder_.GetInsertBlock());
    cgen_state_->ir_builder_.CreateBr(end_bb);
    const auto when_bb = llvm::BasicBlock::Create(cgen_state_->context_, "when_case", cgen_state_->row_func_);
    cgen_state_->ir_builder_.SetInsertPoint(cmp_bb);
    cgen_state_->ir_builder_.CreateCondBr(when_lv, then_bb, when_bb);
    cgen_state_->ir_builder_.SetInsertPoint(when_bb);
  }
  const auto else_expr = case_expr->get_else_expr();
  CHECK(else_expr);
  auto else_lvs = codegen(else_expr, true, co);
  llvm::Value* else_lv{nullptr};
  if (is_real_str && dynamic_cast<const Analyzer::Constant*>(else_expr)) {
    CHECK_EQ(size_t(3), else_lvs.size());
    else_lv = cgen_state_->emitCall("string_pack", {else_lvs[1], else_lvs[2]});
  } else {
    else_lv = else_lvs.front();
  }
  CHECK(else_lv);
  auto else_bb = cgen_state_->ir_builder_.GetInsertBlock();
  cgen_state_->ir_builder_.CreateBr(end_bb);
  cgen_state_->ir_builder_.SetInsertPoint(end_bb);
  auto then_phi = cgen_state_->ir_builder_.CreatePHI(case_llvm_type, expr_pair_list.size() + 1);
  CHECK_EQ(then_bbs.size(), then_lvs.size());
  for (size_t i = 0; i < then_bbs.size(); ++i) {
    then_phi->addIncoming(then_lvs[i], then_bbs[i]);
  }
  then_phi->addIncoming(else_lv, else_bb);
  return then_phi;
}

llvm::Value* Executor::codegen(const Analyzer::ExtractExpr* extract_expr, const CompilationOptions& co) {
  auto from_expr = codegen(extract_expr->get_from_expr(), true, co).front();
  const int32_t extract_field{extract_expr->get_field()};
  const auto& extract_expr_ti = extract_expr->get_from_expr()->get_type_info();
  if (extract_field == kEPOCH) {
    CHECK(extract_expr_ti.get_type() == kTIMESTAMP || extract_expr_ti.get_type() == kDATE);
    if (from_expr->getType()->isIntegerTy(32)) {
      from_expr = cgen_state_->ir_builder_.CreateCast(
          llvm::Instruction::CastOps::SExt, from_expr, get_int_type(64, cgen_state_->context_));
    }
    return from_expr;
  }
  CHECK(from_expr->getType()->isIntegerTy(32) || from_expr->getType()->isIntegerTy(64));
  static_assert(sizeof(time_t) == 4 || sizeof(time_t) == 8, "Unsupported time_t size");
  if (sizeof(time_t) == 4 && from_expr->getType()->isIntegerTy(64)) {
    from_expr = cgen_state_->ir_builder_.CreateCast(
        llvm::Instruction::CastOps::Trunc, from_expr, get_int_type(32, cgen_state_->context_));
  }
  std::vector<llvm::Value*> extract_args{ll_int(static_cast<int32_t>(extract_expr->get_field())), from_expr};
  std::string extract_fname{"ExtractFromTime"};
  if (!extract_expr_ti.get_notnull()) {
    extract_args.push_back(inlineIntNull(extract_expr_ti));
    extract_fname += "Nullable";
  }
  return cgen_state_->emitExternalCall(extract_fname, get_int_type(64, cgen_state_->context_), extract_args);
}

llvm::Value* Executor::codegen(const Analyzer::DatetruncExpr* datetrunc_expr, const CompilationOptions& co) {
  auto from_expr = codegen(datetrunc_expr->get_from_expr(), true, co).front();
  const auto& datetrunc_expr_ti = datetrunc_expr->get_from_expr()->get_type_info();
  CHECK(from_expr->getType()->isIntegerTy(32) || from_expr->getType()->isIntegerTy(64));
  static_assert(sizeof(time_t) == 4 || sizeof(time_t) == 8, "Unsupported time_t size");
  if (sizeof(time_t) == 4 && from_expr->getType()->isIntegerTy(64)) {
    from_expr = cgen_state_->ir_builder_.CreateCast(
        llvm::Instruction::CastOps::Trunc, from_expr, get_int_type(32, cgen_state_->context_));
  }
  std::vector<llvm::Value*> datetrunc_args{ll_int(static_cast<int32_t>(datetrunc_expr->get_field())), from_expr};
  std::string datetrunc_fname{"DateTruncate"};
  if (!datetrunc_expr_ti.get_notnull()) {
    datetrunc_args.push_back(inlineIntNull(datetrunc_expr_ti));
    datetrunc_fname += "Nullable";
  }
  return cgen_state_->emitExternalCall(datetrunc_fname, get_int_type(64, cgen_state_->context_), datetrunc_args);
}

namespace {

llvm::CmpInst::Predicate llvm_icmp_pred(const SQLOps op_type) {
  switch (op_type) {
    case kEQ:
      return llvm::ICmpInst::ICMP_EQ;
    case kNE:
      return llvm::ICmpInst::ICMP_NE;
    case kLT:
      return llvm::ICmpInst::ICMP_SLT;
    case kGT:
      return llvm::ICmpInst::ICMP_SGT;
    case kLE:
      return llvm::ICmpInst::ICMP_SLE;
    case kGE:
      return llvm::ICmpInst::ICMP_SGE;
    default:
      CHECK(false);
  }
}

std::string icmp_name(const SQLOps op_type) {
  switch (op_type) {
    case kEQ:
      return "eq";
    case kNE:
      return "ne";
    case kLT:
      return "lt";
    case kGT:
      return "gt";
    case kLE:
      return "le";
    case kGE:
      return "ge";
    default:
      CHECK(false);
  }
}

std::string icmp_arr_name(const SQLOps op_type) {
  switch (op_type) {
    case kEQ:
      return "eq";
    case kNE:
      return "ne";
    case kLT:
      return "gt";
    case kGT:
      return "lt";
    case kLE:
      return "ge";
    case kGE:
      return "le";
    default:
      CHECK(false);
  }
}

llvm::CmpInst::Predicate llvm_fcmp_pred(const SQLOps op_type) {
  switch (op_type) {
    case kEQ:
      return llvm::CmpInst::FCMP_OEQ;
    case kNE:
      return llvm::CmpInst::FCMP_ONE;
    case kLT:
      return llvm::CmpInst::FCMP_OLT;
    case kGT:
      return llvm::CmpInst::FCMP_OGT;
    case kLE:
      return llvm::CmpInst::FCMP_OLE;
    case kGE:
      return llvm::CmpInst::FCMP_OGE;
    default:
      CHECK(false);
  }
}

}  // namespace

namespace {

std::string string_cmp_func(const SQLOps optype) {
  switch (optype) {
    case kLT:
      return "string_lt";
    case kLE:
      return "string_le";
    case kGT:
      return "string_gt";
    case kGE:
      return "string_ge";
    case kEQ:
      return "string_eq";
    case kNE:
      return "string_ne";
    default:
      CHECK(false);
  }
}

std::string get_null_check_suffix(const SQLTypeInfo& lhs_ti, const SQLTypeInfo& rhs_ti) {
  if (lhs_ti.get_notnull() && rhs_ti.get_notnull()) {
    return "";
  }
  std::string null_check_suffix{"_nullable"};
  if (lhs_ti.get_notnull()) {
    CHECK(!rhs_ti.get_notnull());
    null_check_suffix += "_rhs";
  } else if (rhs_ti.get_notnull()) {
    CHECK(!lhs_ti.get_notnull());
    null_check_suffix += "_lhs";
  }
  return null_check_suffix;
}

}  // namespace

llvm::Value* Executor::codegenCmp(const Analyzer::BinOper* bin_oper, const CompilationOptions& co) {
  for (const auto equi_join_tautology : plan_state_->join_info_.equi_join_tautologies_) {
    if (*equi_join_tautology == *bin_oper) {
      return plan_state_->join_info_.join_hash_table_->codegenSlot(co.hoist_literals_);
    }
  }
  const auto optype = bin_oper->get_optype();
  const auto qualifier = bin_oper->get_qualifier();
  const auto lhs = bin_oper->get_left_operand();
  const auto rhs = bin_oper->get_right_operand();
  if (is_unnest(lhs) || is_unnest(rhs)) {
    throw std::runtime_error("Unnest not supported in comparisons");
  }
  const auto lhs_lvs = codegen(lhs, true, co);
  return codegenCmp(optype, qualifier, lhs_lvs, lhs->get_type_info(), rhs, co);
}

llvm::Value* Executor::codegenCmp(const SQLOps optype,
                                  const SQLQualifier qualifier,
                                  std::vector<llvm::Value*> lhs_lvs,
                                  const SQLTypeInfo& lhs_ti,
                                  const Analyzer::Expr* rhs,
                                  const CompilationOptions& co) {
  CHECK(IS_COMPARISON(optype));
  const auto& rhs_ti = rhs->get_type_info();
  if (rhs_ti.is_array()) {
    return codegenQualifierCmp(optype, qualifier, lhs_lvs, rhs, co);
  }
  auto rhs_lvs = codegen(rhs, true, co);
  CHECK_EQ(kONE, qualifier);
  CHECK((lhs_ti.get_type() == rhs_ti.get_type()) || (lhs_ti.is_string() && rhs_ti.is_string()));
  const auto null_check_suffix = get_null_check_suffix(lhs_ti, rhs_ti);
  if (lhs_ti.is_integer() || lhs_ti.is_decimal() || lhs_ti.is_time() || lhs_ti.is_boolean() || lhs_ti.is_string()) {
    if (lhs_ti.is_string()) {
      CHECK(rhs_ti.is_string());
      CHECK_EQ(lhs_ti.get_compression(), rhs_ti.get_compression());
      if (lhs_ti.get_compression() == kENCODING_NONE) {
        // unpack pointer + length if necessary
        if (lhs_lvs.size() != 3) {
          CHECK_EQ(size_t(1), lhs_lvs.size());
          lhs_lvs.push_back(cgen_state_->emitCall("extract_str_ptr", {lhs_lvs.front()}));
          lhs_lvs.push_back(cgen_state_->emitCall("extract_str_len", {lhs_lvs.front()}));
        }
        if (rhs_lvs.size() != 3) {
          CHECK_EQ(size_t(1), rhs_lvs.size());
          rhs_lvs.push_back(cgen_state_->emitCall("extract_str_ptr", {rhs_lvs.front()}));
          rhs_lvs.push_back(cgen_state_->emitCall("extract_str_len", {rhs_lvs.front()}));
        }
        std::vector<llvm::Value*> str_cmp_args{lhs_lvs[1], lhs_lvs[2], rhs_lvs[1], rhs_lvs[2]};
        if (!null_check_suffix.empty()) {
          str_cmp_args.push_back(inlineIntNull(SQLTypeInfo(kBOOLEAN, false)));
        }
        return cgen_state_->emitCall(string_cmp_func(optype) + (null_check_suffix.empty() ? "" : "_nullable"),
                                     str_cmp_args);
      } else {
        CHECK(optype == kEQ || optype == kNE);
      }
    }
    return null_check_suffix.empty()
               ? cgen_state_->ir_builder_.CreateICmp(llvm_icmp_pred(optype), lhs_lvs.front(), rhs_lvs.front())
               : cgen_state_->emitCall(icmp_name(optype) + "_" + numeric_type_name(lhs_ti) + null_check_suffix,
                                       {lhs_lvs.front(),
                                        rhs_lvs.front(),
                                        ll_int(inline_int_null_val(lhs_ti)),
                                        inlineIntNull(SQLTypeInfo(kBOOLEAN, false))});
  }
  if (lhs_ti.get_type() == kFLOAT || lhs_ti.get_type() == kDOUBLE) {
    return null_check_suffix.empty()
               ? cgen_state_->ir_builder_.CreateFCmp(llvm_fcmp_pred(optype), lhs_lvs.front(), rhs_lvs.front())
               : cgen_state_->emitCall(icmp_name(optype) + "_" + numeric_type_name(lhs_ti) + null_check_suffix,
                                       {lhs_lvs.front(),
                                        rhs_lvs.front(),
                                        lhs_ti.get_type() == kFLOAT ? ll_fp(NULL_FLOAT) : ll_fp(NULL_DOUBLE),
                                        inlineIntNull(SQLTypeInfo(kBOOLEAN, false))});
  }
  CHECK(false);
  return nullptr;
}

llvm::Value* Executor::codegenQualifierCmp(const SQLOps optype,
                                           const SQLQualifier qualifier,
                                           std::vector<llvm::Value*> lhs_lvs,
                                           const Analyzer::Expr* rhs,
                                           const CompilationOptions& co) {
  const auto& rhs_ti = rhs->get_type_info();
  const Analyzer::Expr* arr_expr{rhs};
  if (dynamic_cast<const Analyzer::UOper*>(rhs)) {
    const auto cast_arr = static_cast<const Analyzer::UOper*>(rhs);
    CHECK_EQ(kCAST, cast_arr->get_optype());
    arr_expr = cast_arr->get_operand();
  }
  const auto& arr_ti = arr_expr->get_type_info();
  const auto& elem_ti = arr_ti.get_elem_type();
  auto rhs_lvs = codegen(arr_expr, true, co);
  CHECK_NE(kONE, qualifier);
  std::string fname{std::string("array_") + (qualifier == kANY ? "any" : "all") + "_" + icmp_arr_name(optype)};
  const auto& target_ti = rhs_ti.get_elem_type();
  const bool is_real_string{target_ti.is_string() && target_ti.get_compression() != kENCODING_DICT};
  if (is_real_string) {
    cgen_state_->must_run_on_cpu_ = true;
    CHECK_EQ(kENCODING_NONE, target_ti.get_compression());
    fname += "_str";
  }
  if (elem_ti.is_integer() || elem_ti.is_boolean() || elem_ti.is_string()) {
    fname += ("_" + numeric_type_name(elem_ti));
  } else {
    CHECK(elem_ti.is_fp());
    fname += elem_ti.get_type() == kDOUBLE ? "_double" : "_float";
  }
  if (is_real_string) {
    CHECK_EQ(size_t(3), lhs_lvs.size());
    return cgen_state_->emitExternalCall(
        fname,
        get_int_type(1, cgen_state_->context_),
        {rhs_lvs.front(),
         posArg(arr_expr),
         lhs_lvs[1],
         lhs_lvs[2],
         ll_int(int64_t(getStringDictionary(elem_ti.get_comp_param(), row_set_mem_owner_))),
         inlineIntNull(elem_ti)});
  }
  if (target_ti.is_integer() || target_ti.is_boolean() || target_ti.is_string()) {
    fname += ("_" + numeric_type_name(target_ti));
  } else {
    CHECK(target_ti.is_fp());
    fname += target_ti.get_type() == kDOUBLE ? "_double" : "_float";
  }
  return cgen_state_->emitExternalCall(fname,
                                       get_int_type(1, cgen_state_->context_),
                                       {rhs_lvs.front(),
                                        posArg(arr_expr),
                                        lhs_lvs.front(),
                                        elem_ti.is_fp() ? static_cast<llvm::Value*>(inlineFpNull(elem_ti))
                                                        : static_cast<llvm::Value*>(inlineIntNull(elem_ti))});
}

llvm::Value* Executor::codegenLogical(const Analyzer::BinOper* bin_oper, const CompilationOptions& co) {
  const auto optype = bin_oper->get_optype();
  CHECK(IS_LOGIC(optype));
  const auto lhs = bin_oper->get_left_operand();
  const auto rhs = bin_oper->get_right_operand();
  auto lhs_lv = codegen(lhs, true, co).front();
  auto rhs_lv = codegen(rhs, true, co).front();
  const auto& ti = bin_oper->get_type_info();
  if (ti.get_notnull()) {
    switch (optype) {
      case kAND:
        return cgen_state_->ir_builder_.CreateAnd(lhs_lv, rhs_lv);
      case kOR:
        return cgen_state_->ir_builder_.CreateOr(lhs_lv, rhs_lv);
      default:
        CHECK(false);
    }
  }
  CHECK(lhs_lv->getType()->isIntegerTy(1) || lhs_lv->getType()->isIntegerTy(8));
  CHECK(rhs_lv->getType()->isIntegerTy(1) || rhs_lv->getType()->isIntegerTy(8));
  if (lhs_lv->getType()->isIntegerTy(1)) {
    lhs_lv = castToTypeIn(lhs_lv, 8);
  }
  if (rhs_lv->getType()->isIntegerTy(1)) {
    rhs_lv = castToTypeIn(rhs_lv, 8);
  }
  switch (optype) {
    case kAND:
      return cgen_state_->emitCall("logical_and", {lhs_lv, rhs_lv, inlineIntNull(ti)});
    case kOR:
      return cgen_state_->emitCall("logical_or", {lhs_lv, rhs_lv, inlineIntNull(ti)});
    default:
      CHECK(false);
  }
}

llvm::Value* Executor::toBool(llvm::Value* lv) {
  CHECK(lv->getType()->isIntegerTy());
  if (static_cast<llvm::IntegerType*>(lv->getType())->getBitWidth() > 1) {
    return cgen_state_->ir_builder_.CreateICmp(llvm::ICmpInst::ICMP_SGT, lv, llvm::ConstantInt::get(lv->getType(), 0));
  }
  return lv;
}

llvm::Value* Executor::codegenCast(const Analyzer::UOper* uoper, const CompilationOptions& co) {
  CHECK_EQ(uoper->get_optype(), kCAST);
  const auto& ti = uoper->get_type_info();
  const auto operand = uoper->get_operand();
  const auto operand_as_const = dynamic_cast<const Analyzer::Constant*>(operand);
  // For dictionary encoded constants, the cast holds the dictionary id
  // information as the compression parameter; handle this case separately.
  auto operand_lv = operand_as_const ? codegen(operand_as_const, ti.get_compression(), ti.get_comp_param(), co).front()
                                     : codegen(operand, true, co).front();
  const auto& operand_ti = operand->get_type_info();
  if (operand_lv->getType()->isIntegerTy()) {
    if (operand_ti.is_string()) {
      return codegenCastFromString(operand_lv, operand_ti, ti, operand_as_const);
    }
    CHECK(operand_ti.is_integer() || operand_ti.is_decimal() || operand_ti.is_time() || operand_ti.is_boolean());
    if (operand_ti.is_boolean()) {
      CHECK(operand_lv->getType()->isIntegerTy(1) || operand_lv->getType()->isIntegerTy(8));
      if (operand_lv->getType()->isIntegerTy(1)) {
        operand_lv = castToTypeIn(operand_lv, 8);
      }
    }
    if (ti.is_integer() || ti.is_decimal() || ti.is_time()) {
      return codegenCastBetweenIntTypes(operand_lv, operand_ti, ti);
    } else {
      return codegenCastToFp(operand_lv, operand_ti, ti);
    }
  } else {
    return codegenCastFromFp(operand_lv, operand_ti, ti);
  }
  CHECK(false);
  return nullptr;
}

llvm::Value* Executor::codegenCastFromString(llvm::Value* operand_lv,
                                             const SQLTypeInfo& operand_ti,
                                             const SQLTypeInfo& ti,
                                             const bool operand_is_const) {
  if (!ti.is_string()) {
    throw std::runtime_error("Cast from " + operand_ti.get_type_name() + " to " + ti.get_type_name() +
                             " not supported");
  }
  // dictionary encode non-constant
  if (operand_ti.get_compression() != kENCODING_DICT && !operand_is_const) {
    CHECK_EQ(kENCODING_NONE, operand_ti.get_compression());
    CHECK_EQ(kENCODING_DICT, ti.get_compression());
    CHECK(operand_lv->getType()->isIntegerTy(64));
    cgen_state_->must_run_on_cpu_ = true;
    return cgen_state_->emitExternalCall(
        "string_compress",
        get_int_type(32, cgen_state_->context_),
        {operand_lv, ll_int(int64_t(getStringDictionary(ti.get_comp_param(), row_set_mem_owner_)))});
  }
  CHECK(operand_lv->getType()->isIntegerTy(32));
  if (ti.get_compression() == kENCODING_NONE) {
    CHECK_EQ(kENCODING_DICT, operand_ti.get_compression());
    cgen_state_->must_run_on_cpu_ = true;
    return cgen_state_->emitExternalCall(
        "string_decompress",
        get_int_type(64, cgen_state_->context_),
        {operand_lv, ll_int(int64_t(getStringDictionary(operand_ti.get_comp_param(), row_set_mem_owner_)))});
  }
  CHECK(operand_is_const);
  CHECK_EQ(kENCODING_DICT, ti.get_compression());
  return operand_lv;
}

llvm::Value* Executor::codegenCastBetweenIntTypes(llvm::Value* operand_lv,
                                                  const SQLTypeInfo& operand_ti,
                                                  const SQLTypeInfo& ti) {
  if (ti.is_decimal()) {
    CHECK(!operand_ti.is_decimal() || operand_ti.get_scale() <= ti.get_scale());
    operand_lv = cgen_state_->ir_builder_.CreateSExt(operand_lv, get_int_type(64, cgen_state_->context_));
    const auto scale_lv = llvm::ConstantInt::get(get_int_type(64, cgen_state_->context_),
                                                 exp_to_scale(ti.get_scale() - operand_ti.get_scale()));
    if (operand_ti.get_notnull()) {
      operand_lv = cgen_state_->ir_builder_.CreateMul(operand_lv, scale_lv);
    } else {
      operand_lv = cgen_state_->emitCall(
          "scale_decimal",
          {operand_lv, scale_lv, ll_int(inline_int_null_val(operand_ti)), inlineIntNull(SQLTypeInfo(kBIGINT, false))});
    }
  } else if (operand_ti.is_decimal()) {
    const auto scale_lv = llvm::ConstantInt::get(static_cast<llvm::IntegerType*>(operand_lv->getType()),
                                                 exp_to_scale(operand_ti.get_scale()));
    operand_lv = cgen_state_->emitCall("div_" + numeric_type_name(operand_ti) + "_nullable_lhs",
                                       {operand_lv, scale_lv, ll_int(inline_int_null_val(operand_ti))});
  }
  const auto operand_width = static_cast<llvm::IntegerType*>(operand_lv->getType())->getBitWidth();
  const auto target_width = get_bit_width(ti);
  if (target_width == operand_width) {
    return operand_lv;
  }
  if (operand_ti.get_notnull()) {
    return cgen_state_->ir_builder_.CreateCast(
        target_width > operand_width ? llvm::Instruction::CastOps::SExt : llvm::Instruction::CastOps::Trunc,
        operand_lv,
        get_int_type(target_width, cgen_state_->context_));
  }
  return cgen_state_->emitCall("cast_" + numeric_type_name(operand_ti) + "_to_" + numeric_type_name(ti) + "_nullable",
                               {operand_lv, inlineIntNull(operand_ti), inlineIntNull(ti)});
}

llvm::Value* Executor::codegenCastToFp(llvm::Value* operand_lv, const SQLTypeInfo& operand_ti, const SQLTypeInfo& ti) {
  if (!ti.is_fp()) {
    throw std::runtime_error("Cast from " + operand_ti.get_type_name() + " to " + ti.get_type_name() +
                             " not supported");
  }
  const auto to_tname = numeric_type_name(ti);
  llvm::Value* result_lv{nullptr};
  if (operand_ti.get_notnull()) {
    result_lv =
        cgen_state_->ir_builder_.CreateSIToFP(operand_lv,
                                              ti.get_type() == kFLOAT ? llvm::Type::getFloatTy(cgen_state_->context_)
                                                                      : llvm::Type::getDoubleTy(cgen_state_->context_));
  } else {
    result_lv = cgen_state_->emitCall("cast_" + numeric_type_name(operand_ti) + "_to_" + to_tname + "_nullable",
                                      {operand_lv, inlineIntNull(operand_ti), inlineFpNull(ti)});
  }
  CHECK(result_lv);
  result_lv = cgen_state_->ir_builder_.CreateFDiv(
      result_lv, llvm::ConstantFP::get(result_lv->getType(), exp_to_scale(operand_ti.get_scale())));
  return result_lv;
}

llvm::Value* Executor::codegenCastFromFp(llvm::Value* operand_lv,
                                         const SQLTypeInfo& operand_ti,
                                         const SQLTypeInfo& ti) {
  if (!operand_ti.is_fp()) {
    throw std::runtime_error("Cast from " + operand_ti.get_type_name() + " to " + ti.get_type_name() +
                             " not supported");
  }
  if (operand_ti == ti) {
    return operand_lv;
  }
  CHECK(operand_lv->getType()->isFloatTy() || operand_lv->getType()->isDoubleTy());
  if (operand_ti.get_notnull()) {
    if (ti.get_type() == kDOUBLE) {
      return cgen_state_->ir_builder_.CreateFPExt(operand_lv, llvm::Type::getDoubleTy(cgen_state_->context_));
    } else if (ti.get_type() == kFLOAT) {
      return cgen_state_->ir_builder_.CreateFPTrunc(operand_lv, llvm::Type::getFloatTy(cgen_state_->context_));
    } else if (ti.is_integer()) {
      return cgen_state_->ir_builder_.CreateFPToSI(operand_lv, get_int_type(get_bit_width(ti), cgen_state_->context_));
    } else {
      CHECK(false);
    }
  } else {
    const auto from_tname = numeric_type_name(operand_ti);
    const auto to_tname = numeric_type_name(ti);
    if (ti.is_fp()) {
      return cgen_state_->emitCall("cast_" + from_tname + "_to_" + to_tname + "_nullable",
                                   {operand_lv, inlineFpNull(operand_ti), inlineFpNull(ti)});
    } else if (ti.is_integer()) {
      return cgen_state_->emitCall("cast_" + from_tname + "_to_" + to_tname + "_nullable",
                                   {operand_lv, inlineFpNull(operand_ti), inlineIntNull(ti)});
    } else {
      CHECK(false);
    }
  }
  CHECK(false);
  return nullptr;
}

llvm::Value* Executor::codegenUMinus(const Analyzer::UOper* uoper, const CompilationOptions& co) {
  CHECK_EQ(uoper->get_optype(), kUMINUS);
  const auto operand_lv = codegen(uoper->get_operand(), true, co).front();
  const auto& ti = uoper->get_type_info();
  return ti.get_notnull() ? cgen_state_->ir_builder_.CreateNeg(operand_lv)
                          : cgen_state_->emitCall("uminus_" + numeric_type_name(ti) + "_nullable",
                                                  {operand_lv,
                                                   ti.is_fp() ? static_cast<llvm::Value*>(inlineFpNull(ti))
                                                              : static_cast<llvm::Value*>(inlineIntNull(ti))});
}

llvm::Value* Executor::codegenLogical(const Analyzer::UOper* uoper, const CompilationOptions& co) {
  const auto optype = uoper->get_optype();
  CHECK_EQ(kNOT, optype);
  const auto operand = uoper->get_operand();
  const auto& operand_ti = operand->get_type_info();
  CHECK(operand_ti.is_boolean());
  const auto operand_lv = codegen(operand, true, co).front();
  CHECK(operand_lv->getType()->isIntegerTy());
  CHECK(operand_ti.get_notnull() || operand_lv->getType()->isIntegerTy(8));
  return operand_ti.get_notnull() ? cgen_state_->ir_builder_.CreateNot(toBool(operand_lv))
                                  : cgen_state_->emitCall("logical_not", {operand_lv, inlineIntNull(operand_ti)});
}

llvm::Value* Executor::codegenIsNull(const Analyzer::UOper* uoper, const CompilationOptions& co) {
  const auto operand = uoper->get_operand();
  const auto& ti = operand->get_type_info();
  if (ti.get_type() == kNULLT) {
    // if type is null, short-circuit to false
    return llvm::ConstantInt::get(get_int_type(1, cgen_state_->context_), 1);
  }
  CHECK(ti.is_integer() || ti.is_boolean() || ti.is_decimal() || ti.is_time() || ti.is_string() || ti.is_fp() ||
        ti.is_array());
  // if the type is inferred as non null, short-circuit to false
  if (ti.get_notnull() && !ti.is_array()) {
    return llvm::ConstantInt::get(get_int_type(1, cgen_state_->context_), 0);
  }
  const auto operand_lv = codegen(operand, true, co).front();
  if (ti.is_fp()) {
    return cgen_state_->ir_builder_.CreateFCmp(
        llvm::FCmpInst::FCMP_OEQ, operand_lv, ti.get_type() == kFLOAT ? ll_fp(NULL_FLOAT) : ll_fp(NULL_DOUBLE));
  }
  if (ti.is_array()) {
    return cgen_state_->emitExternalCall(
        "array_is_null", get_int_type(1, cgen_state_->context_), {operand_lv, posArg(operand)});
  }
  return cgen_state_->ir_builder_.CreateICmp(llvm::ICmpInst::ICMP_EQ, operand_lv, inlineIntNull(ti));
}

llvm::Value* Executor::codegenUnnest(const Analyzer::UOper* uoper, const CompilationOptions& co) {
  return codegen(uoper->get_operand(), true, co).front();
}

llvm::Value* Executor::codegenArrayAt(const Analyzer::BinOper* array_at, const CompilationOptions& co) {
  const auto arr_expr = array_at->get_left_operand();
  const auto idx_expr = array_at->get_right_operand();
  const auto& idx_ti = idx_expr->get_type_info();
  CHECK(idx_ti.is_integer());
  auto idx_lvs = codegen(idx_expr, true, co);
  CHECK_EQ(size_t(1), idx_lvs.size());
  auto idx_lv = idx_lvs.front();
  if (idx_ti.get_size() < 8) {
    idx_lv = cgen_state_->ir_builder_.CreateCast(
        llvm::Instruction::CastOps::SExt, idx_lv, get_int_type(64, cgen_state_->context_));
  }
  const auto& array_ti = arr_expr->get_type_info();
  CHECK(array_ti.is_array());
  const auto& elem_ti = array_ti.get_elem_type();
  const std::string array_at_fname{
      elem_ti.is_fp() ? "array_at_" + std::string(elem_ti.get_type() == kDOUBLE ? "double_checked" : "float_checked")
                      : "array_at_int" + std::to_string(elem_ti.get_size() * 8) + "_t_checked"};
  const auto ret_ty = elem_ti.is_fp() ? (elem_ti.get_type() == kDOUBLE ? llvm::Type::getDoubleTy(cgen_state_->context_)
                                                                       : llvm::Type::getFloatTy(cgen_state_->context_))
                                      : get_int_type(elem_ti.get_size() * 8, cgen_state_->context_);
  const auto arr_lvs = codegen(arr_expr, true, co);
  CHECK_EQ(size_t(1), arr_lvs.size());
  return cgen_state_->emitExternalCall(array_at_fname,
                                       ret_ty,
                                       {arr_lvs.front(),
                                        posArg(arr_expr),
                                        idx_lv,
                                        elem_ti.is_fp() ? static_cast<llvm::Value*>(inlineFpNull(elem_ti))
                                                        : static_cast<llvm::Value*>(inlineIntNull(elem_ti))});
}

llvm::ConstantInt* Executor::codegenIntConst(const Analyzer::Constant* constant) {
  const auto& type_info = constant->get_type_info();
  if (constant->get_is_null()) {
    return inlineIntNull(type_info);
  }
  const auto type = type_info.is_decimal() ? decimal_to_int_type(type_info) : type_info.get_type();
  switch (type) {
    case kSMALLINT:
      return ll_int(constant->get_constval().smallintval);
    case kINT:
      return ll_int(constant->get_constval().intval);
    case kBIGINT:
      return ll_int(constant->get_constval().bigintval);
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
      return ll_int(constant->get_constval().timeval);
    default:
      CHECK(false);
  }
}

llvm::ConstantInt* Executor::inlineIntNull(const SQLTypeInfo& type_info) {
  auto type = type_info.is_decimal() ? decimal_to_int_type(type_info) : type_info.get_type();
  if (type_info.is_string()) {
    switch (type_info.get_compression()) {
      case kENCODING_DICT:
        return ll_int(static_cast<int32_t>(inline_int_null_val(type_info)));
      case kENCODING_NONE:
        return ll_int(int64_t(0));
      default:
        CHECK(false);
    }
  }
  switch (type) {
    case kBOOLEAN:
      return ll_int(static_cast<int8_t>(inline_int_null_val(type_info)));
    case kSMALLINT:
      return ll_int(static_cast<int16_t>(inline_int_null_val(type_info)));
    case kINT:
      return ll_int(static_cast<int32_t>(inline_int_null_val(type_info)));
    case kBIGINT:
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
      return ll_int(inline_int_null_val(type_info));
    case kARRAY:
      return ll_int(int64_t(0));
    default:
      CHECK(false);
  }
}

llvm::ConstantFP* Executor::inlineFpNull(const SQLTypeInfo& type_info) {
  CHECK(type_info.is_fp());
  switch (type_info.get_type()) {
    case kFLOAT:
      return ll_fp(NULL_FLOAT);
    case kDOUBLE:
      return ll_fp(NULL_DOUBLE);
    default:
      CHECK(false);
  }
}

std::pair<llvm::ConstantInt*, llvm::ConstantInt*> Executor::inlineIntMaxMin(const size_t byte_width) {
  int64_t max_int{0}, min_int{0};
  std::tie(max_int, min_int) = inline_int_max_min(byte_width);
  switch (byte_width) {
    case 1:
      return std::make_pair(ll_int(static_cast<int8_t>(max_int)), ll_int(static_cast<int8_t>(min_int)));
    case 2:
      return std::make_pair(ll_int(static_cast<int16_t>(max_int)), ll_int(static_cast<int16_t>(min_int)));
    case 4:
      return std::make_pair(ll_int(static_cast<int32_t>(max_int)), ll_int(static_cast<int32_t>(min_int)));
    case 8:
      return std::make_pair(ll_int(max_int), ll_int(min_int));
    default:
      CHECK(false);
  }
}

llvm::Value* Executor::codegenArith(const Analyzer::BinOper* bin_oper, const CompilationOptions& co) {
  const auto optype = bin_oper->get_optype();
  CHECK(IS_ARITHMETIC(optype));
  const auto lhs = bin_oper->get_left_operand();
  const auto rhs = bin_oper->get_right_operand();
  const auto& lhs_type = lhs->get_type_info();
  const auto& rhs_type = rhs->get_type_info();
  auto lhs_lv = codegen(lhs, true, co).front();
  auto rhs_lv = codegen(rhs, true, co).front();
  CHECK_EQ(lhs_type.get_type(), rhs_type.get_type());
  if (lhs_type.is_decimal()) {
    CHECK_EQ(lhs_type.get_scale(), rhs_type.get_scale());
  }
  const auto null_check_suffix = get_null_check_suffix(lhs_type, rhs_type);
  if (lhs_type.is_integer() || lhs_type.is_decimal()) {
    const auto int_typename = numeric_type_name(lhs_type);
    switch (optype) {
      case kMINUS:
        if (null_check_suffix.empty()) {
          return cgen_state_->ir_builder_.CreateSub(lhs_lv, rhs_lv);
        } else {
          return cgen_state_->emitCall("sub_" + int_typename + null_check_suffix,
                                       {lhs_lv, rhs_lv, ll_int(inline_int_null_val(lhs_type))});
        }
      case kPLUS:
        if (null_check_suffix.empty()) {
          return cgen_state_->ir_builder_.CreateAdd(lhs_lv, rhs_lv);
        } else {
          return cgen_state_->emitCall("add_" + int_typename + null_check_suffix,
                                       {lhs_lv, rhs_lv, ll_int(inline_int_null_val(lhs_type))});
        }
      case kMULTIPLY: {
        if (lhs_type.is_decimal()) {
          return cgen_state_->emitCall(
              "mul_" + int_typename + "_decimal",
              {lhs_lv, rhs_lv, ll_int(exp_to_scale(lhs_type.get_scale())), ll_int(inline_int_null_val(lhs_type))});
        }
        llvm::Value* result{nullptr};
        if (null_check_suffix.empty()) {
          result = cgen_state_->ir_builder_.CreateMul(lhs_lv, rhs_lv);
        } else {
          result = cgen_state_->emitCall("mul_" + int_typename + null_check_suffix,
                                         {lhs_lv, rhs_lv, ll_int(inline_int_null_val(lhs_type))});
        }
        return result;
      }
      case kDIVIDE:
        return codegenDiv(lhs_lv, rhs_lv, null_check_suffix.empty() ? "" : int_typename, null_check_suffix, lhs_type);
      case kMODULO:
        return codegenMod(lhs_lv, rhs_lv, null_check_suffix.empty() ? "" : int_typename, null_check_suffix, lhs_type);
      default:
        CHECK(false);
    }
  }
  if (lhs_type.is_fp()) {
    const auto fp_typename = numeric_type_name(lhs_type);
    llvm::ConstantFP* fp_null{lhs_type.get_type() == kFLOAT ? ll_fp(NULL_FLOAT) : ll_fp(NULL_DOUBLE)};
    switch (optype) {
      case kMINUS:
        return null_check_suffix.empty()
                   ? cgen_state_->ir_builder_.CreateFSub(lhs_lv, rhs_lv)
                   : cgen_state_->emitCall("sub_" + fp_typename + null_check_suffix, {lhs_lv, rhs_lv, fp_null});
      case kPLUS:
        return null_check_suffix.empty()
                   ? cgen_state_->ir_builder_.CreateFAdd(lhs_lv, rhs_lv)
                   : cgen_state_->emitCall("add_" + fp_typename + null_check_suffix, {lhs_lv, rhs_lv, fp_null});
      case kMULTIPLY:
        return null_check_suffix.empty()
                   ? cgen_state_->ir_builder_.CreateFMul(lhs_lv, rhs_lv)
                   : cgen_state_->emitCall("mul_" + fp_typename + null_check_suffix, {lhs_lv, rhs_lv, fp_null});
      case kDIVIDE:
        return codegenDiv(lhs_lv, rhs_lv, null_check_suffix.empty() ? "" : fp_typename, null_check_suffix, lhs_type);
      default:
        CHECK(false);
    }
  }
  CHECK(false);
  return nullptr;
}

llvm::Value* Executor::codegenDiv(llvm::Value* lhs_lv,
                                  llvm::Value* rhs_lv,
                                  const std::string& null_typename,
                                  const std::string& null_check_suffix,
                                  const SQLTypeInfo& ti) {
  CHECK_EQ(lhs_lv->getType(), rhs_lv->getType());
  if (ti.is_decimal()) {
    CHECK(lhs_lv->getType()->isIntegerTy());
    const auto scale_lv = llvm::ConstantInt::get(lhs_lv->getType(), exp_to_scale(ti.get_scale()));
    lhs_lv = null_typename.empty() ? cgen_state_->ir_builder_.CreateMul(lhs_lv, scale_lv)
                                   : cgen_state_->emitCall("mul_" + numeric_type_name(ti) + null_check_suffix,
                                                           {lhs_lv, scale_lv, ll_int(inline_int_null_val(ti))});
  }
  cgen_state_->uses_div_ = true;
  auto div_ok = llvm::BasicBlock::Create(cgen_state_->context_, "div_ok", cgen_state_->row_func_);
  auto div_zero = llvm::BasicBlock::Create(cgen_state_->context_, "div_zero", cgen_state_->row_func_);
  auto zero_const = rhs_lv->getType()->isIntegerTy() ? llvm::ConstantInt::get(rhs_lv->getType(), 0, true)
                                                     : llvm::ConstantFP::get(rhs_lv->getType(), 0.);
  cgen_state_->ir_builder_.CreateCondBr(
      zero_const->getType()->isFloatingPointTy()
          ? cgen_state_->ir_builder_.CreateFCmp(llvm::FCmpInst::FCMP_ONE, rhs_lv, zero_const)
          : cgen_state_->ir_builder_.CreateICmp(llvm::ICmpInst::ICMP_NE, rhs_lv, zero_const),
      div_ok,
      div_zero);
  cgen_state_->ir_builder_.SetInsertPoint(div_ok);
  auto ret = zero_const->getType()->isIntegerTy()
                 ? (null_typename.empty() ? cgen_state_->ir_builder_.CreateSDiv(lhs_lv, rhs_lv)
                                          : cgen_state_->emitCall("div_" + null_typename + null_check_suffix,
                                                                  {lhs_lv, rhs_lv, ll_int(inline_int_null_val(ti))}))
                 : (null_typename.empty()
                        ? cgen_state_->ir_builder_.CreateFDiv(lhs_lv, rhs_lv)
                        : cgen_state_->emitCall(
                              "div_" + null_typename + null_check_suffix,
                              {lhs_lv, rhs_lv, ti.get_type() == kFLOAT ? ll_fp(NULL_FLOAT) : ll_fp(NULL_DOUBLE)}));
  cgen_state_->ir_builder_.SetInsertPoint(div_zero);
  cgen_state_->ir_builder_.CreateRet(ll_int(ERR_DIV_BY_ZERO));
  cgen_state_->ir_builder_.SetInsertPoint(div_ok);
  return ret;
}

llvm::Value* Executor::codegenMod(llvm::Value* lhs_lv,
                                  llvm::Value* rhs_lv,
                                  const std::string& null_typename,
                                  const std::string& null_check_suffix,
                                  const SQLTypeInfo& ti) {
  CHECK_EQ(lhs_lv->getType(), rhs_lv->getType());
  CHECK(ti.is_integer());
  cgen_state_->uses_div_ = true;
  auto mod_ok = llvm::BasicBlock::Create(cgen_state_->context_, "mod_ok", cgen_state_->row_func_);
  auto mod_zero = llvm::BasicBlock::Create(cgen_state_->context_, "mod_zero", cgen_state_->row_func_);
  auto zero_const = llvm::ConstantInt::get(rhs_lv->getType(), 0, true);
  cgen_state_->ir_builder_.CreateCondBr(
      cgen_state_->ir_builder_.CreateICmp(llvm::ICmpInst::ICMP_NE, rhs_lv, zero_const), mod_ok, mod_zero);
  cgen_state_->ir_builder_.SetInsertPoint(mod_ok);
  auto ret = null_typename.empty() ? cgen_state_->ir_builder_.CreateSRem(lhs_lv, rhs_lv)
                                   : cgen_state_->emitCall("mod_" + null_typename + null_check_suffix,
                                                           {lhs_lv, rhs_lv, ll_int(inline_int_null_val(ti))});
  cgen_state_->ir_builder_.SetInsertPoint(mod_zero);
  cgen_state_->ir_builder_.CreateRet(ll_int(ERR_DIV_BY_ZERO));
  cgen_state_->ir_builder_.SetInsertPoint(mod_ok);
  return ret;
}

std::unordered_set<llvm::Function*> Executor::markDeadRuntimeFuncs(llvm::Module& module,
                                                                   const std::vector<llvm::Function*>& roots,
                                                                   const std::vector<llvm::Function*>& leaves) {
  std::unordered_set<llvm::Function*> live_funcs;
  live_funcs.insert(roots.begin(), roots.end());
  live_funcs.insert(leaves.begin(), leaves.end());

  if (auto F = module.getFunction("init_shared_mem_nop"))
    live_funcs.insert(F);
  if (auto F = module.getFunction("write_back_nop"))
    live_funcs.insert(F);

  for (const llvm::Function* F : roots) {
    for (const llvm::BasicBlock& BB : *F) {
      for (const llvm::Instruction& I : BB) {
        if (const llvm::CallInst* CI = llvm::dyn_cast<const llvm::CallInst>(&I)) {
          live_funcs.insert(CI->getCalledFunction());
        }
      }
    }
  }

  for (llvm::Function& F : module) {
    if (!live_funcs.count(&F) && !F.isDeclaration()) {
      F.setLinkage(llvm::GlobalValue::InternalLinkage);
    }
  }

  return live_funcs;
}

namespace {

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

std::vector<int64_t*> launch_query_cpu_code(const std::vector<void*>& fn_ptrs,
                                            const bool hoist_literals,
                                            const std::vector<int8_t>& literal_buff,
                                            std::vector<std::vector<const int8_t*>> col_buffers,
                                            const std::vector<int64_t>& num_rows,
                                            const std::vector<uint64_t>& frag_row_offsets,
                                            const int32_t scan_limit,
                                            const QueryMemoryDescriptor& query_mem_desc,
                                            const std::vector<int64_t>& init_agg_vals,
                                            std::vector<int64_t*> group_by_buffers,
                                            std::vector<int64_t*> small_group_by_buffers,
                                            int32_t* error_code,
                                            const uint32_t num_tables,
                                            const int64_t join_hash_table) {
  const bool is_group_by{!group_by_buffers.empty()};
  std::vector<int64_t*> out_vec;
  if (group_by_buffers.empty()) {
    for (size_t i = 0; i < init_agg_vals.size(); ++i) {
      auto buff = new int64_t[1];
      out_vec.push_back(static_cast<int64_t*>(buff));
    }
  }

  std::vector<const int8_t**> multifrag_col_buffers;
  for (auto& col_buffer : col_buffers) {
    multifrag_col_buffers.push_back(&col_buffer[0]);
  }
  const int8_t*** multifrag_cols_ptr{multifrag_col_buffers.empty() ? nullptr : &multifrag_col_buffers[0]};
  int64_t** small_group_by_buffers_ptr{small_group_by_buffers.empty() ? nullptr : &small_group_by_buffers[0]};
  const uint32_t num_fragments = multifrag_cols_ptr ? 1 : 0;

  int64_t rowid_lookup_num_rows{*error_code ? *error_code + 1 : 0};
  auto num_rows_ptr = rowid_lookup_num_rows ? &rowid_lookup_num_rows : &num_rows[0];
  int32_t total_matched_init{0};

  std::vector<int64_t> cmpt_val_buff;
  if (is_group_by) {
    cmpt_val_buff = compact_init_vals(
        align_to_int64(query_mem_desc.getColsSize()) / sizeof(int64_t), init_agg_vals, query_mem_desc.agg_col_widths);
  }

  if (hoist_literals) {
    typedef void (*agg_query)(const int8_t*** col_buffers,
                              const uint32_t* num_fragments,
                              const int8_t* literals,
                              const int64_t* num_rows,
                              const uint64_t* frag_row_offsets,
                              const int32_t* max_matched,
                              int32_t* total_matched,
                              const int64_t* init_agg_value,
                              int64_t** out,
                              int64_t** out2,
                              int32_t* error_code,
                              const uint32_t* num_tables,
                              const int64_t* join_hash_table_ptr);
    if (is_group_by) {
      reinterpret_cast<agg_query>(fn_ptrs[0])(multifrag_cols_ptr,
                                              &num_fragments,
                                              &literal_buff[0],
                                              num_rows_ptr,
                                              &frag_row_offsets[0],
                                              &scan_limit,
                                              &total_matched_init,
                                              &cmpt_val_buff[0],
                                              &group_by_buffers[0],
                                              small_group_by_buffers_ptr,
                                              error_code,
                                              &num_tables,
                                              &join_hash_table);
    } else {
      reinterpret_cast<agg_query>(fn_ptrs[0])(multifrag_cols_ptr,
                                              &num_fragments,
                                              &literal_buff[0],
                                              num_rows_ptr,
                                              &frag_row_offsets[0],
                                              &scan_limit,
                                              &total_matched_init,
                                              &init_agg_vals[0],
                                              &out_vec[0],
                                              nullptr,
                                              error_code,
                                              &num_tables,
                                              &join_hash_table);
    }
  } else {
    typedef void (*agg_query)(const int8_t*** col_buffers,
                              const uint32_t* num_fragments,
                              const int64_t* num_rows,
                              const uint64_t* frag_row_offsets,
                              const int32_t* max_matched,
                              int32_t* total_matched,
                              const int64_t* init_agg_value,
                              int64_t** out,
                              int64_t** out2,
                              int32_t* error_code,
                              const uint32_t* num_tables,
                              const int64_t* join_hash_table_ptr);
    if (is_group_by) {
      reinterpret_cast<agg_query>(fn_ptrs[0])(multifrag_cols_ptr,
                                              &num_fragments,
                                              num_rows_ptr,
                                              &frag_row_offsets[0],
                                              &scan_limit,
                                              &total_matched_init,
                                              &cmpt_val_buff[0],
                                              &group_by_buffers[0],
                                              small_group_by_buffers_ptr,
                                              error_code,
                                              &num_tables,
                                              &join_hash_table);
    } else {
      reinterpret_cast<agg_query>(fn_ptrs[0])(multifrag_cols_ptr,
                                              &num_fragments,
                                              num_rows_ptr,
                                              &frag_row_offsets[0],
                                              &scan_limit,
                                              &total_matched_init,
                                              &init_agg_vals[0],
                                              &out_vec[0],
                                              nullptr,
                                              error_code,
                                              &num_tables,
                                              &join_hash_table);
    }
  }

  if (rowid_lookup_num_rows && *error_code < 0) {
    *error_code = 0;
  }

  return out_vec;
}

#ifdef __clang__
#pragma clang diagnostic pop
#else
#pragma GCC diagnostic pop
#endif

}  // namespace

// TODO(alex): remove or split
std::pair<int64_t, int32_t> Executor::reduceResults(const SQLAgg agg,
                                                    const SQLTypeInfo& ti,
                                                    const int64_t agg_init_val,
                                                    const int8_t out_byte_width,
                                                    const int64_t* out_vec,
                                                    const size_t out_vec_sz,
                                                    const bool is_group_by) {
  const auto error_no = ERR_OVERFLOW_OR_UNDERFLOW;
  switch (agg) {
    case kAVG:
    case kSUM:
      if (0 != agg_init_val) {
        if (ti.is_integer() || ti.is_decimal() || ti.is_time() || ti.is_boolean()) {
          int64_t agg_result = agg_init_val;
          for (size_t i = 0; i < out_vec_sz; ++i) {
            if (detect_overflow_and_underflow(agg_result, out_vec[i], true, agg_init_val, ti)) {
              return {0, error_no};
            }
            agg_sum_skip_val(&agg_result, out_vec[i], agg_init_val);
          }
          return {agg_result, 0};
        } else {
          CHECK(ti.is_fp());
          switch (out_byte_width) {
            case 4: {
              int agg_result = static_cast<int32_t>(agg_init_val);
              for (size_t i = 0; i < out_vec_sz; ++i) {
                agg_sum_float_skip_val(&agg_result,
                                       *reinterpret_cast<const float*>(&out_vec[i]),
                                       *reinterpret_cast<const float*>(&agg_init_val));
              }
              return {float_to_double_bin(static_cast<int32_t>(agg_result), true), 0};
            } break;
            case 8: {
              int64_t agg_result = agg_init_val;
              for (size_t i = 0; i < out_vec_sz; ++i) {
                agg_sum_double_skip_val(&agg_result,
                                        *reinterpret_cast<const double*>(&out_vec[i]),
                                        *reinterpret_cast<const double*>(&agg_init_val));
              }
              return {agg_result, 0};
            } break;
            default:
              CHECK(false);
          }
        }
      }
      if (ti.is_integer() || ti.is_decimal() || ti.is_time()) {
        int64_t agg_result = 0;
        for (size_t i = 0; i < out_vec_sz; ++i) {
          if (detect_overflow_and_underflow(agg_result, out_vec[i], false, int64_t(0), ti)) {
            return {0, error_no};
          }
          agg_result += out_vec[i];
        }
        return {agg_result, 0};
      } else {
        CHECK(ti.is_fp());
        switch (out_byte_width) {
          case 4: {
            double r = 0.;
            for (size_t i = 0; i < out_vec_sz; ++i) {
              r += *reinterpret_cast<const float*>(&out_vec[i]);
            }
            return {*reinterpret_cast<const int64_t*>(&r), 0};
          }
          case 8: {
            double r = 0.;
            for (size_t i = 0; i < out_vec_sz; ++i) {
              r += *reinterpret_cast<const double*>(&out_vec[i]);
            }
            return {*reinterpret_cast<const int64_t*>(&r), 0};
          }
          default:
            CHECK(false);
        }
      }
      break;
    case kCOUNT: {
      int64_t agg_result = 0;
      for (size_t i = 0; i < out_vec_sz; ++i) {
        if (detect_overflow_and_underflow(agg_result, out_vec[i], false, int64_t(0), ti)) {
          return {0, error_no};
        }
        agg_result += out_vec[i];
      }
      return {agg_result, 0};
    }
    case kMIN: {
      if (ti.is_integer() || ti.is_decimal() || ti.is_time() || ti.is_boolean()) {
        int64_t agg_result = agg_init_val;
        for (size_t i = 0; i < out_vec_sz; ++i) {
          agg_min_skip_val(&agg_result, out_vec[i], agg_init_val);
        }
        return {agg_result, 0};
      } else {
        switch (out_byte_width) {
          case 4: {
            int32_t agg_result = static_cast<int32_t>(agg_init_val);
            for (size_t i = 0; i < out_vec_sz; ++i) {
              agg_min_float_skip_val(&agg_result,
                                     *reinterpret_cast<const float*>(&out_vec[i]),
                                     *reinterpret_cast<const float*>(&agg_init_val));
            }
            return {float_to_double_bin(agg_result, true), 0};
          }
          case 8: {
            int64_t agg_result = agg_init_val;
            for (size_t i = 0; i < out_vec_sz; ++i) {
              agg_min_double_skip_val(&agg_result,
                                      *reinterpret_cast<const double*>(&out_vec[i]),
                                      *reinterpret_cast<const double*>(&agg_init_val));
            }
            return {agg_result, 0};
          }
          default:
            CHECK(false);
        }
      }
    }
    case kMAX:
      if (ti.is_integer() || ti.is_decimal() || ti.is_time() || ti.is_boolean()) {
        int64_t agg_result = agg_init_val;
        for (size_t i = 0; i < out_vec_sz; ++i) {
          agg_max_skip_val(&agg_result, out_vec[i], agg_init_val);
        }
        return {agg_result, 0};
      } else {
        switch (out_byte_width) {
          case 4: {
            int32_t agg_result = static_cast<int32_t>(agg_init_val);
            for (size_t i = 0; i < out_vec_sz; ++i) {
              agg_max_float_skip_val(&agg_result,
                                     *reinterpret_cast<const float*>(&out_vec[i]),
                                     *reinterpret_cast<const float*>(&agg_init_val));
            }
            return {float_to_double_bin(agg_result, !ti.get_notnull()), 0};
          }
          case 8: {
            int64_t agg_result = agg_init_val;
            for (size_t i = 0; i < out_vec_sz; ++i) {
              agg_max_double_skip_val(&agg_result,
                                      *reinterpret_cast<const double*>(&out_vec[i]),
                                      *reinterpret_cast<const double*>(&agg_init_val));
            }
            return {agg_result, 0};
          }
          default:
            CHECK(false);
        }
      }
    default:
      CHECK(false);
  }
  CHECK(false);
}

ResultRows Executor::reduceMultiDeviceResults(
    std::vector<std::pair<ResultRows, std::vector<size_t>>>& results_per_device,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const QueryMemoryDescriptor& query_mem_desc,
    const bool output_columnar) const {
  if (results_per_device.empty()) {
    return ResultRows(query_mem_desc, {}, nullptr, nullptr, {}, ExecutorDeviceType::CPU);
  }

  auto reduced_results = results_per_device.front().first;

  for (size_t i = 1; i < results_per_device.size(); ++i) {
    const auto error_code = reduced_results.reduce(results_per_device[i].first, query_mem_desc, output_columnar);
    if (error_code) {
      CHECK_EQ(error_code, ERR_OVERFLOW_OR_UNDERFLOW);
      throw OverflowOrUnderflow();
    }
  }

  row_set_mem_owner->addLiteralStringDict(lit_str_dict_);

  return reduced_results;
}

namespace {

std::vector<std::string> get_agg_fnames(const std::vector<Analyzer::Expr*>& target_exprs, const bool is_group_by) {
  std::vector<std::string> result;
  for (size_t target_idx = 0, agg_col_idx = 0; target_idx < target_exprs.size(); ++target_idx, ++agg_col_idx) {
    const auto target_expr = target_exprs[target_idx];
    CHECK(target_expr);
    const auto target_type_info = target_expr->get_type_info();
    const auto target_type = target_type_info.get_type();
    const auto agg_expr = dynamic_cast<Analyzer::AggExpr*>(target_expr);
    if (!agg_expr) {
      result.push_back((target_type == kFLOAT || target_type == kDOUBLE) ? "agg_id_double" : "agg_id");
      if (target_type_info.is_string() && target_type_info.get_compression() == kENCODING_NONE) {
        result.push_back("agg_id");
      }
      continue;
    }
    const auto agg_type = agg_expr->get_aggtype();
    const auto& agg_type_info = agg_type != kCOUNT ? agg_expr->get_arg()->get_type_info() : target_type_info;
    switch (agg_type) {
      case kAVG: {
        if (!agg_type_info.is_integer() && !agg_type_info.is_decimal() && !agg_type_info.is_fp()) {
          throw std::runtime_error("AVG is only valid on integer and floating point");
        }
        result.push_back((agg_type_info.is_integer() || agg_type_info.is_time()) ? "agg_sum" : "agg_sum_double");
        result.push_back((agg_type_info.is_integer() || agg_type_info.is_time()) ? "agg_count" : "agg_count_double");
        break;
      }
      case kMIN: {
        if (agg_type_info.is_string() || agg_type_info.is_array()) {
          throw std::runtime_error("MIN on strings or arrays not supported yet");
        }
        result.push_back((agg_type_info.is_integer() || agg_type_info.is_time()) ? "agg_min" : "agg_min_double");
        break;
      }
      case kMAX: {
        if (agg_type_info.is_string() || agg_type_info.is_array()) {
          throw std::runtime_error("MAX on strings or arrays not supported yet");
        }
        result.push_back((agg_type_info.is_integer() || agg_type_info.is_time()) ? "agg_max" : "agg_max_double");
        break;
      }
      case kSUM: {
        if (!agg_type_info.is_integer() && !agg_type_info.is_decimal() && !agg_type_info.is_fp()) {
          throw std::runtime_error("SUM is only valid on integer and floating point");
        }
        result.push_back((agg_type_info.is_integer() || agg_type_info.is_time()) ? "agg_sum" : "agg_sum_double");
        break;
      }
      case kCOUNT:
        result.push_back(agg_expr->get_is_distinct() ? "agg_count_distinct" : "agg_count");
        break;
      default:
        CHECK(false);
    }
  }
  return result;
}

ResultRows results_union(std::vector<std::pair<ResultRows, std::vector<size_t>>>& results_per_device) {
  if (results_per_device.empty()) {
    return ResultRows({}, {}, nullptr, nullptr, {}, ExecutorDeviceType::CPU);
  }
  typedef std::pair<ResultRows, std::vector<size_t>> IndexedResultRows;
  std::sort(results_per_device.begin(),
            results_per_device.end(),
            [](const IndexedResultRows& lhs, const IndexedResultRows& rhs) {
    CHECK_EQ(size_t(1), lhs.second.size());
    CHECK_EQ(size_t(1), rhs.second.size());
    return lhs.second < rhs.second;
  });
  auto all_results = results_per_device.front().first;
  for (size_t dev_idx = 1; dev_idx < results_per_device.size(); ++dev_idx) {
    all_results.append(results_per_device[dev_idx].first);
  }
  return all_results;
}

}  // namespace

ResultRows Executor::executeResultPlan(const Planner::Result* result_plan,
                                       const bool hoist_literals,
                                       const ExecutorDeviceType device_type,
                                       const ExecutorOptLevel opt_level,
                                       const Catalog_Namespace::Catalog& cat,
                                       size_t& max_groups_buffer_entry_guess,
                                       int32_t* error_code,
                                       const Planner::Sort* sort_plan,
                                       const bool allow_multifrag,
                                       const bool just_explain,
                                       const bool allow_loop_joins) {
  const auto agg_plan = dynamic_cast<const Planner::AggPlan*>(result_plan->get_child_plan());
  if (!agg_plan) {  // TODO(alex)
    throw std::runtime_error("Query not supported yet, child plan needs to be an aggregate plan.");
  }
  row_set_mem_owner_ = std::make_shared<RowSetMemoryOwner>();
  lit_str_dict_ = nullptr;
  const auto scan_plan = dynamic_cast<const Planner::Scan*>(agg_plan->get_child_plan());
  auto simple_quals = scan_plan ? scan_plan->get_simple_quals() : std::list<std::shared_ptr<Analyzer::Expr>>{};
  auto quals = scan_plan ? scan_plan->get_quals() : std::list<std::shared_ptr<Analyzer::Expr>>{};
  std::vector<InputDescriptor> input_descs;
  std::list<InputColDescriptor> input_col_descs;
  collect_input_descs(input_descs, input_col_descs, agg_plan, cat);
  const auto join_plan = get_join_child(agg_plan);
  if (join_plan) {
    collect_quals_from_join(simple_quals, quals, join_plan);
  }
  const auto join_quals = join_plan ? join_plan->get_quals() : std::list<std::shared_ptr<Analyzer::Expr>>{};
  CHECK(check_plan_sanity(agg_plan));
  const auto query_infos = get_table_infos(input_descs, cat, TemporaryTables{});
  const auto ra_exe_unit_in = RelAlgExecutionUnit{input_descs,
                                                  input_col_descs,
                                                  simple_quals,
                                                  quals,
                                                  JoinType::INVALID,
                                                  join_quals,
                                                  {},
                                                  agg_plan->get_groupby_list(),
                                                  get_agg_target_exprs(agg_plan),
                                                  {},
                                                  0};
  QueryRewriter query_rewriter(ra_exe_unit_in, query_infos, this, result_plan);
  const auto ra_exe_unit = query_rewriter.rewrite();
  auto result_rows = executeWorkUnit(error_code,
                                     max_groups_buffer_entry_guess,
                                     true,
                                     query_infos,
                                     ra_exe_unit,
                                     {device_type, hoist_literals, opt_level},
                                     {false, allow_multifrag, just_explain, allow_loop_joins, g_enable_watchdog},
                                     cat,
                                     row_set_mem_owner_,
                                     nullptr);
  if (just_explain) {
    return result_rows;
  }

  const int in_col_count{static_cast<int>(agg_plan->get_targetlist().size())};
  std::list<InputColDescriptor> pseudo_input_col_descs;
  for (int pseudo_col = 1; pseudo_col <= in_col_count; ++pseudo_col) {
    pseudo_input_col_descs.emplace_back(pseudo_col, 0, -1);
  }
  const auto order_entries = sort_plan ? sort_plan->get_order_entries() : std::list<Analyzer::OrderEntry>{};
  const RelAlgExecutionUnit res_ra_unit{{},
                                        pseudo_input_col_descs,
                                        result_plan->get_constquals(),
                                        result_plan->get_quals(),
                                        JoinType::INVALID,
                                        {},
                                        {},
                                        {nullptr},
                                        get_agg_target_exprs(result_plan),
                                        order_entries,
                                        0};

  if (*error_code) {
    return ResultRows({}, {}, nullptr, nullptr, {}, ExecutorDeviceType::CPU);
  }
  const auto& targets = result_plan->get_targetlist();
  CHECK(!targets.empty());
  std::vector<AggInfo> agg_infos;
  for (size_t target_idx = 0; target_idx < targets.size(); ++target_idx) {
    const auto target_entry = targets[target_idx];
    const auto target_type = target_entry->get_expr()->get_type_info().get_type();
    agg_infos.emplace_back((target_type == kFLOAT || target_type == kDOUBLE) ? "agg_id_double" : "agg_id",
                           target_entry->get_expr(),
                           0,
                           target_idx);
  }
  std::vector<SQLTypeInfo> target_types;
  for (auto in_col : agg_plan->get_targetlist()) {
    target_types.push_back(in_col->get_expr()->get_type_info());
  }
  ColumnarResults result_columns(result_rows, in_col_count, target_types);
  std::vector<llvm::Value*> col_heads;
  // Nested query, let the compiler know
  ResetIsNested reset_is_nested(this);
  is_nested_ = true;
  std::vector<Analyzer::Expr*> target_exprs;
  for (auto target_entry : targets) {
    target_exprs.emplace_back(target_entry->get_expr());
  }
  const auto row_count = result_rows.rowCount();
  if (!row_count) {
    return ResultRows({}, {}, nullptr, nullptr, {}, ExecutorDeviceType::CPU);
  }
  std::vector<ColWidths> agg_col_widths;
  for (auto wid : get_col_byte_widths(target_exprs)) {
    agg_col_widths.push_back(
        {wid, int8_t(compact_byte_width(wid, pick_target_compact_width(res_ra_unit, {}, get_min_byte_width())))});
  }
  QueryMemoryDescriptor query_mem_desc{this,
                                       allow_multifrag,
                                       GroupByColRangeType::OneColGuessedRange,
                                       false,
                                       false,
                                       -1,
                                       0,
                                       {sizeof(int64_t)},
                                       agg_col_widths,
                                       row_count,
                                       small_groups_buffer_entry_count_,
                                       0,
                                       0,
                                       0,
                                       false,
                                       GroupByMemSharing::Shared,
                                       CountDistinctDescriptors{},
                                       false,
                                       true,
                                       false,
                                       false};
  auto compilation_result =
      compileWorkUnit(false,
                      {},
                      res_ra_unit,
                      {ExecutorDeviceType::CPU, hoist_literals, opt_level},
                      {false, allow_multifrag, just_explain, allow_loop_joins, g_enable_watchdog},
                      nullptr,
                      false,
                      row_set_mem_owner_,
                      row_count,
                      small_groups_buffer_entry_count_,
                      get_min_byte_width(),
                      JoinInfo(JoinImplType::Invalid, std::vector<std::shared_ptr<Analyzer::BinOper>>{}, nullptr));
  auto column_buffers = result_columns.getColumnBuffers();
  CHECK_EQ(column_buffers.size(), static_cast<size_t>(in_col_count));
  std::vector<int64_t> init_agg_vals(query_mem_desc.agg_col_widths.size());
  auto query_exe_context = query_mem_desc.getQueryExecutionContext(
      init_agg_vals, this, ExecutorDeviceType::CPU, 0, {}, row_set_mem_owner_, false, false, nullptr);
  const auto hoist_buf = serializeLiterals(compilation_result.literal_values, 0);
  *error_code = 0;
  std::vector<std::vector<const int8_t*>> multi_frag_col_buffers{column_buffers};
  launch_query_cpu_code(compilation_result.native_functions,
                        hoist_literals,
                        hoist_buf,
                        multi_frag_col_buffers,
                        {static_cast<int64_t>(result_columns.size())},
                        {0},
                        0,
                        query_mem_desc,
                        init_agg_vals,
                        query_exe_context->group_by_buffers_,
                        query_exe_context->small_group_by_buffers_,
                        error_code,
                        1,
                        0);
  CHECK_GE(*error_code, 0);
  return query_exe_context->groupBufferToResults(0, target_exprs, false);
}

ResultRows Executor::executeSortPlan(const Planner::Sort* sort_plan,
                                     const int64_t limit,
                                     const int64_t offset,
                                     const bool hoist_literals,
                                     const ExecutorDeviceType device_type,
                                     const ExecutorOptLevel opt_level,
                                     const Catalog_Namespace::Catalog& cat,
                                     size_t& max_groups_buffer_entry_guess,
                                     int32_t* error_code,
                                     const bool allow_multifrag,
                                     const bool just_explain,
                                     const bool allow_loop_joins) {
  *error_code = 0;
  auto rows_to_sort = executeSelectPlan(sort_plan->get_child_plan(),
                                        0,
                                        0,
                                        hoist_literals,
                                        device_type,
                                        opt_level,
                                        cat,
                                        max_groups_buffer_entry_guess,
                                        error_code,
                                        sort_plan,
                                        allow_multifrag,
                                        just_explain,
                                        allow_loop_joins,
                                        nullptr);
  if (just_explain) {
    return rows_to_sort;
  }
  rows_to_sort.sort(sort_plan->get_order_entries(), sort_plan->get_remove_duplicates(), limit + offset);
  if (limit || offset) {
    rows_to_sort.dropFirstN(offset);
    if (limit) {
      rows_to_sort.keepFirstN(limit);
    }
  }
  return rows_to_sort;
}

namespace {

size_t compute_buffer_entry_guess(const std::vector<Fragmenter_Namespace::TableInfo>& query_infos) {
  using Fragmenter_Namespace::FragmentInfo;
  size_t max_groups_buffer_entry_guess = 1;
  for (const auto& query_info : query_infos) {
    CHECK(!query_info.fragments.empty());
    auto it =
        std::max_element(query_info.fragments.begin(),
                         query_info.fragments.end(),
                         [](const FragmentInfo& f1, const FragmentInfo& f2) { return f1.numTuples < f2.numTuples; });
    max_groups_buffer_entry_guess *= it->numTuples;
  }
  return max_groups_buffer_entry_guess;
}

std::unordered_set<int> get_available_gpus(const Catalog_Namespace::Catalog& cat) {
  std::unordered_set<int> available_gpus;
  if (cat.get_dataMgr().gpusPresent()) {
    int gpu_count = cat.get_dataMgr().cudaMgr_->getDeviceCount();
    CHECK_GT(gpu_count, 0);
    for (int gpu_id = 0; gpu_id < gpu_count; ++gpu_id) {
      available_gpus.insert(gpu_id);
    }
  }
  return available_gpus;
}

size_t get_context_count(const ExecutorDeviceType device_type, const size_t cpu_count, const size_t gpu_count) {
  return device_type == ExecutorDeviceType::GPU ? gpu_count : device_type == ExecutorDeviceType::Hybrid
                                                                  ? std::max(static_cast<size_t>(cpu_count), gpu_count)
                                                                  : static_cast<size_t>(cpu_count);
}

void checkWorkUnitWatchdog(const RelAlgExecutionUnit& ra_exe_unit) {
  for (const auto target_expr : ra_exe_unit.target_exprs) {
    if (dynamic_cast<const Analyzer::AggExpr*>(target_expr)) {
      return;
    }
  }
  if (ra_exe_unit.groupby_exprs.size() == 1 && !ra_exe_unit.groupby_exprs.front() && !ra_exe_unit.scan_limit) {
    throw WatchdogException("Query would require a scan without a limit");
  }
}

}  // namespace

ResultRows Executor::executeWorkUnit(int32_t* error_code,
                                     size_t& max_groups_buffer_entry_guess,
                                     const bool is_agg,
                                     const std::vector<Fragmenter_Namespace::TableInfo>& query_infos,
                                     const RelAlgExecutionUnit& ra_exe_unit,
                                     const CompilationOptions& co,
                                     const ExecutionOptions& options,
                                     const Catalog_Namespace::Catalog& cat,
                                     std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                                     RenderAllocatorMap* render_allocator_map) {
  const auto device_type = getDeviceTypeForTargets(ra_exe_unit, co.device_type_);
  CHECK(!query_infos.empty());
  if (!max_groups_buffer_entry_guess) {
    // The query has failed the first execution attempt because of running out
    // of group by slots. Make the conservative choice: allocate fragment size
    // slots and run on the CPU.
    CHECK(device_type == ExecutorDeviceType::CPU);
    max_groups_buffer_entry_guess = compute_buffer_entry_guess(query_infos);
  }

  auto join_info = chooseJoinType(ra_exe_unit.inner_join_quals, query_infos, device_type);
  if (join_info.join_impl_type_ == JoinImplType::Loop) {
    join_info = chooseJoinType(ra_exe_unit.outer_join_quals, query_infos, device_type);
  }

  int8_t crt_min_byte_width{get_min_byte_width()};
  do {
    *error_code = 0;
    // could use std::thread::hardware_concurrency(), but some
    // slightly out-of-date compilers (gcc 4.7) implement it as always 0.
    // Play it POSIX.1 safe instead.
    int available_cpus = cpu_threads();
    auto available_gpus = get_available_gpus(cat);

    const auto context_count = get_context_count(device_type, available_cpus, available_gpus.size());

    ExecutionDispatch execution_dispatch(this,
                                         ra_exe_unit,
                                         query_infos,
                                         cat,
                                         {device_type, co.hoist_literals_, co.opt_level_},
                                         context_count,
                                         row_set_mem_owner,
                                         error_code,
                                         render_allocator_map);
    crt_min_byte_width =
        execution_dispatch.compile(join_info, max_groups_buffer_entry_guess, crt_min_byte_width, options);

    if (options.just_explain) {
      return executeExplain(execution_dispatch);
    }

    for (const auto target_expr : ra_exe_unit.target_exprs) {
      plan_state_->target_exprs_.push_back(target_expr);
    }

    std::condition_variable scheduler_cv;
    std::mutex scheduler_mutex;
    auto dispatch = [this, &execution_dispatch, &available_cpus, &available_gpus, &scheduler_mutex, &scheduler_cv](
        const ExecutorDeviceType chosen_device_type,
        int chosen_device_id,
        const std::map<int, std::vector<size_t>>& frag_ids,
        const size_t ctx_idx,
        const int64_t rowid_lookup_key) {
      execution_dispatch.run(chosen_device_type, chosen_device_id, frag_ids, ctx_idx, rowid_lookup_key);
      if (execution_dispatch.getDeviceType() == ExecutorDeviceType::Hybrid) {
        std::unique_lock<std::mutex> scheduler_lock(scheduler_mutex);
        if (chosen_device_type == ExecutorDeviceType::CPU) {
          ++available_cpus;
        } else {
          CHECK(chosen_device_type == ExecutorDeviceType::GPU);
          auto it_ok = available_gpus.insert(chosen_device_id);
          CHECK(it_ok.second);
        }
        scheduler_lock.unlock();
        scheduler_cv.notify_one();
      }
    };

    std::map<int, const TableFragments*> all_tables_fragments;
    CHECK_EQ(query_infos.size(), ra_exe_unit.input_descs.size());
    for (size_t table_idx = 0; table_idx < ra_exe_unit.input_descs.size(); ++table_idx) {
      all_tables_fragments[ra_exe_unit.input_descs[table_idx].getTableId()] = &query_infos[table_idx].fragments;
    }
    const QueryMemoryDescriptor& query_mem_desc = execution_dispatch.getQueryMemoryDescriptor();
    dispatchFragments(dispatch,
                      execution_dispatch,
                      options,
                      is_agg,
                      all_tables_fragments,
                      context_count,
                      scheduler_cv,
                      scheduler_mutex,
                      available_gpus,
                      available_cpus);
    cat.get_dataMgr().freeAllBuffers();
    if (*error_code == ERR_OVERFLOW_OR_UNDERFLOW) {
      crt_min_byte_width <<= 1;
      continue;
    }
    if (is_agg) {
      try {
        return collectAllDeviceResults(execution_dispatch,
                                       ra_exe_unit.target_exprs,
                                       query_mem_desc,
                                       row_set_mem_owner,
                                       execution_dispatch.outputColumnar());
      } catch (ReductionRanOutOfSlots&) {
        *error_code = ERR_OUT_OF_SLOTS;
        return ResultRows(query_mem_desc,
                          plan_state_->target_exprs_,
                          nullptr,
                          {},
                          nullptr,
                          0,
                          false,
                          {},
                          execution_dispatch.getDeviceType(),
                          -1);
      } catch (OverflowOrUnderflow&) {
        crt_min_byte_width <<= 1;
        continue;
      }
    }
    return results_union(execution_dispatch.getFragmentResults());

  } while (static_cast<size_t>(crt_min_byte_width) <= sizeof(int64_t));

  return ResultRows({}, {}, nullptr, nullptr, {}, ExecutorDeviceType::CPU);
}

ResultRows Executor::executeExplain(const ExecutionDispatch& execution_dispatch) {
  std::string explained_plan;
  const auto llvm_ir_cpu = execution_dispatch.getIR(ExecutorDeviceType::CPU);
  if (!llvm_ir_cpu.empty()) {
    explained_plan += ("IR for the CPU:\n===============\n" + llvm_ir_cpu);
  }
  const auto llvm_ir_gpu = execution_dispatch.getIR(ExecutorDeviceType::GPU);
  if (!llvm_ir_gpu.empty()) {
    explained_plan +=
        (std::string(llvm_ir_cpu.empty() ? "" : "\n") + "IR for the GPU:\n===============\n" + llvm_ir_gpu);
  }
  return ResultRows(explained_plan);
}

// Looks at the targets and returns a feasible device type. We only punt
// to CPU for count distinct and we should probably fix it and remove this.
ExecutorDeviceType Executor::getDeviceTypeForTargets(const RelAlgExecutionUnit& ra_exe_unit,
                                                     const ExecutorDeviceType requested_device_type) {
  auto agg_fnames = get_agg_fnames(ra_exe_unit.target_exprs, !ra_exe_unit.groupby_exprs.empty());
  for (const auto& agg_name : agg_fnames) {
    // TODO(alex): count distinct can't be executed on the GPU yet, punt to CPU
    if (agg_name == "agg_count_distinct") {
      return ExecutorDeviceType::CPU;
    }
  }
  return requested_device_type;
}

Executor::ExecutionDispatch::ExecutionDispatch(Executor* executor,
                                               const RelAlgExecutionUnit& ra_exe_unit,
                                               const std::vector<Fragmenter_Namespace::TableInfo>& query_infos,
                                               const Catalog_Namespace::Catalog& cat,
                                               const CompilationOptions& co,
                                               const size_t context_count,
                                               const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                                               int32_t* error_code,
                                               RenderAllocatorMap* render_allocator_map)
    : executor_(executor),
      ra_exe_unit_(ra_exe_unit),
      query_infos_(query_infos),
      cat_(cat),
      co_(co),
      query_contexts_(context_count),
      query_context_mutexes_(context_count),
      row_set_mem_owner_(row_set_mem_owner),
      error_code_(error_code),
      render_allocator_map_(render_allocator_map) {
  all_frag_row_offsets_.resize(query_infos.front().fragments.size() + 1);
  for (size_t i = 1; i <= query_infos.front().fragments.size(); ++i) {
    all_frag_row_offsets_[i] = all_frag_row_offsets_[i - 1] + query_infos_.front().fragments[i - 1].numTuples;
  }
  all_fragment_results_.reserve(query_infos_.front().fragments.size());
}

void Executor::ExecutionDispatch::run(const ExecutorDeviceType chosen_device_type,
                                      int chosen_device_id,
                                      const std::map<int, std::vector<size_t>>& frag_ids,
                                      const size_t ctx_idx,
                                      const int64_t rowid_lookup_key) noexcept {
  static std::mutex reduce_mutex;
  const auto memory_level =
      chosen_device_type == ExecutorDeviceType::GPU ? Data_Namespace::GPU_LEVEL : Data_Namespace::CPU_LEVEL;
  std::vector<int64_t> num_rows;
  std::vector<uint64_t> dev_frag_row_offsets;
  const int outer_table_id = ra_exe_unit_.input_descs.front().getTableId();
  const auto outer_it = frag_ids.find(outer_table_id);
  CHECK(outer_it != frag_ids.end());
  for (const auto frag_id : outer_it->second) {
    const auto& outer_fragment = query_infos_.front().fragments[frag_id];
    if (co_.device_type_ != ExecutorDeviceType::Hybrid) {
      chosen_device_id = outer_fragment.deviceIds[static_cast<int>(memory_level)];
    }
    num_rows.push_back(outer_fragment.numTuples);
    if (ra_exe_unit_.input_descs.size() > 1) {
      for (size_t table_idx = 1; table_idx < ra_exe_unit_.input_descs.size(); ++table_idx) {
        const int inner_table_id = ra_exe_unit_.input_descs[table_idx].getTableId();
        const auto inner_it = frag_ids.find(inner_table_id);
        if (inner_it->second.empty()) {
          num_rows.push_back(0);
        }
        CHECK(inner_it != frag_ids.end());
        for (const auto inner_frag_id : inner_it->second) {
          const auto& inner_fragment = query_infos_[table_idx].fragments[inner_frag_id];
          num_rows.push_back(inner_fragment.numTuples);
        }
      }
    }
    dev_frag_row_offsets.push_back(all_frag_row_offsets_[frag_id]);
  }
  CHECK_GE(chosen_device_id, 0);
  CHECK_LT(chosen_device_id, max_gpu_count);
  // need to own them while query executes
  std::list<ChunkIter> chunk_iterators;
  std::list<std::shared_ptr<Chunk_NS::Chunk>> chunks;
  std::unique_ptr<std::lock_guard<std::mutex>> gpu_lock;
  if (chosen_device_type == ExecutorDeviceType::GPU) {
    gpu_lock.reset(new std::lock_guard<std::mutex>(executor_->gpu_exec_mutex_[chosen_device_id]));
  }
  std::vector<std::vector<const int8_t*>> col_buffers;
  try {
    std::map<int, const TableFragments*> all_tables_fragments;
    for (size_t table_idx = 0; table_idx < ra_exe_unit_.input_descs.size(); ++table_idx) {
      int table_id = ra_exe_unit_.input_descs[table_idx].getTableId();
      const auto& fragments = query_infos_[table_idx].fragments;
      auto it_ok = all_tables_fragments.insert(std::make_pair(table_id, &fragments));
      if (!it_ok.second) {
        std::lock_guard<std::mutex> lock(reduce_mutex);
        *error_code_ = ERR_UNSUPPORTED_SELF_JOIN;
        return;
      }
    }
    col_buffers = executor_->fetchChunks(*this,
                                         ra_exe_unit_.input_col_descs,
                                         chosen_device_id,
                                         memory_level,
                                         ra_exe_unit_.input_descs,
                                         all_tables_fragments,
                                         frag_ids,
                                         cat_,
                                         chunk_iterators,
                                         chunks);
  } catch (const OutOfMemory&) {
    std::lock_guard<std::mutex> lock(reduce_mutex);
    *error_code_ = ERR_OUT_OF_GPU_MEM;
    return;
  }
  CHECK(chosen_device_type != ExecutorDeviceType::Hybrid);
  const CompilationResult& compilation_result =
      chosen_device_type == ExecutorDeviceType::GPU ? compilation_result_gpu_ : compilation_result_cpu_;
  CHECK(!compilation_result.query_mem_desc.usesCachedContext() || !ra_exe_unit_.scan_limit);
  std::unique_ptr<QueryExecutionContext> query_exe_context_owned;
  try {
    query_exe_context_owned =
        compilation_result.query_mem_desc.usesCachedContext()
            ? nullptr
            : compilation_result.query_mem_desc.getQueryExecutionContext(executor_->plan_state_->init_agg_vals_,
                                                                         executor_,
                                                                         chosen_device_type,
                                                                         chosen_device_id,
                                                                         col_buffers,
                                                                         row_set_mem_owner_,
                                                                         compilation_result.output_columnar,
                                                                         compilation_result.query_mem_desc.sortOnGpu(),
                                                                         render_allocator_map_);
  } catch (const OutOfHostMemory& e) {
    LOG(ERROR) << e.what();
    *error_code_ = ERR_OUT_OF_CPU_MEM;
    return;
  }
  QueryExecutionContext* query_exe_context{query_exe_context_owned.get()};
  std::unique_ptr<std::lock_guard<std::mutex>> query_ctx_lock;
  if (compilation_result.query_mem_desc.usesCachedContext()) {
    query_ctx_lock.reset(new std::lock_guard<std::mutex>(query_context_mutexes_[ctx_idx]));
    if (!query_contexts_[ctx_idx]) {
      try {
        query_contexts_[ctx_idx] =
            compilation_result.query_mem_desc.getQueryExecutionContext(executor_->plan_state_->init_agg_vals_,
                                                                       executor_,
                                                                       chosen_device_type,
                                                                       chosen_device_id,
                                                                       col_buffers,
                                                                       row_set_mem_owner_,
                                                                       compilation_result.output_columnar,
                                                                       compilation_result.query_mem_desc.sortOnGpu(),
                                                                       render_allocator_map_);
      } catch (const OutOfHostMemory& e) {
        LOG(ERROR) << e.what();
        *error_code_ = ERR_OUT_OF_CPU_MEM;
        return;
      }
    }
    query_exe_context = query_contexts_[ctx_idx].get();
  }
  CHECK(query_exe_context);
  int32_t err{0};
  ResultRows device_results(
      compilation_result.query_mem_desc, {}, nullptr, {}, nullptr, 0, false, {}, chosen_device_type, chosen_device_id);
  uint32_t start_rowid{0};
  if (rowid_lookup_key >= 0) {
    CHECK_LE(frag_ids.size(), size_t(1));
    if (!frag_ids.empty()) {
      start_rowid = rowid_lookup_key - all_frag_row_offsets_[frag_ids.begin()->second.front()];
    }
  }
  if (ra_exe_unit_.groupby_exprs.empty()) {
    err = executor_->executePlanWithoutGroupBy(compilation_result,
                                               co_.hoist_literals_,
                                               device_results,
                                               ra_exe_unit_.target_exprs,
                                               chosen_device_type,
                                               col_buffers,
                                               query_exe_context,
                                               num_rows,
                                               dev_frag_row_offsets,
                                               &cat_.get_dataMgr(),
                                               chosen_device_id,
                                               start_rowid,
                                               ra_exe_unit_.input_descs.size(),
                                               render_allocator_map_);
  } else {
    err = executor_->executePlanWithGroupBy(compilation_result,
                                            co_.hoist_literals_,
                                            device_results,
                                            ra_exe_unit_.target_exprs,
                                            ra_exe_unit_.groupby_exprs.size(),
                                            chosen_device_type,
                                            col_buffers,
                                            query_exe_context,
                                            num_rows,
                                            dev_frag_row_offsets,
                                            &cat_.get_dataMgr(),
                                            chosen_device_id,
                                            ra_exe_unit_.scan_limit,
                                            co_.device_type_ == ExecutorDeviceType::Hybrid,
                                            start_rowid,
                                            ra_exe_unit_.input_descs.size(),
                                            render_allocator_map_);
  }
  {
    std::lock_guard<std::mutex> lock(reduce_mutex);
    if (err) {
      *error_code_ = err;
    }
    if (!device_results.definitelyHasNoRows()) {
      all_fragment_results_.emplace_back(device_results, frag_ids.begin()->second);
    }
  }
}

const int8_t* Executor::ExecutionDispatch::getColumn(const ResultRows* rows,
                                                     const int col_id,
                                                     const Data_Namespace::MemoryLevel memory_level,
                                                     const int device_id) const {
  CHECK(rows);
  static std::mutex columnar_conversion_mutex;
  {
    std::lock_guard<std::mutex> columnar_conversion_guard(columnar_conversion_mutex);
    if (!ra_node_input_) {
      ra_node_input_.reset(rows_to_columnar_results(rows));
    }
  }
  CHECK_GE(col_id, 0);
  return getColumn(ra_node_input_.get(), col_id, &cat_.get_dataMgr(), memory_level, device_id);
}

const int8_t* Executor::ExecutionDispatch::getColumn(const ColumnarResults* columnar_results,
                                                     const int col_id,
                                                     Data_Namespace::DataMgr* data_mgr,
                                                     const Data_Namespace::MemoryLevel memory_level,
                                                     const int device_id) {
  const auto& col_buffers = columnar_results->getColumnBuffers();
  CHECK_LT(static_cast<size_t>(col_id), col_buffers.size());
  if (memory_level == Data_Namespace::GPU_LEVEL) {
    const auto num_bytes = columnar_results->size() * get_bit_width(columnar_results->getColumnType(col_id)) * 8;
    auto gpu_col_buffer = alloc_gpu_mem(data_mgr, num_bytes, device_id, nullptr);
    copy_to_gpu(data_mgr, gpu_col_buffer, col_buffers[col_id], num_bytes, device_id);
    return reinterpret_cast<const int8_t*>(gpu_col_buffer);
  }
  return col_buffers[col_id];
}

int8_t Executor::ExecutionDispatch::compile(const Executor::JoinInfo& join_info,
                                            const size_t max_groups_buffer_entry_guess,
                                            const int8_t crt_min_byte_width,
                                            const ExecutionOptions& options) {
  int8_t actual_min_byte_wdith{MAX_BYTE_WIDTH_SUPPORTED};
  auto compile_on_cpu = [&]() {
    const CompilationOptions co_cpu{ExecutorDeviceType::CPU, co_.hoist_literals_, co_.opt_level_};
    try {
      compilation_result_cpu_ =
          executor_->compileWorkUnit(false,
                                     query_infos_,
                                     ra_exe_unit_,
                                     co_cpu,
                                     options,
                                     cat_.get_dataMgr().cudaMgr_,
                                     true,
                                     row_set_mem_owner_,
                                     max_groups_buffer_entry_guess,
                                     render_allocator_map_ ? executor_->render_small_groups_buffer_entry_count_
                                                           : executor_->small_groups_buffer_entry_count_,
                                     crt_min_byte_width,
                                     join_info);
    } catch (const CompilationRetryNoLazyFetch&) {
      compilation_result_cpu_ =
          executor_->compileWorkUnit(false,
                                     query_infos_,
                                     ra_exe_unit_,
                                     co_cpu,
                                     options,
                                     cat_.get_dataMgr().cudaMgr_,
                                     false,
                                     row_set_mem_owner_,
                                     max_groups_buffer_entry_guess,
                                     render_allocator_map_ ? executor_->render_small_groups_buffer_entry_count_
                                                           : executor_->small_groups_buffer_entry_count_,
                                     crt_min_byte_width,
                                     join_info);
    }
    for (auto wids : compilation_result_cpu_.query_mem_desc.agg_col_widths) {
      actual_min_byte_wdith = std::min(actual_min_byte_wdith, wids.compact);
    }
  };

  if (co_.device_type_ == ExecutorDeviceType::CPU || co_.device_type_ == ExecutorDeviceType::Hybrid) {
    compile_on_cpu();
  }

  if (co_.device_type_ == ExecutorDeviceType::GPU ||
      (co_.device_type_ == ExecutorDeviceType::Hybrid && cat_.get_dataMgr().gpusPresent())) {
    const CompilationOptions co_gpu{ExecutorDeviceType::GPU, co_.hoist_literals_, co_.opt_level_};
    try {
      compilation_result_gpu_ =
          executor_->compileWorkUnit(render_allocator_map_,
                                     query_infos_,
                                     ra_exe_unit_,
                                     co_gpu,
                                     options,
                                     cat_.get_dataMgr().cudaMgr_,
                                     render_allocator_map_ ? false : true,
                                     row_set_mem_owner_,
                                     max_groups_buffer_entry_guess,
                                     render_allocator_map_ ? executor_->render_small_groups_buffer_entry_count_
                                                           : executor_->small_groups_buffer_entry_count_,
                                     crt_min_byte_width,
                                     join_info);
    } catch (const CompilationRetryNoLazyFetch&) {
      compilation_result_gpu_ =
          executor_->compileWorkUnit(render_allocator_map_,
                                     query_infos_,
                                     ra_exe_unit_,
                                     co_gpu,
                                     options,
                                     cat_.get_dataMgr().cudaMgr_,
                                     false,
                                     row_set_mem_owner_,
                                     max_groups_buffer_entry_guess,
                                     render_allocator_map_ ? executor_->render_small_groups_buffer_entry_count_
                                                           : executor_->small_groups_buffer_entry_count_,
                                     crt_min_byte_width,
                                     join_info);
    }
    for (auto wids : compilation_result_gpu_.query_mem_desc.agg_col_widths) {
      actual_min_byte_wdith = std::min(actual_min_byte_wdith, wids.compact);
    }
  }

  if (executor_->cgen_state_->must_run_on_cpu_) {
    if (co_.device_type_ == ExecutorDeviceType::GPU) {  // override user choice
      compile_on_cpu();
    }
    co_.device_type_ = ExecutorDeviceType::CPU;
  }
  return std::max(actual_min_byte_wdith, crt_min_byte_width);
}

std::string Executor::ExecutionDispatch::getIR(const ExecutorDeviceType device_type) const {
  CHECK(device_type == ExecutorDeviceType::CPU || device_type == ExecutorDeviceType::GPU);
  if (device_type == ExecutorDeviceType::CPU) {
    return compilation_result_cpu_.llvm_ir;
  }
  return compilation_result_gpu_.llvm_ir;
}

ExecutorDeviceType Executor::ExecutionDispatch::getDeviceType() const {
  return co_.device_type_;
}

const RelAlgExecutionUnit& Executor::ExecutionDispatch::getExecutionUnit() const {
  return ra_exe_unit_;
}

const QueryMemoryDescriptor& Executor::ExecutionDispatch::getQueryMemoryDescriptor() const {
  // TODO(alex): make query_mem_desc easily available
  return compilation_result_cpu_.native_functions.empty() ? compilation_result_gpu_.query_mem_desc
                                                          : compilation_result_cpu_.query_mem_desc;
}

const bool Executor::ExecutionDispatch::outputColumnar() const {
  return compilation_result_cpu_.native_functions.empty() ? compilation_result_gpu_.output_columnar : false;
}

const std::vector<uint64_t>& Executor::ExecutionDispatch::getFragOffsets() const {
  return all_frag_row_offsets_;
}

const std::vector<std::unique_ptr<QueryExecutionContext>>& Executor::ExecutionDispatch::getQueryContexts() const {
  return query_contexts_;
}

std::vector<std::pair<ResultRows, std::vector<size_t>>>& Executor::ExecutionDispatch::getFragmentResults() {
  return all_fragment_results_;
}

ResultRows Executor::collectAllDeviceResults(ExecutionDispatch& execution_dispatch,
                                             const std::vector<Analyzer::Expr*>& target_exprs,
                                             const QueryMemoryDescriptor& query_mem_desc,
                                             std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                                             const bool output_columnar) {
  for (const auto& query_exe_context : execution_dispatch.getQueryContexts()) {
    if (!query_exe_context) {
      continue;
    }
    execution_dispatch.getFragmentResults().emplace_back(
        query_exe_context->getRowSet(
            target_exprs, query_mem_desc, execution_dispatch.getDeviceType() == ExecutorDeviceType::Hybrid),
        std::vector<size_t>{});
  }
  return reduceMultiDeviceResults(
      execution_dispatch.getFragmentResults(), row_set_mem_owner, query_mem_desc, output_columnar);
}

void Executor::dispatchFragments(const std::function<void(const ExecutorDeviceType chosen_device_type,
                                                          int chosen_device_id,
                                                          const std::map<int, std::vector<size_t>>& frag_ids,
                                                          const size_t ctx_idx,
                                                          const int64_t rowid_lookup_key)> dispatch,
                                 const ExecutionDispatch& execution_dispatch,
                                 const ExecutionOptions& eo,
                                 const bool is_agg,
                                 const std::map<int, const TableFragments*>& all_tables_fragments,
                                 const size_t context_count,
                                 std::condition_variable& scheduler_cv,
                                 std::mutex& scheduler_mutex,
                                 std::unordered_set<int>& available_gpus,
                                 int& available_cpus) {
  size_t frag_list_idx{0};
  std::vector<std::thread> query_threads;
  int64_t rowid_lookup_key{-1};
  const auto& ra_exe_unit = execution_dispatch.getExecutionUnit();
  CHECK(!ra_exe_unit.input_descs.empty());
  const int outer_table_id = ra_exe_unit.input_descs.front().getTableId();
  auto it = all_tables_fragments.find(outer_table_id);
  CHECK(it != all_tables_fragments.end());
  const auto fragments = it->second;
  const auto device_type = execution_dispatch.getDeviceType();
  const auto& all_frag_row_offsets = execution_dispatch.getFragOffsets();

  const auto& query_mem_desc = execution_dispatch.getQueryMemoryDescriptor();
  const bool allow_multifrag =
      eo.allow_multifrag && (ra_exe_unit.groupby_exprs.empty() || query_mem_desc.usesCachedContext());

  if ((device_type == ExecutorDeviceType::GPU) && allow_multifrag && is_agg) {
    // NB: We should never be on this path when the query is retried because of
    //     running out of group by slots; also, for scan only queries (!agg_plan)
    //     we want the high-granularity, fragment by fragment execution instead.
    std::unordered_map<int, std::map<int, std::vector<size_t>>> fragments_per_device;
    for (size_t frag_id = 0; frag_id < fragments->size(); ++frag_id) {
      const auto& fragment = (*fragments)[frag_id];
      const auto skip_frag =
          skipFragment(outer_table_id, fragment, ra_exe_unit.simple_quals, all_frag_row_offsets, frag_id);
      if (skip_frag.first) {
        continue;
      }
      const int device_id = fragment.deviceIds[static_cast<int>(Data_Namespace::GPU_LEVEL)];
      for (const auto& inner_frags : all_tables_fragments) {
        if (inner_frags.first == outer_table_id) {
          fragments_per_device[device_id][inner_frags.first].push_back(frag_id);
        } else {
          std::vector<size_t> all_frag_ids(inner_frags.second->size());
          if (all_frag_ids.size() > 1) {
            throw std::runtime_error("Multi-fragment inner table not supported yet");
          }
          std::iota(all_frag_ids.begin(), all_frag_ids.end(), 0);
          fragments_per_device[device_id][inner_frags.first] = all_frag_ids;
        }
      }
      rowid_lookup_key = std::max(rowid_lookup_key, skip_frag.second);
    }
    if (eo.with_watchdog && rowid_lookup_key < 0) {
      checkWorkUnitWatchdog(ra_exe_unit);
    }
    for (const auto& kv : fragments_per_device) {
      query_threads.push_back(std::thread(
          dispatch, ExecutorDeviceType::GPU, kv.first, kv.second, kv.first % context_count, rowid_lookup_key));
    }
  } else {
    for (size_t i = 0; i < fragments->size(); ++i) {
      const auto skip_frag =
          skipFragment(outer_table_id, (*fragments)[i], ra_exe_unit.simple_quals, all_frag_row_offsets, i);
      if (skip_frag.first) {
        continue;
      }
      rowid_lookup_key = std::max(rowid_lookup_key, skip_frag.second);
      auto chosen_device_type = device_type;
      int chosen_device_id = 0;
      if (device_type == ExecutorDeviceType::Hybrid) {
        std::unique_lock<std::mutex> scheduler_lock(scheduler_mutex);
        scheduler_cv.wait(scheduler_lock, [this, &available_cpus, &available_gpus] {
          return available_cpus || !available_gpus.empty();
        });
        if (!available_gpus.empty()) {
          chosen_device_type = ExecutorDeviceType::GPU;
          auto device_id_it = available_gpus.begin();
          chosen_device_id = *device_id_it;
          available_gpus.erase(device_id_it);
        } else {
          chosen_device_type = ExecutorDeviceType::CPU;
          CHECK_GT(available_cpus, 0);
          --available_cpus;
        }
      }
      std::map<int, std::vector<size_t>> frag_ids_for_table;
      for (const auto& inner_frags : all_tables_fragments) {
        if (inner_frags.first == outer_table_id) {
          frag_ids_for_table[inner_frags.first] = {i};
        } else {
          std::vector<size_t> all_frag_ids(inner_frags.second->size());
          if (all_frag_ids.size() > 1) {
            throw std::runtime_error("Multi-fragment inner table not supported yet");
          }
          std::iota(all_frag_ids.begin(), all_frag_ids.end(), 0);
          frag_ids_for_table[inner_frags.first] = all_frag_ids;
        }
      }
      if (eo.with_watchdog && rowid_lookup_key < 0) {
        checkWorkUnitWatchdog(ra_exe_unit);
      }
      query_threads.push_back(std::thread(dispatch,
                                          chosen_device_type,
                                          chosen_device_id,
                                          frag_ids_for_table,
                                          frag_list_idx % context_count,
                                          rowid_lookup_key));
      ++frag_list_idx;
    }
  }
  for (auto& child : query_threads) {
    child.join();
  }
}

std::vector<std::vector<const int8_t*>> Executor::fetchChunks(
    const ExecutionDispatch& execution_dispatch,
    const std::list<InputColDescriptor>& col_global_ids,
    const int device_id,
    const Data_Namespace::MemoryLevel memory_level,
    const std::vector<InputDescriptor>& input_descs,
    const std::map<int, const TableFragments*>& all_tables_fragments,
    const std::map<int, std::vector<size_t>>& selected_fragments,
    const Catalog_Namespace::Catalog& cat,
    std::list<ChunkIter>& chunk_iterators,
    std::list<std::shared_ptr<Chunk_NS::Chunk>>& chunks) {
  static std::mutex str_dec_mutex;  // TODO(alex): remove

  CHECK_EQ(all_tables_fragments.size(), selected_fragments.size());

  std::vector<std::vector<size_t>> selected_fragments_crossjoin;
  std::vector<size_t> local_col_to_frag_pos;
  buildSelectedFragsMapping(
      selected_fragments_crossjoin, local_col_to_frag_pos, col_global_ids, selected_fragments, input_descs);

  CartesianProduct<std::vector<std::vector<size_t>>> frag_ids_crossjoin(selected_fragments_crossjoin);

  std::vector<std::vector<const int8_t*>> all_frag_col_buffers;

  for (const auto& selected_frag_ids : frag_ids_crossjoin) {
    std::vector<const int8_t*> frag_col_buffers(plan_state_->global_to_local_col_ids_.size());
    for (const auto& col_id : col_global_ids) {
      const int table_id = col_id.getScanDesc().getTableId();
      const auto cd = get_column_descriptor_maybe(col_id.getColId(), table_id, cat);
      if (cd && cd->isVirtualCol) {
        CHECK_EQ("rowid", cd->columnName);
        continue;
      }
      const auto fragments_it = all_tables_fragments.find(table_id);
      CHECK(fragments_it != all_tables_fragments.end());
      const auto fragments = fragments_it->second;
      auto it = plan_state_->global_to_local_col_ids_.find(col_id);
      CHECK(it != plan_state_->global_to_local_col_ids_.end());
      CHECK_LT(static_cast<size_t>(it->second), plan_state_->global_to_local_col_ids_.size());
      const size_t frag_id = selected_frag_ids[local_col_to_frag_pos[it->second]];
      CHECK_LT(frag_id, fragments->size());
      const auto& fragment = (*fragments)[frag_id];
      auto memory_level_for_column = memory_level;
      if (plan_state_->columns_to_fetch_.find(col_id.getColId()) == plan_state_->columns_to_fetch_.end()) {
        memory_level_for_column = Data_Namespace::CPU_LEVEL;
      }
      std::shared_ptr<Chunk_NS::Chunk> chunk;
      auto chunk_meta_it = fragment.chunkMetadataMap.find(col_id.getColId());
      if (cd) {
        CHECK(chunk_meta_it != fragment.chunkMetadataMap.end());
        ChunkKey chunk_key{cat.get_currentDB().dbId, table_id, col_id.getColId(), fragment.fragmentId};
        std::lock_guard<std::mutex> lock(str_dec_mutex);
        chunk = Chunk_NS::Chunk::getChunk(cd,
                                          &cat.get_dataMgr(),
                                          chunk_key,
                                          memory_level_for_column,
                                          memory_level_for_column == Data_Namespace::CPU_LEVEL ? 0 : device_id,
                                          chunk_meta_it->second.numBytes,
                                          chunk_meta_it->second.numElements);
        chunks.push_back(chunk);
      }
      const auto col_type = get_column_type(col_id.getColId(), table_id, cd, temporary_tables_);
      const bool is_real_string = col_type.is_string() && col_type.get_compression() == kENCODING_NONE;
      if (is_real_string || col_type.is_array()) {
        CHECK_GT(table_id, 0);
        CHECK(chunk_meta_it != fragment.chunkMetadataMap.end());
        chunk_iterators.push_back(chunk->begin_iterator(chunk_meta_it->second));
        auto& chunk_iter = chunk_iterators.back();
        if (memory_level_for_column == Data_Namespace::CPU_LEVEL) {
          frag_col_buffers[it->second] = reinterpret_cast<int8_t*>(&chunk_iter);
        } else {
          CHECK_EQ(Data_Namespace::GPU_LEVEL, memory_level_for_column);
          auto& data_mgr = cat.get_dataMgr();
          auto chunk_iter_gpu = alloc_gpu_mem(&data_mgr, sizeof(ChunkIter), device_id, nullptr);
          copy_to_gpu(&data_mgr, chunk_iter_gpu, &chunk_iter, sizeof(ChunkIter), device_id);
          frag_col_buffers[it->second] = reinterpret_cast<int8_t*>(chunk_iter_gpu);
        }
      } else {
        const bool input_is_result = col_id.getScanDesc().getSourceType() == InputSourceType::RESULT;
        CHECK_NE(input_is_result, static_cast<bool>(chunk));
        if (input_is_result) {
          CHECK_EQ(size_t(0), frag_id);
          frag_col_buffers[it->second] = execution_dispatch.getColumn(
              get_temporary_table(temporary_tables_, table_id), col_id.getColId(), memory_level_for_column, device_id);
        } else {
          auto ab = chunk->get_buffer();
          CHECK(ab->getMemoryPtr());
          frag_col_buffers[it->second] = ab->getMemoryPtr();  // @TODO(alex) change to use ChunkIter
        }
      }
    }
    all_frag_col_buffers.push_back(frag_col_buffers);
  }
  return all_frag_col_buffers;
}

void Executor::buildSelectedFragsMapping(std::vector<std::vector<size_t>>& selected_fragments_crossjoin,
                                         std::vector<size_t>& local_col_to_frag_pos,
                                         const std::list<InputColDescriptor>& col_global_ids,
                                         const std::map<int, std::vector<size_t>>& selected_fragments,
                                         const std::vector<InputDescriptor>& input_descs) {
  local_col_to_frag_pos.resize(plan_state_->global_to_local_col_ids_.size());
  size_t frag_pos{0};
  for (size_t scan_idx = 0; scan_idx < input_descs.size(); ++scan_idx) {
    const int table_id = input_descs[scan_idx].getTableId();
    const auto selected_fragments_it = selected_fragments.find(table_id);
    CHECK(selected_fragments_it != selected_fragments.end());
    selected_fragments_crossjoin.push_back(selected_fragments_it->second);
    for (const auto& col_id : col_global_ids) {
      const auto& input_desc = col_id.getScanDesc();
      if (input_desc.getTableId() != table_id || input_desc.getNestLevel() != static_cast<int>(scan_idx)) {
        continue;
      }
      auto it = plan_state_->global_to_local_col_ids_.find(col_id);
      CHECK(it != plan_state_->global_to_local_col_ids_.end());
      CHECK_LT(static_cast<size_t>(it->second), plan_state_->global_to_local_col_ids_.size());
      local_col_to_frag_pos[it->second] = frag_pos;
    }
    ++frag_pos;
  }
}

namespace {

class OutVecOwner {
 public:
  OutVecOwner(const std::vector<int64_t*>& out_vec) : out_vec_(out_vec) {}
  ~OutVecOwner() {
    for (auto out : out_vec_) {
      delete[] out;
    }
  }

 private:
  std::vector<int64_t*> out_vec_;
};

}  // namespace

int32_t Executor::executePlanWithoutGroupBy(const CompilationResult& compilation_result,
                                            const bool hoist_literals,
                                            ResultRows& results,
                                            const std::vector<Analyzer::Expr*>& target_exprs,
                                            const ExecutorDeviceType device_type,
                                            std::vector<std::vector<const int8_t*>>& col_buffers,
                                            const QueryExecutionContext* query_exe_context,
                                            const std::vector<int64_t>& num_rows,
                                            const std::vector<uint64_t>& dev_frag_row_offsets,
                                            Data_Namespace::DataMgr* data_mgr,
                                            const int device_id,
                                            const uint32_t start_rowid,
                                            const uint32_t num_tables,
                                            RenderAllocatorMap* render_allocator_map) noexcept {
  if (col_buffers.empty()) {
    results = ResultRows({}, {}, nullptr, nullptr, {}, device_type);
    return 0;
  }
  int32_t error_code = device_type == ExecutorDeviceType::GPU ? 0 : start_rowid;
  std::vector<int64_t*> out_vec;
  const auto hoist_buf = serializeLiterals(compilation_result.literal_values, device_id);
  const auto join_hash_table_ptr = getJoinHashTablePtr(device_type, device_id);
  std::unique_ptr<OutVecOwner> output_memory_scope;
  if (device_type == ExecutorDeviceType::CPU) {
    out_vec = launch_query_cpu_code(compilation_result.native_functions,
                                    hoist_literals,
                                    hoist_buf,
                                    col_buffers,
                                    num_rows,
                                    dev_frag_row_offsets,
                                    0,
                                    query_exe_context->query_mem_desc_,
                                    query_exe_context->init_agg_vals_,
                                    {},
                                    {},
                                    &error_code,
                                    num_tables,
                                    join_hash_table_ptr);
    output_memory_scope.reset(new OutVecOwner(out_vec));
  } else {
    try {
      out_vec = query_exe_context->launchGpuCode(compilation_result.native_functions,
                                                 hoist_literals,
                                                 hoist_buf,
                                                 col_buffers,
                                                 num_rows,
                                                 dev_frag_row_offsets,
                                                 0,
                                                 query_exe_context->init_agg_vals_,
                                                 data_mgr,
                                                 blockSize(),
                                                 gridSize(),
                                                 device_id,
                                                 &error_code,
                                                 num_tables,
                                                 join_hash_table_ptr,
                                                 render_allocator_map);
      output_memory_scope.reset(new OutVecOwner(out_vec));
    } catch (const OutOfMemory&) {
      return ERR_OUT_OF_GPU_MEM;
    } catch (const std::exception& e) {
      LOG(FATAL) << "Error launching the GPU kernel: " << e.what();
    }
  }
  if (error_code == Executor::ERR_OVERFLOW_OR_UNDERFLOW || error_code == Executor::ERR_DIV_BY_ZERO) {
    return error_code;
  }
  results = ResultRows(query_exe_context->query_mem_desc_,
                       target_exprs,
                       this,
                       query_exe_context->row_set_mem_owner_,
                       query_exe_context->init_agg_vals_,
                       device_type);
  results.beginRow();
  size_t out_vec_idx = 0;
  for (const auto target_expr : target_exprs) {
    const auto agg_info = target_info(target_expr);
    CHECK(agg_info.is_agg);
    uint32_t num_fragments = col_buffers.size();
    int64_t val1;
    std::tie(val1, error_code) =
        reduceResults(agg_info.agg_kind,
                      agg_info.sql_type,
                      query_exe_context->init_agg_vals_[out_vec_idx],
                      query_exe_context->query_mem_desc_.agg_col_widths[out_vec_idx].compact,
                      out_vec[out_vec_idx],
                      device_type == ExecutorDeviceType::GPU ? num_fragments * blockSize() * gridSize() : 1,
                      false);
    if (error_code) {
      break;
    }
    if (agg_info.agg_kind == kAVG) {
      ++out_vec_idx;
      int64_t val2;
      std::tie(val2, error_code) =
          reduceResults(kCOUNT,
                        agg_info.sql_type,
                        query_exe_context->init_agg_vals_[out_vec_idx],
                        query_exe_context->query_mem_desc_.agg_col_widths[out_vec_idx].compact,
                        out_vec[out_vec_idx],
                        device_type == ExecutorDeviceType::GPU ? num_fragments * blockSize() * gridSize() : 1,
                        false);
      if (error_code) {
        break;
      }
      results.addValue(val1, val2);
    } else {
      results.addValue(val1);
    }
    ++out_vec_idx;
  }
  return error_code;
}

int32_t Executor::executePlanWithGroupBy(const CompilationResult& compilation_result,
                                         const bool hoist_literals,
                                         ResultRows& results,
                                         const std::vector<Analyzer::Expr*>& target_exprs,
                                         const size_t group_by_col_count,
                                         const ExecutorDeviceType device_type,
                                         std::vector<std::vector<const int8_t*>>& col_buffers,
                                         const QueryExecutionContext* query_exe_context,
                                         const std::vector<int64_t>& num_rows,
                                         const std::vector<uint64_t>& dev_frag_row_offsets,
                                         Data_Namespace::DataMgr* data_mgr,
                                         const int device_id,
                                         const int64_t scan_limit,
                                         const bool was_auto_device,
                                         const uint32_t start_rowid,
                                         const uint32_t num_tables,
                                         RenderAllocatorMap* render_allocator_map) noexcept {
  CHECK_NE(group_by_col_count, size_t(0));
  // TODO(alex):
  // 1. Optimize size (make keys more compact).
  // 2. Resize on overflow.
  // 3. Optimize runtime.
  const auto hoist_buf = serializeLiterals(compilation_result.literal_values, device_id);
  int32_t error_code = device_type == ExecutorDeviceType::GPU ? 0 : start_rowid;
  const auto join_hash_table_ptr = getJoinHashTablePtr(device_type, device_id);
  if (device_type == ExecutorDeviceType::CPU) {
    launch_query_cpu_code(compilation_result.native_functions,
                          hoist_literals,
                          hoist_buf,
                          col_buffers,
                          num_rows,
                          dev_frag_row_offsets,
                          scan_limit,
                          query_exe_context->query_mem_desc_,
                          query_exe_context->init_agg_vals_,
                          query_exe_context->group_by_buffers_,
                          query_exe_context->small_group_by_buffers_,
                          &error_code,
                          num_tables,
                          join_hash_table_ptr);
  } else {
    try {
      query_exe_context->launchGpuCode(compilation_result.native_functions,
                                       hoist_literals,
                                       hoist_buf,
                                       col_buffers,
                                       num_rows,
                                       dev_frag_row_offsets,
                                       scan_limit,
                                       query_exe_context->init_agg_vals_,
                                       data_mgr,
                                       blockSize(),
                                       gridSize(),
                                       device_id,
                                       &error_code,
                                       num_tables,
                                       join_hash_table_ptr,
                                       render_allocator_map);
    } catch (const OutOfMemory&) {
      return ERR_OUT_OF_GPU_MEM;
    } catch (const OutOfRenderMemory&) {
      return ERR_OUT_OF_RENDER_MEM;
    } catch (const std::exception& e) {
      LOG(FATAL) << "Error launching the GPU kernel: " << e.what();
    }
  }
  if (error_code != Executor::ERR_OVERFLOW_OR_UNDERFLOW && error_code != Executor::ERR_DIV_BY_ZERO &&
      !query_exe_context->query_mem_desc_.usesCachedContext() && !render_allocator_map) {
    CHECK(!query_exe_context->query_mem_desc_.sortOnGpu());
    results = query_exe_context->getRowSet(target_exprs, query_exe_context->query_mem_desc_, was_auto_device);
  }
  if (error_code && (render_allocator_map || (!scan_limit || results.rowCount() < static_cast<size_t>(scan_limit)))) {
    return error_code;  // unlucky, not enough results and we ran out of slots
  }
  return 0;
}

int64_t Executor::getJoinHashTablePtr(const ExecutorDeviceType device_type, const int device_id) {
  const auto join_hash_table = plan_state_->join_info_.join_hash_table_;
  if (!join_hash_table) {
    return 0;
  }
  return join_hash_table->getJoinHashBuffer(device_type, device_type == ExecutorDeviceType::GPU ? device_id : 0);
}

void Executor::executeSimpleInsert(const Planner::RootPlan* root_plan) {
  const auto plan = root_plan->get_plan();
  CHECK(plan);
  const auto values_plan = dynamic_cast<const Planner::ValuesScan*>(plan);
  CHECK(values_plan);
  const auto& targets = values_plan->get_targetlist();
  const int table_id = root_plan->get_result_table_id();
  const auto& col_id_list = root_plan->get_result_col_list();
  std::vector<const ColumnDescriptor*> col_descriptors;
  std::vector<int> col_ids;
  std::unordered_map<int, int8_t*> col_buffers;
  std::unordered_map<int, std::vector<std::string>> str_col_buffers;
  auto& cat = root_plan->get_catalog();
  for (const int col_id : col_id_list) {
    const auto cd = get_column_descriptor(col_id, table_id, cat);
    const auto col_enc = cd->columnType.get_compression();
    if (cd->columnType.is_string()) {
      switch (col_enc) {
        case kENCODING_NONE: {
          auto it_ok = str_col_buffers.insert(std::make_pair(col_id, std::vector<std::string>{}));
          CHECK(it_ok.second);
          break;
        }
        case kENCODING_DICT: {
          const auto dd = cat.getMetadataForDict(cd->columnType.get_comp_param());
          CHECK(dd);
          auto it_ok = col_buffers.insert(std::make_pair(col_id, nullptr));
          CHECK(it_ok.second);
          break;
        }
        default:
          CHECK(false);
      }
    } else {
      auto it_ok = col_buffers.insert(std::make_pair(col_id, nullptr));
      CHECK(it_ok.second);
    }
    col_descriptors.push_back(cd);
    col_ids.push_back(col_id);
  }
  size_t col_idx = 0;
  Fragmenter_Namespace::InsertData insert_data;
  insert_data.databaseId = cat.get_currentDB().dbId;
  insert_data.tableId = table_id;
  for (auto target_entry : targets) {
    auto col_cv = dynamic_cast<const Analyzer::Constant*>(target_entry->get_expr());
    if (!col_cv) {
      auto col_cast = dynamic_cast<const Analyzer::UOper*>(target_entry->get_expr());
      CHECK(col_cast);
      CHECK_EQ(kCAST, col_cast->get_optype());
      col_cv = dynamic_cast<const Analyzer::Constant*>(col_cast->get_operand());
    }
    CHECK(col_cv);
    const auto cd = col_descriptors[col_idx];
    auto col_datum = col_cv->get_constval();
    auto col_type = cd->columnType.is_decimal() ? decimal_to_int_type(cd->columnType) : cd->columnType.get_type();
    switch (col_type) {
      case kBOOLEAN: {
        auto col_data = reinterpret_cast<int8_t*>(checked_malloc(sizeof(int8_t)));
        *col_data = col_cv->get_is_null() ? inline_int_null_val(cd->columnType) : (col_datum.boolval ? 1 : 0);
        col_buffers[col_ids[col_idx]] = col_data;
        break;
      }
      case kSMALLINT: {
        auto col_data = reinterpret_cast<int16_t*>(checked_malloc(sizeof(int16_t)));
        *col_data = col_cv->get_is_null() ? inline_int_null_val(cd->columnType) : col_datum.smallintval;
        col_buffers[col_ids[col_idx]] = reinterpret_cast<int8_t*>(col_data);
        break;
      }
      case kINT: {
        auto col_data = reinterpret_cast<int32_t*>(checked_malloc(sizeof(int32_t)));
        *col_data = col_cv->get_is_null() ? inline_int_null_val(cd->columnType) : col_datum.intval;
        col_buffers[col_ids[col_idx]] = reinterpret_cast<int8_t*>(col_data);
        break;
      }
      case kBIGINT: {
        auto col_data = reinterpret_cast<int64_t*>(checked_malloc(sizeof(int64_t)));
        *col_data = col_cv->get_is_null() ? inline_int_null_val(cd->columnType) : col_datum.bigintval;
        col_buffers[col_ids[col_idx]] = reinterpret_cast<int8_t*>(col_data);
        break;
      }
      case kFLOAT: {
        auto col_data = reinterpret_cast<float*>(checked_malloc(sizeof(float)));
        *col_data = col_datum.floatval;
        col_buffers[col_ids[col_idx]] = reinterpret_cast<int8_t*>(col_data);
        break;
      }
      case kDOUBLE: {
        auto col_data = reinterpret_cast<double*>(checked_malloc(sizeof(double)));
        *col_data = col_datum.doubleval;
        col_buffers[col_ids[col_idx]] = reinterpret_cast<int8_t*>(col_data);
        break;
      }
      case kTEXT:
      case kVARCHAR:
      case kCHAR: {
        switch (cd->columnType.get_compression()) {
          case kENCODING_NONE:
            str_col_buffers[col_ids[col_idx]].push_back(col_datum.stringval ? *col_datum.stringval : "");
            break;
          case kENCODING_DICT: {
            auto col_data = reinterpret_cast<int32_t*>(checked_malloc(sizeof(int32_t)));
            if (col_cv->get_is_null()) {
              *col_data = NULL_INT;
            } else {
              const int dict_id = cd->columnType.get_comp_param();
              const int32_t str_id = getStringDictionary(dict_id, row_set_mem_owner_)->getOrAdd(*col_datum.stringval);
              *col_data = str_id;
            }
            col_buffers[col_ids[col_idx]] = reinterpret_cast<int8_t*>(col_data);
            break;
          }
          default:
            CHECK(false);
        }
        break;
      }
      case kTIME:
      case kTIMESTAMP:
      case kDATE: {
        auto col_data = reinterpret_cast<time_t*>(checked_malloc(sizeof(time_t)));
        *col_data = col_datum.timeval;
        col_buffers[col_ids[col_idx]] = reinterpret_cast<int8_t*>(col_data);
        break;
      }
      default:
        CHECK(false);
    }
    ++col_idx;
  }
  for (const auto kv : col_buffers) {
    insert_data.columnIds.push_back(kv.first);
    DataBlockPtr p;
    p.numbersPtr = kv.second;
    insert_data.data.push_back(p);
  }
  for (auto& kv : str_col_buffers) {
    insert_data.columnIds.push_back(kv.first);
    DataBlockPtr p;
    p.stringsPtr = &kv.second;
    insert_data.data.push_back(p);
  }
  insert_data.numRows = 1;
  const auto table_descriptor = cat.getMetadataForTable(table_id);
  table_descriptor->fragmenter->insertData(insert_data);
  cat.get_dataMgr().checkpoint();
  for (const auto kv : col_buffers) {
    free(kv.second);
  }
}

namespace {

llvm::Module* read_template_module(llvm::LLVMContext& context) {
  llvm::SMDiagnostic err;

  auto buffer_or_error = llvm::MemoryBuffer::getFile(mapd_root_abs_path() + "/QueryEngine/RuntimeFunctions.bc");
  CHECK(!buffer_or_error.getError());
  llvm::MemoryBuffer* buffer = buffer_or_error.get().get();
#if LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR == 5
  auto module = llvm::parseBitcodeFile(buffer, context).get();
#else
  auto owner = llvm::parseBitcodeFile(buffer->getMemBufferRef(), context);
  CHECK(!owner.getError());
  auto module = owner.get().release();
#endif
  CHECK(module);

  return module;
}

void bind_pos_placeholders(const std::string& pos_fn_name,
                           const bool use_resume_param,
                           llvm::Function* query_func,
                           llvm::Module* module) {
  for (auto it = llvm::inst_begin(query_func), e = llvm::inst_end(query_func); it != e; ++it) {
    if (!llvm::isa<llvm::CallInst>(*it)) {
      continue;
    }
    auto& pos_call = llvm::cast<llvm::CallInst>(*it);
    if (std::string(pos_call.getCalledFunction()->getName()) == pos_fn_name) {
      if (use_resume_param) {
        auto& resume_param = query_func->getArgumentList().back();
        llvm::ReplaceInstWithInst(&pos_call,
                                  llvm::CallInst::Create(module->getFunction(pos_fn_name + "_impl"),
                                                         std::vector<llvm::Value*>{&resume_param}));
      } else {
        llvm::ReplaceInstWithInst(&pos_call, llvm::CallInst::Create(module->getFunction(pos_fn_name + "_impl")));
      }
      break;
    }
  }
}

std::vector<llvm::Value*> generate_column_heads_load(const int num_columns,
                                                     llvm::Function* query_func,
                                                     llvm::LLVMContext& context) {
  auto max_col_local_id = num_columns - 1;
  auto& fetch_bb = query_func->front();
  llvm::IRBuilder<> fetch_ir_builder(&fetch_bb);
  fetch_ir_builder.SetInsertPoint(fetch_bb.begin());
  auto& in_arg_list = query_func->getArgumentList();
  CHECK_GE(in_arg_list.size(), size_t(4));
  auto& byte_stream_arg = in_arg_list.front();
  std::vector<llvm::Value*> col_heads;
  for (int col_id = 0; col_id <= max_col_local_id; ++col_id) {
    col_heads.emplace_back(fetch_ir_builder.CreateLoad(
        fetch_ir_builder.CreateGEP(&byte_stream_arg, llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), col_id))));
  }
  return col_heads;
}

void set_row_func_argnames(llvm::Function* row_func,
                           const size_t in_col_count,
                           const size_t agg_col_count,
                           const bool hoist_literals) {
  auto arg_it = row_func->arg_begin();

  if (agg_col_count) {
    for (size_t i = 0; i < agg_col_count; ++i) {
      arg_it->setName("out");
      ++arg_it;
    }
  } else {
    arg_it->setName("group_by_buff");
    ++arg_it;
    arg_it->setName("small_group_by_buff");
    ++arg_it;
    arg_it->setName("crt_match");
    ++arg_it;
    arg_it->setName("total_matched");
    ++arg_it;
    arg_it->setName("old_total_matched");
    ++arg_it;
  }

  arg_it->setName("agg_init_val");
  ++arg_it;

  arg_it->setName("pos");
  ++arg_it;

  arg_it->setName("frag_row_off");
  ++arg_it;

  arg_it->setName("num_rows_per_scan");
  ++arg_it;

  if (hoist_literals) {
    arg_it->setName("literals");
    ++arg_it;
  }

  for (size_t i = 0; i < in_col_count; ++i) {
    arg_it->setName("col_buf");
    ++arg_it;
  }

  arg_it->setName("join_hash_table");
}

std::pair<llvm::Function*, std::vector<llvm::Value*>> create_row_function(const size_t in_col_count,
                                                                          const size_t agg_col_count,
                                                                          const bool hoist_literals,
                                                                          llvm::Function* query_func,
                                                                          llvm::Module* module,
                                                                          llvm::LLVMContext& context) {
  std::vector<llvm::Type*> row_process_arg_types;

  if (agg_col_count) {
    // output (aggregate) arguments
    for (size_t i = 0; i < agg_col_count; ++i) {
      row_process_arg_types.push_back(llvm::Type::getInt64PtrTy(context));
    }
  } else {
    // group by buffer
    row_process_arg_types.push_back(llvm::Type::getInt64PtrTy(context));
    // small group by buffer
    row_process_arg_types.push_back(llvm::Type::getInt64PtrTy(context));
    // current match count
    row_process_arg_types.push_back(llvm::Type::getInt32PtrTy(context));
    // total match count passed from the caller
    row_process_arg_types.push_back(llvm::Type::getInt32PtrTy(context));
    // old total match count returned to the caller
    row_process_arg_types.push_back(llvm::Type::getInt32PtrTy(context));
  }

  // aggregate init values
  row_process_arg_types.push_back(llvm::Type::getInt64PtrTy(context));

  // position argument
  row_process_arg_types.push_back(llvm::Type::getInt64Ty(context));

  // fragment row offset argument
  row_process_arg_types.push_back(llvm::Type::getInt64Ty(context));

  // number of rows for each scan
  row_process_arg_types.push_back(llvm::Type::getInt64PtrTy(context));

  // literals buffer argument
  if (hoist_literals) {
    row_process_arg_types.push_back(llvm::Type::getInt8PtrTy(context));
  }

  // Generate the function signature and column head fetches s.t.
  // double indirection isn't needed in the inner loop
  auto col_heads = generate_column_heads_load(in_col_count, query_func, context);
  CHECK_EQ(in_col_count, col_heads.size());

  // column buffer arguments
  for (size_t i = 0; i < in_col_count; ++i) {
    row_process_arg_types.emplace_back(llvm::Type::getInt8PtrTy(context));
  }

  // join hash table argument
  row_process_arg_types.push_back(llvm::Type::getInt64Ty(context));

  // generate the function
  auto ft = llvm::FunctionType::get(get_int_type(32, context), row_process_arg_types, false);

  auto row_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "row_func", module);

  // set the row function argument names; for debugging purposes only
  set_row_func_argnames(row_func, in_col_count, agg_col_count, hoist_literals);

  return std::make_pair(row_func, col_heads);
}

void bind_query(llvm::Function* query_func,
                const std::string& query_fname,
                llvm::Function* multifrag_query_func,
                llvm::Module* module) {
  std::vector<llvm::CallInst*> query_stubs;
  for (auto it = llvm::inst_begin(multifrag_query_func), e = llvm::inst_end(multifrag_query_func); it != e; ++it) {
    if (!llvm::isa<llvm::CallInst>(*it)) {
      continue;
    }
    auto& query_call = llvm::cast<llvm::CallInst>(*it);
    if (std::string(query_call.getCalledFunction()->getName()) == query_fname) {
      query_stubs.push_back(&query_call);
    }
  }
  for (auto& S : query_stubs) {
    std::vector<llvm::Value*> args;
    for (size_t i = 0; i < S->getNumArgOperands(); ++i) {
      args.push_back(S->getArgOperand(i));
    }
    llvm::ReplaceInstWithInst(S, llvm::CallInst::Create(query_func, args, ""));
  }
}

template <class T>
std::string serialize_llvm_object(const T* llvm_obj) {
  std::stringstream ss;
  llvm::raw_os_ostream os(ss);
  llvm_obj->print(os);
  os.flush();
  return ss.str();
}

bool should_defer_eval(const std::shared_ptr<Analyzer::Expr> expr) {
  if (std::dynamic_pointer_cast<Analyzer::LikeExpr>(expr)) {
    return true;
  }
  if (!std::dynamic_pointer_cast<Analyzer::BinOper>(expr)) {
    return false;
  }
  const auto bin_expr = std::static_pointer_cast<Analyzer::BinOper>(expr);
  const auto rhs = bin_expr->get_right_operand();
  return rhs->get_type_info().is_array();
}

void verify_function_ir(const llvm::Function* func) {
  std::stringstream err_ss;
  llvm::raw_os_ostream err_os(err_ss);
  if (llvm::verifyFunction(*func, &err_os)) {
    func->dump();
    LOG(FATAL) << err_ss.str();
  }
}

}  // namespace

void Executor::nukeOldState(const bool allow_lazy_fetch, const JoinInfo& join_info) {
  cgen_state_.reset(new CgenState());
  plan_state_.reset(new PlanState(allow_lazy_fetch, join_info, this));
}

Executor::CompilationResult Executor::compileWorkUnit(const bool render_output,
                                                      const std::vector<Fragmenter_Namespace::TableInfo>& query_infos,
                                                      const RelAlgExecutionUnit& ra_exe_unit,
                                                      const CompilationOptions& co,
                                                      const ExecutionOptions& eo,
                                                      const CudaMgr_Namespace::CudaMgr* cuda_mgr,
                                                      const bool allow_lazy_fetch,
                                                      std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                                                      const size_t max_groups_buffer_entry_guess,
                                                      const size_t small_groups_buffer_entry_count,
                                                      const int8_t crt_min_byte_width,
                                                      const JoinInfo& join_info) {
  nukeOldState(allow_lazy_fetch && ra_exe_unit.outer_join_quals.empty(), join_info);

  GroupByAndAggregate group_by_and_aggregate(this,
                                             co.device_type_,
                                             ra_exe_unit,
                                             render_output,
                                             query_infos,
                                             row_set_mem_owner,
                                             max_groups_buffer_entry_guess,
                                             small_groups_buffer_entry_count,
                                             crt_min_byte_width,
                                             eo.allow_multifrag,
                                             eo.output_columnar_hint && co.device_type_ == ExecutorDeviceType::GPU);
  const auto& query_mem_desc = group_by_and_aggregate.getQueryMemoryDescriptor();

  const bool output_columnar = group_by_and_aggregate.outputColumnar();

  if (co.device_type_ == ExecutorDeviceType::GPU &&
      query_mem_desc.hash_type == GroupByColRangeType::MultiColPerfectHash) {
    const size_t required_memory{(gridSize() * query_mem_desc.getBufferSizeBytes(ExecutorDeviceType::GPU))};
    CHECK(catalog_->get_dataMgr().cudaMgr_);
    const size_t max_memory{catalog_->get_dataMgr().cudaMgr_->deviceProperties[0].globalMem / 5};
    cgen_state_->must_run_on_cpu_ = required_memory > max_memory;
  }

  // Read the module template and target either CPU or GPU
  // by binding the stream position functions to the right implementation:
  // stride access for GPU, contiguous for CPU
  cgen_state_->module_ = read_template_module(cgen_state_->context_);

  auto agg_fnames = get_agg_fnames(ra_exe_unit.target_exprs, !ra_exe_unit.groupby_exprs.empty());

  const bool is_group_by{!query_mem_desc.group_col_widths.empty()};
  auto query_func = is_group_by
                        ? query_group_by_template(cgen_state_->module_,
                                                  is_nested_,
                                                  co.hoist_literals_,
                                                  query_mem_desc,
                                                  co.device_type_,
                                                  ra_exe_unit.scan_limit)
                        : query_template(cgen_state_->module_, agg_fnames.size(), is_nested_, co.hoist_literals_);
  bind_pos_placeholders("pos_start", true, query_func, cgen_state_->module_);
  bind_pos_placeholders("group_buff_idx", false, query_func, cgen_state_->module_);
  bind_pos_placeholders("pos_step", false, query_func, cgen_state_->module_);
  if (is_group_by) {
    bindInitGroupByBuffer(query_func, query_mem_desc, co.device_type_);
  }

  std::vector<llvm::Value*> col_heads;
  std::tie(cgen_state_->row_func_, col_heads) = create_row_function(ra_exe_unit.input_col_descs.size(),
                                                                    is_group_by ? 0 : agg_fnames.size(),
                                                                    co.hoist_literals_,
                                                                    query_func,
                                                                    cgen_state_->module_,
                                                                    cgen_state_->context_);
  CHECK(cgen_state_->row_func_);

  // make sure it's in-lined, we don't want register spills in the inner loop
  cgen_state_->row_func_->addAttribute(llvm::AttributeSet::FunctionIndex, llvm::Attribute::AlwaysInline);

  auto bb = llvm::BasicBlock::Create(cgen_state_->context_, "entry", cgen_state_->row_func_);
  cgen_state_->ir_builder_.SetInsertPoint(bb);

  allocateInnerScansIterators(ra_exe_unit.input_descs, eo.allow_loop_joins);

  // generate the code for the filter
  allocateLocalColumnIds(ra_exe_unit.input_col_descs);

  // Generate the expression for outer join first, the isOuterJoin() method relies
  // on it and ExpressionRange module calls isOuterJoin() when computing range.
  if (!ra_exe_unit.outer_join_quals.empty()) {
    cgen_state_->outer_join_cond_lv_ =
        llvm::ConstantInt::get(llvm::IntegerType::getInt1Ty(cgen_state_->context_), true);
    for (auto expr : ra_exe_unit.outer_join_quals) {
      cgen_state_->outer_join_cond_lv_ = cgen_state_->ir_builder_.CreateAnd(
          cgen_state_->outer_join_cond_lv_, toBool(codegen(expr.get(), true, co).front()));
    }
  }

  std::vector<Analyzer::Expr*> deferred_quals;
  llvm::Value* filter_lv = llvm::ConstantInt::get(llvm::IntegerType::getInt1Ty(cgen_state_->context_), true);

  for (auto expr : ra_exe_unit.inner_join_quals) {
    filter_lv = cgen_state_->ir_builder_.CreateAnd(filter_lv, toBool(codegen(expr.get(), true, co).front()));
  }

  for (auto expr : ra_exe_unit.simple_quals) {
    if (should_defer_eval(expr)) {
      deferred_quals.push_back(expr.get());
      continue;
    }
    filter_lv = cgen_state_->ir_builder_.CreateAnd(filter_lv, toBool(codegen(expr.get(), true, co).front()));
  }
  for (auto expr : ra_exe_unit.quals) {
    if (should_defer_eval(expr)) {
      deferred_quals.push_back(expr.get());
      continue;
    }
    auto qual_expr = rewrite_expr(expr.get());
    if (!qual_expr) {
      qual_expr = expr;
    }
    filter_lv = cgen_state_->ir_builder_.CreateAnd(filter_lv, toBool(codegen(qual_expr.get(), true, co).front()));
  }

  if (!deferred_quals.empty()) {
    auto sc_true = llvm::BasicBlock::Create(cgen_state_->context_, "sc_true", cgen_state_->row_func_);
    auto sc_false = llvm::BasicBlock::Create(cgen_state_->context_, "sc_false", cgen_state_->row_func_);
    cgen_state_->ir_builder_.CreateCondBr(filter_lv, sc_true, sc_false);
    cgen_state_->ir_builder_.SetInsertPoint(sc_false);
    codegenInnerScanNextRow();
    cgen_state_->ir_builder_.SetInsertPoint(sc_true);
  }

  for (auto expr : deferred_quals) {
    filter_lv = cgen_state_->ir_builder_.CreateAnd(filter_lv, toBool(codegen(expr, true, co).front()));
  }

  CHECK(filter_lv->getType()->isIntegerTy(1));

  const bool needs_error_check = group_by_and_aggregate.codegen(filter_lv, co);

  if (needs_error_check || cgen_state_->uses_div_) {
    createErrorCheckControlFlow(query_func);
  }

  // iterate through all the instruction in the query template function and
  // replace the call to the filter placeholder with the call to the actual filter
  for (auto it = llvm::inst_begin(query_func), e = llvm::inst_end(query_func); it != e; ++it) {
    if (!llvm::isa<llvm::CallInst>(*it)) {
      continue;
    }
    auto& filter_call = llvm::cast<llvm::CallInst>(*it);
    if (std::string(filter_call.getCalledFunction()->getName()) == unique_name("row_process", is_nested_)) {
      std::vector<llvm::Value*> args;
      for (size_t i = 0; i < filter_call.getNumArgOperands(); ++i) {
        args.push_back(filter_call.getArgOperand(i));
      }
      args.insert(args.end(), col_heads.begin(), col_heads.end());
      args.push_back(get_arg_by_name(query_func, "join_hash_table"));
      llvm::ReplaceInstWithInst(&filter_call, llvm::CallInst::Create(cgen_state_->row_func_, args, ""));
      break;
    }
  }

  is_nested_ = false;
  plan_state_->init_agg_vals_ = init_agg_val_vec(ra_exe_unit.target_exprs,
                                                 ra_exe_unit.quals,
                                                 query_mem_desc.agg_col_widths.size(),
                                                 is_group_by,
                                                 query_mem_desc.getCompactByteWidth());

  if (co.device_type_ == ExecutorDeviceType::GPU && cgen_state_->must_run_on_cpu_) {
    return {};
  }

  auto multifrag_query_func =
      cgen_state_->module_->getFunction("multifrag_query" + std::string(co.hoist_literals_ ? "_hoisted_literals" : ""));
  CHECK(multifrag_query_func);

  bind_query(query_func,
             "query_stub" + std::string(co.hoist_literals_ ? "_hoisted_literals" : ""),
             multifrag_query_func,
             cgen_state_->module_);

  auto live_funcs =
      markDeadRuntimeFuncs(*cgen_state_->module_, {query_func, cgen_state_->row_func_}, {multifrag_query_func});

  std::string llvm_ir;
  if (eo.just_explain) {
    llvm_ir = serialize_llvm_object(query_func) + serialize_llvm_object(cgen_state_->row_func_);
  }
  verify_function_ir(cgen_state_->row_func_);
  return Executor::CompilationResult{
      co.device_type_ == ExecutorDeviceType::CPU
          ? optimizeAndCodegenCPU(query_func, multifrag_query_func, live_funcs, cgen_state_->module_, co)
          : optimizeAndCodegenGPU(
                query_func, multifrag_query_func, live_funcs, cgen_state_->module_, is_group_by, cuda_mgr, co),
      cgen_state_->getLiterals(),
      query_mem_desc,
      output_columnar,
      llvm_ir};
}

void Executor::createErrorCheckControlFlow(llvm::Function* query_func) {
  // check whether the row processing was successful; currently, it can
  // fail by running out of group by buffer slots
  bool done_splitting = false;
  for (auto bb_it = query_func->begin(); bb_it != query_func->end() && !done_splitting; ++bb_it) {
    for (auto inst_it = bb_it->begin(); inst_it != bb_it->end(); ++inst_it) {
      if (!llvm::isa<llvm::CallInst>(*inst_it)) {
        continue;
      }
      auto& filter_call = llvm::cast<llvm::CallInst>(*inst_it);
      if (std::string(filter_call.getCalledFunction()->getName()) == unique_name("row_process", is_nested_)) {
        auto next_inst_it = inst_it;
        ++next_inst_it;
        auto new_bb = bb_it->splitBasicBlock(next_inst_it);
        auto& br_instr = bb_it->back();
        llvm::IRBuilder<> ir_builder(&br_instr);
        llvm::Value* err_lv = inst_it;
        auto& error_code_arg = query_func->getArgumentList().back();
        CHECK(error_code_arg.getName() == "error_code");
        err_lv = ir_builder.CreateCall(cgen_state_->module_->getFunction("record_error_code"),
                                       std::vector<llvm::Value*>{err_lv, &error_code_arg});
        err_lv = ir_builder.CreateICmp(llvm::ICmpInst::ICMP_NE, err_lv, ll_int(int32_t(0)));
        auto error_bb = llvm::BasicBlock::Create(cgen_state_->context_, ".error_exit", query_func, new_bb);
        llvm::ReturnInst::Create(cgen_state_->context_, error_bb);
        llvm::ReplaceInstWithInst(&br_instr, llvm::BranchInst::Create(error_bb, new_bb, err_lv));
        done_splitting = true;
        break;
      }
    }
  }
  CHECK(done_splitting);
}

void Executor::codegenInnerScanNextRow() {
  if (cgen_state_->inner_scan_labels_.empty()) {
    cgen_state_->ir_builder_.CreateRet(ll_int(int32_t(0)));
  } else {
    CHECK_EQ(size_t(1), cgen_state_->scan_to_iterator_.size());
    auto inner_it_val_and_ptr = cgen_state_->scan_to_iterator_.begin()->second;
    auto inner_it_inc = cgen_state_->ir_builder_.CreateAdd(inner_it_val_and_ptr.first, ll_int(int64_t(1)));
    cgen_state_->ir_builder_.CreateStore(inner_it_inc, inner_it_val_and_ptr.second);
    CHECK_EQ(size_t(1), cgen_state_->inner_scan_labels_.size());
    cgen_state_->ir_builder_.CreateBr(cgen_state_->inner_scan_labels_.front());
  }
}

void Executor::allocateInnerScansIterators(const std::vector<InputDescriptor>& input_descs,
                                           const bool allow_loop_joins) {
  if (input_descs.size() <= 1) {
    return;
  }
  if (plan_state_->join_info_.join_impl_type_ == JoinImplType::HashOneToOne) {
    return;
  }
  if (!allow_loop_joins) {
    throw std::runtime_error("Loop joins are disabled; run the server with --allow-loop-joins to enable them.");
  }
  CHECK(plan_state_->join_info_.join_impl_type_ == JoinImplType::Loop);
  auto preheader = cgen_state_->ir_builder_.GetInsertBlock();
  for (auto it = input_descs.begin() + 1; it != input_descs.end(); ++it) {
    const int inner_scan_idx = it - input_descs.begin();
    auto inner_scan_pos_ptr = cgen_state_->ir_builder_.CreateAlloca(
        get_int_type(64, cgen_state_->context_), nullptr, "inner_scan_" + std::to_string(inner_scan_idx));
    cgen_state_->ir_builder_.CreateStore(ll_int(int64_t(0)), inner_scan_pos_ptr);
    auto scan_loop_head = llvm::BasicBlock::Create(
        cgen_state_->context_, "scan_loop_head", cgen_state_->row_func_, preheader->getNextNode());
    cgen_state_->inner_scan_labels_.push_back(scan_loop_head);
    cgen_state_->ir_builder_.CreateBr(scan_loop_head);
    cgen_state_->ir_builder_.SetInsertPoint(scan_loop_head);
    auto inner_scan_pos = cgen_state_->ir_builder_.CreateLoad(inner_scan_pos_ptr, "load_inner_it");
    {
      const auto it_ok = cgen_state_->scan_to_iterator_.insert(
          std::make_pair(*it, std::make_pair(inner_scan_pos, inner_scan_pos_ptr)));
      CHECK(it_ok.second);
    }
    {
      auto rows_per_scan_ptr = cgen_state_->ir_builder_.CreateGEP(rowsPerScan(), {ll_int(int32_t(inner_scan_idx))});
      auto rows_per_scan = cgen_state_->ir_builder_.CreateLoad(rows_per_scan_ptr, "rows_per_scan");
      auto have_more_inner_rows =
          cgen_state_->ir_builder_.CreateICmp(llvm::ICmpInst::ICMP_ULT, inner_scan_pos, rows_per_scan);
      auto inner_scan_ret = llvm::BasicBlock::Create(cgen_state_->context_, "inner_scan_ret", cgen_state_->row_func_);
      auto inner_scan_cont = llvm::BasicBlock::Create(cgen_state_->context_, "inner_scan_cont", cgen_state_->row_func_);
      cgen_state_->ir_builder_.CreateCondBr(have_more_inner_rows, inner_scan_cont, inner_scan_ret);
      cgen_state_->ir_builder_.SetInsertPoint(inner_scan_ret);
      cgen_state_->ir_builder_.CreateRet(ll_int(int32_t(0)));
      cgen_state_->ir_builder_.SetInsertPoint(inner_scan_cont);
    }
  }
}

Executor::JoinInfo Executor::chooseJoinType(const std::list<std::shared_ptr<Analyzer::Expr>>& join_quals,
                                            const std::vector<Fragmenter_Namespace::TableInfo>& query_infos,
                                            const ExecutorDeviceType device_type) {
  CHECK(device_type != ExecutorDeviceType::Hybrid);
  const MemoryLevel memory_level{device_type == ExecutorDeviceType::GPU ? MemoryLevel::GPU_LEVEL
                                                                        : MemoryLevel::CPU_LEVEL};
  for (auto qual : join_quals) {
    auto qual_bin_oper = std::dynamic_pointer_cast<Analyzer::BinOper>(qual);
    if (!qual_bin_oper) {
      const auto bool_const = std::dynamic_pointer_cast<Analyzer::Constant>(qual);
      CHECK(bool_const);
      const auto& bool_const_ti = bool_const->get_type_info();
      CHECK(bool_const_ti.is_boolean());
      continue;
    }
    if (qual_bin_oper->get_optype() == kEQ) {
      const int device_count =
          device_type == ExecutorDeviceType::GPU ? catalog_->get_dataMgr().cudaMgr_->getDeviceCount() : 1;
      CHECK_GT(device_count, 0);
      const auto join_hash_table =
          JoinHashTable::getInstance(qual_bin_oper, *catalog_, query_infos, memory_level, device_count, this);
      if (join_hash_table) {
        return Executor::JoinInfo(JoinImplType::HashOneToOne,
                                  std::vector<std::shared_ptr<Analyzer::BinOper>>{qual_bin_oper},
                                  join_hash_table);
      }
    }
  }
  return Executor::JoinInfo(JoinImplType::Loop, std::vector<std::shared_ptr<Analyzer::BinOper>>{}, nullptr);
}

void Executor::bindInitGroupByBuffer(llvm::Function* query_func,
                                     const QueryMemoryDescriptor& query_mem_desc,
                                     const ExecutorDeviceType device_type) {
  if (!query_mem_desc.lazyInitGroups(device_type) || query_mem_desc.hash_type != GroupByColRangeType::MultiCol) {
    return;
  }
  for (auto& inst : *query_func->begin()) {
    if (!llvm::isa<llvm::CallInst>(inst)) {
      continue;
    }
    auto& init_group_by_buffer_call = llvm::cast<llvm::CallInst>(inst);
    if (std::string(init_group_by_buffer_call.getCalledFunction()->getName()) != "init_group_by_buffer") {
      continue;
    }
    CHECK(!query_mem_desc.output_columnar);
    std::vector<llvm::Value*> args;
    for (size_t i = 0; i < init_group_by_buffer_call.getNumArgOperands(); ++i) {
      args.push_back(init_group_by_buffer_call.getArgOperand(i));
    }
    auto keyless_lv = llvm::ConstantInt::get(get_int_type(1, cgen_state_->context_), query_mem_desc.keyless_hash);
    args.push_back(keyless_lv);
    if (!query_mem_desc.output_columnar) {
      const int8_t warp_count = query_mem_desc.interleavedBins(device_type) ? warpSize() : 1;
      args.push_back(ll_int<int8_t>(warp_count));
    }

    llvm::ReplaceInstWithInst(
        &init_group_by_buffer_call,
        llvm::CallInst::Create(query_func->getParent()->getFunction("init_group_by_buffer_gpu"), args));
    break;
  }
}

namespace {

void eliminateDeadSelfRecursiveFuncs(llvm::Module& M, std::unordered_set<llvm::Function*>& live_funcs) {
  std::vector<llvm::Function*> dead_funcs;
  for (auto& F : M) {
    bool bAlive = false;
    if (live_funcs.count(&F))
      continue;
    for (auto U : F.users()) {
      auto* C = llvm::dyn_cast<const llvm::CallInst>(U);
      if (!C || C->getParent()->getParent() != &F) {
        bAlive = true;
        break;
      }
    }
    if (!bAlive)
      dead_funcs.push_back(&F);
  }
  for (auto pFn : dead_funcs) {
    pFn->eraseFromParent();
  }
}

void optimizeIR(llvm::Function* query_func,
                llvm::Module* module,
                std::unordered_set<llvm::Function*>& live_funcs,
                const CompilationOptions& co,
                const std::string& debug_dir,
                const std::string& debug_file) {
  llvm::legacy::PassManager pass_manager;
  pass_manager.add(llvm::createAlwaysInlinerPass());
  pass_manager.add(llvm::createPromoteMemoryToRegisterPass());
  pass_manager.add(llvm::createInstructionSimplifierPass());
  pass_manager.add(llvm::createInstructionCombiningPass());
  pass_manager.add(llvm::createGlobalOptimizerPass());
// FIXME(miyu): need investigate how 3.7+ dump debug IR.
#if LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR == 5
  if (!debug_dir.empty()) {
    CHECK(!debug_file.empty());
    pass_manager.add(llvm::createDebugIRPass(false, false, debug_dir, debug_file));
  }
#endif
  if (co.hoist_literals_) {
    pass_manager.add(llvm::createLICMPass());
  }
  if (co.opt_level_ == ExecutorOptLevel::LoopStrengthReduction) {
    pass_manager.add(llvm::createLoopStrengthReducePass());
  }
  pass_manager.run(*module);

  eliminateDeadSelfRecursiveFuncs(*module, live_funcs);

  // optimizations might add attributes to the function
  // and NVPTX doesn't understand all of them; play it
  // safe and clear all attributes
  llvm::AttributeSet no_attributes;
  query_func->setAttributes(no_attributes);
  verify_function_ir(query_func);
}

}  // namespace

std::vector<void*> Executor::getCodeFromCache(
    const CodeCacheKey& key,
    const std::map<CodeCacheKey, std::pair<CodeCacheVal, llvm::Module*>>& cache) {
  auto it = cache.find(key);
  if (it != cache.end()) {
    delete cgen_state_->module_;
    cgen_state_->module_ = it->second.second;
    std::vector<void*> native_functions;
    for (auto& native_code : it->second.first) {
      native_functions.push_back(std::get<0>(native_code));
    }
    return native_functions;
  }
  return {};
}

void Executor::addCodeToCache(
    const CodeCacheKey& key,
    const std::vector<std::tuple<void*, llvm::ExecutionEngine*, GpuCompilationContext*>>& native_code,
    llvm::Module* module,
    std::map<CodeCacheKey, std::pair<CodeCacheVal, llvm::Module*>>& cache) {
  CHECK(!native_code.empty());
  CodeCacheVal cache_val;
  for (const auto& native_func : native_code) {
    cache_val.emplace_back(std::get<0>(native_func),
                           std::unique_ptr<llvm::ExecutionEngine>(std::get<1>(native_func)),
                           std::unique_ptr<GpuCompilationContext>(std::get<2>(native_func)));
  }
  auto it_ok = cache.insert(std::make_pair(key, std::make_pair(std::move(cache_val), module)));
  CHECK(it_ok.second);
}

std::vector<void*> Executor::optimizeAndCodegenCPU(llvm::Function* query_func,
                                                   llvm::Function* multifrag_query_func,
                                                   std::unordered_set<llvm::Function*>& live_funcs,
                                                   llvm::Module* module,
                                                   const CompilationOptions& co) {
  CodeCacheKey key{serialize_llvm_object(query_func), serialize_llvm_object(cgen_state_->row_func_)};
  for (const auto helper : cgen_state_->helper_functions_) {
    key.push_back(serialize_llvm_object(helper));
  }
  auto cached_code = getCodeFromCache(key, cpu_code_cache_);
  if (!cached_code.empty()) {
    return cached_code;
  }

  // run optimizations
  optimizeIR(query_func, module, live_funcs, co, debug_dir_, debug_file_);

  llvm::ExecutionEngine* execution_engine{nullptr};

  auto init_err = llvm::InitializeNativeTarget();
  CHECK(!init_err);

  llvm::InitializeAllTargetMCs();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  std::string err_str;
#if LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR == 5
  llvm::EngineBuilder eb(module);
  eb.setUseMCJIT(true);
#else
  std::unique_ptr<llvm::Module> owner(module);
  llvm::EngineBuilder eb(std::move(owner));
#endif
  eb.setErrorStr(&err_str);
  eb.setEngineKind(llvm::EngineKind::JIT);
  llvm::TargetOptions to;
  to.EnableFastISel = true;
  eb.setTargetOptions(to);
  execution_engine = eb.create();
  CHECK(execution_engine);

  execution_engine->finalizeObject();
  auto native_code = execution_engine->getPointerToFunction(multifrag_query_func);

  CHECK(native_code);
  addCodeToCache(key, {{std::make_tuple(native_code, execution_engine, nullptr)}}, module, cpu_code_cache_);

  return {native_code};
}

namespace {

std::string cpp_to_llvm_name(const std::string& s) {
  if (s == "int8_t") {
    return "i8";
  }
  if (s == "int16_t") {
    return "i16";
  }
  if (s == "int32_t") {
    return "i32";
  }
  if (s == "int64_t") {
    return "i64";
  }
  CHECK(s == "float" || s == "double");
  return s;
}

std::string gen_array_any_all_sigs() {
  std::string result;
  for (const std::string any_or_all : {"any", "all"}) {
    for (const std::string elem_type : {"int8_t", "int16_t", "int32_t", "int64_t", "float", "double"}) {
      for (const std::string needle_type : {"int8_t", "int16_t", "int32_t", "int64_t", "float", "double"}) {
        for (const std::string op_name : {"eq", "ne", "lt", "le", "gt", "ge"}) {
          result += ("declare i1 @array_" + any_or_all + "_" + op_name + "_" + elem_type + "_" + needle_type +
                     "(i8*, i64, " + cpp_to_llvm_name(needle_type) + ", " + cpp_to_llvm_name(elem_type) + ");\n");
        }
      }
    }
  }
  return result;
}

const std::string cuda_rt_decls =
    R"(
declare void @llvm.lifetime.start(i64, i8* nocapture) nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) nounwind
declare i32 @pos_start_impl(i32*);
declare i32 @group_buff_idx_impl();
declare i32 @pos_step_impl();
declare i8 @thread_warp_idx(i8);
declare i64* @init_shared_mem(i64*, i32);
declare i64* @init_shared_mem_nop(i64*, i32);
declare void @write_back(i64*, i64*, i32);
declare void @write_back_nop(i64*, i64*, i32);
declare void @init_group_by_buffer_gpu(i64*, i64*, i32, i32, i32, i1, i8);
declare i64* @get_group_value(i64*, i32, i64*, i32, i32, i64*);
declare i64* @get_group_value_fast(i64*, i64, i64, i64, i32);
declare i32 @get_columnar_group_bin_offset(i64*, i64, i64, i64);
declare i64* @get_group_value_one_key(i64*, i32, i64*, i32, i64, i64, i32, i64*);
declare i64 @agg_count_shared(i64*, i64);
declare i64 @agg_count_skip_val_shared(i64*, i64, i64);
declare i32 @agg_count_int32_shared(i32*, i32);
declare i32 @agg_count_int32_skip_val_shared(i32*, i32, i32);
declare i64 @agg_count_double_shared(i64*, double);
declare i64 @agg_count_double_skip_val_shared(i64*, double, double);
declare i32 @agg_count_float_shared(i32*, float);
declare i32 @agg_count_float_skip_val_shared(i32*, float, float);
declare i64 @agg_sum_shared(i64*, i64);
declare i64 @agg_sum_skip_val_shared(i64*, i64, i64);
declare i32 @agg_sum_int32_shared(i32*, i32);
declare i32 @agg_sum_int32_skip_val_shared(i32*, i32, i32);
declare void @agg_sum_double_shared(i64*, double);
declare void @agg_sum_double_skip_val_shared(i64*, double, double);
declare void @agg_sum_float_shared(i32*, float);
declare void @agg_sum_float_skip_val_shared(i32*, float, float);
declare void @agg_max_shared(i64*, i64);
declare void @agg_max_skip_val_shared(i64*, i64, i64);
declare void @agg_max_int32_shared(i32*, i32);
declare void @agg_max_int32_skip_val_shared(i32*, i32, i32);
declare void @agg_max_double_shared(i64*, double);
declare void @agg_max_double_skip_val_shared(i64*, double, double);
declare void @agg_max_float_shared(i32*, float);
declare void @agg_max_float_skip_val_shared(i32*, float, float);
declare void @agg_min_shared(i64*, i64);
declare void @agg_min_skip_val_shared(i64*, i64, i64);
declare void @agg_min_int32_shared(i32*, i32);
declare void @agg_min_int32_skip_val_shared(i32*, i32, i32);
declare void @agg_min_double_shared(i64*, double);
declare void @agg_min_double_skip_val_shared(i64*, double, double);
declare void @agg_min_float_shared(i32*, float);
declare void @agg_min_float_skip_val_shared(i32*, float, float);
declare void @agg_id_shared(i64*, i64);
declare void @agg_id_int32_shared(i32*, i32);
declare void @agg_id_double_shared(i64*, double);
declare void @agg_id_float_shared(i32*, float);
declare i64 @ExtractFromTime(i32, i64);
declare i64 @ExtractFromTimeNullable(i32, i64, i64);
declare i64 @DateTruncate(i32, i64);
declare i64 @DateTruncateNullable(i32, i64, i64);
declare i64 @string_decode(i8*, i64);
declare i32 @array_size(i8*, i64, i32);
declare i1 @array_is_null(i8*, i64);
declare i8* @array_buff(i8*, i64);
declare i8 @array_at_int8_t(i8*, i64, i32);
declare i16 @array_at_int16_t(i8*, i64, i32);
declare i32 @array_at_int32_t(i8*, i64, i32);
declare i64 @array_at_int64_t(i8*, i64, i32);
declare float @array_at_float(i8*, i64, i32);
declare double @array_at_double(i8*, i64, i32);
declare i8 @array_at_int8_t_checked(i8*, i64, i64, i8);
declare i16 @array_at_int16_t_checked(i8*, i64, i64, i16);
declare i32 @array_at_int32_t_checked(i8*, i64, i64, i32);
declare i64 @array_at_int64_t_checked(i8*, i64, i64, i64);
declare float @array_at_float_checked(i8*, i64, i64, float);
declare double @array_at_double_checked(i8*, i64, i64, double);
declare i32 @char_length(i8*, i32);
declare i32 @char_length_nullable(i8*, i32, i32);
declare i32 @char_length_encoded(i8*, i32);
declare i32 @char_length_encoded_nullable(i8*, i32, i32);
declare i1 @string_like(i8*, i32, i8*, i32, i8);
declare i1 @string_ilike(i8*, i32, i8*, i32, i8);
declare i8 @string_like_nullable(i8*, i32, i8*, i32, i8, i8);
declare i8 @string_ilike_nullable(i8*, i32, i8*, i32, i8, i8);
declare i1 @string_like_simple(i8*, i32, i8*, i32);
declare i1 @string_ilike_simple(i8*, i32, i8*, i32);
declare i8 @string_like_simple_nullable(i8*, i32, i8*, i32, i8);
declare i8 @string_ilike_simple_nullable(i8*, i32, i8*, i32, i8);
declare i1 @string_lt(i8*, i32, i8*, i32);
declare i1 @string_le(i8*, i32, i8*, i32);
declare i1 @string_gt(i8*, i32, i8*, i32);
declare i1 @string_ge(i8*, i32, i8*, i32);
declare i1 @string_eq(i8*, i32, i8*, i32);
declare i1 @string_ne(i8*, i32, i8*, i32);
declare i8 @string_lt_nullable(i8*, i32, i8*, i32, i8);
declare i8 @string_le_nullable(i8*, i32, i8*, i32, i8);
declare i8 @string_gt_nullable(i8*, i32, i8*, i32, i8);
declare i8 @string_ge_nullable(i8*, i32, i8*, i32, i8);
declare i8 @string_eq_nullable(i8*, i32, i8*, i32, i8);
declare i8 @string_ne_nullable(i8*, i32, i8*, i32, i8);
declare i32 @record_error_code(i32, i32*);
)" +
    gen_array_any_all_sigs();

}  // namespace

std::vector<void*> Executor::optimizeAndCodegenGPU(llvm::Function* query_func,
                                                   llvm::Function* multifrag_query_func,
                                                   std::unordered_set<llvm::Function*>& live_funcs,
                                                   llvm::Module* module,
                                                   const bool no_inline,
                                                   const CudaMgr_Namespace::CudaMgr* cuda_mgr,
                                                   const CompilationOptions& co) {
#ifdef HAVE_CUDA
  CHECK(cuda_mgr);
  CodeCacheKey key{serialize_llvm_object(query_func), serialize_llvm_object(cgen_state_->row_func_)};
  for (const auto helper : cgen_state_->helper_functions_) {
    key.push_back(serialize_llvm_object(helper));
  }
  auto cached_code = getCodeFromCache(key, gpu_code_cache_);
  if (!cached_code.empty()) {
    return cached_code;
  }

  auto get_group_value_func = module->getFunction("get_group_value_one_key");
  CHECK(get_group_value_func);
  get_group_value_func->setAttributes(llvm::AttributeSet{});

  bool row_func_not_inlined = false;
  if (no_inline) {
    for (auto it = llvm::inst_begin(cgen_state_->row_func_), e = llvm::inst_end(cgen_state_->row_func_); it != e;
         ++it) {
      if (llvm::isa<llvm::CallInst>(*it)) {
        auto& get_gv_call = llvm::cast<llvm::CallInst>(*it);
        if (get_gv_call.getCalledFunction()->getName() == "get_group_value" ||
            get_gv_call.getCalledFunction()->getName() == "get_matching_group_value_perfect_hash" ||
            get_gv_call.getCalledFunction()->getName() == "string_decode" ||
            get_gv_call.getCalledFunction()->getName() == "array_size") {
          llvm::AttributeSet no_inline_attrs;
          no_inline_attrs = no_inline_attrs.addAttribute(cgen_state_->context_, 0, llvm::Attribute::NoInline);
          cgen_state_->row_func_->setAttributes(no_inline_attrs);
          row_func_not_inlined = true;
          break;
        }
      }
    }
  }

  module->setDataLayout(
      "e-p:64:64:64-i1:8:8-i8:8:8-"
      "i16:16:16-i32:32:32-i64:64:64-"
      "f32:32:32-f64:64:64-v16:16:16-"
      "v32:32:32-v64:64:64-v128:128:128-n16:32:64");
  module->setTargetTriple("nvptx64-nvidia-cuda");

  // run optimizations
  optimizeIR(query_func, module, live_funcs, co, "", "");

  std::stringstream ss;
  llvm::raw_os_ostream os(ss);

  llvm::LLVMContext& ctx = module->getContext();
  // Get "nvvm.annotations" metadata node
  llvm::NamedMDNode* md = module->getOrInsertNamedMetadata("nvvm.annotations");

#if LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR == 5
  llvm::Value* md_vals[] = {
      multifrag_query_func, llvm::MDString::get(ctx, "kernel"), llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 1)};
#else
  llvm::Metadata* md_vals[] = {llvm::ConstantAsMetadata::get(multifrag_query_func),
                               llvm::MDString::get(ctx, "kernel"),
                               llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 1))};
#endif
  // Append metadata to nvvm.annotations
  md->addOperand(llvm::MDNode::get(ctx, md_vals));

  std::unordered_set<llvm::Function*> roots{multifrag_query_func, query_func};
  if (row_func_not_inlined) {
    llvm::AttributeSet no_attributes;
    cgen_state_->row_func_->setAttributes(no_attributes);
    roots.insert(cgen_state_->row_func_);
  }

  std::vector<llvm::Function*> rt_funcs;
  for (auto& Fn : *module) {
    if (roots.count(&Fn))
      continue;
    rt_funcs.push_back(&Fn);
  }
  for (auto& pFn : rt_funcs)
    pFn->removeFromParent();
  module->print(os, nullptr);
  os.flush();
  for (auto& pFn : rt_funcs) {
    module->getFunctionList().push_back(pFn);
  }
  module->eraseNamedMetadata(md);

  auto cuda_llir = cuda_rt_decls + ss.str();

  std::vector<void*> native_functions;
  std::vector<std::tuple<void*, llvm::ExecutionEngine*, GpuCompilationContext*>> cached_functions;

  const auto ptx = generatePTX(cuda_llir);

  auto cubin_result = ptx_to_cubin(ptx, blockSize(), cuda_mgr);
  auto& option_keys = cubin_result.option_keys;
  auto& option_values = cubin_result.option_values;
  auto cubin = cubin_result.cubin;
  auto link_state = cubin_result.link_state;
  const auto num_options = option_keys.size();

  auto func_name = multifrag_query_func->getName().str();
  for (int device_id = 0; device_id < cuda_mgr->getDeviceCount(); ++device_id) {
    auto gpu_context = new GpuCompilationContext(
        cubin, func_name, device_id, cuda_mgr, num_options, &option_keys[0], &option_values[0]);
    auto native_code = gpu_context->kernel();
    CHECK(native_code);
    native_functions.push_back(native_code);
    cached_functions.emplace_back(native_code, nullptr, gpu_context);
  }
  addCodeToCache(key, cached_functions, module, gpu_code_cache_);

  checkCudaErrors(cuLinkDestroy(link_state));

  return native_functions;
#else
  return {};
#endif
}

std::string Executor::generatePTX(const std::string& cuda_llir) const {
  initializeNVPTXBackend();
  auto mem_buff = llvm::MemoryBuffer::getMemBuffer(cuda_llir, "", false);

  llvm::SMDiagnostic err;

#if LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR == 5
  auto module = llvm::ParseIR(mem_buff, err, cgen_state_->context_);
#else
  auto module = llvm::parseIR(mem_buff->getMemBufferRef(), err, cgen_state_->context_);
#endif
  if (!module) {
    LOG(FATAL) << err.getMessage().str();
  }

#if LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR == 5
  std::stringstream ss;
  llvm::raw_os_ostream raw_os(ss);
  llvm::formatted_raw_ostream formatted_os(raw_os);
#else
  llvm::SmallString<256> code_str;
  llvm::raw_svector_ostream formatted_os(code_str);
#endif
  CHECK(nvptx_target_machine_);
  {
    llvm::legacy::PassManager ptxgen_pm;
#if LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR == 5
    ptxgen_pm.add(new llvm::DataLayoutPass(module));
#else
    module->setDataLayout(nvptx_target_machine_->createDataLayout());
#endif

    nvptx_target_machine_->addPassesToEmitFile(ptxgen_pm, formatted_os, llvm::TargetMachine::CGFT_AssemblyFile);
    ptxgen_pm.run(*module);
    formatted_os.flush();
  }

#if LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR == 5
  return ss.str();
#else
  return code_str.str();
#endif
}

void Executor::initializeNVPTXBackend() const {
  if (nvptx_target_machine_) {
    return;
  }
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  std::string err;
  auto target = llvm::TargetRegistry::lookupTarget("nvptx64", err);
  if (!target) {
    LOG(FATAL) << err;
  }
  nvptx_target_machine_.reset(target->createTargetMachine("nvptx64-nvidia-cuda",
                                                          "sm_30",
                                                          "",
                                                          llvm::TargetOptions(),
                                                          llvm::Reloc::Default,
                                                          llvm::CodeModel::Default,
                                                          llvm::CodeGenOpt::Aggressive));
}

int8_t Executor::warpSize() const {
  CHECK(catalog_);
  CHECK(catalog_->get_dataMgr().cudaMgr_);
  const auto& dev_props = catalog_->get_dataMgr().cudaMgr_->deviceProperties;
  CHECK(!dev_props.empty());
  return dev_props.front().warpSize;
}

unsigned Executor::gridSize() const {
  CHECK(catalog_);
  CHECK(catalog_->get_dataMgr().cudaMgr_);
  const auto& dev_props = catalog_->get_dataMgr().cudaMgr_->deviceProperties;
  return grid_size_x_ ? grid_size_x_ : 2 * dev_props.front().numMPs;
}

unsigned Executor::blockSize() const {
  CHECK(catalog_);
  CHECK(catalog_->get_dataMgr().cudaMgr_);
  const auto& dev_props = catalog_->get_dataMgr().cudaMgr_->deviceProperties;
  return block_size_x_ ? block_size_x_ : dev_props.front().maxThreadsPerBlock;
}

llvm::Value* Executor::castToFP(llvm::Value* val) {
  if (!val->getType()->isIntegerTy()) {
    return val;
  }

  auto val_width = static_cast<llvm::IntegerType*>(val->getType())->getBitWidth();
  llvm::Type* dest_ty{nullptr};
  switch (val_width) {
    case 32:
      dest_ty = llvm::Type::getFloatTy(cgen_state_->context_);
      break;
    case 64:
      dest_ty = llvm::Type::getDoubleTy(cgen_state_->context_);
      break;
    default:
      CHECK(false);
  }
  return cgen_state_->ir_builder_.CreateSIToFP(val, dest_ty);
}

llvm::Value* Executor::castToTypeIn(llvm::Value* val, const size_t bit_width) {
  if (val->getType()->isIntegerTy()) {
    auto val_width = static_cast<llvm::IntegerType*>(val->getType())->getBitWidth();
    if (val_width == bit_width) {
      return val;
    }

    CHECK_LT(val_width, bit_width);
    const auto cast_op = val_width == 1 ? llvm::Instruction::CastOps::ZExt : llvm::Instruction::CastOps::SExt;
    return val_width < bit_width
               ? cgen_state_->ir_builder_.CreateCast(cast_op, val, get_int_type(bit_width, cgen_state_->context_))
               : val;
  }
  // real (not dictionary-encoded) strings; store the pointer to the payload
  if (val->getType()->isPointerTy()) {
    const auto val_ptr_type = static_cast<llvm::PointerType*>(val->getType());
    CHECK(val_ptr_type->getElementType()->isIntegerTy(8));
    return cgen_state_->ir_builder_.CreatePointerCast(val, get_int_type(bit_width, cgen_state_->context_));
  }

  CHECK(val->getType()->isFloatTy() || val->getType()->isDoubleTy());

  return val->getType()->isFloatTy() && bit_width == 64
             ? cgen_state_->ir_builder_.CreateFPExt(val, llvm::Type::getDoubleTy(cgen_state_->context_))
             : val;
}

llvm::Value* Executor::castToIntPtrTyIn(llvm::Value* val, const size_t bitWidth) {
  CHECK(val->getType()->isPointerTy());

  const auto val_ptr_type = static_cast<llvm::PointerType*>(val->getType());
  const auto val_width = val_ptr_type->getElementType()->getIntegerBitWidth();
  CHECK_LT(size_t(0), val_width);
  if (bitWidth == val_width) {
    return val;
  }
  return cgen_state_->ir_builder_.CreateBitCast(
      val, llvm::PointerType::get(get_int_type(bitWidth, cgen_state_->context_), 0));
}

#define EXECUTE_INCLUDE
#include "ArrayOps.cpp"
#include "StringFunctions.cpp"
#undef EXECUTE_INCLUDE

llvm::Value* Executor::groupByColumnCodegen(Analyzer::Expr* group_by_col,
                                            const CompilationOptions& co,
                                            const bool translate_null_val,
                                            const int64_t translated_null_val,
                                            GroupByAndAggregate::DiamondCodegen& diamond_codegen,
                                            std::stack<llvm::BasicBlock*>& array_loops) {
  auto group_key = codegen(group_by_col, true, co).front();
  if (dynamic_cast<Analyzer::UOper*>(group_by_col) &&
      static_cast<Analyzer::UOper*>(group_by_col)->get_optype() == kUNNEST) {
    auto preheader = cgen_state_->ir_builder_.GetInsertBlock();
    auto array_loop_head = llvm::BasicBlock::Create(
        cgen_state_->context_, "array_loop_head", cgen_state_->row_func_, preheader->getNextNode());
    diamond_codegen.setFalseTarget(array_loop_head);
    const auto ret_ty = get_int_type(32, cgen_state_->context_);
    auto array_idx_ptr = cgen_state_->ir_builder_.CreateAlloca(ret_ty);
    CHECK(array_idx_ptr);
    cgen_state_->ir_builder_.CreateStore(ll_int(int32_t(0)), array_idx_ptr);
    const auto arr_expr = static_cast<Analyzer::UOper*>(group_by_col)->get_operand();
    const auto& array_ti = arr_expr->get_type_info();
    CHECK(array_ti.is_array());
    const auto& elem_ti = array_ti.get_elem_type();
    auto array_len = cgen_state_->emitExternalCall(
        "array_size", ret_ty, {group_key, posArg(arr_expr), ll_int(log2_bytes(elem_ti.get_size()))});
    cgen_state_->ir_builder_.CreateBr(array_loop_head);
    cgen_state_->ir_builder_.SetInsertPoint(array_loop_head);
    CHECK(array_len);
    auto array_idx = cgen_state_->ir_builder_.CreateLoad(array_idx_ptr);
    auto bound_check = cgen_state_->ir_builder_.CreateICmp(llvm::ICmpInst::ICMP_SLT, array_idx, array_len);
    auto array_loop_body = llvm::BasicBlock::Create(cgen_state_->context_, "array_loop_body", cgen_state_->row_func_);
    cgen_state_->ir_builder_.CreateCondBr(
        bound_check, array_loop_body, array_loops.empty() ? diamond_codegen.orig_cond_false_ : array_loops.top());
    cgen_state_->ir_builder_.SetInsertPoint(array_loop_body);
    cgen_state_->ir_builder_.CreateStore(cgen_state_->ir_builder_.CreateAdd(array_idx, ll_int(int32_t(1))),
                                         array_idx_ptr);
    const auto array_at_fname = "array_at_" + numeric_type_name(elem_ti);
    const auto ar_ret_ty = elem_ti.is_fp()
                               ? (elem_ti.get_type() == kDOUBLE ? llvm::Type::getDoubleTy(cgen_state_->context_)
                                                                : llvm::Type::getFloatTy(cgen_state_->context_))
                               : get_int_type(elem_ti.get_size() * 8, cgen_state_->context_);
    group_key = cgen_state_->emitExternalCall(array_at_fname, ar_ret_ty, {group_key, posArg(arr_expr), array_idx});
    CHECK(array_loop_head);
    array_loops.push(array_loop_head);
  }
  cgen_state_->group_by_expr_cache_.push_back(group_key);
  if (translate_null_val) {
    const auto& ti = group_by_col->get_type_info();
    const auto key_type = get_int_type(ti.get_size() * 8, cgen_state_->context_);
    group_key =
        cgen_state_->emitCall("translate_null_key_" + numeric_type_name(ti),
                              {group_key,
                               static_cast<llvm::Value*>(llvm::ConstantInt::get(key_type, inline_int_null_val(ti))),
                               static_cast<llvm::Value*>(llvm::ConstantInt::get(key_type, translated_null_val))});
  }
  group_key =
      cgen_state_->ir_builder_.CreateBitCast(castToTypeIn(group_key, 64), get_int_type(64, cgen_state_->context_));
  return group_key;
}

void Executor::allocateLocalColumnIds(const std::list<InputColDescriptor>& global_col_ids) {
  for (const auto& col_id : global_col_ids) {
    const auto local_col_id = plan_state_->global_to_local_col_ids_.size();
    const auto it_ok = plan_state_->global_to_local_col_ids_.insert(std::make_pair(col_id, local_col_id));
    plan_state_->local_to_global_col_ids_.push_back(col_id.getColId());
    // enforce uniqueness of the column ids in the scan plan
    CHECK(it_ok.second);
  }
}

int Executor::getLocalColumnId(const Analyzer::ColumnVar* col_var, const bool fetch_column) const {
  CHECK(col_var);
  const int table_id = is_nested_ ? 0 : col_var->get_table_id();
  int global_col_id = col_var->get_column_id();
  if (is_nested_) {
    const auto var = dynamic_cast<const Analyzer::Var*>(col_var);
    CHECK(var);
    global_col_id = var->get_varno();
  }
  const int scan_idx = is_nested_ ? -1 : col_var->get_rte_idx();
  InputColDescriptor scan_col_desc(global_col_id, table_id, scan_idx);
  const auto it = plan_state_->global_to_local_col_ids_.find(scan_col_desc);
  if (it == plan_state_->global_to_local_col_ids_.end()) {
    CHECK(false);
  }
  if (fetch_column) {
    plan_state_->columns_to_fetch_.insert(global_col_id);
  }
  return it->second;
}

std::pair<bool, int64_t> Executor::skipFragment(const int table_id,
                                                const Fragmenter_Namespace::FragmentInfo& fragment,
                                                const std::list<std::shared_ptr<Analyzer::Expr>>& simple_quals,
                                                const std::vector<uint64_t>& all_frag_row_offsets,
                                                const size_t frag_idx) {
  for (const auto simple_qual : simple_quals) {
    const auto comp_expr = std::dynamic_pointer_cast<const Analyzer::BinOper>(simple_qual);
    if (!comp_expr) {
      // is this possible?
      return {false, -1};
    }
    const auto lhs = comp_expr->get_left_operand();
    const auto lhs_col = dynamic_cast<const Analyzer::ColumnVar*>(lhs);
    if (!lhs_col || !lhs_col->get_table_id() || lhs_col->get_rte_idx()) {
      return {false, -1};
    }
    const auto rhs = comp_expr->get_right_operand();
    const auto rhs_const = dynamic_cast<const Analyzer::Constant*>(rhs);
    if (!rhs_const) {
      // is this possible?
      return {false, -1};
    }
    if (!lhs->get_type_info().is_integer() && !lhs->get_type_info().is_time()) {
      return {false, -1};
    }
    const int col_id = lhs_col->get_column_id();
    auto chunk_meta_it = fragment.chunkMetadataMap.find(col_id);
    int64_t chunk_min{0};
    int64_t chunk_max{0};
    bool is_rowid{false};
    if (chunk_meta_it == fragment.chunkMetadataMap.end()) {
      auto cd = get_column_descriptor(col_id, table_id, *catalog_);
      CHECK(cd->isVirtualCol && cd->columnName == "rowid");
      chunk_min = all_frag_row_offsets[frag_idx];
      chunk_max = all_frag_row_offsets[frag_idx + 1] - 1;
      is_rowid = true;
    } else {
      const auto& chunk_type = lhs->get_type_info();
      chunk_min = extract_min_stat(chunk_meta_it->second.chunkStats, chunk_type);
      chunk_max = extract_max_stat(chunk_meta_it->second.chunkStats, chunk_type);
    }
    const auto rhs_val = codegenIntConst(rhs_const)->getSExtValue();
    switch (comp_expr->get_optype()) {
      case kGE:
        if (chunk_max < rhs_val) {
          return {true, -1};
        }
        break;
      case kGT:
        if (chunk_max <= rhs_val) {
          return {true, -1};
        }
        break;
      case kLE:
        if (chunk_min > rhs_val) {
          return {true, -1};
        }
        break;
      case kLT:
        if (chunk_min >= rhs_val) {
          return {true, -1};
        }
        break;
      case kEQ:
        if (chunk_min > rhs_val || chunk_max < rhs_val) {
          return {true, -1};
        } else if (is_rowid) {
          return {false, rhs_val};
        }
        break;
      default:
        break;
    }
  }
  return {false, -1};
}

std::map<std::pair<int, ::QueryRenderer::QueryRenderManager*>, std::shared_ptr<Executor>> Executor::executors_;
std::mutex Executor::execute_mutex_;
mapd_shared_mutex Executor::executors_cache_mutex_;
