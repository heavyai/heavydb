#include "Execute.h"

#include "AggregateUtils.h"
#include "CartesianProduct.h"
#include "Codec.h"
#include "ExpressionRewrite.h"
#include "ExtensionFunctionsBinding.h"
#include "ExtensionFunctionsWhitelist.h"
#include "ExtensionFunctions.hpp"
#include "GpuMemUtils.h"
#include "InPlaceSort.h"
#include "JsonAccessors.h"
#include "MaxwellCodegenPatch.h"
#include "NullableValue.h"
#include "OutputBufferInitialization.h"
#include "QueryRewrite.h"
#include "QueryTemplateGenerator.h"
#include "RuntimeFunctions.h"
#include "DynamicWatchdog.h"
#include "SpeculativeTopN.h"

#include "CudaMgr/CudaMgr.h"
#include "DataMgr/BufferMgr/BufferMgr.h"
#include "Parser/ParserNode.h"
#include "Shared/checked_alloc.h"
#include "Shared/mapdpath.h"
#include "Shared/MapDParameters.h"
#include "Shared/scope.h"

#if LLVM_VERSION_MAJOR >= 4
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#else
#include <llvm/Bitcode/ReaderWriter.h>
#endif
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/InstIterator.h>
#include "llvm/IR/Intrinsics.h"
#include "AggregatedColRange.h"
#include "StringDictionaryGenerations.h"
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/FormattedStream.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Transforms/Instrumentation.h>
#include <llvm/Transforms/IPO.h>
#if LLVM_VERSION_MAJOR >= 4
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#endif
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#ifdef HAVE_CUDA
#include <cuda.h>
#endif  // HAVE_CUDA
#include <future>
#include <memory>
#include <numeric>
#include <thread>
#include <set>

bool g_enable_watchdog{false};
bool g_enable_dynamic_watchdog{false};
unsigned g_dynamic_watchdog_time_limit{10000};
bool g_allow_cpu_retry{false};

std::mutex Executor::ExecutionDispatch::reduce_mutex_;

Executor::Executor(const int db_id,
                   const size_t block_size_x,
                   const size_t grid_size_x,
                   const std::string& debug_dir,
                   const std::string& debug_file,
                   ::QueryRenderer::QueryRenderManager* render_manager)
    : cgen_state_(new CgenState({}, false)),
      is_nested_(false),
      gpu_active_modules_device_mask_(0x0),
      interrupted_(false),
      render_manager_(render_manager),
      block_size_x_(block_size_x),
      grid_size_x_(grid_size_x),
      debug_dir_(debug_dir),
      debug_file_(debug_file),
      db_id_(db_id),
      catalog_(nullptr),
      temporary_tables_(nullptr),
      input_table_info_cache_(this) {}

std::shared_ptr<Executor> Executor::getExecutor(const int db_id,
                                                const std::string& debug_dir,
                                                const std::string& debug_file,
                                                const MapDParameters mapd_parameters,
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
    auto executor = std::make_shared<Executor>(
        db_id, mapd_parameters.cuda_block_size, mapd_parameters.cuda_grid_size, debug_dir, debug_file, render_manager);
    auto it_ok = executors_.insert(std::make_pair(executor_key, executor));
    CHECK(it_ok.second);
    return executor;
  }
}

namespace {

bool is_unnest(const Analyzer::Expr* expr) {
  return dynamic_cast<const Analyzer::UOper*>(expr) &&
         static_cast<const Analyzer::UOper*>(expr)->get_optype() == kUNNEST;
}

Likelihood get_likelihood(const Analyzer::Expr* expr) {
  Likelihood truth{1.0};
  auto likelihood_expr = dynamic_cast<const Analyzer::LikelihoodExpr*>(expr);
  if (likelihood_expr) {
    return Likelihood(likelihood_expr->get_likelihood());
  }
  auto u_oper = dynamic_cast<const Analyzer::UOper*>(expr);
  if (u_oper) {
    Likelihood oper_likelihood = get_likelihood(u_oper->get_operand());
    if (oper_likelihood.isInvalid())
      return Likelihood();
    if (u_oper->get_optype() == kNOT)
      return truth - oper_likelihood;
    return oper_likelihood;
  }
  auto bin_oper = dynamic_cast<const Analyzer::BinOper*>(expr);
  if (bin_oper) {
    auto lhs = bin_oper->get_left_operand();
    auto rhs = bin_oper->get_right_operand();
    Likelihood lhs_likelihood = get_likelihood(lhs);
    Likelihood rhs_likelihood = get_likelihood(rhs);
    if (lhs_likelihood.isInvalid() && rhs_likelihood.isInvalid())
      return Likelihood();
    const auto optype = bin_oper->get_optype();
    if (optype == kOR) {
      auto both_false = (truth - lhs_likelihood) * (truth - rhs_likelihood);
      return truth - both_false;
    }
    if (optype == kAND) {
      return lhs_likelihood * rhs_likelihood;
    }
    return (lhs_likelihood + rhs_likelihood) / 2.0;
  }

  return Likelihood();
}

Weight get_weight(const Analyzer::Expr* expr, int depth = 0) {
  auto like_expr = dynamic_cast<const Analyzer::LikeExpr*>(expr);
  if (like_expr) {
    // heavy weight expr, start valid weight propagation
    return Weight((like_expr->get_is_simple()) ? 200 : 1000);
  }
  auto regexp_expr = dynamic_cast<const Analyzer::RegexpExpr*>(expr);
  if (regexp_expr) {
    // heavy weight expr, start valid weight propagation
    return Weight(2000);
  }
  auto u_oper = dynamic_cast<const Analyzer::UOper*>(expr);
  if (u_oper) {
    auto weight = get_weight(u_oper->get_operand(), depth + 1);
    return weight + 1;
  }
  auto bin_oper = dynamic_cast<const Analyzer::BinOper*>(expr);
  if (bin_oper) {
    auto lhs = bin_oper->get_left_operand();
    auto rhs = bin_oper->get_right_operand();
    auto lhs_weight = get_weight(lhs, depth + 1);
    auto rhs_weight = get_weight(rhs, depth + 1);
    if (rhs->get_type_info().is_array()) {
      // heavy weight expr, start valid weight propagation
      rhs_weight = rhs_weight + Weight(100);
    }
    auto weight = lhs_weight + rhs_weight;
    return weight + 1;
  }

  if (depth > 4)
    return Weight(1);

  return Weight();
}

bool contains_unsafe_division(const Analyzer::Expr* expr) {
  auto is_div = [](const Analyzer::Expr* e) -> bool {
    auto bin_oper = dynamic_cast<const Analyzer::BinOper*>(e);
    if (bin_oper && bin_oper->get_optype() == kDIVIDE) {
      auto rhs = bin_oper->get_right_operand();
      auto rhs_constant = dynamic_cast<const Analyzer::Constant*>(rhs);
      if (!rhs_constant || rhs_constant->get_is_null())
        return true;
      const auto& datum = rhs_constant->get_constval();
      const auto& ti = rhs_constant->get_type_info();
      const auto type = ti.is_decimal() ? decimal_to_int_type(ti) : ti.get_type();
      if ((type == kBOOLEAN && datum.boolval == 0) || (type == kSMALLINT && datum.smallintval == 0) ||
          (type == kINT && datum.intval == 0) || (type == kBIGINT && datum.bigintval == 0LL) ||
          (type == kFLOAT && datum.floatval == 0.0) || (type == kDOUBLE && datum.doubleval == 0.0)) {
        return true;
      }
    }
    return false;
  };
  std::list<const Analyzer::Expr*> binoper_list;
  expr->find_expr(is_div, binoper_list);
  return !binoper_list.empty();
}

}  // namespace

StringDictionaryProxy* Executor::getStringDictionaryProxy(const int dict_id_in,
                                                          std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                                                          const bool with_generation) const {
  const int dict_id{dict_id_in < 0 ? REGULAR_DICT(dict_id_in) : dict_id_in};
  CHECK(catalog_);
  const auto dd = catalog_->getMetadataForDict(dict_id);
  std::lock_guard<std::mutex> lock(str_dict_mutex_);
  if (dd) {
    CHECK(dd->stringDict);
    CHECK_LE(dd->dictNBits, 32);
    if (row_set_mem_owner) {
#ifdef HAVE_RAVM
      CHECK(!with_generation || !execute_mutex_.try_lock());
      const auto generation = with_generation ? string_dictionary_generations_.getGeneration(dict_id) : ssize_t(-1);
#else
      const ssize_t generation = dd->stringDict->storageEntryCount();
#endif  // HAVE_RAVM
      return row_set_mem_owner->addStringDict(dd->stringDict, dict_id, generation);
    }
  }
  CHECK_EQ(0, dict_id);
  if (!lit_str_dict_proxy_) {
    std::shared_ptr<StringDictionary> tsd = std::make_shared<StringDictionary>("");
    lit_str_dict_proxy_.reset(new StringDictionaryProxy(tsd, 0));
  }
  return lit_str_dict_proxy_.get();
}

bool Executor::isCPUOnly() const {
  CHECK(catalog_);
  return !catalog_->get_dataMgr().cudaMgr_;
}

const ColumnDescriptor* Executor::getColumnDescriptor(const Analyzer::ColumnVar* col_var) const {
  return get_column_descriptor_maybe(col_var->get_column_id(), col_var->get_table_id(), *catalog_);
}

const Catalog_Namespace::Catalog* Executor::getCatalog() const {
  return catalog_;
}

const std::shared_ptr<RowSetMemoryOwner> Executor::getRowSetMemoryOwner() const {
  return row_set_mem_owner_;
}

const TemporaryTables* Executor::getTemporaryTables() const {
  return temporary_tables_;
}

Fragmenter_Namespace::TableInfo Executor::getTableInfo(const int table_id) {
  CHECK(!execute_mutex_.try_lock());
  return input_table_info_cache_.getTableInfo(table_id);
}

const TableGeneration& Executor::getTableGeneration(const int table_id) const {
  CHECK(!execute_mutex_.try_lock());
  return table_generations_.getGeneration(table_id);
}

ExpressionRange Executor::getColRange(const PhysicalInput& phys_input) const {
  CHECK(!execute_mutex_.try_lock());
  return agg_col_range_cache_.getColRange(phys_input);
}

void Executor::clearMetaInfoCache() {
  CHECK(!execute_mutex_.try_lock());
  input_table_info_cache_.clear();
  agg_col_range_cache_.clear();
  string_dictionary_generations_.clear();
  table_generations_.clear();
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
        const auto str_id = getStringDictionaryProxy(p->second, row_set_mem_owner_, true)->getIdOfString(p->first);
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
  auto iter_expr = dynamic_cast<const Analyzer::IterExpr*>(expr);
  if (iter_expr) {
#ifdef ENABLE_MULTIFRAG_JOIN
    if (iter_expr->get_rte_idx() > 0) {
      const auto offset = cgen_state_->frag_offsets_[iter_expr->get_rte_idx()];
      if (offset) {
        return {cgen_state_->ir_builder_.CreateAdd(posArg(iter_expr), offset)};
      } else {
        return {posArg(iter_expr)};
      }
    }
#endif
    return {posArg(iter_expr)};
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
    return codegen(col_var, fetch_columns, co);
  }
  auto constant = dynamic_cast<const Analyzer::Constant*>(expr);
  if (constant) {
    if (constant->get_is_null()) {
      const auto& ti = constant->get_type_info();
      return {ti.is_fp() ? static_cast<llvm::Value*>(inlineFpNull(ti)) : static_cast<llvm::Value*>(inlineIntNull(ti))};
    }
    // The dictionary encoding case should be handled by the parent expression
    // (cast, for now), here is too late to know the dictionary id
    CHECK_NE(kENCODING_DICT, constant->get_type_info().get_compression());
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
  auto datediff_expr = dynamic_cast<const Analyzer::DatediffExpr*>(expr);
  if (datediff_expr) {
    return {codegen(datediff_expr, co)};
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
  auto regexp_expr = dynamic_cast<const Analyzer::RegexpExpr*>(expr);
  if (regexp_expr) {
    return {codegen(regexp_expr, co)};
  }
  auto likelihood_expr = dynamic_cast<const Analyzer::LikelihoodExpr*>(expr);
  if (likelihood_expr) {
    return {codegen(likelihood_expr->get_arg(), fetch_columns, co)};
  }
  auto in_expr = dynamic_cast<const Analyzer::InValues*>(expr);
  if (in_expr) {
    return {codegen(in_expr, co)};
  }
  auto in_integer_set_expr = dynamic_cast<const Analyzer::InIntegerSet*>(expr);
  if (in_integer_set_expr) {
    return {codegen(in_integer_set_expr, co)};
  }
  auto function_oper_with_custom_type_handling_expr =
      dynamic_cast<const Analyzer::FunctionOperWithCustomTypeHandling*>(expr);
  if (function_oper_with_custom_type_handling_expr) {
    return {codegenFunctionOperWithCustomTypeHandling(function_oper_with_custom_type_handling_expr, co)};
  }
  auto function_oper_expr = dynamic_cast<const Analyzer::FunctionOper*>(expr);
  if (function_oper_expr) {
    return {codegenFunctionOper(function_oper_expr, co)};
  }
#ifdef HAVE_CALCITE
  abort();
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
  auto string_dict_proxy = reinterpret_cast<const StringDictionaryProxy*>(string_dict_handle);
  auto string_bytes = string_dict_proxy->getStringBytes(string_id);
  CHECK(string_bytes.first);
  return (reinterpret_cast<uint64_t>(string_bytes.first) & 0xffffffffffff) |
         (static_cast<uint64_t>(string_bytes.second) << 48);
}

extern "C" int32_t string_compress(const int64_t ptr_and_len, const int64_t string_dict_handle) {
  std::string raw_str(reinterpret_cast<char*>(extract_str_ptr_noinline(ptr_and_len)),
                      extract_str_len_noinline(ptr_and_len));
  auto string_dict_proxy = reinterpret_cast<const StringDictionaryProxy*>(string_dict_handle);
  return string_dict_proxy->getIdOfString(raw_str);
}

llvm::Value* Executor::codegen(const Analyzer::CharLengthExpr* expr, const CompilationOptions& co) {
  auto str_lv = codegen(expr->get_arg(), true, co);
  if (str_lv.size() != 3) {
    CHECK_EQ(size_t(1), str_lv.size());
    if (g_enable_watchdog) {
      throw WatchdogException("LENGTH / CHAR_LENGTH on dictionary-encoded strings would be slow");
    }
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
  const auto& ti = expr->get_arg()->get_type_info();
  CHECK(ti.is_string());
  if (g_enable_watchdog && ti.get_compression() != kENCODING_NONE) {
    throw WatchdogException("Cannot do LIKE / ILIKE on this dictionary encoded column, its cardinality is too high");
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
  const auto sdp = getStringDictionaryProxy(dict_like_arg_ti.get_comp_param(), row_set_mem_owner_, true);
  if (sdp->storageEntryCount() > 200000000) {
    return nullptr;
  }
  const auto& pattern_ti = pattern->get_type_info();
  CHECK(pattern_ti.is_string());
  CHECK_EQ(kENCODING_NONE, pattern_ti.get_compression());
  const auto& pattern_datum = pattern->get_constval();
  const auto& pattern_str = *pattern_datum.stringval;
  const auto matching_strings = sdp->getLike(pattern_str, ilike, is_simple, escape_char);
  std::list<std::shared_ptr<Analyzer::Expr>> matching_str_exprs;
  for (const auto& matching_string : matching_strings) {
    auto const_val = Parser::StringLiteral::analyzeValue(matching_string);
    matching_str_exprs.push_back(const_val->add_cast(dict_like_arg_ti));
  }
  const auto in_values = makeExpr<Analyzer::InValues>(dict_like_arg, matching_str_exprs);
  return codegen(in_values.get(), co);
}

llvm::Value* Executor::codegen(const Analyzer::RegexpExpr* expr, const CompilationOptions& co) {
  if (is_unnest(extract_cast_arg(expr->get_arg()))) {
    throw std::runtime_error("REGEXP not supported for unnested expressions");
  }
  char escape_char{'\\'};
  if (expr->get_escape_expr()) {
    auto escape_char_expr = dynamic_cast<const Analyzer::Constant*>(expr->get_escape_expr());
    CHECK(escape_char_expr);
    CHECK(escape_char_expr->get_type_info().is_string());
    CHECK_EQ(size_t(1), escape_char_expr->get_constval().stringval->size());
    escape_char = (*escape_char_expr->get_constval().stringval)[0];
  }
  auto pattern = dynamic_cast<const Analyzer::Constant*>(expr->get_pattern_expr());
  CHECK(pattern);
  auto fast_dict_pattern_lv = codegenDictRegexp(expr->get_own_arg(), pattern, escape_char, co);
  if (fast_dict_pattern_lv) {
    return fast_dict_pattern_lv;
  }
  const auto& ti = expr->get_arg()->get_type_info();
  CHECK(ti.is_string());
  if (g_enable_watchdog && ti.get_compression() != kENCODING_NONE) {
    throw WatchdogException("Cannot do REGEXP_LIKE on this dictionary encoded column, its cardinality is too high");
  }
  auto str_lv = codegen(expr->get_arg(), true, co);
  // Running on CPU for now.
  cgen_state_->must_run_on_cpu_ = true;
  if (str_lv.size() != 3) {
    CHECK_EQ(size_t(1), str_lv.size());
    str_lv.push_back(cgen_state_->emitCall("extract_str_ptr", {str_lv.front()}));
    str_lv.push_back(cgen_state_->emitCall("extract_str_len", {str_lv.front()}));
    cgen_state_->must_run_on_cpu_ = true;
  }
  auto regexp_expr_arg_lvs = codegen(expr->get_pattern_expr(), true, co);
  CHECK_EQ(size_t(3), regexp_expr_arg_lvs.size());
  const bool is_nullable{!expr->get_arg()->get_type_info().get_notnull()};
  std::vector<llvm::Value*> regexp_args{str_lv[1], str_lv[2], regexp_expr_arg_lvs[1], regexp_expr_arg_lvs[2]};
  std::string fn_name("regexp_like");
  regexp_args.push_back(ll_int(int8_t(escape_char)));
  if (is_nullable) {
    fn_name += "_nullable";
    regexp_args.push_back(inlineIntNull(expr->get_type_info()));
    return cgen_state_->emitExternalCall(fn_name, get_int_type(8, cgen_state_->context_), regexp_args);
  }
  return cgen_state_->emitExternalCall(fn_name, get_int_type(1, cgen_state_->context_), regexp_args);
}

llvm::Value* Executor::codegenDictRegexp(const std::shared_ptr<Analyzer::Expr> pattern_arg,
                                         const Analyzer::Constant* pattern,
                                         const char escape_char,
                                         const CompilationOptions& co) {
  const auto cast_oper = std::dynamic_pointer_cast<Analyzer::UOper>(pattern_arg);
  if (!cast_oper) {
    return nullptr;
  }
  CHECK(cast_oper);
  CHECK_EQ(kCAST, cast_oper->get_optype());
  const auto dict_regexp_arg = cast_oper->get_own_operand();
  const auto& dict_regexp_arg_ti = dict_regexp_arg->get_type_info();
  CHECK(dict_regexp_arg_ti.is_string());
  CHECK_EQ(kENCODING_DICT, dict_regexp_arg_ti.get_compression());
  const auto sdp = getStringDictionaryProxy(dict_regexp_arg_ti.get_comp_param(), row_set_mem_owner_, true);
  if (sdp->storageEntryCount() > 15000000) {
    return nullptr;
  }
  const auto& pattern_ti = pattern->get_type_info();
  CHECK(pattern_ti.is_string());
  CHECK_EQ(kENCODING_NONE, pattern_ti.get_compression());
  const auto& pattern_datum = pattern->get_constval();
  const auto& pattern_str = *pattern_datum.stringval;
  const auto matching_strings = sdp->getRegexpLike(pattern_str, escape_char);
  std::list<std::shared_ptr<Analyzer::Expr>> matching_str_exprs;
  for (const auto& matching_string : matching_strings) {
    auto const_val = Parser::StringLiteral::analyzeValue(matching_string);
    matching_str_exprs.push_back(const_val->add_cast(dict_regexp_arg_ti));
  }
  const auto in_values = makeExpr<Analyzer::InValues>(dict_regexp_arg, matching_str_exprs);
  return codegen(in_values.get(), co);
}

llvm::Value* Executor::codegen(const Analyzer::InValues* expr, const CompilationOptions& co) {
  const auto in_arg = expr->get_arg();
  if (is_unnest(in_arg)) {
    throw std::runtime_error("IN not supported for unnested expressions");
  }
  const auto& expr_ti = expr->get_type_info();
  CHECK(expr_ti.is_boolean());
  const auto lhs_lvs = codegen(in_arg, true, co);
  llvm::Value* result{nullptr};
  if (expr_ti.get_notnull()) {
    result = llvm::ConstantInt::get(llvm::IntegerType::getInt1Ty(cgen_state_->context_), false);
  } else {
    result = ll_int(int8_t(0));
  }
  CHECK(result);
  if (co.hoist_literals_) {  // TODO(alex): remove this constraint
    auto in_vals_bitmap = createInValuesBitmap(expr, co);
    if (in_vals_bitmap) {
      if (in_vals_bitmap->isEmpty()) {
        return in_vals_bitmap->hasNull() ? inlineIntNull(SQLTypeInfo(kBOOLEAN, false)) : result;
      }
      CHECK_EQ(size_t(1), lhs_lvs.size());
      return cgen_state_->addInValuesBitmap(in_vals_bitmap)->codegen(lhs_lvs.front(), this);
    }
  }
  if (expr_ti.get_notnull()) {
    for (auto in_val : expr->get_value_list()) {
      result = cgen_state_->ir_builder_.CreateOr(
          result, toBool(codegenCmp(kEQ, kONE, lhs_lvs, in_arg->get_type_info(), in_val.get(), co)));
    }
  } else {
    for (auto in_val : expr->get_value_list()) {
      const auto crt = codegenCmp(kEQ, kONE, lhs_lvs, in_arg->get_type_info(), in_val.get(), co);
      result = cgen_state_->emitCall("logical_or", {result, crt, inlineIntNull(expr_ti)});
    }
  }
  return result;
}

llvm::Value* Executor::codegen(const Analyzer::InIntegerSet* in_integer_set, const CompilationOptions& co) {
  const auto in_arg = in_integer_set->get_arg();
  if (is_unnest(in_arg)) {
    throw std::runtime_error("IN not supported for unnested expressions");
  }
  const auto& ti = in_integer_set->get_arg()->get_type_info();
  const auto needle_null_val = inline_int_null_val(ti);
  if (!co.hoist_literals_) {
    // We never run without literal hoisting in real world scenarios, this avoids a crash when testing.
    throw std::runtime_error(
        "IN subquery with many right-hand side values not supported when literal hoisting is disabled");
  }
  auto in_vals_bitmap = boost::make_unique<InValuesBitmap>(
      in_integer_set->get_value_list(),
      needle_null_val,
      co.device_type_ == ExecutorDeviceType::GPU ? Data_Namespace::GPU_LEVEL : Data_Namespace::CPU_LEVEL,
      deviceCount(co.device_type_),
      &catalog_->get_dataMgr());
  const auto& in_integer_set_ti = in_integer_set->get_type_info();
  CHECK(in_integer_set_ti.is_boolean());
  const auto lhs_lvs = codegen(in_arg, true, co);
  llvm::Value* result{nullptr};
  if (in_integer_set_ti.get_notnull()) {
    result = llvm::ConstantInt::get(llvm::IntegerType::getInt1Ty(cgen_state_->context_), false);
  } else {
    result = ll_int(int8_t(0));
  }
  CHECK(result);
  if (in_vals_bitmap->isEmpty()) {
    return in_vals_bitmap->hasNull() ? inlineIntNull(SQLTypeInfo(kBOOLEAN, false)) : result;
  }
  CHECK_EQ(size_t(1), lhs_lvs.size());
  return cgen_state_->addInValuesBitmap(in_vals_bitmap)->codegen(lhs_lvs.front(), this);
}

std::unique_ptr<InValuesBitmap> Executor::createInValuesBitmap(const Analyzer::InValues* in_values,
                                                               const CompilationOptions& co) {
  const auto& value_list = in_values->get_value_list();
  const auto val_count = value_list.size();
  const auto& ti = in_values->get_arg()->get_type_info();
  if (!(ti.is_integer() || (ti.is_string() && ti.get_compression() == kENCODING_DICT))) {
    return nullptr;
  }
  const auto sdp = ti.is_string() ? getStringDictionaryProxy(ti.get_comp_param(), row_set_mem_owner_, true) : nullptr;
  if (val_count > 3) {
    typedef decltype(value_list.begin()) ListIterator;
    std::vector<int64_t> values;
    const auto needle_null_val = inline_int_null_val(ti);
    const int worker_count = val_count > 10000 ? cpu_threads() : int(1);
    std::vector<std::vector<int64_t>> values_set(worker_count, std::vector<int64_t>());
    std::vector<std::future<bool>> worker_threads;
    auto start_it = value_list.begin();
    for (size_t i = 0, start_val = 0, stride = (val_count + worker_count - 1) / worker_count;
         i < val_count && start_val < val_count;
         ++i, start_val += stride, std::advance(start_it, stride)) {
      auto end_it = start_it;
      std::advance(end_it, std::min(stride, val_count - start_val));
      const auto do_work = [&](
          std::vector<int64_t>& out_vals, const ListIterator start, const ListIterator end) -> bool {
        for (auto val_it = start; val_it != end; ++val_it) {
          const auto& in_val = *val_it;
          const auto in_val_const = dynamic_cast<const Analyzer::Constant*>(extract_cast_arg(in_val.get()));
          if (!in_val_const) {
            return false;
          }
          const auto& in_val_ti = in_val->get_type_info();
          CHECK(in_val_ti == ti);
          if (ti.is_string()) {
            CHECK(sdp);
            const auto string_id = in_val_const->get_is_null()
                                       ? needle_null_val
                                       : sdp->getIdOfString(*in_val_const->get_constval().stringval);
            if (string_id != StringDictionary::INVALID_STR_ID) {
              out_vals.push_back(string_id);
            }
          } else {
            out_vals.push_back(codegenIntConst(in_val_const)->getSExtValue());
          }
        }
        return true;
      };
      if (worker_count > 1) {
        worker_threads.push_back(std::async(std::launch::async, do_work, std::ref(values_set[i]), start_it, end_it));
      } else {
        do_work(std::ref(values), start_it, end_it);
      }
    }
    bool success = true;
    for (auto& worker : worker_threads) {
      success &= worker.get();
    }
    if (!success) {
      return nullptr;
    }
    if (worker_count > 1) {
      size_t total_val_count = 0;
      for (auto& vals : values_set) {
        total_val_count += vals.size();
      }
      values.reserve(total_val_count);
      for (auto& vals : values_set) {
        values.insert(values.end(), vals.begin(), vals.end());
      }
    }
    try {
      return boost::make_unique<InValuesBitmap>(
          values,
          needle_null_val,
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
  abort();
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
      abort();
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
            return std::make_shared<FixedWidthInt>(ti.get_size());
          }
          CHECK(false);
      }
    }
    case kENCODING_DICT:
      CHECK(ti.is_string());
      if (ti.get_size() < ti.get_logical_size()) {
        return std::make_shared<FixedWidthUnsigned>(ti.get_size());
      }
      return std::make_shared<FixedWidthInt>(ti.get_size());
    case kENCODING_FIXED: {
      const auto bit_width = col_var->get_comp_param();
      CHECK_EQ(0, bit_width % 8);
      return std::make_shared<FixedWidthInt>(bit_width / 8);
    }
    default:
      abort();
  }
}

size_t get_col_bit_width(const Analyzer::ColumnVar* col_var) {
  const auto& type_info = col_var->get_type_info();
  return get_bit_width(type_info);
}

}  // namespace

std::vector<llvm::Value*> Executor::codegen(const Analyzer::ColumnVar* col_var,
                                            const bool fetch_column,
                                            const CompilationOptions& co) {
  const auto col_var_lvs = codegenColVar(col_var, fetch_column, co);
  if (!cgen_state_->outer_join_cond_lv_ || col_var->get_rte_idx() == 0) {
    return col_var_lvs;
  }
  return codegenOuterJoinNullPlaceholder(col_var_lvs, col_var);
}

std::vector<llvm::Value*> Executor::codegenColVar(const Analyzer::ColumnVar* col_var,
                                                  const bool fetch_column,
                                                  const CompilationOptions& co) {
  const bool hoist_literals = co.hoist_literals_;
  auto col_id = col_var->get_column_id();
  const auto rte_idx = col_var->get_rte_idx() == -1 ? int(0) : col_var->get_rte_idx();
#ifdef ENABLE_MULTIFRAG_JOIN
  CHECK_LT(rte_idx, cgen_state_->frag_offsets_.size());
#endif
  if (col_var->get_table_id() > 0) {
    auto cd = get_column_descriptor(col_id, col_var->get_table_id(), *catalog_);
    if (cd->isVirtualCol) {
      CHECK(cd->columnName == "rowid");
#ifndef ENABLE_MULTIFRAG_JOIN
      if (rte_idx > 0) {
        // rowid for inner scan, the fragment offset from the outer scan
        // is meaningless, the relative position in the scan is the rowid
        return {posArg(col_var)};
      }
#endif
      const auto offset = cgen_state_->frag_offsets_[rte_idx];
      if (offset) {
        const auto& table_generation = getTableGeneration(col_var->get_table_id());
        if (table_generation.start_rowid > 0) {
          Datum d;
          d.bigintval = table_generation.start_rowid;
          const auto start_rowid = makeExpr<Analyzer::Constant>(kBIGINT, false, d);
          const auto start_rowid_lvs = codegen(start_rowid.get(), kENCODING_NONE, -1, co);
          CHECK_EQ(size_t(1), start_rowid_lvs.size());
          return {cgen_state_->ir_builder_.CreateAdd(cgen_state_->ir_builder_.CreateAdd(posArg(col_var), offset),
                                                     start_rowid_lvs.front())};
        } else {
          return {cgen_state_->ir_builder_.CreateAdd(posArg(col_var), offset)};
        }
      } else {
        return {posArg(col_var)};
      }
    }
  }
  const auto grouped_col_lv = resolveGroupedColumnReference(col_var);
  if (grouped_col_lv) {
    return {grouped_col_lv};
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
    if (plan_state_->isLazyFetchColumn(col_var)) {
      plan_state_->columns_to_fetch_.insert(std::make_pair(col_var->get_table_id(), col_var->get_column_id()));
    }
    return codegen(hash_join_lhs, fetch_column, co);
  }
  auto pos_arg = posArg(col_var);
  auto col_byte_stream = colByteStream(col_var, fetch_column, hoist_literals);
  if (plan_state_->isLazyFetchColumn(col_var)) {
    plan_state_->columns_to_not_fetch_.insert(std::make_pair(col_var->get_table_id(), col_var->get_column_id()));
#ifdef ENABLE_MULTIFRAG_JOIN
    if (rte_idx > 0) {
      const auto offset = cgen_state_->frag_offsets_[rte_idx];
      if (offset) {
        return {cgen_state_->ir_builder_.CreateAdd(pos_arg, offset)};
      } else {
        return {pos_arg};
      }
    }
#endif
    return {pos_arg};
  }
  const auto& col_ti = col_var->get_type_info();
  if (col_ti.is_string() && col_ti.get_compression() == kENCODING_NONE) {
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
  if (col_ti.is_array()) {
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
    if ((col_ti.get_compression() == kENCODING_FIXED ||
         (col_ti.get_compression() == kENCODING_DICT && col_ti.get_size() < 4)) &&
        !col_ti.get_notnull()) {
      dec_val_cast = codgenAdjustFixedEncNull(dec_val_cast, col_ti);
    }
  } else {
    CHECK_EQ(kENCODING_NONE, col_ti.get_compression());
    CHECK(dec_type->isFloatTy() || dec_type->isDoubleTy());
    if (dec_type->isDoubleTy()) {
      CHECK(col_ti.get_type() == kDOUBLE);
    } else if (dec_type->isFloatTy()) {
      CHECK(col_ti.get_type() == kFLOAT);
    }
    dec_val_cast = dec_val;
  }
  CHECK(dec_val_cast);
  auto it_ok = cgen_state_->fetch_cache_.insert(std::make_pair(local_col_id, std::vector<llvm::Value*>{dec_val_cast}));
  CHECK(it_ok.second);
  return {it_ok.first->second};
}

namespace {

SQLTypes get_phys_int_type(const size_t byte_sz) {
  switch (byte_sz) {
    case 1:
      return kBOOLEAN;
    case 2:
      return kSMALLINT;
    case 4:
      return kINT;
    case 8:
      return kBIGINT;
    default:
      CHECK(false);
  }
  return kNULLT;
}

}  // namespace

llvm::Value* Executor::codgenAdjustFixedEncNull(llvm::Value* val, const SQLTypeInfo& col_ti) {
  CHECK_LT(col_ti.get_size(), col_ti.get_logical_size());
  const auto col_phys_width = col_ti.get_size() * 8;
  auto from_typename = "int" + std::to_string(col_phys_width) + "_t";
  auto adjusted = cgen_state_->ir_builder_.CreateCast(
      llvm::Instruction::CastOps::Trunc, val, get_int_type(col_phys_width, cgen_state_->context_));
  if (col_ti.get_compression() == kENCODING_DICT) {
    from_typename = "u" + from_typename;
    llvm::Value* from_null{nullptr};
    switch (col_ti.get_size()) {
      case 1:
        from_null = ll_int(std::numeric_limits<uint8_t>::max());
        break;
      case 2:
        from_null = ll_int(std::numeric_limits<uint16_t>::max());
        break;
      default:
        CHECK(false);
    }
    return cgen_state_->emitCall("cast_" + from_typename + "_to_" + numeric_type_name(col_ti) + "_nullable",
                                 {adjusted, from_null, inlineIntNull(col_ti)});
  }
  SQLTypeInfo col_phys_ti(get_phys_int_type(col_ti.get_size()),
                          col_ti.get_dimension(),
                          col_ti.get_scale(),
                          false,
                          kENCODING_NONE,
                          0,
                          col_ti.get_subtype());
  return cgen_state_->emitCall("cast_" + from_typename + "_to_" + numeric_type_name(col_ti) + "_nullable",
                               {adjusted, inlineIntNull(col_phys_ti), inlineIntNull(col_ti)});
}

std::vector<llvm::Value*> Executor::codegenOuterJoinNullPlaceholder(const std::vector<llvm::Value*>& orig_lvs,
                                                                    const Analyzer::ColumnVar* col_var) {
  const auto grouped_col_lv = resolveGroupedColumnReference(col_var);
  if (grouped_col_lv) {
    return {grouped_col_lv};
  }
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
  const auto& null_ti = col_var->get_type_info();
  const auto null_constant = makeExpr<Analyzer::Constant>(null_ti, true, Datum{0});
  const auto null_target_lvs = codegen(
      null_constant.get(), false, CompilationOptions{ExecutorDeviceType::CPU, false, ExecutorOptLevel::Default, false});
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

llvm::Value* Executor::resolveGroupedColumnReference(const Analyzer::ColumnVar* col_var) {
  auto col_id = col_var->get_column_id();
  if (col_var->get_rte_idx() >= 0 && !is_nested_) {
    return nullptr;
  }
  CHECK((col_id == 0) || (col_var->get_rte_idx() >= 0 && col_var->get_table_id() > 0));
  const auto var = dynamic_cast<const Analyzer::Var*>(col_var);
  CHECK(var);
  col_id = var->get_varno();
  CHECK_GE(col_id, 1);
  if (var->get_which_row() == Analyzer::Var::kGROUPBY) {
    CHECK_LE(static_cast<size_t>(col_id), cgen_state_->group_by_expr_cache_.size());
    return cgen_state_->group_by_expr_cache_[col_id - 1];
  }
  return nullptr;
}

// returns the byte stream argument and the position for the given column
llvm::Value* Executor::colByteStream(const Analyzer::ColumnVar* col_var,
                                     const bool fetch_column,
                                     const bool hoist_literals) {
  CHECK_GE(cgen_state_->row_func_->arg_size(), size_t(3));
  const auto stream_arg_name = "col_buf" + std::to_string(getLocalColumnId(col_var, fetch_column));
  for (auto& arg : cgen_state_->row_func_->args()) {
    if (arg.getName() == stream_arg_name) {
      CHECK(arg.getType() == llvm::Type::getInt8PtrTy(cgen_state_->context_));
      return &arg;
    }
  }
  CHECK(false);
  return nullptr;
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
#ifdef ENABLE_JOIN_EXEC
  else if (dynamic_cast<const Analyzer::IterExpr*>(expr)) {
    const auto iter = static_cast<const Analyzer::IterExpr*>(expr);
    const auto hash_pos_it = cgen_state_->scan_idx_to_hash_pos_.find(iter->get_rte_idx());
    if (hash_pos_it != cgen_state_->scan_idx_to_hash_pos_.end()) {
      return hash_pos_it->second;
    }
    const auto inner_it =
        cgen_state_->scan_to_iterator_.find(InputDescriptor(iter->get_table_id(), iter->get_rte_idx()));
    if (inner_it != cgen_state_->scan_to_iterator_.end()) {
      CHECK(inner_it->second.first);
      CHECK(inner_it->second.first->getType()->isIntegerTy(64));
      return inner_it->second.first;
    }
  }
#endif
  for (auto& arg : cgen_state_->row_func_->args()) {
    if (arg.getName() == "pos") {
      CHECK(arg.getType()->isIntegerTy(64));
      return &arg;
    }
  }
  abort();
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
      return type_info.get_notnull() ? std::vector<llvm::Value*>{llvm::ConstantInt::get(
                                           get_int_type(1, cgen_state_->context_), constant->get_constval().boolval)}
                                     : std::vector<llvm::Value*>{llvm::ConstantInt::get(
                                           get_int_type(8, cgen_state_->context_), constant->get_constval().boolval)};
    case kSMALLINT:
    case kINT:
    case kBIGINT:
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH:
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
        if (enc_type == kENCODING_DICT) {
          return {ll_int(static_cast<int32_t>(inline_int_null_val(type_info)))};
        }
        return {ll_int(int64_t(0)),
                llvm::Constant::getNullValue(llvm::PointerType::get(get_int_type(8, cgen_state_->context_), 0)),
                ll_int(int32_t(0))};
      }
      const auto& str_const = *constant->get_constval().stringval;
      if (enc_type == kENCODING_DICT) {
        return {ll_int(getStringDictionaryProxy(dict_id, row_set_mem_owner_, true)->getIdOfString(str_const))};
      }
      return {ll_int(int64_t(0)),
              cgen_state_->addStringConstant(str_const),
              ll_int(static_cast<int32_t>(str_const.size()))};
    }
    default:
      CHECK(false);
  }
  abort();
}

std::vector<llvm::Value*> Executor::codegenHoistedConstants(const std::vector<const Analyzer::Constant*>& constants,
                                                            const EncodingType enc_type,
                                                            const int dict_id) {
  CHECK(!constants.empty());
  const auto& type_info = constants.front()->get_type_info();
  auto lit_buff_lv = get_arg_by_name(cgen_state_->row_func_, "literals");
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
  if (type_info.is_integer() || type_info.is_decimal() || type_info.is_time() || type_info.is_timeinterval() ||
      type_info.is_string() || type_info.is_boolean()) {
    val_ptr_type = llvm::PointerType::get(llvm::IntegerType::get(cgen_state_->context_, val_bits), 0);
  } else {
    CHECK(type_info.get_type() == kFLOAT || type_info.get_type() == kDOUBLE);
    val_ptr_type = (type_info.get_type() == kFLOAT) ? llvm::Type::getFloatPtrTy(cgen_state_->context_)
                                                    : llvm::Type::getDoublePtrTy(cgen_state_->context_);
  }
  auto lit_lv =
      cgen_state_->ir_builder_.CreateLoad(cgen_state_->ir_builder_.CreateBitCast(lit_buf_start, val_ptr_type));
  if (type_info.is_boolean() && type_info.get_notnull()) {
    return {toBool(lit_lv)};
  }
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
      case_llvm_type = get_int_type(8 * case_ti.get_logical_size(), cgen_state_->context_);
    } else {
      is_real_str = true;
      case_llvm_type = get_int_type(64, cgen_state_->context_);
    }
  } else if (case_ti.is_boolean()) {
    case_llvm_type = get_int_type(8 * case_ti.get_logical_size(), cgen_state_->context_);
  }
  CHECK(case_llvm_type);
  const auto& else_ti = case_expr->get_else_expr()->get_type_info();
  CHECK_EQ(else_ti.get_type(), case_ti.get_type());
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
  if (else_lvs.size() == 3) {
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

llvm::Value* Executor::codegen(const Analyzer::DatediffExpr* datediff_expr, const CompilationOptions& co) {
  static_assert(sizeof(time_t) == 4 || sizeof(time_t) == 8, "Unsupported time_t size");
  auto start = codegen(datediff_expr->get_start_expr(), true, co).front();
  CHECK(start->getType()->isIntegerTy(32) || start->getType()->isIntegerTy(64));
  if (sizeof(time_t) == 4 && start->getType()->isIntegerTy(64)) {
    start = cgen_state_->ir_builder_.CreateCast(
        llvm::Instruction::CastOps::Trunc, start, get_int_type(32, cgen_state_->context_));
  }
  auto end = codegen(datediff_expr->get_end_expr(), true, co).front();
  CHECK(end->getType()->isIntegerTy(32) || end->getType()->isIntegerTy(64));
  if (sizeof(time_t) == 4 && end->getType()->isIntegerTy(64)) {
    end = cgen_state_->ir_builder_.CreateCast(
        llvm::Instruction::CastOps::Trunc, end, get_int_type(32, cgen_state_->context_));
  }
  std::vector<llvm::Value*> datediff_args{ll_int(static_cast<int32_t>(datediff_expr->get_field())), start, end};
  std::string datediff_fname{"DateDiff"};
  const auto& start_ti = datediff_expr->get_start_expr()->get_type_info();
  const auto& end_ti = datediff_expr->get_end_expr()->get_type_info();
  const auto& ret_ti = datediff_expr->get_type_info();
  if (!start_ti.get_notnull() || !end_ti.get_notnull()) {
    datediff_args.push_back(inlineIntNull(ret_ti));
    datediff_fname += "Nullable";
  }
  return cgen_state_->emitExternalCall(datediff_fname, get_int_type(64, cgen_state_->context_), datediff_args);
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
      abort();
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
      abort();
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
      abort();
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
      abort();
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
      abort();
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
      return plan_state_->join_info_.join_hash_table_->codegenSlot(co);
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
  if (lhs_ti.is_integer() || lhs_ti.is_decimal() || lhs_ti.is_time() || lhs_ti.is_boolean() || lhs_ti.is_string() ||
      lhs_ti.is_timeinterval()) {
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
    if (g_cluster) {
      throw std::runtime_error(
          "Comparison between a dictionary-encoded and a none-encoded string not supported for distributed queries");
    }
    if (g_enable_watchdog) {
      throw WatchdogException("Comparison between a dictionary-encoded and a none-encoded string would be slow");
    }
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
         ll_int(int64_t(getStringDictionaryProxy(elem_ti.get_comp_param(), row_set_mem_owner_, true))),
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

llvm::Value* Executor::codegenLogicalShortCircuit(const Analyzer::BinOper* bin_oper, const CompilationOptions& co) {
  const auto optype = bin_oper->get_optype();
  auto lhs = bin_oper->get_left_operand();
  auto rhs = bin_oper->get_right_operand();

  if (contains_unsafe_division(rhs)) {
    // rhs contains a possible div-by-0: short-circuit
  } else if (contains_unsafe_division(lhs)) {
    // lhs contains a possible div-by-0: swap and short-circuit
    std::swap(rhs, lhs);
  } else if (((optype == kOR && get_likelihood(lhs) > 0.90) || (optype == kAND && get_likelihood(lhs) < 0.10)) &&
             get_weight(rhs) > 10) {
    // short circuit if we're likely to see either (trueA || heavyB) or (falseA && heavyB)
  } else if (((optype == kOR && get_likelihood(rhs) > 0.90) || (optype == kAND && get_likelihood(rhs) < 0.10)) &&
             get_weight(lhs) > 10) {
    // swap and short circuit if we're likely to see either (heavyA || trueB) or (heavyA && falseB)
    std::swap(rhs, lhs);
  } else {
    // no motivation to short circuit
    return nullptr;
  }

  const auto& ti = bin_oper->get_type_info();
  auto lhs_lv = codegen(lhs, true, co).front();

  // Here the linear control flow will diverge and expressions cached during the
  // code branch code generation (currently just column decoding) are not going
  // to be available once we're done generating the short-circuited logic.
  // Take a snapshot of the cache with FetchCacheAnchor and restore it once
  // the control flow converges.
  FetchCacheAnchor anchor(cgen_state_.get());

  auto rhs_bb = llvm::BasicBlock::Create(cgen_state_->context_, "rhs_bb", cgen_state_->row_func_);
  auto ret_bb = llvm::BasicBlock::Create(cgen_state_->context_, "ret_bb", cgen_state_->row_func_);
  llvm::BasicBlock* nullcheck_ok_bb{nullptr};
  llvm::BasicBlock* nullcheck_fail_bb{nullptr};

  if (!ti.get_notnull()) {
    // need lhs nullcheck before short circuiting
    nullcheck_ok_bb = llvm::BasicBlock::Create(cgen_state_->context_, "nullcheck_ok_bb", cgen_state_->row_func_);
    nullcheck_fail_bb = llvm::BasicBlock::Create(cgen_state_->context_, "nullcheck_fail_bb", cgen_state_->row_func_);
    if (lhs_lv->getType()->isIntegerTy(1)) {
      lhs_lv = castToTypeIn(lhs_lv, 8);
    }
    auto lhs_nullcheck = cgen_state_->ir_builder_.CreateICmpEQ(lhs_lv, inlineIntNull(ti));
    cgen_state_->ir_builder_.CreateCondBr(lhs_nullcheck, nullcheck_fail_bb, nullcheck_ok_bb);
    cgen_state_->ir_builder_.SetInsertPoint(nullcheck_ok_bb);
  }

  auto sc_check_bb = cgen_state_->ir_builder_.GetInsertBlock();
  auto cnst_lv = llvm::ConstantInt::get(lhs_lv->getType(), (optype == kOR));
  // Branch to codegen rhs if NOT getting (true || rhs) or (false && rhs), likelihood of the branch is < 0.10
  cgen_state_->ir_builder_.CreateCondBr(cgen_state_->ir_builder_.CreateICmpNE(lhs_lv, cnst_lv),
                                        rhs_bb,
                                        ret_bb,
                                        llvm::MDBuilder(cgen_state_->context_).createBranchWeights(10, 90));

  // Codegen rhs when unable to short circuit.
  cgen_state_->ir_builder_.SetInsertPoint(rhs_bb);
  auto rhs_lv = codegen(rhs, true, co).front();
  if (!ti.get_notnull()) {
    // need rhs nullcheck as well
    if (rhs_lv->getType()->isIntegerTy(1)) {
      rhs_lv = castToTypeIn(rhs_lv, 8);
    }
    auto rhs_nullcheck = cgen_state_->ir_builder_.CreateICmpEQ(rhs_lv, inlineIntNull(ti));
    cgen_state_->ir_builder_.CreateCondBr(rhs_nullcheck, nullcheck_fail_bb, ret_bb);
  } else {
    cgen_state_->ir_builder_.CreateBr(ret_bb);
  }
  auto rhs_codegen_bb = cgen_state_->ir_builder_.GetInsertBlock();

  if (!ti.get_notnull()) {
    cgen_state_->ir_builder_.SetInsertPoint(nullcheck_fail_bb);
    cgen_state_->ir_builder_.CreateBr(ret_bb);
  }

  cgen_state_->ir_builder_.SetInsertPoint(ret_bb);
  auto result_phi = cgen_state_->ir_builder_.CreatePHI(lhs_lv->getType(), (!ti.get_notnull()) ? 3 : 2);
  if (!ti.get_notnull())
    result_phi->addIncoming(inlineIntNull(ti), nullcheck_fail_bb);
  result_phi->addIncoming(cnst_lv, sc_check_bb);
  result_phi->addIncoming(rhs_lv, rhs_codegen_bb);
  return result_phi;
}

llvm::Value* Executor::codegenLogical(const Analyzer::BinOper* bin_oper, const CompilationOptions& co) {
  const auto optype = bin_oper->get_optype();
  CHECK(IS_LOGIC(optype));

  if (llvm::Value* short_circuit = codegenLogicalShortCircuit(bin_oper, co))
    return short_circuit;

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
      abort();
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
  llvm::Value* operand_lv{nullptr};
  if (operand_as_const) {
    const auto operand_lvs = codegen(operand_as_const, ti.get_compression(), ti.get_comp_param(), co);
    if (operand_lvs.size() == 3) {
      operand_lv = cgen_state_->emitCall("string_pack", {operand_lvs[1], operand_lvs[2]});
    } else {
      operand_lv = operand_lvs.front();
    }
  } else {
    operand_lv = codegen(operand, true, co).front();
  }
  const auto& operand_ti = operand->get_type_info();
  return codegenCast(operand_lv, operand_ti, ti, operand_as_const);
}

llvm::Value* Executor::codegenCast(llvm::Value* operand_lv,
                                   const SQLTypeInfo& operand_ti,
                                   const SQLTypeInfo& ti,
                                   const bool operand_is_const) {
  if (operand_lv->getType()->isIntegerTy()) {
    if (operand_ti.is_string()) {
      return codegenCastFromString(operand_lv, operand_ti, ti, operand_is_const);
    }
    CHECK(operand_ti.is_integer() || operand_ti.is_decimal() || operand_ti.is_time() || operand_ti.is_boolean());
    if (operand_ti.is_boolean()) {
      CHECK(operand_lv->getType()->isIntegerTy(1) || operand_lv->getType()->isIntegerTy(8));
      if (operand_lv->getType()->isIntegerTy(1)) {
        operand_lv = castToTypeIn(operand_lv, 8);
      }
    }
    if (operand_ti.get_type() == kTIMESTAMP && ti.get_type() == kDATE) {
      // Maybe we should instead generate DatetruncExpr directly from RelAlgTranslator
      // for this pattern. However, DatetruncExpr is supposed to return a timestamp,
      // whereas this cast returns a date. The underlying type for both is still the same,
      // but it still doesn't look like a good idea to misuse DatetruncExpr.
      return codegenCastTimestampToDate(operand_lv, !ti.get_notnull());
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

llvm::Value* Executor::codegenCastTimestampToDate(llvm::Value* ts_lv, const bool nullable) {
  static_assert(sizeof(time_t) == 4 || sizeof(time_t) == 8, "Unsupported time_t size");
  CHECK(ts_lv->getType()->isIntegerTy(32) || ts_lv->getType()->isIntegerTy(64));
  if (sizeof(time_t) == 4 && ts_lv->getType()->isIntegerTy(64)) {
    ts_lv = cgen_state_->ir_builder_.CreateCast(
        llvm::Instruction::CastOps::Trunc, ts_lv, get_int_type(32, cgen_state_->context_));
  }
  std::vector<llvm::Value*> datetrunc_args{ll_int(static_cast<int32_t>(dtDAY)), ts_lv};
  std::string datetrunc_fname{"DateTruncate"};
  if (nullable) {
    datetrunc_args.push_back(inlineIntNull(SQLTypeInfo(ts_lv->getType()->isIntegerTy(64) ? kBIGINT : kINT, false)));
    datetrunc_fname += "Nullable";
  }
  return cgen_state_->emitExternalCall(datetrunc_fname, get_int_type(64, cgen_state_->context_), datetrunc_args);
}

llvm::Value* Executor::codegenCastFromString(llvm::Value* operand_lv,
                                             const SQLTypeInfo& operand_ti,
                                             const SQLTypeInfo& ti,
                                             const bool operand_is_const) {
  if (!ti.is_string()) {
    throw std::runtime_error("Cast from " + operand_ti.get_type_name() + " to " + ti.get_type_name() +
                             " not supported");
  }
  if (operand_ti.get_compression() == kENCODING_NONE && ti.get_compression() == kENCODING_NONE) {
    return operand_lv;
  }
  // dictionary encode non-constant
  if (operand_ti.get_compression() != kENCODING_DICT && !operand_is_const) {
    if (g_cluster) {
      throw std::runtime_error(
          "Cast from none-encoded string to dictionary-encoded not supported for distributed queries");
    }
    if (g_enable_watchdog) {
      throw WatchdogException("Cast from none-encoded string to dictionary-encoded would be slow");
    }
    CHECK_EQ(kENCODING_NONE, operand_ti.get_compression());
    CHECK_EQ(kENCODING_DICT, ti.get_compression());
    CHECK(operand_lv->getType()->isIntegerTy(64));
    cgen_state_->must_run_on_cpu_ = true;
    return cgen_state_->emitExternalCall(
        "string_compress",
        get_int_type(32, cgen_state_->context_),
        {operand_lv, ll_int(int64_t(getStringDictionaryProxy(ti.get_comp_param(), row_set_mem_owner_, true)))});
  }
  CHECK(operand_lv->getType()->isIntegerTy(32));
  if (ti.get_compression() == kENCODING_NONE) {
    if (g_cluster) {
      throw std::runtime_error(
          "Cast from dictionary-encoded string to none-encoded not supported for distributed queries");
    }
    if (g_enable_watchdog) {
      throw WatchdogException("Cast from dictionary-encoded string to none-encoded would be slow");
    }
    CHECK_EQ(kENCODING_DICT, operand_ti.get_compression());
    cgen_state_->must_run_on_cpu_ = true;
    return cgen_state_->emitExternalCall(
        "string_decompress",
        get_int_type(64, cgen_state_->context_),
        {operand_lv, ll_int(int64_t(getStringDictionaryProxy(operand_ti.get_comp_param(), row_set_mem_owner_, true)))});
  }
  CHECK(operand_is_const);
  CHECK_EQ(kENCODING_DICT, ti.get_compression());
  return operand_lv;
}

llvm::Value* Executor::codegenCastBetweenIntTypes(llvm::Value* operand_lv,
                                                  const SQLTypeInfo& operand_ti,
                                                  const SQLTypeInfo& ti,
                                                  bool upscale) {
  if (ti.is_decimal()) {
    if (upscale) {
      CHECK(!operand_ti.is_decimal() || operand_ti.get_scale() <= ti.get_scale());
      operand_lv = cgen_state_->ir_builder_.CreateSExt(operand_lv, get_int_type(64, cgen_state_->context_));
      const auto scale_lv = llvm::ConstantInt::get(get_int_type(64, cgen_state_->context_),
                                                   exp_to_scale(ti.get_scale() - operand_ti.get_scale()));
      if (operand_ti.get_notnull()) {
        operand_lv = cgen_state_->ir_builder_.CreateMul(operand_lv, scale_lv);
      } else {
        operand_lv = cgen_state_->emitCall("scale_decimal",
                                           {operand_lv,
                                            scale_lv,
                                            ll_int(inline_int_null_val(operand_ti)),
                                            inlineIntNull(SQLTypeInfo(kBIGINT, false))});
      }
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
  if (operand_ti.get_scale()) {
    result_lv = cgen_state_->ir_builder_.CreateFDiv(
        result_lv, llvm::ConstantFP::get(result_lv->getType(), exp_to_scale(operand_ti.get_scale())));
  }
  return result_lv;
}

llvm::Value* Executor::codegenCastFromFp(llvm::Value* operand_lv,
                                         const SQLTypeInfo& operand_ti,
                                         const SQLTypeInfo& ti) {
  if (!operand_ti.is_fp() || !ti.is_number() || ti.is_decimal()) {
    throw std::runtime_error("Cast from " + operand_ti.get_type_name() + " to " + ti.get_type_name() +
                             " not supported");
  }
  if (operand_ti.get_type() == ti.get_type()) {
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

bool Executor::checkExpressionRanges(const Analyzer::UOper* uoper, int64_t min, int64_t max) {
  if (uoper->get_type_info().is_decimal())
    return false;

  auto expr_range_info = cgen_state_->query_infos_.size() > 0
                             ? getExpressionRange(uoper, cgen_state_->query_infos_, this)
                             : ExpressionRange::makeInvalidRange();
  if (expr_range_info.getType() != ExpressionRangeType::Integer)
    return false;
  if (expr_range_info.getIntMin() >= min && expr_range_info.getIntMax() <= max)
    return true;

  return false;
}

llvm::Value* Executor::codegenUMinus(const Analyzer::UOper* uoper, const CompilationOptions& co) {
  CHECK_EQ(uoper->get_optype(), kUMINUS);
  const auto operand_lv = codegen(uoper->get_operand(), true, co).front();
  const auto& ti = uoper->get_type_info();
  llvm::Value* chosen_max{nullptr};
  llvm::Value* chosen_min{nullptr};
  bool need_overflow_check = false;
  if (ti.is_integer() || ti.is_decimal() || ti.is_timeinterval()) {
    std::tie(chosen_max, chosen_min) = inlineIntMaxMin(ti.get_size(), true);
    need_overflow_check = !checkExpressionRanges(uoper,
                                                 static_cast<llvm::ConstantInt*>(chosen_min)->getSExtValue(),
                                                 static_cast<llvm::ConstantInt*>(chosen_max)->getSExtValue());
  }
  llvm::BasicBlock* uminus_ok{nullptr};
  llvm::BasicBlock* uminus_fail{nullptr};
  if (need_overflow_check) {
    cgen_state_->needs_error_check_ = true;
    uminus_ok = llvm::BasicBlock::Create(cgen_state_->context_, "uminus_ok", cgen_state_->row_func_);
    uminus_fail = llvm::BasicBlock::Create(cgen_state_->context_, "uminus_fail", cgen_state_->row_func_);
    auto const_min = llvm::ConstantInt::get(
        operand_lv->getType(), static_cast<llvm::ConstantInt*>(chosen_min)->getSExtValue(), true);
    auto overflow = cgen_state_->ir_builder_.CreateICmpEQ(operand_lv, const_min);
    cgen_state_->ir_builder_.CreateCondBr(overflow, uminus_fail, uminus_ok);
    cgen_state_->ir_builder_.SetInsertPoint(uminus_ok);
  }
  auto ret = ti.get_notnull() ? (ti.is_fp() ? cgen_state_->ir_builder_.CreateFNeg(operand_lv)
                                            : cgen_state_->ir_builder_.CreateNeg(operand_lv))
                              : cgen_state_->emitCall("uminus_" + numeric_type_name(ti) + "_nullable",
                                                      {operand_lv,
                                                       ti.is_fp() ? static_cast<llvm::Value*>(inlineFpNull(ti))
                                                                  : static_cast<llvm::Value*>(inlineIntNull(ti))});
  if (need_overflow_check) {
    cgen_state_->ir_builder_.SetInsertPoint(uminus_fail);
    cgen_state_->ir_builder_.CreateRet(ll_int(ERR_OVERFLOW_OR_UNDERFLOW));
    cgen_state_->ir_builder_.SetInsertPoint(uminus_ok);
  }
  return ret;
}

namespace {

bool is_qualified_bin_oper(const Analyzer::Expr* expr) {
  const auto bin_oper = dynamic_cast<const Analyzer::BinOper*>(expr);
  return bin_oper && bin_oper->get_qualifier() != kONE;
}

}  // namespace

llvm::Value* Executor::codegenLogical(const Analyzer::UOper* uoper, const CompilationOptions& co) {
  const auto optype = uoper->get_optype();
  CHECK_EQ(kNOT, optype);
  const auto operand = uoper->get_operand();
  const auto& operand_ti = operand->get_type_info();
  CHECK(operand_ti.is_boolean());
  const auto operand_lv = codegen(operand, true, co).front();
  CHECK(operand_lv->getType()->isIntegerTy());
  const bool not_null = (operand_ti.get_notnull() || is_qualified_bin_oper(operand));
  CHECK(not_null || operand_lv->getType()->isIntegerTy(8));
  return not_null ? cgen_state_->ir_builder_.CreateNot(toBool(operand_lv))
                  : cgen_state_->emitCall("logical_not", {operand_lv, inlineIntNull(operand_ti)});
}

llvm::Value* Executor::codegenIsNull(const Analyzer::UOper* uoper, const CompilationOptions& co) {
  const auto operand = uoper->get_operand();
  if (dynamic_cast<const Analyzer::Constant*>(operand) &&
      dynamic_cast<const Analyzer::Constant*>(operand)->get_is_null()) {
    // for null constants, short-circuit to true
    return llvm::ConstantInt::get(get_int_type(1, cgen_state_->context_), 1);
  }
  const auto& ti = operand->get_type_info();
  CHECK(ti.is_integer() || ti.is_boolean() || ti.is_decimal() || ti.is_time() || ti.is_string() || ti.is_fp() ||
        ti.is_array());
  // if the type is inferred as non null, short-circuit to false
  if (ti.get_notnull() && !ti.is_array()) {
    return llvm::ConstantInt::get(get_int_type(1, cgen_state_->context_), 0);
  }
  const auto operand_lv = codegen(operand, true, co).front();
  if (ti.is_array()) {
    return cgen_state_->emitExternalCall(
        "array_is_null", get_int_type(1, cgen_state_->context_), {operand_lv, posArg(operand)});
  }
  return codegenIsNullNumber(operand_lv, ti);
}

llvm::Value* Executor::codegenIsNullNumber(llvm::Value* operand_lv, const SQLTypeInfo& ti) {
  if (ti.is_fp()) {
    return cgen_state_->ir_builder_.CreateFCmp(
        llvm::FCmpInst::FCMP_OEQ, operand_lv, ti.get_type() == kFLOAT ? ll_fp(NULL_FLOAT) : ll_fp(NULL_DOUBLE));
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
  if (idx_ti.get_logical_size() < 8) {
    idx_lv = cgen_state_->ir_builder_.CreateCast(
        llvm::Instruction::CastOps::SExt, idx_lv, get_int_type(64, cgen_state_->context_));
  }
  const auto& array_ti = arr_expr->get_type_info();
  CHECK(array_ti.is_array());
  const auto& elem_ti = array_ti.get_elem_type();
  const std::string array_at_fname{
      elem_ti.is_fp() ? "array_at_" + std::string(elem_ti.get_type() == kDOUBLE ? "double_checked" : "float_checked")
                      : "array_at_int" + std::to_string(elem_ti.get_logical_size() * 8) + "_t_checked"};
  const auto ret_ty = elem_ti.is_fp() ? (elem_ti.get_type() == kDOUBLE ? llvm::Type::getDoubleTy(cgen_state_->context_)
                                                                       : llvm::Type::getFloatTy(cgen_state_->context_))
                                      : get_int_type(elem_ti.get_logical_size() * 8, cgen_state_->context_);
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

namespace {

llvm::Type* ext_arg_type_to_llvm_type(const ExtArgumentType ext_arg_type, llvm::LLVMContext& ctx) {
  switch (ext_arg_type) {
    case ExtArgumentType::Int16:
      return get_int_type(16, ctx);
    case ExtArgumentType::Int32:
      return get_int_type(32, ctx);
    case ExtArgumentType::Int64:
      return get_int_type(64, ctx);
    case ExtArgumentType::Float:
      return llvm::Type::getFloatTy(ctx);
    case ExtArgumentType::Double:
      return llvm::Type::getDoubleTy(ctx);
    default:
      CHECK(false);
  }
  CHECK(false);
  return nullptr;
}

bool ext_func_call_requires_nullcheck(const Analyzer::FunctionOper* function_oper) {
  for (size_t i = 0; i < function_oper->getArity(); ++i) {
    const auto arg = function_oper->getArg(i);
    const auto& arg_ti = arg->get_type_info();
    if (!arg_ti.get_notnull()) {
      return true;
    }
  }
  return false;
}

}  // namespace

llvm::Value* Executor::codegenFunctionOper(const Analyzer::FunctionOper* function_oper, const CompilationOptions& co) {
  const auto ext_func_sigs = ExtensionFunctionsWhitelist::get(function_oper->getName());
  if (!ext_func_sigs) {
    throw std::runtime_error("Runtime function " + function_oper->getName() + " not supported");
  }
  CHECK(!ext_func_sigs->empty());
  const auto& ext_func_sig = bind_function(function_oper, *ext_func_sigs);
  const auto& ret_ti = function_oper->get_type_info();
  CHECK(ret_ti.is_integer() || ret_ti.is_fp());
  const auto ret_ty = ret_ti.is_fp() ? (ret_ti.get_type() == kDOUBLE ? llvm::Type::getDoubleTy(cgen_state_->context_)
                                                                     : llvm::Type::getFloatTy(cgen_state_->context_))
                                     : get_int_type(ret_ti.get_logical_size() * 8, cgen_state_->context_);
  if (ret_ty != ext_arg_type_to_llvm_type(ext_func_sig.getRet(), cgen_state_->context_)) {
    throw std::runtime_error("Inconsistent return type for " + function_oper->getName());
  }
  std::vector<llvm::Value*> orig_arg_lvs;
  for (size_t i = 0; i < function_oper->getArity(); ++i) {
    const auto arg_lvs = codegen(function_oper->getArg(i), true, co);
    CHECK_EQ(size_t(1), arg_lvs.size());
    orig_arg_lvs.push_back(arg_lvs.front());
  }
  const auto bbs = beginArgsNullcheck(function_oper, orig_arg_lvs);
  CHECK_EQ(orig_arg_lvs.size(), function_oper->getArity());
  const auto args = codegenFunctionOperCastArgs(function_oper, &ext_func_sig, orig_arg_lvs);
  auto ext_call = cgen_state_->emitExternalCall(ext_func_sig.getName(), ret_ty, args);
  return endArgsNullcheck(bbs, ext_call, function_oper);
}

Executor::ArgNullcheckBBs Executor::beginArgsNullcheck(const Analyzer::FunctionOper* function_oper,
                                                       const std::vector<llvm::Value*>& orig_arg_lvs) {
  llvm::BasicBlock* args_null_bb{nullptr};
  llvm::BasicBlock* args_notnull_bb{nullptr};
  llvm::BasicBlock* orig_bb = cgen_state_->ir_builder_.GetInsertBlock();
  if (ext_func_call_requires_nullcheck(function_oper)) {
    const auto args_notnull_lv =
        cgen_state_->ir_builder_.CreateNot(codegenFunctionOperNullArg(function_oper, orig_arg_lvs));
    args_notnull_bb = llvm::BasicBlock::Create(cgen_state_->context_, "args_notnull", cgen_state_->row_func_);
    args_null_bb = llvm::BasicBlock::Create(cgen_state_->context_, "args_null", cgen_state_->row_func_);
    cgen_state_->ir_builder_.CreateCondBr(args_notnull_lv, args_notnull_bb, args_null_bb);
    cgen_state_->ir_builder_.SetInsertPoint(args_notnull_bb);
  }
  return {args_null_bb, args_notnull_bb, orig_bb};
}

llvm::Value* Executor::endArgsNullcheck(const ArgNullcheckBBs& bbs,
                                        llvm::Value* fn_ret_lv,
                                        const Analyzer::FunctionOper* function_oper) {
  if (bbs.args_null_bb) {
    CHECK(bbs.args_notnull_bb);
    cgen_state_->ir_builder_.CreateBr(bbs.args_null_bb);
    cgen_state_->ir_builder_.SetInsertPoint(bbs.args_null_bb);
    auto ext_call_phi = cgen_state_->ir_builder_.CreatePHI(fn_ret_lv->getType(), 2);
    ext_call_phi->addIncoming(fn_ret_lv, bbs.args_notnull_bb);
    const auto& ret_ti = function_oper->get_type_info();
    const auto null_lv = ret_ti.is_fp() ? static_cast<llvm::Value*>(inlineFpNull(ret_ti))
                                        : static_cast<llvm::Value*>(inlineIntNull(ret_ti));
    ext_call_phi->addIncoming(null_lv, bbs.orig_bb);
    return ext_call_phi;
  }
  return fn_ret_lv;
}

namespace {

bool call_requires_custom_type_handling(const Analyzer::FunctionOper* function_oper) {
  const auto& ret_ti = function_oper->get_type_info();
  if (!ret_ti.is_integer() && !ret_ti.is_fp()) {
    return true;
  }
  for (size_t i = 0; i < function_oper->getArity(); ++i) {
    const auto arg = function_oper->getArg(i);
    const auto& arg_ti = arg->get_type_info();
    if (!arg_ti.is_integer() && !arg_ti.is_fp()) {
      return true;
    }
  }
  return false;
}

}  // namespace

llvm::Value* Executor::codegenFunctionOperWithCustomTypeHandling(
    const Analyzer::FunctionOperWithCustomTypeHandling* function_oper,
    const CompilationOptions& co) {
  if (call_requires_custom_type_handling(function_oper)) {
    if (function_oper->getName() == "FLOOR" || function_oper->getName() == "CEIL") {
      CHECK_EQ(size_t(1), function_oper->getArity());
      const auto arg = function_oper->getArg(0);
      const auto& arg_ti = arg->get_type_info();
      CHECK(arg_ti.is_decimal());
      const auto arg_lvs = codegen(arg, true, co);
      CHECK_EQ(size_t(1), arg_lvs.size());
      const auto arg_lv = arg_lvs.front();
      CHECK(arg_lv->getType()->isIntegerTy(64));
      const auto bbs = beginArgsNullcheck(function_oper, {arg_lvs});
      const std::string func_name = (function_oper->getName() == "FLOOR") ? "decimal_floor" : "decimal_ceil";
      const auto covar_result_lv = cgen_state_->emitCall(func_name, {arg_lv, ll_int(exp_to_scale(arg_ti.get_scale()))});
      const auto ret_ti = function_oper->get_type_info();
      CHECK(ret_ti.is_decimal());
      CHECK_EQ(0, ret_ti.get_scale());
      const auto result_lv =
          cgen_state_->ir_builder_.CreateSDiv(covar_result_lv, ll_int(exp_to_scale(arg_ti.get_scale())));
      return endArgsNullcheck(bbs, result_lv, function_oper);
    }
    throw std::runtime_error("Type combination not supported for function " + function_oper->getName());
  }
  return codegenFunctionOper(function_oper, co);
}

llvm::Value* Executor::codegenFunctionOperNullArg(const Analyzer::FunctionOper* function_oper,
                                                  const std::vector<llvm::Value*>& orig_arg_lvs) {
  llvm::Value* one_arg_null = llvm::ConstantInt::get(llvm::IntegerType::getInt1Ty(cgen_state_->context_), false);
  for (size_t i = 0; i < function_oper->getArity(); ++i) {
    const auto arg = function_oper->getArg(i);
    const auto& arg_ti = arg->get_type_info();
    CHECK(arg_ti.is_number());
    one_arg_null = cgen_state_->ir_builder_.CreateOr(one_arg_null, codegenIsNullNumber(orig_arg_lvs[i], arg_ti));
  }
  return one_arg_null;
}

std::vector<llvm::Value*> Executor::codegenFunctionOperCastArgs(const Analyzer::FunctionOper* function_oper,
                                                                const ExtensionFunction* ext_func_sig,
                                                                const std::vector<llvm::Value*>& orig_arg_lvs) {
  CHECK(ext_func_sig);
  const auto& ext_func_args = ext_func_sig->getArgs();
  CHECK_EQ(function_oper->getArity(), ext_func_args.size());
  std::vector<llvm::Value*> args;
  for (size_t i = 0; i < function_oper->getArity(); ++i) {
    const auto arg = function_oper->getArg(i);
    const auto& arg_ti = arg->get_type_info();
    const auto arg_target_ti = ext_arg_type_to_type_info(ext_func_args[i]);
    llvm::Value* arg_lv{nullptr};
    if (arg_ti.get_type() != arg_target_ti.get_type()) {
      arg_lv = codegenCast(orig_arg_lvs[i], arg_ti, arg_target_ti, false);
    } else {
      arg_lv = orig_arg_lvs[i];
    }
    CHECK_EQ(arg_lv->getType(), ext_arg_type_to_llvm_type(ext_func_args[i], cgen_state_->context_));
    args.push_back(arg_lv);
  }
  return args;
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
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH:
      return ll_int(constant->get_constval().timeval);
    default:
      abort();
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
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH:
      return ll_int(inline_int_null_val(type_info));
    case kARRAY:
      return ll_int(int64_t(0));
    default:
      abort();
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
      abort();
  }
}

std::pair<llvm::ConstantInt*, llvm::ConstantInt*> Executor::inlineIntMaxMin(const size_t byte_width,
                                                                            const bool is_signed) {
  int64_t max_int{0}, min_int{0};
  if (is_signed) {
    std::tie(max_int, min_int) = inline_int_max_min(byte_width);
  } else {
    uint64_t max_uint{0}, min_uint{0};
    std::tie(max_uint, min_uint) = inline_uint_max_min(byte_width);
    max_int = static_cast<int64_t>(max_uint);
    CHECK_EQ(uint64_t(0), min_uint);
  }
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
      abort();
  }
}

namespace {

std::string numeric_or_time_interval_type_name(const SQLTypeInfo& ti1, const SQLTypeInfo& ti2) {
  if (ti2.is_timeinterval()) {
    return numeric_type_name(ti2);
  }
  return numeric_type_name(ti1);
}

}  // namespace

llvm::Value* Executor::codegenArith(const Analyzer::BinOper* bin_oper, const CompilationOptions& co) {
  const auto optype = bin_oper->get_optype();
  CHECK(IS_ARITHMETIC(optype));
  const auto lhs = bin_oper->get_left_operand();
  const auto rhs = bin_oper->get_right_operand();
  const auto& lhs_type = lhs->get_type_info();
  const auto& rhs_type = rhs->get_type_info();

  if (lhs_type.is_decimal() && rhs_type.is_decimal() && lhs_type.get_scale() == rhs_type.get_scale() &&
      optype == kMULTIPLY) {
    if (auto ret = codegenDeciMul(bin_oper, false, co))
      return ret;
    if (auto ret = codegenDeciMul(bin_oper, true, co))
      return ret;
  }

  auto lhs_lv = codegen(lhs, true, co).front();
  auto rhs_lv = codegen(rhs, true, co).front();
  if (lhs_type.is_timeinterval()) {
    rhs_lv = codegenCastBetweenIntTypes(rhs_lv, rhs_type, lhs_type);
  } else if (rhs_type.is_timeinterval()) {
    lhs_lv = codegenCastBetweenIntTypes(lhs_lv, lhs_type, rhs_type);
  } else {
    CHECK_EQ(lhs_type.get_type(), rhs_type.get_type());
  }
  const auto& oper_type = rhs_type.is_timeinterval() ? rhs_type : lhs_type;
  if (lhs_type.is_decimal()) {
    CHECK_EQ(lhs_type.get_scale(), rhs_type.get_scale());
  }
  const auto null_check_suffix = get_null_check_suffix(lhs_type, rhs_type);
  if (lhs_type.is_integer() || lhs_type.is_decimal() || lhs_type.is_timeinterval()) {
    const auto int_typename = numeric_or_time_interval_type_name(lhs_type, rhs_type);
    switch (optype) {
      case kMINUS:
        return codegenSub(
            bin_oper, lhs_lv, rhs_lv, null_check_suffix.empty() ? "" : int_typename, null_check_suffix, oper_type);
      case kPLUS:
        return codegenAdd(
            bin_oper, lhs_lv, rhs_lv, null_check_suffix.empty() ? "" : int_typename, null_check_suffix, oper_type);
      case kMULTIPLY:
        return codegenMul(
            bin_oper, lhs_lv, rhs_lv, null_check_suffix.empty() ? "" : int_typename, null_check_suffix, oper_type);
      case kDIVIDE:
        return codegenDiv(lhs_lv, rhs_lv, null_check_suffix.empty() ? "" : int_typename, null_check_suffix, oper_type);
      case kMODULO:
        return codegenMod(lhs_lv, rhs_lv, null_check_suffix.empty() ? "" : int_typename, null_check_suffix, oper_type);
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

namespace {

bool is_temporary_column(const Analyzer::Expr* expr) {
  const auto col_expr = dynamic_cast<const Analyzer::ColumnVar*>(expr);
  if (!col_expr) {
    return false;
  }
  return col_expr->get_table_id() < 0;
}

}  // namespace

bool Executor::checkExpressionRanges(const Analyzer::BinOper* bin_oper, int64_t min, int64_t max) {
  if (is_temporary_column(bin_oper->get_left_operand()) || is_temporary_column(bin_oper->get_right_operand())) {
    // Computing the range for temporary columns is a lot more expensive than the overflow check.
    return false;
  }
  if (bin_oper->get_type_info().is_decimal())
    return false;

  auto expr_range_info = cgen_state_->query_infos_.size() > 0
                             ? getExpressionRange(bin_oper, cgen_state_->query_infos_, this)
                             : ExpressionRange::makeInvalidRange();
  if (expr_range_info.getType() != ExpressionRangeType::Integer)
    return false;
  if (expr_range_info.getIntMin() >= min && expr_range_info.getIntMax() <= max)
    return true;

  return false;
}

llvm::Value* Executor::codegenAdd(const Analyzer::BinOper* bin_oper,
                                  llvm::Value* lhs_lv,
                                  llvm::Value* rhs_lv,
                                  const std::string& null_typename,
                                  const std::string& null_check_suffix,
                                  const SQLTypeInfo& ti) {
  CHECK_EQ(lhs_lv->getType(), rhs_lv->getType());
  CHECK(ti.is_integer() || ti.is_decimal() || ti.is_timeinterval());
  llvm::Value* chosen_max{nullptr};
  llvm::Value* chosen_min{nullptr};
  std::tie(chosen_max, chosen_min) = inlineIntMaxMin(ti.get_size(), true);
  auto need_overflow_check = !checkExpressionRanges(bin_oper,
                                                    static_cast<llvm::ConstantInt*>(chosen_min)->getSExtValue(),
                                                    static_cast<llvm::ConstantInt*>(chosen_max)->getSExtValue());
  llvm::BasicBlock* add_ok{nullptr};
  llvm::BasicBlock* add_fail{nullptr};
  if (need_overflow_check) {
    cgen_state_->needs_error_check_ = true;
    add_ok = llvm::BasicBlock::Create(cgen_state_->context_, "add_ok", cgen_state_->row_func_);
    add_fail = llvm::BasicBlock::Create(cgen_state_->context_, "add_fail", cgen_state_->row_func_);
    llvm::Value* detected{nullptr};
    auto const_zero = llvm::ConstantInt::get(lhs_lv->getType(), 0, true);
    auto overflow = cgen_state_->ir_builder_.CreateAnd(
        cgen_state_->ir_builder_.CreateICmpSGT(lhs_lv, const_zero),
        cgen_state_->ir_builder_.CreateICmpSGT(rhs_lv, cgen_state_->ir_builder_.CreateSub(chosen_max, lhs_lv)));
    auto underflow = cgen_state_->ir_builder_.CreateAnd(
        cgen_state_->ir_builder_.CreateICmpSLT(lhs_lv, const_zero),
        cgen_state_->ir_builder_.CreateICmpSLT(rhs_lv, cgen_state_->ir_builder_.CreateSub(chosen_min, lhs_lv)));
    detected = cgen_state_->ir_builder_.CreateOr(overflow, underflow);
    cgen_state_->ir_builder_.CreateCondBr(detected, add_fail, add_ok);
    cgen_state_->ir_builder_.SetInsertPoint(add_ok);
  }
  auto ret = null_check_suffix.empty() ? cgen_state_->ir_builder_.CreateAdd(lhs_lv, rhs_lv)
                                       : cgen_state_->emitCall("add_" + null_typename + null_check_suffix,
                                                               {lhs_lv, rhs_lv, ll_int(inline_int_null_val(ti))});
  if (need_overflow_check) {
    cgen_state_->ir_builder_.SetInsertPoint(add_fail);
    cgen_state_->ir_builder_.CreateRet(ll_int(ERR_OVERFLOW_OR_UNDERFLOW));
    cgen_state_->ir_builder_.SetInsertPoint(add_ok);
  }
  return ret;
}

llvm::Value* Executor::codegenSub(const Analyzer::BinOper* bin_oper,
                                  llvm::Value* lhs_lv,
                                  llvm::Value* rhs_lv,
                                  const std::string& null_typename,
                                  const std::string& null_check_suffix,
                                  const SQLTypeInfo& ti) {
  CHECK_EQ(lhs_lv->getType(), rhs_lv->getType());
  CHECK(ti.is_integer() || ti.is_decimal() || ti.is_timeinterval());
  llvm::Value* chosen_max{nullptr};
  llvm::Value* chosen_min{nullptr};
  std::tie(chosen_max, chosen_min) = inlineIntMaxMin(ti.get_size(), true);
  auto need_overflow_check = !checkExpressionRanges(bin_oper,
                                                    static_cast<llvm::ConstantInt*>(chosen_min)->getSExtValue(),
                                                    static_cast<llvm::ConstantInt*>(chosen_max)->getSExtValue());
  llvm::BasicBlock* sub_ok{nullptr};
  llvm::BasicBlock* sub_fail{nullptr};
  if (need_overflow_check) {
    cgen_state_->needs_error_check_ = true;
    sub_ok = llvm::BasicBlock::Create(cgen_state_->context_, "sub_ok", cgen_state_->row_func_);
    sub_fail = llvm::BasicBlock::Create(cgen_state_->context_, "sub_fail", cgen_state_->row_func_);
    llvm::Value* detected{nullptr};
    auto const_zero = llvm::ConstantInt::get(lhs_lv->getType(), 0, true);
    auto overflow = cgen_state_->ir_builder_.CreateAnd(
        cgen_state_->ir_builder_.CreateICmpSLT(lhs_lv, const_zero),
        cgen_state_->ir_builder_.CreateICmpSGT(rhs_lv, cgen_state_->ir_builder_.CreateAdd(chosen_max, lhs_lv)));
    auto underflow = cgen_state_->ir_builder_.CreateAnd(
        cgen_state_->ir_builder_.CreateICmpSGT(lhs_lv, const_zero),
        cgen_state_->ir_builder_.CreateICmpSLT(rhs_lv, cgen_state_->ir_builder_.CreateAdd(chosen_min, lhs_lv)));
    detected = cgen_state_->ir_builder_.CreateOr(overflow, underflow);
    cgen_state_->ir_builder_.CreateCondBr(detected, sub_fail, sub_ok);
    cgen_state_->ir_builder_.SetInsertPoint(sub_ok);
  }
  auto ret = null_check_suffix.empty() ? cgen_state_->ir_builder_.CreateSub(lhs_lv, rhs_lv)
                                       : cgen_state_->emitCall("sub_" + null_typename + null_check_suffix,
                                                               {lhs_lv, rhs_lv, ll_int(inline_int_null_val(ti))});
  if (need_overflow_check) {
    cgen_state_->ir_builder_.SetInsertPoint(sub_fail);
    cgen_state_->ir_builder_.CreateRet(ll_int(ERR_OVERFLOW_OR_UNDERFLOW));
    cgen_state_->ir_builder_.SetInsertPoint(sub_ok);
  }
  return ret;
}

llvm::Value* Executor::codegenMul(const Analyzer::BinOper* bin_oper,
                                  llvm::Value* lhs_lv,
                                  llvm::Value* rhs_lv,
                                  const std::string& null_typename,
                                  const std::string& null_check_suffix,
                                  const SQLTypeInfo& ti,
                                  bool downscale) {
  CHECK_EQ(lhs_lv->getType(), rhs_lv->getType());
  CHECK(ti.is_integer() || ti.is_decimal() || ti.is_timeinterval());
  llvm::Value* chosen_max{nullptr};
  llvm::Value* chosen_min{nullptr};
  std::tie(chosen_max, chosen_min) = inlineIntMaxMin(ti.get_size(), true);
  auto need_overflow_check = !checkExpressionRanges(bin_oper,
                                                    static_cast<llvm::ConstantInt*>(chosen_min)->getSExtValue(),
                                                    static_cast<llvm::ConstantInt*>(chosen_max)->getSExtValue());
  llvm::BasicBlock* mul_ok{nullptr};
  llvm::BasicBlock* mul_fail{nullptr};
  if (need_overflow_check) {
    cgen_state_->needs_error_check_ = true;
    mul_ok = llvm::BasicBlock::Create(cgen_state_->context_, "mul_ok", cgen_state_->row_func_);
    mul_fail = llvm::BasicBlock::Create(cgen_state_->context_, "mul_fail", cgen_state_->row_func_);
    auto mul_check = llvm::BasicBlock::Create(cgen_state_->context_, "mul_check", cgen_state_->row_func_);
    auto const_zero = llvm::ConstantInt::get(rhs_lv->getType(), 0, true);
    cgen_state_->ir_builder_.CreateCondBr(cgen_state_->ir_builder_.CreateICmpEQ(rhs_lv, const_zero), mul_ok, mul_check);
    cgen_state_->ir_builder_.SetInsertPoint(mul_check);
    auto detected = cgen_state_->ir_builder_.CreateOr(  // overflow
        cgen_state_->ir_builder_.CreateICmpSGT(lhs_lv, cgen_state_->ir_builder_.CreateSDiv(chosen_max, rhs_lv)),
        // underflow
        cgen_state_->ir_builder_.CreateICmpSLT(lhs_lv, cgen_state_->ir_builder_.CreateSDiv(chosen_min, rhs_lv)));
    cgen_state_->ir_builder_.CreateCondBr(detected, mul_fail, mul_ok);
    cgen_state_->ir_builder_.SetInsertPoint(mul_ok);
  }
  llvm::Value* ret{nullptr};
  if (ti.is_decimal()) {
    const auto decimal_null_typename = null_typename.empty() ? "int64_t" : null_typename;
    if (downscale) {
      ret = cgen_state_->emitCall(
          "mul_" + decimal_null_typename + "_decimal",
          {lhs_lv, rhs_lv, ll_int(exp_to_scale(ti.get_scale())), ll_int(inline_int_null_val(ti))});
    } else {
      ret = cgen_state_->emitCall("mul_" + decimal_null_typename + "_decimal_no_downscale",
                                  {lhs_lv, rhs_lv, ll_int(inline_int_null_val(ti))});
    }
  } else {
    ret = null_check_suffix.empty() ? cgen_state_->ir_builder_.CreateMul(lhs_lv, rhs_lv)
                                    : cgen_state_->emitCall("mul_" + null_typename + null_check_suffix,
                                                            {lhs_lv, rhs_lv, ll_int(inline_int_null_val(ti))});
  }
  if (need_overflow_check) {
    cgen_state_->ir_builder_.SetInsertPoint(mul_fail);
    cgen_state_->ir_builder_.CreateRet(ll_int(ERR_OVERFLOW_OR_UNDERFLOW));
    cgen_state_->ir_builder_.SetInsertPoint(mul_ok);
  }
  return ret;
}

llvm::Value* Executor::codegenDeciMul(const Analyzer::BinOper* bin_oper, bool swap, const CompilationOptions& co) {
  auto lhs = swap ? bin_oper->get_right_operand() : bin_oper->get_left_operand();
  auto rhs = swap ? bin_oper->get_left_operand() : bin_oper->get_right_operand();
  const auto& lhs_type = lhs->get_type_info();
  const auto& rhs_type = rhs->get_type_info();
  CHECK(lhs_type.is_decimal() && rhs_type.is_decimal() && lhs_type.get_scale() == rhs_type.get_scale());

  auto rhs_constant = dynamic_cast<const Analyzer::Constant*>(rhs);
  auto rhs_cast = dynamic_cast<const Analyzer::UOper*>(rhs);
  if (rhs_constant && (rhs_constant->get_constval().bigintval % exp_to_scale(rhs_type.get_scale())) == 0) {
    // can safely downscale a scaled constant
  } else if (rhs_cast && rhs_cast->get_optype() == kCAST && rhs_cast->get_operand()->get_type_info().is_integer()) {
    // can skip upscale in the int to dec cast
  } else {
    return nullptr;
  }

  auto lhs_lv = codegen(lhs, true, co).front();
  llvm::Value* rhs_lv{nullptr};
  if (rhs_constant) {
    const auto rhs_lit =
        Parser::IntLiteral::analyzeValue(rhs_constant->get_constval().bigintval / exp_to_scale(rhs_type.get_scale()));
    auto rhs_lit_lv = codegenIntConst(dynamic_cast<const Analyzer::Constant*>(rhs_lit.get()));
    rhs_lv = codegenCastBetweenIntTypes(rhs_lit_lv, rhs_lit->get_type_info(), lhs_type, /*upscale*/ false);
  } else if (rhs_cast) {
    auto rhs_cast_oper = rhs_cast->get_operand();
    const auto& rhs_cast_oper_ti = rhs_cast_oper->get_type_info();
    auto rhs_cast_oper_lv = codegen(rhs_cast_oper, true, co).front();
    rhs_lv = codegenCastBetweenIntTypes(rhs_cast_oper_lv, rhs_cast_oper_ti, lhs_type, /*upscale*/ false);
  } else {
    CHECK(false);
  }
  const auto int_typename = numeric_or_time_interval_type_name(lhs_type, rhs_type);
  const auto null_check_suffix = get_null_check_suffix(lhs_type, rhs_type);
  return codegenMul(bin_oper,
                    lhs_lv,
                    rhs_lv,
                    null_check_suffix.empty() ? "" : int_typename,
                    null_check_suffix,
                    lhs_type,
                    /*downscale*/ false);
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
  cgen_state_->needs_error_check_ = true;
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
  cgen_state_->needs_error_check_ = true;
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
      uint64_t agg_result = 0;
      for (size_t i = 0; i < out_vec_sz; ++i) {
        const uint64_t out = static_cast<uint64_t>(out_vec[i]);
        if (detect_overflow_and_underflow(agg_result, out, false, uint64_t(0), ti)) {
          return {0, error_no};
        }
        agg_result += out;
      }
      return {static_cast<int64_t>(agg_result), 0};
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
  abort();
}

namespace {

template <typename PtrTy>
PtrTy get_merged_result(const std::vector<std::pair<ResultPtr, std::vector<size_t>>>& results_per_device) {
  auto& first = boost::get<PtrTy>(results_per_device.front().first);
  CHECK(first);
  auto copy = boost::make_unique<typename PtrTy::element_type>(*first);
  CHECK(copy);
  for (size_t dev_idx = 1; dev_idx < results_per_device.size(); ++dev_idx) {
    const auto& next = boost::get<PtrTy>(results_per_device[dev_idx].first);
    CHECK(next);
    copy->append(*next);
  }
  return copy;
}

}  // namespace

ResultPtr Executor::resultsUnion(ExecutionDispatch& execution_dispatch) {
  auto& results_per_device = execution_dispatch.getFragmentResults();
  if (results_per_device.empty()) {
    const auto& ra_exe_unit = execution_dispatch.getExecutionUnit();
    return boost::make_unique<ResultRows>(QueryMemoryDescriptor{},
                                          ra_exe_unit.target_exprs,
                                          nullptr,
                                          nullptr,
                                          std::vector<int64_t>{},
                                          ExecutorDeviceType::CPU);
  }
  typedef std::pair<ResultPtr, std::vector<size_t>> IndexedResultRows;
  std::sort(results_per_device.begin(),
            results_per_device.end(),
            [](const IndexedResultRows& lhs, const IndexedResultRows& rhs) {
              CHECK_EQ(size_t(1), lhs.second.size());
              CHECK_EQ(size_t(1), rhs.second.size());
              return lhs.second < rhs.second;
            });

  if (boost::get<RowSetPtr>(&results_per_device.front().first)) {
    return get_merged_result<RowSetPtr>(results_per_device);
  } else if (boost::get<IterTabPtr>(&results_per_device.front().first)) {
    return get_merged_result<IterTabPtr>(results_per_device);
  }
  CHECK(false);
  return RowSetPtr(nullptr);
}

namespace {

RowSetPtr reduce_estimator_results(const RelAlgExecutionUnit& ra_exe_unit,
                                   std::vector<std::pair<ResultPtr, std::vector<size_t>>>& results_per_device) {
  if (results_per_device.empty()) {
    return nullptr;
  }
  auto first = boost::get<RowSetPtr>(&results_per_device.front().first);
  CHECK(first && *first);
  const auto result_set = (*first)->getResultSet();
  CHECK(result_set);
  auto estimator_buffer = result_set->getHostEstimatorBuffer();
  CHECK(estimator_buffer);
  for (size_t i = 1; i < results_per_device.size(); ++i) {
    auto next = boost::get<RowSetPtr>(&results_per_device[i].first);
    CHECK(next && *next);
    const auto next_result_set = (*next)->getResultSet();
    CHECK(next_result_set);
    const auto other_estimator_buffer = next_result_set->getHostEstimatorBuffer();
    for (size_t off = 0; off < ra_exe_unit.estimator->getEstimatorBufferSize(); ++off) {
      estimator_buffer[off] |= other_estimator_buffer[off];
    }
  }
  return std::move(*first);
}

}  // namespace

// TODO(miyu): remove dt_for_all along w/ can_use_result_set
RowSetPtr Executor::reduceMultiDeviceResults(const RelAlgExecutionUnit& ra_exe_unit,
                                             std::vector<std::pair<ResultPtr, std::vector<size_t>>>& results_per_device,
                                             std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                                             const QueryMemoryDescriptor& query_mem_desc,
                                             const bool output_columnar,
                                             const ExecutorDeviceType dt_for_all) const {
  if (ra_exe_unit.estimator) {
    return reduce_estimator_results(ra_exe_unit, results_per_device);
  }

  if (results_per_device.empty()) {
    return boost::make_unique<ResultRows>(
        query_mem_desc, ra_exe_unit.target_exprs, nullptr, nullptr, std::vector<int64_t>{}, ExecutorDeviceType::CPU);
  }

  if (can_use_result_set(query_mem_desc, dt_for_all)) {
    return reduceMultiDeviceResultSets(
        results_per_device, row_set_mem_owner, ResultSet::fixupQueryMemoryDescriptor(query_mem_desc));
  }

  auto first = boost::get<RowSetPtr>(&results_per_device.front().first);
  CHECK(first && *first);

  auto reduced_results = boost::make_unique<ResultRows>(**first);
  CHECK(reduced_results);

  for (size_t i = 1; i < results_per_device.size(); ++i) {
    auto next = boost::get<RowSetPtr>(&results_per_device[i].first);
    CHECK(next && *next);
    reduced_results->reduce(**next, query_mem_desc, output_columnar);
  }

  row_set_mem_owner->addLiteralStringDictProxy(lit_str_dict_proxy_);

  return reduced_results;
}

RowSetPtr Executor::reduceMultiDeviceResultSets(
    std::vector<std::pair<ResultPtr, std::vector<size_t>>>& results_per_device,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const QueryMemoryDescriptor& query_mem_desc) const {
  std::shared_ptr<ResultSet> reduced_results;

  const auto& first = boost::get<RowSetPtr>(results_per_device.front().first);
  CHECK(first);

  if (query_mem_desc.hash_type == GroupByColRangeType::MultiCol && results_per_device.size() > 1) {
    const auto total_entry_count =
        std::accumulate(results_per_device.begin(),
                        results_per_device.end(),
                        size_t(0),
                        [](const size_t init, const std::pair<ResultPtr, std::vector<size_t>>& rs) {
                          const auto& r = boost::get<RowSetPtr>(rs.first);
                          return init + r->getResultSet()->getQueryMemDesc().entry_count;
                        });
    CHECK(total_entry_count);
    const auto first_result = first->getResultSet();
    CHECK(first_result);
    auto query_mem_desc = first_result->getQueryMemDesc();
    query_mem_desc.entry_count = total_entry_count;
    reduced_results = std::make_shared<ResultSet>(
        first_result->getTargetInfos(), ExecutorDeviceType::CPU, query_mem_desc, row_set_mem_owner, this);
    auto result_storage = reduced_results->allocateStorage(plan_state_->init_agg_vals_);
    reduced_results->initializeStorage();
    first_result->getStorage()->moveEntriesToBuffer(result_storage->getUnderlyingBuffer(), query_mem_desc.entry_count);
  } else {
    reduced_results = first->getResultSet();
  }

  for (size_t i = 1; i < results_per_device.size(); ++i) {
    const auto& result = boost::get<RowSetPtr>(results_per_device[i].first);
    const auto result_set = result->getResultSet();
    CHECK(result_set);
    reduced_results->getStorage()->reduce(*(result_set->getStorage()));
  }

  return boost::make_unique<ResultRows>(ResultRows(reduced_results));
}

RowSetPtr Executor::reduceSpeculativeTopN(const RelAlgExecutionUnit& ra_exe_unit,
                                          std::vector<std::pair<ResultPtr, std::vector<size_t>>>& results_per_device,
                                          std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                                          const QueryMemoryDescriptor& query_mem_desc) const {
  if (results_per_device.size() == 1) {
    auto rows = boost::get<RowSetPtr>(&results_per_device.front().first);
    CHECK(rows);
    return std::move(*rows);
  }
  const auto top_n = ra_exe_unit.sort_info.limit + ra_exe_unit.sort_info.offset;
  SpeculativeTopNMap m;
  for (const auto& result : results_per_device) {
    auto rows = boost::get<RowSetPtr>(&result.first);
    CHECK(rows);
    if (!*rows) {
      continue;
    }
    SpeculativeTopNMap that(
        **rows, ra_exe_unit.target_exprs, std::max(size_t(10000 * std::max(1, static_cast<int>(log(top_n)))), top_n));
    m.reduce(that);
  }
  CHECK_EQ(size_t(1), ra_exe_unit.sort_info.order_entries.size());
  const auto desc = ra_exe_unit.sort_info.order_entries.front().is_desc;
  return m.asRows(ra_exe_unit, row_set_mem_owner, query_mem_desc, plan_state_->init_agg_vals_, this, top_n, desc);
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
      case kAPPROX_COUNT_DISTINCT:
        result.push_back("agg_approximate_count_distinct");
        break;
      default:
        CHECK(false);
    }
  }
  return result;
}

size_t compute_buffer_entry_guess(const std::vector<InputTableInfo>& query_infos) {
  using Fragmenter_Namespace::FragmentInfo;
  size_t max_groups_buffer_entry_guess = 1;
  for (const auto& query_info : query_infos) {
    CHECK(!query_info.info.fragments.empty());
    auto it = std::max_element(
        query_info.info.fragments.begin(),
        query_info.info.fragments.end(),
        [](const FragmentInfo& f1, const FragmentInfo& f2) { return f1.getNumTuples() < f2.getNumTuples(); });
    max_groups_buffer_entry_guess *= it->getNumTuples();
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

std::string get_table_name(const InputDescriptor& input_desc, const Catalog_Namespace::Catalog& cat) {
  const auto source_type = input_desc.getSourceType();
  if (source_type == InputSourceType::TABLE) {
    const auto td = cat.getMetadataForTable(input_desc.getTableId());
    CHECK(td);
    return td->tableName;
  } else {
    return "$TEMPORARY_TABLE" + std::to_string(-input_desc.getTableId());
  }
}

void checkWorkUnitWatchdog(const RelAlgExecutionUnit& ra_exe_unit, const Catalog_Namespace::Catalog& cat) {
  for (const auto target_expr : ra_exe_unit.target_exprs) {
    if (dynamic_cast<const Analyzer::AggExpr*>(target_expr)) {
      return;
    }
  }
  if (ra_exe_unit.groupby_exprs.size() == 1 && !ra_exe_unit.groupby_exprs.front() &&
      (!ra_exe_unit.scan_limit || ra_exe_unit.scan_limit > Executor::high_scan_limit)) {
    std::vector<std::string> table_names;
    const auto& input_descs = ra_exe_unit.input_descs;
    for (const auto& input_desc : input_descs) {
      table_names.push_back(get_table_name(input_desc, cat));
    }
    throw WatchdogException("Query would require a scan without a limit on table(s): " +
                            boost::algorithm::join(table_names, ", "));
  }
}

bool is_sample_query(const RelAlgExecutionUnit& ra_exe_unit) {
  const bool result = ra_exe_unit.input_descs.size() == 1 && ra_exe_unit.simple_quals.empty() &&
                      ra_exe_unit.quals.empty() && ra_exe_unit.sort_info.order_entries.empty() &&
                      ra_exe_unit.scan_limit;
  if (result) {
    CHECK(ra_exe_unit.join_type == JoinType::INVALID);
    CHECK(ra_exe_unit.inner_join_quals.empty());
    CHECK(ra_exe_unit.outer_join_quals.empty());
    CHECK_EQ(size_t(1), ra_exe_unit.groupby_exprs.size());
    CHECK(!ra_exe_unit.groupby_exprs.front());
  }
  return result;
}

bool is_trivial_loop_join(const std::vector<InputTableInfo>& query_infos, const RelAlgExecutionUnit& ra_exe_unit) {
  if (ra_exe_unit.input_descs.size() < 2) {
    return false;
  }
  CHECK_EQ(size_t(2), ra_exe_unit.input_descs.size());
  const auto inner_table_id = ra_exe_unit.input_descs[1].getTableId();
  ssize_t inner_table_idx = -1;
  for (size_t i = 0; i < query_infos.size(); ++i) {
    if (query_infos[i].table_id == inner_table_id) {
      inner_table_idx = i;
      break;
    }
  }
  CHECK_NE(ssize_t(-1), inner_table_idx);
  return query_infos[inner_table_idx].info.getNumTuples() == 1;
}

}  // namespace

ResultPtr Executor::executeWorkUnit(int32_t* error_code,
                                    size_t& max_groups_buffer_entry_guess,
                                    const bool is_agg,
                                    const std::vector<InputTableInfo>& query_infos,
                                    const RelAlgExecutionUnit& ra_exe_unit,
                                    const CompilationOptions& co,
                                    const ExecutionOptions& options,
                                    const Catalog_Namespace::Catalog& cat,
                                    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                                    RenderAllocatorMap* render_allocator_map,
                                    const bool has_cardinality_estimation) {
  CHECK(!execute_mutex_.try_lock());
  const auto device_type = getDeviceTypeForTargets(ra_exe_unit, co.device_type_);
  CHECK(!query_infos.empty());
  if (!max_groups_buffer_entry_guess) {
    // The query has failed the first execution attempt because of running out
    // of group by slots. Make the conservative choice: allocate fragment size
    // slots and run on the CPU.
    CHECK(device_type == ExecutorDeviceType::CPU);
    max_groups_buffer_entry_guess = compute_buffer_entry_guess(query_infos);
  }

  auto join_info = JoinInfo(JoinImplType::Invalid, std::vector<std::shared_ptr<Analyzer::BinOper>>{}, nullptr, "");
  if (ra_exe_unit.input_descs.size() > 1) {
    join_info = chooseJoinType(ra_exe_unit.inner_join_quals, query_infos, ra_exe_unit.input_col_descs, device_type);
  }
  if (join_info.join_impl_type_ == JoinImplType::Loop && !ra_exe_unit.outer_join_quals.empty()) {
    join_info = chooseJoinType(ra_exe_unit.outer_join_quals, query_infos, ra_exe_unit.input_col_descs, device_type);
  }

  if (join_info.join_impl_type_ == JoinImplType::Loop &&
      !(options.allow_loop_joins || is_trivial_loop_join(query_infos, ra_exe_unit))) {
    throw std::runtime_error("Hash join failed, reason: " + join_info.hash_join_fail_reason_);
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
                                         {device_type, co.hoist_literals_, co.opt_level_, co.with_dynamic_watchdog_},
                                         context_count,
                                         row_set_mem_owner,
                                         error_code,
                                         render_allocator_map);
    try {
      crt_min_byte_width = execution_dispatch.compile(
          join_info, max_groups_buffer_entry_guess, crt_min_byte_width, options, has_cardinality_estimation);
    } catch (CompilationRetryNoCompaction&) {
      crt_min_byte_width = MAX_BYTE_WIDTH_SUPPORTED;
      continue;
    }

    if (options.just_explain) {
      return executeExplain(execution_dispatch);
    }

    for (const auto target_expr : ra_exe_unit.target_exprs) {
      plan_state_->target_exprs_.push_back(target_expr);
    }

    std::condition_variable scheduler_cv;
    std::mutex scheduler_mutex;
    auto dispatch =
        [this, &execution_dispatch, &available_cpus, &available_gpus, &options, &scheduler_mutex, &scheduler_cv](
            const ExecutorDeviceType chosen_device_type,
            int chosen_device_id,
            const std::vector<std::pair<int, std::vector<size_t>>>& frag_ids,
            const size_t ctx_idx,
            const int64_t rowid_lookup_key) {
          execution_dispatch.run(chosen_device_type, chosen_device_id, options, frag_ids, ctx_idx, rowid_lookup_key);
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

    const size_t input_desc_count{ra_exe_unit.input_descs.size()};
    std::map<int, const TableFragments*> selected_tables_fragments;
    CHECK_EQ(query_infos.size(), (input_desc_count + ra_exe_unit.extra_input_descs.size()));
    for (size_t table_idx = 0; table_idx < input_desc_count; ++table_idx) {
      const auto table_id = ra_exe_unit.input_descs[table_idx].getTableId();
      if (!selected_tables_fragments.count(table_id)) {
        selected_tables_fragments[ra_exe_unit.input_descs[table_idx].getTableId()] =
            &query_infos[table_idx].info.fragments;
      }
    }
    const QueryMemoryDescriptor& query_mem_desc = execution_dispatch.getQueryMemoryDescriptor();
    if (render_allocator_map && cgen_state_->must_run_on_cpu_) {
      throw std::runtime_error("Query has to run on CPU, cannot render its results");
    }
    if (!options.just_validate) {
      dispatchFragments(dispatch,
                        execution_dispatch,
                        options,
                        is_agg,
                        selected_tables_fragments,
                        context_count,
                        scheduler_cv,
                        scheduler_mutex,
                        available_gpus,
                        available_cpus);
    }
    if (options.with_dynamic_watchdog && interrupted_ && *error_code == ERR_OUT_OF_TIME) {
      *error_code = ERR_INTERRUPTED;
    }
    cat.get_dataMgr().freeAllBuffers();
    if (*error_code == ERR_OVERFLOW_OR_UNDERFLOW) {
      crt_min_byte_width <<= 1;
      continue;
    }
    if (*error_code != 0) {
      return boost::make_unique<ResultRows>(QueryMemoryDescriptor{},
                                            std::vector<Analyzer::Expr*>{},
                                            nullptr,
                                            nullptr,
                                            std::vector<int64_t>{},
                                            ExecutorDeviceType::CPU);
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
        return boost::make_unique<ResultRows>(query_mem_desc,
                                              plan_state_->target_exprs_,
                                              nullptr,
                                              std::vector<int64_t>{},
                                              nullptr,
                                              0,
                                              false,
                                              std::vector<std::vector<const int8_t*>>{},
                                              execution_dispatch.getDeviceType(),
                                              -1);
      } catch (OverflowOrUnderflow&) {
        crt_min_byte_width <<= 1;
        continue;
      }
    }
    return resultsUnion(execution_dispatch);

  } while (static_cast<size_t>(crt_min_byte_width) <= sizeof(int64_t));

  return boost::make_unique<ResultRows>(QueryMemoryDescriptor{},
                                        std::vector<Analyzer::Expr*>{},
                                        nullptr,
                                        nullptr,
                                        std::vector<int64_t>{},
                                        ExecutorDeviceType::CPU);
}

RowSetPtr Executor::executeExplain(const ExecutionDispatch& execution_dispatch) {
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
  return (g_cluster || g_use_result_set) ? boost::make_unique<ResultRows>(std::make_shared<ResultSet>(explained_plan))
                                         : boost::make_unique<ResultRows>(explained_plan);
}

// Looks at the targets and returns a feasible device type. We only punt
// to CPU for count distinct and we should probably fix it and remove this.
ExecutorDeviceType Executor::getDeviceTypeForTargets(const RelAlgExecutionUnit& ra_exe_unit,
                                                     const ExecutorDeviceType requested_device_type) {
  for (const auto target_expr : ra_exe_unit.target_exprs) {
    const auto agg_info = target_info(target_expr);
    if (!ra_exe_unit.groupby_exprs.empty() && !isArchPascal(requested_device_type)) {
      if ((agg_info.agg_kind == kAVG || agg_info.agg_kind == kSUM) && agg_info.agg_arg_type.is_fp()) {
        return ExecutorDeviceType::CPU;
      }
    }
    if (dynamic_cast<const Analyzer::RegexpExpr*>(target_expr)) {
      return ExecutorDeviceType::CPU;
    }
  }
  return requested_device_type;
}

Executor::ExecutionDispatch::ExecutionDispatch(Executor* executor,
                                               const RelAlgExecutionUnit& ra_exe_unit,
                                               const std::vector<InputTableInfo>& query_infos,
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
  all_fragment_results_.reserve(query_infos_.front().info.fragments.size());
}

namespace {

inline bool needs_skip_result(const ResultPtr& res) {
  if (auto rows = boost::get<RowSetPtr>(&res)) {
    return !*rows || (*rows)->definitelyHasNoRows();
  } else if (auto tab = boost::get<IterTabPtr>(&res)) {
    // Keep empty table in result array when no error
    return !*tab;
  }

  return false;
}

}  // namespace

void Executor::ExecutionDispatch::run(const ExecutorDeviceType chosen_device_type,
                                      int chosen_device_id,
                                      const ExecutionOptions& options,
                                      const std::vector<std::pair<int, std::vector<size_t>>>& frag_ids,
                                      const size_t ctx_idx,
                                      const int64_t rowid_lookup_key) noexcept {
  try {
    runImpl(chosen_device_type, chosen_device_id, options, frag_ids, ctx_idx, rowid_lookup_key);
  } catch (const std::bad_alloc& e) {
    std::lock_guard<std::mutex> lock(reduce_mutex_);
    LOG(ERROR) << e.what();
    *error_code_ = ERR_OUT_OF_CPU_MEM;
  } catch (const OutOfMemory& e) {
    std::lock_guard<std::mutex> lock(reduce_mutex_);
    LOG(ERROR) << e.what();
    *error_code_ = ERR_OUT_OF_GPU_MEM;
  }
}

void Executor::ExecutionDispatch::runImpl(const ExecutorDeviceType chosen_device_type,
                                          int chosen_device_id,
                                          const ExecutionOptions& options,
                                          const std::vector<std::pair<int, std::vector<size_t>>>& frag_ids,
                                          const size_t ctx_idx,
                                          const int64_t rowid_lookup_key) {
  const auto memory_level =
      chosen_device_type == ExecutorDeviceType::GPU ? Data_Namespace::GPU_LEVEL : Data_Namespace::CPU_LEVEL;
  const int outer_table_id = ra_exe_unit_.input_descs[0].getTableId();
  CHECK_GE(frag_ids.size(), size_t(1));
  CHECK_LE(frag_ids.size(), size_t(2));
  CHECK_EQ(frag_ids[0].first, outer_table_id);
  const auto& outer_tab_frag_ids = frag_ids[0].second;
  for (const auto frag_id : frag_ids[0].second) {
    const auto& outer_fragment = query_infos_.front().info.fragments[frag_id];
    if (co_.device_type_ != ExecutorDeviceType::Hybrid) {
      chosen_device_id = outer_fragment.deviceIds[static_cast<int>(memory_level)];
    }
  }
  CHECK_GE(chosen_device_id, 0);
  CHECK_LT(chosen_device_id, max_gpu_count);
  // need to own them while query executes
  auto chunk_iterators_ptr = std::make_shared<std::list<ChunkIter>>();
  std::list<std::shared_ptr<Chunk_NS::Chunk>> chunks;
  std::unique_ptr<std::lock_guard<std::mutex>> gpu_lock;
  if (chosen_device_type == ExecutorDeviceType::GPU) {
    gpu_lock.reset(new std::lock_guard<std::mutex>(executor_->gpu_exec_mutex_[chosen_device_id]));
  }
  FetchResult fetch_result;
  try {
    std::map<int, const TableFragments*> all_tables_fragments;
    for (size_t tab_idx = 0, tab_cnt = ra_exe_unit_.input_descs.size(); tab_idx < tab_cnt; ++tab_idx) {
      int table_id = ra_exe_unit_.input_descs[tab_idx].getTableId();
      CHECK_EQ(query_infos_[tab_idx].table_id, table_id);
      const auto& fragments = query_infos_[tab_idx].info.fragments;
      if (!all_tables_fragments.count(table_id)) {
        all_tables_fragments.insert(std::make_pair(table_id, &fragments));
      }
    }
    for (size_t tab_idx = 0,
                extra_tab_base = ra_exe_unit_.input_descs.size(),
                tab_cnt = ra_exe_unit_.extra_input_descs.size();
         tab_idx < tab_cnt;
         ++tab_idx) {
      int table_id = ra_exe_unit_.extra_input_descs[tab_idx].getTableId();
      const auto& fragments = query_infos_[extra_tab_base + tab_idx].info.fragments;
      all_tables_fragments.insert(std::make_pair(table_id, &fragments));
    }
    fetch_result = executor_->fetchChunks(*this,
                                          ra_exe_unit_,
                                          chosen_device_id,
                                          memory_level,
                                          all_tables_fragments,
                                          frag_ids,
                                          cat_,
                                          *chunk_iterators_ptr,
                                          chunks);
    if (options.with_dynamic_watchdog && !dynamic_watchdog_set_.test_and_set(std::memory_order_acquire)) {
      CHECK_GT(options.dynamic_watchdog_time_limit, 0);
      auto cycle_budget = dynamic_watchdog_init(options.dynamic_watchdog_time_limit);
      LOG(INFO) << "Dynamic Watchdog budget: CPU: " << std::to_string(options.dynamic_watchdog_time_limit) << "ms, "
                << std::to_string(cycle_budget) << " cycles";
    }
  } catch (const OutOfMemory&) {
    std::lock_guard<std::mutex> lock(reduce_mutex_);
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
            : compilation_result.query_mem_desc.getQueryExecutionContext(ra_exe_unit_,
                                                                         executor_->plan_state_->init_agg_vals_,
                                                                         executor_,
                                                                         chosen_device_type,
                                                                         chosen_device_id,
                                                                         fetch_result.col_buffers,
                                                                         fetch_result.iter_buffers,
                                                                         fetch_result.frag_offsets,
                                                                         row_set_mem_owner_,
                                                                         compilation_result.output_columnar,
                                                                         compilation_result.query_mem_desc.sortOnGpu(),
                                                                         render_allocator_map_);
  } catch (const OutOfHostMemory& e) {
    std::lock_guard<std::mutex> lock(reduce_mutex_);
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
            compilation_result.query_mem_desc.getQueryExecutionContext(ra_exe_unit_,
                                                                       executor_->plan_state_->init_agg_vals_,
                                                                       executor_,
                                                                       chosen_device_type,
                                                                       chosen_device_id,
                                                                       fetch_result.col_buffers,
                                                                       fetch_result.iter_buffers,
                                                                       fetch_result.frag_offsets,
                                                                       row_set_mem_owner_,
                                                                       compilation_result.output_columnar,
                                                                       compilation_result.query_mem_desc.sortOnGpu(),
                                                                       render_allocator_map_);
      } catch (const OutOfHostMemory& e) {
        std::lock_guard<std::mutex> lock(reduce_mutex_);
        LOG(ERROR) << e.what();
        *error_code_ = ERR_OUT_OF_CPU_MEM;
        return;
      }
    }
    query_exe_context = query_contexts_[ctx_idx].get();
  }
  CHECK(query_exe_context);
  int32_t err{0};
  uint32_t start_rowid{0};
  if (rowid_lookup_key >= 0) {
    CHECK_LE(frag_ids.size(), size_t(1));
    if (!frag_ids.empty()) {
      const auto& all_frag_row_offsets = getFragOffsets();
      start_rowid = rowid_lookup_key - all_frag_row_offsets[frag_ids.begin()->second.front()];
    }
  }

  ResultPtr device_results;
  if (ra_exe_unit_.groupby_exprs.empty()) {
    err = executor_->executePlanWithoutGroupBy(ra_exe_unit_,
                                               compilation_result,
                                               co_.hoist_literals_,
                                               device_results,
                                               ra_exe_unit_.target_exprs,
                                               chosen_device_type,
                                               fetch_result.col_buffers,
                                               query_exe_context,
                                               fetch_result.num_rows,
                                               fetch_result.frag_offsets,
                                               getFragmentStride(frag_ids),
                                               &cat_.get_dataMgr(),
                                               chosen_device_id,
                                               start_rowid,
                                               ra_exe_unit_.input_descs.size(),
                                               render_allocator_map_);
  } else {
    err = executor_->executePlanWithGroupBy(ra_exe_unit_,
                                            compilation_result,
                                            co_.hoist_literals_,
                                            device_results,
                                            chosen_device_type,
                                            fetch_result.col_buffers,
                                            outer_tab_frag_ids,
                                            query_exe_context,
                                            fetch_result.num_rows,
                                            fetch_result.frag_offsets,
                                            getFragmentStride(frag_ids),
                                            &cat_.get_dataMgr(),
                                            chosen_device_id,
                                            ra_exe_unit_.scan_limit,
                                            co_.device_type_ == ExecutorDeviceType::Hybrid,
                                            start_rowid,
                                            ra_exe_unit_.input_descs.size(),
                                            render_allocator_map_);
  }
  if (auto rows_pp = boost::get<RowSetPtr>(&device_results)) {
    if (auto& rows_ptr = *rows_pp) {
      rows_ptr->holdChunks(chunks);
      rows_ptr->holdChunkIterators(chunk_iterators_ptr);
    }
  }
  {
    std::lock_guard<std::mutex> lock(reduce_mutex_);
    if (err) {
      *error_code_ = err;
    }
    if (!needs_skip_result(device_results)) {
      all_fragment_results_.emplace_back(std::move(device_results), outer_tab_frag_ids);
    }
  }
}

namespace {

size_t get_mapped_frag_id_of_src_table(const std::vector<std::pair<int, size_t>>& join_dimensions,
                                       const int src_tab_id,
                                       const size_t dst_frag_id) {
  CHECK(join_dimensions.size());
  std::unordered_map<int, size_t> tab_id_to_frag_cnt;
  size_t combination_count{1};
  for (const auto& table : join_dimensions) {
    tab_id_to_frag_cnt.insert(table);
    CHECK(table.second);
    combination_count *= table.second;
  }

  auto cnt_it = tab_id_to_frag_cnt.find(src_tab_id);
  CHECK(cnt_it != tab_id_to_frag_cnt.end());
  if (size_t(1) == cnt_it->second) {
    return size_t(0);
  }
  size_t crt_frag_id{dst_frag_id};
  for (auto dim_it = join_dimensions.rbegin(); dim_it->first != src_tab_id && dim_it != join_dimensions.rend();
       ++dim_it) {
    crt_frag_id %= combination_count;
    combination_count /= dim_it->second;
  }
  combination_count /= cnt_it->second;
  return crt_frag_id / combination_count;
}

}  // namespace

const int8_t* Executor::ExecutionDispatch::getScanColumn(
    const int table_id,
    const int frag_id,
    const int col_id,
    const std::map<int, const TableFragments*>& all_tables_fragments,
    std::list<std::shared_ptr<Chunk_NS::Chunk>>& chunk_holder,
    std::list<ChunkIter>& chunk_iter_holder,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_id) const {
  static std::mutex str_dec_mutex;  // TODO(alex): remove
  const auto fragments_it = all_tables_fragments.find(table_id);
  CHECK(fragments_it != all_tables_fragments.end());
  const auto fragments = fragments_it->second;
  const auto& fragment = (*fragments)[frag_id];
  std::shared_ptr<Chunk_NS::Chunk> chunk;
  auto chunk_meta_it = fragment.getChunkMetadataMap().find(col_id);
  CHECK(chunk_meta_it != fragment.getChunkMetadataMap().end());
  CHECK(table_id > 0);
  auto cd = get_column_descriptor(col_id, table_id, cat_);
  CHECK(cd);
  {
    ChunkKey chunk_key{cat_.get_currentDB().dbId, table_id, col_id, fragment.fragmentId};
    std::lock_guard<std::mutex> lock(str_dec_mutex);
    chunk = Chunk_NS::Chunk::getChunk(cd,
                                      &cat_.get_dataMgr(),
                                      chunk_key,
                                      memory_level,
                                      memory_level == Data_Namespace::CPU_LEVEL ? 0 : device_id,
                                      chunk_meta_it->second.numBytes,
                                      chunk_meta_it->second.numElements);
    chunk_holder.push_back(chunk);
  }
  const auto col_type = get_column_type(col_id, table_id, cd, executor_->temporary_tables_);
  const bool is_real_string = col_type.is_string() && col_type.get_compression() == kENCODING_NONE;
  if (is_real_string || col_type.is_array()) {
    CHECK_GT(table_id, 0);
    CHECK(chunk_meta_it != fragment.getChunkMetadataMap().end());
    chunk_iter_holder.push_back(chunk->begin_iterator(chunk_meta_it->second));
    auto& chunk_iter = chunk_iter_holder.back();
    if (memory_level == Data_Namespace::CPU_LEVEL) {
      return reinterpret_cast<int8_t*>(&chunk_iter);
    } else {
      CHECK_EQ(Data_Namespace::GPU_LEVEL, memory_level);
      auto& data_mgr = cat_.get_dataMgr();
      auto chunk_iter_gpu = alloc_gpu_mem(&data_mgr, sizeof(ChunkIter), device_id, nullptr);
      copy_to_gpu(&data_mgr, chunk_iter_gpu, &chunk_iter, sizeof(ChunkIter), device_id);
      return reinterpret_cast<int8_t*>(chunk_iter_gpu);
    }
  } else {
    auto ab = chunk->get_buffer();
    CHECK(ab->getMemoryPtr());
    return ab->getMemoryPtr();  // @TODO(alex) change to use ChunkIter
  }
}

uint32_t Executor::ExecutionDispatch::getFragmentStride(
    const std::vector<std::pair<int, std::vector<size_t>>>& frag_ids) const {
#ifdef ENABLE_MULTIFRAG_JOIN
  const bool is_hash_join = executor_->plan_state_->join_info_.join_impl_type_ == Executor::JoinImplType::HashOneToOne;
  if (is_hash_join) {
    CHECK_EQ(ra_exe_unit_.input_descs.size(), size_t(2));
    CHECK_EQ(frag_ids.size(), size_t(2));
    CHECK_EQ(ra_exe_unit_.input_descs.back().getTableId(), frag_ids[1].first);
    return static_cast<uint32_t>(frag_ids[1].second.size());
  }
#endif
  return 1u;
}

#ifdef ENABLE_MULTIFRAG_JOIN
const int8_t* Executor::ExecutionDispatch::getAllScanColumnFrags(
    const int table_id,
    const int col_id,
    const std::map<int, const TableFragments*>& all_tables_fragments,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_id) const {
  const auto fragments_it = all_tables_fragments.find(table_id);
  CHECK(fragments_it != all_tables_fragments.end());
  const auto fragments = fragments_it->second;
  const auto frag_count = fragments->size();
  std::vector<std::unique_ptr<ColumnarResults>> column_frags;
  const ColumnarResults* table_column = nullptr;
  const InputColDescriptor col_desc(col_id, table_id, int(0));
  CHECK(col_desc.getScanDesc().getSourceType() == InputSourceType::TABLE);
  {
    std::lock_guard<std::mutex> columnar_conversion_guard(columnar_conversion_mutex_);
    auto column_it = columnarized_scan_table_cache_.find(col_desc);
    if (column_it == columnarized_scan_table_cache_.end()) {
      columnarized_scan_table_cache_.insert(std::make_pair(col_desc, nullptr));
      column_it = columnarized_scan_table_cache_.find(col_desc);
      for (size_t frag_id = 0; frag_id < frag_count; ++frag_id) {
        std::list<std::shared_ptr<Chunk_NS::Chunk>> chunk_holder;
        std::list<ChunkIter> chunk_iter_holder;
        const auto& fragment = (*fragments)[frag_id];
        auto chunk_meta_it = fragment.getChunkMetadataMap().find(col_id);
        auto col_buffer = getScanColumn(table_id,
                                        static_cast<int>(frag_id),
                                        col_id,
                                        all_tables_fragments,
                                        chunk_holder,
                                        chunk_iter_holder,
                                        Data_Namespace::CPU_LEVEL,
                                        int(0));
        column_frags.push_back(boost::make_unique<ColumnarResults>(
            row_set_mem_owner_, col_buffer, fragment.getNumTuples(), chunk_meta_it->second.sqlType));
      }
      column_it->second = ColumnarResults::mergeResults(row_set_mem_owner_, column_frags);
    }
    CHECK(column_it != columnarized_scan_table_cache_.end());
    table_column = column_it->second.get();
  }
  return getColumn(table_column, 0, &cat_.get_dataMgr(), memory_level, device_id);
}
#endif

std::vector<const ColumnarResults*> Executor::ExecutionDispatch::getAllScanColumnFrags(
    const int table_id,
    const int col_id,
    const std::map<int, const TableFragments*>& all_tables_fragments) const {
  const auto fragments_it = all_tables_fragments.find(table_id);
  CHECK(fragments_it != all_tables_fragments.end());
  const auto fragments = fragments_it->second;
  const auto frag_count = fragments->size();
  std::vector<const ColumnarResults*> results(frag_count, nullptr);
  const InputColDescriptor desc(col_id, table_id, int(0));
  CHECK(desc.getScanDesc().getSourceType() == InputSourceType::TABLE);
  auto frags_it = columnarized_ref_table_cache_.find(desc);
  if (frags_it == columnarized_ref_table_cache_.end()) {
    columnarized_ref_table_cache_.insert(
        std::make_pair(desc, std::unordered_map<CacheKey, std::unique_ptr<const ColumnarResults>>()));
    frags_it = columnarized_ref_table_cache_.find(desc);
    for (int frag_id = 0; frag_id < static_cast<int>(frag_count); ++frag_id) {
      std::list<std::shared_ptr<Chunk_NS::Chunk>> chunk_holder;
      std::list<ChunkIter> chunk_iter_holder;
      const auto& fragment = (*fragments)[frag_id];
      auto chunk_meta_it = fragment.getChunkMetadataMap().find(col_id);
      auto col_buffer = getScanColumn(table_id,
                                      frag_id,
                                      col_id,
                                      all_tables_fragments,
                                      chunk_holder,
                                      chunk_iter_holder,
                                      Data_Namespace::CPU_LEVEL,
                                      int(0));
      frags_it->second.insert(
          std::make_pair(CacheKey{frag_id},
                         boost::make_unique<ColumnarResults>(
                             row_set_mem_owner_, col_buffer, fragment.getNumTuples(), chunk_meta_it->second.sqlType)));
    }
  }
  CHECK(frags_it != columnarized_ref_table_cache_.end());
  CHECK_EQ(frag_count, frags_it->second.size());
  for (int frag_id = 0; frag_id < static_cast<int>(frag_count); ++frag_id) {
    results[frag_id] = frags_it->second[{frag_id}].get();
  }
  return results;
}

const int8_t* Executor::ExecutionDispatch::getColumn(
    const InputColDescriptor* col_desc,
    const int frag_id,
    const std::map<int, const TableFragments*>& all_tables_fragments,
    const std::map<size_t, std::vector<uint64_t>>& tab_id_to_frag_offsets,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_id,
    const bool is_rowid) const {
  CHECK(col_desc);
  auto ind_col_desc = dynamic_cast<const IndirectInputColDescriptor*>(col_desc);
  if (!ind_col_desc) {
    const auto table_id = col_desc->getScanDesc().getTableId();
    return getColumn(get_temporary_table(executor_->temporary_tables_, table_id),
                     table_id,
                     frag_id,
                     col_desc->getColId(),
                     memory_level,
                     device_id);
  }

  const auto ref_table_id = ind_col_desc->getIndirectDesc().getTableId();
  const auto ref_col_id = ind_col_desc->getRefColIndex();
  const auto iter_table_id = ind_col_desc->getIterDesc().getTableId();
  const auto iter_col_id = ind_col_desc->getIterIndex();
  const auto& iter_buffer = get_temporary_table(executor_->temporary_tables_, iter_table_id);
  const bool ref_tab_is_result = ind_col_desc->getIndirectDesc().getSourceType() == InputSourceType::RESULT;

  const InputColDescriptor iter_desc(iter_col_id, iter_table_id, ind_col_desc->getIterDesc().getNestLevel());
  CHECK_LE(size_t(3), ra_exe_unit_.join_dimensions.size());
  const std::vector<std::pair<int, size_t>> previous_join_dims(ra_exe_unit_.join_dimensions.begin(),
                                                               std::prev(ra_exe_unit_.join_dimensions.end()));
  const auto ref_frag_id = get_mapped_frag_id_of_src_table(previous_join_dims, ref_table_id, frag_id);
  CacheKey sub_key;
  auto ref_col_id_for_cache = ref_col_id;
  const ColumnarResults* result{nullptr};
  {
    std::lock_guard<std::mutex> columnar_conversion_guard(columnar_conversion_mutex_);
    if (columnarized_table_cache_.empty() || !columnarized_table_cache_.count(ref_table_id)) {
      columnarized_table_cache_.insert(
          std::make_pair(iter_table_id, std::unordered_map<int, std::unique_ptr<const ColumnarResults>>()));
    }
    auto& frag_id_to_iters = columnarized_table_cache_[iter_table_id];
    if (frag_id_to_iters.empty() || !frag_id_to_iters.count(frag_id)) {
      frag_id_to_iters.insert(std::make_pair(
          frag_id,
          std::unique_ptr<const ColumnarResults>(columnarize_result(row_set_mem_owner_, iter_buffer, frag_id))));
    }

    if (columnarized_ref_table_cache_.empty() || !columnarized_ref_table_cache_.count(iter_desc)) {
      columnarized_ref_table_cache_.insert(
          std::make_pair(iter_desc, std::unordered_map<CacheKey, std::unique_ptr<const ColumnarResults>>()));
    }
    auto& frag_id_to_result = columnarized_ref_table_cache_[iter_desc];
    if (ref_tab_is_result) {
      CHECK(!is_rowid);
      const auto& ref_buffer = get_temporary_table(executor_->temporary_tables_, ref_table_id);
      if (columnarized_table_cache_.empty() || !columnarized_table_cache_.count(ref_table_id)) {
        columnarized_table_cache_.insert(
            std::make_pair(ref_table_id, std::unordered_map<int, std::unique_ptr<const ColumnarResults>>()));
      }
      auto& frag_id_to_ref = columnarized_table_cache_[ref_table_id];
      if (frag_id_to_ref.empty() || !frag_id_to_ref.count(ref_frag_id)) {
        frag_id_to_ref.insert(std::make_pair(
            ref_frag_id,
            std::unique_ptr<const ColumnarResults>(columnarize_result(row_set_mem_owner_, ref_buffer, ref_frag_id))));
      }
      sub_key = {frag_id};
      if (frag_id_to_result.empty() || !frag_id_to_result.count(sub_key)) {
        frag_id_to_result.insert(std::make_pair(
            sub_key,
            ColumnarResults::createIndexedResults(
                row_set_mem_owner_, *frag_id_to_ref[ref_frag_id], *frag_id_to_iters[frag_id], iter_col_id)));
      }
    } else {
      sub_key = {frag_id, ref_col_id};
      if (frag_id_to_result.empty() || !frag_id_to_result.count(sub_key)) {
        const auto frag_offsets_it = tab_id_to_frag_offsets.find(ref_table_id);
        CHECK(frag_offsets_it != tab_id_to_frag_offsets.end());
        const auto& frag_offsets = frag_offsets_it->second;
        CHECK_LT(ref_frag_id, frag_offsets.size());
        // TODO(miyu): Check rowid offseting
        if (is_rowid) {
          frag_id_to_result.insert(std::make_pair(
              sub_key,
              ColumnarResults::createOffsetResults(
                  row_set_mem_owner_, *frag_id_to_iters[frag_id], iter_col_id, frag_offsets[ref_frag_id])));
        } else {
#ifdef ENABLE_MULTIFRAG_JOIN
          // Each dispatch has only one fragment of outer table.
          if (ref_table_id != ra_exe_unit_.join_dimensions[0].first) {
            auto ref_frags = getAllScanColumnFrags(ref_table_id, ref_col_id, all_tables_fragments);
            frag_id_to_result.insert(std::make_pair(
                sub_key,
                ColumnarResults::createIndexedResults(
                    row_set_mem_owner_, ref_frags, frag_offsets, *frag_id_to_iters[frag_id], iter_col_id)));
          } else
#endif
          {
            const auto fragments_it = all_tables_fragments.find(ref_table_id);
            CHECK(fragments_it != all_tables_fragments.end());
            const auto fragments = fragments_it->second;
            const auto& fragment = (*fragments)[ref_frag_id];
            auto chunk_meta_it = fragment.getChunkMetadataMap().find(ref_col_id);
            CHECK(chunk_meta_it != fragment.getChunkMetadataMap().end());
            std::list<std::shared_ptr<Chunk_NS::Chunk>> chunk_holder;
            std::list<ChunkIter> chunk_iter_holder;
            auto col_buffer = getScanColumn(ref_table_id,
                                            ref_frag_id,
                                            ref_col_id,
                                            all_tables_fragments,
                                            chunk_holder,
                                            chunk_iter_holder,
                                            Data_Namespace::CPU_LEVEL,
                                            device_id);
            ColumnarResults ref_values(
                row_set_mem_owner_, col_buffer, fragment.getNumTuples(), chunk_meta_it->second.sqlType);
            frag_id_to_result.insert(
                std::make_pair(sub_key,
                               ColumnarResults::createIndexedResults(
                                   row_set_mem_owner_, ref_values, *frag_id_to_iters[frag_id], iter_col_id)));
          }
        }
      }
      ref_col_id_for_cache = 0;
    }
    CHECK_NE(size_t(0), columnarized_ref_table_cache_.count(iter_desc));
    result = columnarized_ref_table_cache_[iter_desc][sub_key].get();
  }
  CHECK_GE(ref_col_id, 0);
  return getColumn(result, ref_col_id_for_cache, &cat_.get_dataMgr(), memory_level, device_id);
}

const int8_t* Executor::ExecutionDispatch::getColumn(const ResultPtr& buffer,
                                                     const int table_id,
                                                     const int frag_id,
                                                     const int col_id,
                                                     const Data_Namespace::MemoryLevel memory_level,
                                                     const int device_id) const {
  const ColumnarResults* result{nullptr};
  {
    std::lock_guard<std::mutex> columnar_conversion_guard(columnar_conversion_mutex_);
    if (columnarized_table_cache_.empty() || !columnarized_table_cache_.count(table_id)) {
      columnarized_table_cache_.insert(
          std::make_pair(table_id, std::unordered_map<int, std::unique_ptr<const ColumnarResults>>()));
    }
    auto& frag_id_to_result = columnarized_table_cache_[table_id];
    if (frag_id_to_result.empty() || !frag_id_to_result.count(frag_id)) {
      frag_id_to_result.insert(std::make_pair(
          frag_id, std::unique_ptr<const ColumnarResults>(columnarize_result(row_set_mem_owner_, buffer, frag_id))));
    }
    CHECK_NE(size_t(0), columnarized_table_cache_.count(table_id));
    result = columnarized_table_cache_[table_id][frag_id].get();
  }
  CHECK_GE(col_id, 0);
  return getColumn(result, col_id, &cat_.get_dataMgr(), memory_level, device_id);
}

const int8_t* Executor::ExecutionDispatch::getColumn(const ColumnarResults* columnar_results,
                                                     const int col_id,
                                                     Data_Namespace::DataMgr* data_mgr,
                                                     const Data_Namespace::MemoryLevel memory_level,
                                                     const int device_id) {
  const auto& col_buffers = columnar_results->getColumnBuffers();
  CHECK_LT(static_cast<size_t>(col_id), col_buffers.size());
  if (memory_level == Data_Namespace::GPU_LEVEL) {
    const auto& col_ti = columnar_results->getColumnType(col_id);
    const auto num_bytes = columnar_results->size() * col_ti.get_size();
    auto gpu_col_buffer = alloc_gpu_mem(data_mgr, num_bytes, device_id, nullptr);
    copy_to_gpu(data_mgr, gpu_col_buffer, col_buffers[col_id], num_bytes, device_id);
    return reinterpret_cast<const int8_t*>(gpu_col_buffer);
  }
  return col_buffers[col_id];
}

int8_t Executor::ExecutionDispatch::compile(const Executor::JoinInfo& join_info,
                                            const size_t max_groups_buffer_entry_guess,
                                            const int8_t crt_min_byte_width,
                                            const ExecutionOptions& options,
                                            const bool has_cardinality_estimation) {
  int8_t actual_min_byte_width{MAX_BYTE_WIDTH_SUPPORTED};
  auto compile_on_cpu = [&]() {
    const CompilationOptions co_cpu{
        ExecutorDeviceType::CPU, co_.hoist_literals_, co_.opt_level_, co_.with_dynamic_watchdog_};
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
                                     join_info,
                                     has_cardinality_estimation);
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
                                     join_info,
                                     has_cardinality_estimation);
    }
    for (auto wids : compilation_result_cpu_.query_mem_desc.agg_col_widths) {
      actual_min_byte_width = std::min(actual_min_byte_width, wids.compact);
    }
  };

  if (co_.device_type_ == ExecutorDeviceType::CPU || co_.device_type_ == ExecutorDeviceType::Hybrid) {
    compile_on_cpu();
  }

  if (co_.device_type_ == ExecutorDeviceType::GPU ||
      (co_.device_type_ == ExecutorDeviceType::Hybrid && cat_.get_dataMgr().gpusPresent())) {
    const CompilationOptions co_gpu{
        ExecutorDeviceType::GPU, co_.hoist_literals_, co_.opt_level_, co_.with_dynamic_watchdog_};
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
                                     join_info,
                                     has_cardinality_estimation);
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
                                     join_info,
                                     has_cardinality_estimation);
    }
    for (auto wids : compilation_result_gpu_.query_mem_desc.agg_col_widths) {
      actual_min_byte_width = std::min(actual_min_byte_width, wids.compact);
    }
  }

  if (executor_->cgen_state_->must_run_on_cpu_) {
    if (co_.device_type_ == ExecutorDeviceType::GPU) {  // override user choice
      LOG(INFO) << "Query cannot run on GPU, punt to CPU";
      compile_on_cpu();
    }
    co_.device_type_ = ExecutorDeviceType::CPU;
  }
  return std::max(actual_min_byte_width, crt_min_byte_width);
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
  std::lock_guard<std::mutex> lock(all_frag_row_offsets_mutex_);
  if (all_frag_row_offsets_.empty()) {
    all_frag_row_offsets_.resize(query_infos_.front().info.fragments.size() + 1);
    for (size_t i = 1; i <= query_infos_.front().info.fragments.size(); ++i) {
      all_frag_row_offsets_[i] =
          all_frag_row_offsets_[i - 1] + query_infos_.front().info.fragments[i - 1].getNumTuples();
    }
  }
  return all_frag_row_offsets_;
}

const std::vector<std::unique_ptr<QueryExecutionContext>>& Executor::ExecutionDispatch::getQueryContexts() const {
  return query_contexts_;
}

std::vector<std::pair<ResultPtr, std::vector<size_t>>>& Executor::ExecutionDispatch::getFragmentResults() {
  return all_fragment_results_;
}

namespace {

int64_t inline_null_val(const SQLTypeInfo& ti) {
  CHECK(ti.is_number() || ti.is_time() || ti.is_boolean());
  if (ti.is_fp()) {
    const auto double_null_val = inline_fp_null_val(ti);
    return *reinterpret_cast<const int64_t*>(&double_null_val);
  }
  return inline_int_null_val(ti);
}

void fill_entries_for_empty_input(std::vector<TargetInfo>& target_infos,
                                  std::vector<int64_t>& entry,
                                  const std::vector<Analyzer::Expr*>& target_exprs,
                                  const QueryMemoryDescriptor& query_mem_desc) {
  for (size_t target_idx = 0; target_idx < target_exprs.size(); ++target_idx) {
    const auto target_expr = target_exprs[target_idx];
    const auto agg_info = target_info(target_expr);
    CHECK(agg_info.is_agg);
    target_infos.push_back(agg_info);
    if (g_cluster) {
      CHECK(query_mem_desc.executor_);
      auto row_set_mem_owner = query_mem_desc.executor_->getRowSetMemoryOwner();
      CHECK(row_set_mem_owner);
      CHECK_LT(target_idx, query_mem_desc.count_distinct_descriptors_.size());
      const auto& count_distinct_desc = query_mem_desc.count_distinct_descriptors_[target_idx];
      if (count_distinct_desc.impl_type_ == CountDistinctImplType::Bitmap) {
        auto count_distinct_buffer =
            static_cast<int8_t*>(checked_calloc(count_distinct_desc.bitmapPaddedSizeBytes(), 1));
        CHECK(row_set_mem_owner);
        row_set_mem_owner->addCountDistinctBuffer(
            count_distinct_buffer, count_distinct_desc.bitmapPaddedSizeBytes(), true);
        entry.push_back(reinterpret_cast<int64_t>(count_distinct_buffer));
        continue;
      }
      if (count_distinct_desc.impl_type_ == CountDistinctImplType::StdSet) {
        auto count_distinct_set = new std::set<int64_t>();
        CHECK(row_set_mem_owner);
        row_set_mem_owner->addCountDistinctSet(count_distinct_set);
        entry.push_back(reinterpret_cast<int64_t>(count_distinct_set));
        continue;
      }
    }
    if (agg_info.agg_kind == kCOUNT || agg_info.agg_kind == kAPPROX_COUNT_DISTINCT) {
      entry.push_back(0);
    } else if (agg_info.agg_kind == kAVG) {
      entry.push_back(inline_null_val(agg_info.agg_arg_type));
      entry.push_back(0);
    } else {
      entry.push_back(inline_null_val(agg_info.sql_type));
    }
  }
}

RowSetPtr build_row_for_empty_input(const std::vector<Analyzer::Expr*>& target_exprs_in,
                                    const QueryMemoryDescriptor& query_mem_desc,
                                    const ExecutorDeviceType device_type) {
  std::vector<std::shared_ptr<Analyzer::Expr>> target_exprs_owned_copies;
  std::vector<Analyzer::Expr*> target_exprs;
  for (const auto target_expr : target_exprs_in) {
    const auto target_expr_copy = std::dynamic_pointer_cast<Analyzer::AggExpr>(target_expr->deep_copy());
    CHECK(target_expr_copy);
    auto ti = target_expr->get_type_info();
    ti.set_notnull(false);
    target_expr_copy->set_type_info(ti);
    if (target_expr_copy->get_arg()) {
      auto arg_ti = target_expr_copy->get_arg()->get_type_info();
      arg_ti.set_notnull(false);
      target_expr_copy->get_arg()->set_type_info(arg_ti);
    }
    target_exprs_owned_copies.push_back(target_expr_copy);
    target_exprs.push_back(target_expr_copy.get());
  }
  std::vector<TargetInfo> target_infos;
  std::vector<int64_t> entry;
  fill_entries_for_empty_input(target_infos, entry, target_exprs, query_mem_desc);
  if (can_use_result_set(query_mem_desc, ExecutorDeviceType::CPU)) {
    const auto executor = query_mem_desc.executor_;
    CHECK(executor);
    auto row_set_mem_owner = executor->getRowSetMemoryOwner();
    CHECK(row_set_mem_owner);
    auto rs = std::make_shared<ResultSet>(target_infos, device_type, query_mem_desc, row_set_mem_owner, executor);
    rs->allocateStorage();
    rs->fillOneEntry(entry);
    return boost::make_unique<ResultRows>(rs);
  }
  auto result_rows = boost::make_unique<ResultRows>(
      query_mem_desc, target_exprs, nullptr, nullptr, std::vector<int64_t>{}, ExecutorDeviceType::CPU);

  result_rows->beginRow();
  for (size_t target_idx = 0, entry_idx = 0; target_idx < target_infos.size(); ++target_idx, ++entry_idx) {
    const auto agg_info = target_infos[target_idx];
    CHECK(agg_info.is_agg);
    CHECK_LT(entry_idx, entry.size());
    if (agg_info.agg_kind == kAVG) {
      CHECK_LT(entry_idx + 1, entry.size());
      result_rows->addValue(entry[entry_idx], entry[entry_idx + 1]);
      ++entry_idx;
    } else {
      result_rows->addValue(entry[entry_idx]);
    }
  }
  return result_rows;
}

}  // namespace

RowSetPtr Executor::collectAllDeviceResults(ExecutionDispatch& execution_dispatch,
                                            const std::vector<Analyzer::Expr*>& target_exprs,
                                            const QueryMemoryDescriptor& query_mem_desc,
                                            std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                                            const bool output_columnar) {
  const auto& ra_exe_unit = execution_dispatch.getExecutionUnit();
  for (const auto& query_exe_context : execution_dispatch.getQueryContexts()) {
    if (!query_exe_context) {
      continue;
    }
    execution_dispatch.getFragmentResults().emplace_back(
        query_exe_context->getRowSet(
            ra_exe_unit, query_mem_desc, execution_dispatch.getDeviceType() == ExecutorDeviceType::Hybrid),
        std::vector<size_t>{});
  }
  auto& result_per_device = execution_dispatch.getFragmentResults();
  if (result_per_device.empty() && query_mem_desc.hash_type == GroupByColRangeType::Scan) {
    return build_row_for_empty_input(target_exprs, query_mem_desc, execution_dispatch.getDeviceType());
  }
  if (use_speculative_top_n(ra_exe_unit, query_mem_desc)) {
    return reduceSpeculativeTopN(ra_exe_unit, result_per_device, row_set_mem_owner, query_mem_desc);
  }
  return reduceMultiDeviceResults(ra_exe_unit,
                                  result_per_device,
                                  row_set_mem_owner,
                                  query_mem_desc,
                                  output_columnar,
                                  execution_dispatch.getDeviceType());
}

void Executor::dispatchFragments(
    const std::function<void(const ExecutorDeviceType chosen_device_type,
                             int chosen_device_id,
                             const std::vector<std::pair<int, std::vector<size_t>>>& frag_ids,
                             const size_t ctx_idx,
                             const int64_t rowid_lookup_key)> dispatch,
    const ExecutionDispatch& execution_dispatch,
    const ExecutionOptions& eo,
    const bool is_agg,
    std::map<int, const TableFragments*>& selected_tables_fragments,
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
  const auto& outer_table_desc = ra_exe_unit.input_descs.front();
  const int outer_table_id = outer_table_desc.getTableId();
  auto it = selected_tables_fragments.find(outer_table_id);
  CHECK(it != selected_tables_fragments.end());
  const auto fragments = it->second;
  const auto device_type = execution_dispatch.getDeviceType();

  const auto& query_mem_desc = execution_dispatch.getQueryMemoryDescriptor();
  const bool allow_multifrag =
      eo.allow_multifrag && (ra_exe_unit.groupby_exprs.empty() || query_mem_desc.usesCachedContext() ||
                             query_mem_desc.hash_type == GroupByColRangeType::MultiCol ||
                             query_mem_desc.hash_type == GroupByColRangeType::Projection);

  if ((device_type == ExecutorDeviceType::GPU) && allow_multifrag && is_agg) {
    // NB: We should never be on this path when the query is retried because of
    //     running out of group by slots; also, for scan only queries (!agg_plan)
    //     we want the high-granularity, fragment by fragment execution instead.
    std::unordered_map<int, std::vector<std::pair<int, std::vector<size_t>>>> fragments_per_device;
    for (size_t frag_id = 0; frag_id < fragments->size(); ++frag_id) {
      const auto& fragment = (*fragments)[frag_id];
      const auto skip_frag =
          skipFragment(outer_table_desc, fragment, ra_exe_unit.simple_quals, execution_dispatch, frag_id);
      if (skip_frag.first) {
        continue;
      }
      const int device_id = fragment.deviceIds[static_cast<int>(Data_Namespace::GPU_LEVEL)];
      for (size_t j = 0; j < ra_exe_unit.input_descs.size(); ++j) {
        const auto table_id = ra_exe_unit.input_descs[j].getTableId();
        auto table_frags_it = selected_tables_fragments.find(table_id);
        CHECK(table_frags_it != selected_tables_fragments.end());
        const auto& frag_ids = [&]() -> std::vector<size_t> {
          if (!j) {
            return {frag_id};
          } else {
            auto& inner_frags = table_frags_it->second;
            CHECK_LT(size_t(1), ra_exe_unit.input_descs.size());
            CHECK_EQ(table_id, ra_exe_unit.input_descs[1].getTableId());
            std::vector<size_t> all_frag_ids(inner_frags->size());
#ifndef ENABLE_MULTIFRAG_JOIN
            if (all_frag_ids.size() > 1) {
              throw std::runtime_error("Multi-fragment inner table '" +
                                       get_table_name(ra_exe_unit.input_descs[1], *catalog_) + "' not supported yet");
            }
#endif
            std::iota(all_frag_ids.begin(), all_frag_ids.end(), 0);
            return all_frag_ids;
          }
        }();
        if (fragments_per_device[device_id].size() < j + 1) {
          fragments_per_device[device_id].emplace_back(table_id, frag_ids);
        } else if (!j) {
          CHECK_EQ(fragments_per_device[device_id][j].first, table_id);
          CHECK_EQ(frag_ids.size(), size_t(1));
          auto& curr_frag_ids = fragments_per_device[device_id][j].second;
          curr_frag_ids.insert(curr_frag_ids.end(), frag_ids.begin(), frag_ids.end());
        }
      }
      rowid_lookup_key = std::max(rowid_lookup_key, skip_frag.second);
    }
    if (eo.with_watchdog && rowid_lookup_key < 0) {
      checkWorkUnitWatchdog(ra_exe_unit, *catalog_);
    }
    for (const auto& kv : fragments_per_device) {
      query_threads.push_back(std::thread(
          dispatch, ExecutorDeviceType::GPU, kv.first, kv.second, kv.first % context_count, rowid_lookup_key));
    }
  } else {
    for (size_t i = 0; i < fragments->size(); ++i) {
      const auto& fragment = (*fragments)[i];
      const auto skip_frag = skipFragment(outer_table_desc, fragment, ra_exe_unit.simple_quals, execution_dispatch, i);
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
      std::vector<std::pair<int, std::vector<size_t>>> frag_ids_for_table;
      for (size_t j = 0; j < ra_exe_unit.input_descs.size(); ++j) {
        const auto table_id = ra_exe_unit.input_descs[j].getTableId();
        auto table_frags_it = selected_tables_fragments.find(table_id);
        CHECK(table_frags_it != selected_tables_fragments.end());
        if (!j) {
          frag_ids_for_table.emplace_back(table_id, std::vector<size_t>{i});
        } else {
          auto& inner_frags = table_frags_it->second;
          CHECK_LT(size_t(1), ra_exe_unit.input_descs.size());
          CHECK_EQ(table_id, ra_exe_unit.input_descs[1].getTableId());
          std::vector<size_t> all_frag_ids(inner_frags->size());
#ifndef ENABLE_MULTIFRAG_JOIN
          if (all_frag_ids.size() > 1) {
            throw std::runtime_error("Multi-fragment inner table '" +
                                     get_table_name(ra_exe_unit.input_descs[1], *catalog_) + "' not supported yet");
          }
#endif
          std::iota(all_frag_ids.begin(), all_frag_ids.end(), 0);
          frag_ids_for_table.emplace_back(table_id, all_frag_ids);
        }
      }
      if (eo.with_watchdog && rowid_lookup_key < 0) {
        checkWorkUnitWatchdog(ra_exe_unit, *catalog_);
      }
      query_threads.push_back(std::thread(dispatch,
                                          chosen_device_type,
                                          chosen_device_id,
                                          frag_ids_for_table,
                                          frag_list_idx % context_count,
                                          rowid_lookup_key));
      ++frag_list_idx;
      if (is_sample_query(ra_exe_unit) && fragment.getNumTuples() >= ra_exe_unit.scan_limit) {
        break;
      }
    }
  }
  for (auto& child : query_threads) {
    child.join();
  }
}

std::vector<const int8_t*> Executor::fetchIterTabFrags(const size_t frag_id,
                                                       const ExecutionDispatch& execution_dispatch,
                                                       const InputDescriptor& table_desc,
                                                       const int device_id) {
  CHECK(table_desc.getSourceType() == InputSourceType::RESULT);
  const auto& temp = get_temporary_table(temporary_tables_, table_desc.getTableId());
  const auto table = boost::get<IterTabPtr>(&temp);
  CHECK(table && *table);
  std::vector<const int8_t*> frag_iter_buffers;
  for (size_t i = 0; i < (*table)->colCount(); ++i) {
    const InputColDescriptor desc(i, table_desc.getTableId(), 0);
    frag_iter_buffers.push_back(
        execution_dispatch.getColumn(&desc, frag_id, {}, {}, Data_Namespace::CPU_LEVEL, device_id, false));
  }
  return frag_iter_buffers;
}

namespace {

const ColumnDescriptor* try_get_column_descriptor(const InputColDescriptor* col_desc,
                                                  const Catalog_Namespace::Catalog& cat) {
  const auto ind_col = dynamic_cast<const IndirectInputColDescriptor*>(col_desc);
  const int ref_table_id = ind_col ? ind_col->getIndirectDesc().getTableId() : col_desc->getScanDesc().getTableId();
  const int ref_col_id = ind_col ? ind_col->getRefColIndex() : col_desc->getColId();
  return get_column_descriptor_maybe(ref_col_id, ref_table_id, cat);
}

const SQLTypeInfo get_column_type(const InputColDescriptor* col_desc,
                                  const ColumnDescriptor* cd,
                                  const TemporaryTables* temporary_tables) {
  const auto ind_col = dynamic_cast<const IndirectInputColDescriptor*>(col_desc);
  const int ref_table_id = ind_col ? ind_col->getIndirectDesc().getTableId() : col_desc->getScanDesc().getTableId();
  const int ref_col_id = ind_col ? ind_col->getRefColIndex() : col_desc->getColId();
  return get_column_type(ref_col_id, ref_table_id, cd, temporary_tables);
}

}  // namespace

std::map<size_t, std::vector<uint64_t>> Executor::getAllFragOffsets(
    const std::vector<InputDescriptor>& input_descs,
    const std::map<int, const TableFragments*>& all_tables_fragments) {
  std::map<size_t, std::vector<uint64_t>> tab_id_to_frag_offsets;
  for (auto& desc : input_descs) {
    const auto fragments_it = all_tables_fragments.find(desc.getTableId());
    CHECK(fragments_it != all_tables_fragments.end());
    const auto& fragments = *fragments_it->second;
    std::vector<uint64_t> frag_offsets(fragments.size(), 0);
    for (size_t i = 0, off = 0; i < fragments.size(); ++i) {
      frag_offsets[i] = off;
      off += fragments[i].getNumTuples();
    }
    tab_id_to_frag_offsets.insert(std::make_pair(desc.getTableId(), frag_offsets));
  }
  return tab_id_to_frag_offsets;
}

#ifdef ENABLE_MULTIFRAG_JOIN
// Only fetch columns of hash-joined inner fact table whose fetch are not deferred from all the table fragments.
bool Executor::needFetchAllFragments(const InputColDescriptor& inner_col_desc,
                                     const std::vector<InputDescriptor>& input_descs) const {
  if (inner_col_desc.getScanDesc().getNestLevel() < 1 ||
      inner_col_desc.getScanDesc().getSourceType() != InputSourceType::TABLE ||
      plan_state_->join_info_.join_impl_type_ != JoinImplType::HashOneToOne || input_descs.size() < 2 ||
      plan_state_->isLazyFetchColumn(inner_col_desc)) {
    return false;
  }
  auto inner_table_desc = input_descs[1];
  return inner_col_desc.getScanDesc().getTableId() == inner_table_desc.getTableId();
}
#endif

Executor::FetchResult Executor::fetchChunks(const ExecutionDispatch& execution_dispatch,
                                            const RelAlgExecutionUnit& ra_exe_unit,
                                            const int device_id,
                                            const Data_Namespace::MemoryLevel memory_level,
                                            const std::map<int, const TableFragments*>& all_tables_fragments,
                                            const std::vector<std::pair<int, std::vector<size_t>>>& selected_fragments,
                                            const Catalog_Namespace::Catalog& cat,
                                            std::list<ChunkIter>& chunk_iterators,
                                            std::list<std::shared_ptr<Chunk_NS::Chunk>>& chunks) {
  const auto& col_global_ids = ra_exe_unit.input_col_descs;
  const auto& input_descs = ra_exe_unit.input_descs;
  std::vector<std::vector<size_t>> selected_fragments_crossjoin;
  std::vector<size_t> local_col_to_frag_pos;
  buildSelectedFragsMapping(
      selected_fragments_crossjoin, local_col_to_frag_pos, col_global_ids, selected_fragments, input_descs);

  CartesianProduct<std::vector<std::vector<size_t>>> frag_ids_crossjoin(selected_fragments_crossjoin);

  std::vector<std::vector<const int8_t*>> all_frag_col_buffers;
  std::vector<std::vector<const int8_t*>> all_frag_iter_buffers;
  std::vector<std::vector<int64_t>> all_num_rows;
  std::vector<std::vector<uint64_t>> all_frag_offsets;
  const auto extra_tab_id_to_frag_offsets = getAllFragOffsets(ra_exe_unit.extra_input_descs, all_tables_fragments);
  const bool needs_fetch_iterators =
      ra_exe_unit.join_dimensions.size() > 2 && dynamic_cast<Analyzer::IterExpr*>(ra_exe_unit.target_exprs.front());

  for (const auto& selected_frag_ids : frag_ids_crossjoin) {
    std::vector<const int8_t*> frag_col_buffers(plan_state_->global_to_local_col_ids_.size());
    for (const auto& col_id : col_global_ids) {
      CHECK(col_id);
      const int table_id = col_id->getScanDesc().getTableId();
      const auto cd = try_get_column_descriptor(col_id.get(), cat);
      bool is_rowid = false;
      if (cd && cd->isVirtualCol) {
        CHECK_EQ("rowid", cd->columnName);
        is_rowid = true;
        if (!std::dynamic_pointer_cast<const IndirectInputColDescriptor>(col_id)) {
          continue;
        }
      }
      const auto fragments_it = all_tables_fragments.find(table_id);
      CHECK(fragments_it != all_tables_fragments.end());
      const auto fragments = fragments_it->second;
      auto it = plan_state_->global_to_local_col_ids_.find(*col_id);
      CHECK(it != plan_state_->global_to_local_col_ids_.end());
      CHECK_LT(static_cast<size_t>(it->second), plan_state_->global_to_local_col_ids_.size());
      const size_t frag_id = selected_frag_ids[local_col_to_frag_pos[it->second]];
      CHECK_LT(frag_id, fragments->size());
      auto memory_level_for_column = memory_level;
      if (plan_state_->columns_to_fetch_.find(std::make_pair(col_id->getScanDesc().getTableId(), col_id->getColId())) ==
          plan_state_->columns_to_fetch_.end()) {
        memory_level_for_column = Data_Namespace::CPU_LEVEL;
      }
      const auto col_type = get_column_type(col_id.get(), cd, temporary_tables_);
      const bool is_real_string = col_type.is_string() && col_type.get_compression() == kENCODING_NONE;
      if (col_id->getScanDesc().getSourceType() == InputSourceType::RESULT) {
        CHECK(!is_real_string && !col_type.is_array());
        frag_col_buffers[it->second] = execution_dispatch.getColumn(col_id.get(),
                                                                    frag_id,
                                                                    all_tables_fragments,
                                                                    extra_tab_id_to_frag_offsets,
                                                                    memory_level_for_column,
                                                                    device_id,
                                                                    is_rowid);
      } else {
#ifdef ENABLE_MULTIFRAG_JOIN
        if (needFetchAllFragments(*col_id, input_descs)) {
          frag_col_buffers[it->second] = execution_dispatch.getAllScanColumnFrags(
              table_id, col_id->getColId(), all_tables_fragments, memory_level_for_column, device_id);
        } else
#endif
        {
          frag_col_buffers[it->second] = execution_dispatch.getScanColumn(table_id,
                                                                          frag_id,
                                                                          col_id->getColId(),
                                                                          all_tables_fragments,
                                                                          chunks,
                                                                          chunk_iterators,
                                                                          memory_level_for_column,
                                                                          device_id);
        }
      }
    }
    all_frag_col_buffers.push_back(frag_col_buffers);
    if (needs_fetch_iterators) {
      CHECK_EQ(size_t(2), selected_fragments_crossjoin.size());
      all_frag_iter_buffers.push_back(
          fetchIterTabFrags(selected_frag_ids[0], execution_dispatch, ra_exe_unit.input_descs[0], device_id));
    }
  }
  const auto tab_id_to_frag_offsets = getAllFragOffsets(input_descs, all_tables_fragments);
  for (const auto& selected_frag_ids : frag_ids_crossjoin) {
    std::vector<int64_t> num_rows;
    std::vector<uint64_t> frag_offsets;
    CHECK_EQ(selected_frag_ids.size(), input_descs.size());
    for (size_t tab_idx = 0; tab_idx < input_descs.size(); ++tab_idx) {
      const auto frag_id = selected_frag_ids[tab_idx];
      const auto fragments_it = all_tables_fragments.find(input_descs[tab_idx].getTableId());
      CHECK(fragments_it != all_tables_fragments.end());
      const auto& fragments = *fragments_it->second;
      const auto& fragment = fragments[frag_id];
      num_rows.push_back(fragment.getNumTuples());
      const auto frag_offsets_it = tab_id_to_frag_offsets.find(input_descs[tab_idx].getTableId());
      CHECK(frag_offsets_it != tab_id_to_frag_offsets.end());
      const auto& offsets = frag_offsets_it->second;
      CHECK_LT(frag_id, offsets.size());
      frag_offsets.push_back(offsets[frag_id]);
    }
    all_num_rows.push_back(num_rows);
    // Fragment offsets of outer table should be ONLY used by rowid for now.
    all_frag_offsets.push_back(frag_offsets);
  }
  return {all_frag_col_buffers, all_frag_iter_buffers, all_num_rows, all_frag_offsets};
}

void Executor::buildSelectedFragsMapping(std::vector<std::vector<size_t>>& selected_fragments_crossjoin,
                                         std::vector<size_t>& local_col_to_frag_pos,
                                         const std::list<std::shared_ptr<const InputColDescriptor>>& col_global_ids,
                                         const std::vector<std::pair<int, std::vector<size_t>>>& selected_fragments,
                                         const std::vector<InputDescriptor>& input_descs) {
  local_col_to_frag_pos.resize(plan_state_->global_to_local_col_ids_.size());
  size_t frag_pos{0};
  for (size_t scan_idx = 0; scan_idx < input_descs.size(); ++scan_idx) {
    const int table_id = input_descs[scan_idx].getTableId();
    CHECK_EQ(selected_fragments[scan_idx].first, table_id);
    selected_fragments_crossjoin.push_back(selected_fragments[scan_idx].second);
    for (const auto& col_id : col_global_ids) {
      CHECK(col_id);
      const auto& input_desc = col_id->getScanDesc();
      if (input_desc.getTableId() != table_id || input_desc.getNestLevel() != static_cast<int>(scan_idx)) {
        continue;
      }
      auto it = plan_state_->global_to_local_col_ids_.find(*col_id);
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

int32_t Executor::executePlanWithoutGroupBy(const RelAlgExecutionUnit& ra_exe_unit,
                                            const CompilationResult& compilation_result,
                                            const bool hoist_literals,
                                            ResultPtr& results,
                                            const std::vector<Analyzer::Expr*>& target_exprs,
                                            const ExecutorDeviceType device_type,
                                            std::vector<std::vector<const int8_t*>>& col_buffers,
                                            QueryExecutionContext* query_exe_context,
                                            const std::vector<std::vector<int64_t>>& num_rows,
                                            const std::vector<std::vector<uint64_t>>& frag_offsets,
                                            const uint32_t frag_stride,
                                            Data_Namespace::DataMgr* data_mgr,
                                            const int device_id,
                                            const uint32_t start_rowid,
                                            const uint32_t num_tables,
                                            RenderAllocatorMap* render_allocator_map) {
  results = RowSetPtr(nullptr);
  if (col_buffers.empty()) {
    return 0;
  }

  int32_t error_code = device_type == ExecutorDeviceType::GPU ? 0 : start_rowid;
  std::vector<int64_t*> out_vec;
  const auto hoist_buf = serializeLiterals(compilation_result.literal_values, device_id);
  const auto join_hash_table_ptr = getJoinHashTablePtr(device_type, device_id);
  std::unique_ptr<OutVecOwner> output_memory_scope;
  if (g_enable_dynamic_watchdog && interrupted_) {
    return ERR_INTERRUPTED;
  }
  if (device_type == ExecutorDeviceType::CPU) {
    out_vec = query_exe_context->launchCpuCode(ra_exe_unit,
                                               compilation_result.native_functions,
                                               hoist_literals,
                                               hoist_buf,
                                               col_buffers,
                                               num_rows,
                                               frag_offsets,
                                               frag_stride,
                                               0,
                                               query_exe_context->init_agg_vals_,
                                               &error_code,
                                               num_tables,
                                               join_hash_table_ptr);
    output_memory_scope.reset(new OutVecOwner(out_vec));
  } else {
    try {
      out_vec = query_exe_context->launchGpuCode(ra_exe_unit,
                                                 compilation_result.native_functions,
                                                 hoist_literals,
                                                 hoist_buf,
                                                 col_buffers,
                                                 num_rows,
                                                 frag_offsets,
                                                 frag_stride,
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
  if (error_code == Executor::ERR_OVERFLOW_OR_UNDERFLOW || error_code == Executor::ERR_DIV_BY_ZERO ||
      error_code == Executor::ERR_OUT_OF_TIME || error_code == Executor::ERR_INTERRUPTED) {
    return error_code;
  }
  if (ra_exe_unit.estimator) {
    CHECK(!error_code);
    results =
        boost::make_unique<ResultRows>(std::shared_ptr<ResultSet>(query_exe_context->estimator_result_set_.release()));
    return 0;
  }
  std::vector<int64_t> reduced_outs;
  CHECK_EQ(col_buffers.size() % frag_stride, size_t(0));
  const auto num_out_frags = col_buffers.size() / frag_stride;
  const size_t entry_count =
      device_type == ExecutorDeviceType::GPU ? num_out_frags * blockSize() * gridSize() : num_out_frags;
  if (size_t(1) == entry_count) {
    for (auto out : out_vec) {
      CHECK(out);
      reduced_outs.push_back(*out);
    }
  } else {
    size_t out_vec_idx = 0;
    for (const auto target_expr : target_exprs) {
      const auto agg_info = target_info(target_expr);
      CHECK(agg_info.is_agg);
      int64_t val1;
      if (is_distinct_target(agg_info)) {
        CHECK(agg_info.agg_kind == kCOUNT || agg_info.agg_kind == kAPPROX_COUNT_DISTINCT);
        val1 = out_vec[out_vec_idx][0];
        error_code = 0;
      } else {
        std::tie(val1, error_code) =
            reduceResults(agg_info.agg_kind,
                          agg_info.sql_type,
                          query_exe_context->init_agg_vals_[out_vec_idx],
                          query_exe_context->query_mem_desc_.agg_col_widths[out_vec_idx].compact,
                          out_vec[out_vec_idx],
                          entry_count,
                          false);
      }
      if (error_code) {
        break;
      }
      reduced_outs.push_back(val1);
      if (agg_info.agg_kind == kAVG) {
        int64_t val2;
        std::tie(val2, error_code) =
            reduceResults(kCOUNT,
                          agg_info.sql_type,
                          query_exe_context->init_agg_vals_[out_vec_idx + 1],
                          query_exe_context->query_mem_desc_.agg_col_widths[out_vec_idx + 1].compact,
                          out_vec[out_vec_idx + 1],
                          entry_count,
                          false);
        if (error_code) {
          break;
        }
        reduced_outs.push_back(val2);
        ++out_vec_idx;
      }
      ++out_vec_idx;
    }
  }

  RowSetPtr rows_ptr{nullptr};
  if (can_use_result_set(query_exe_context->query_mem_desc_, device_type)) {
    CHECK_EQ(size_t(1), query_exe_context->result_sets_.size());
    rows_ptr = boost::make_unique<ResultRows>(std::shared_ptr<ResultSet>(query_exe_context->result_sets_[0].release()));
  } else {
    rows_ptr = boost::make_unique<ResultRows>(query_exe_context->query_mem_desc_,
                                              target_exprs,
                                              this,
                                              query_exe_context->row_set_mem_owner_,
                                              query_exe_context->init_agg_vals_,
                                              device_type);
  }
  CHECK(rows_ptr);
  rows_ptr->fillOneRow(reduced_outs);
  results = std::move(rows_ptr);
  return error_code;
}

namespace {

bool check_rows_less_than_needed(const ResultPtr& results, const size_t scan_limit) {
  CHECK(scan_limit);
  if (const auto rows = boost::get<RowSetPtr>(&results)) {
    return (*rows && (*rows)->rowCount() < scan_limit);
  } else if (const auto tab = boost::get<IterTabPtr>(&results)) {
    return (*tab && (*tab)->rowCount() < scan_limit);
  }
  abort();
}

}  // namespace

int32_t Executor::executePlanWithGroupBy(const RelAlgExecutionUnit& ra_exe_unit,
                                         const CompilationResult& compilation_result,
                                         const bool hoist_literals,
                                         ResultPtr& results,
                                         const ExecutorDeviceType device_type,
                                         std::vector<std::vector<const int8_t*>>& col_buffers,
                                         const std::vector<size_t> outer_tab_frag_ids,
                                         QueryExecutionContext* query_exe_context,
                                         const std::vector<std::vector<int64_t>>& num_rows,
                                         const std::vector<std::vector<uint64_t>>& frag_offsets,
                                         const uint32_t frag_stride,
                                         Data_Namespace::DataMgr* data_mgr,
                                         const int device_id,
                                         const int64_t scan_limit,
                                         const bool was_auto_device,
                                         const uint32_t start_rowid,
                                         const uint32_t num_tables,
                                         RenderAllocatorMap* render_allocator_map) {
  if (contains_iter_expr(ra_exe_unit.target_exprs)) {
    results = IterTabPtr(nullptr);
  } else {
    results = RowSetPtr(nullptr);
  }
  if (col_buffers.empty()) {
    return 0;
  }
  CHECK_NE(ra_exe_unit.groupby_exprs.size(), size_t(0));
  // TODO(alex):
  // 1. Optimize size (make keys more compact).
  // 2. Resize on overflow.
  // 3. Optimize runtime.
  auto hoist_buf = serializeLiterals(compilation_result.literal_values, device_id);
  int32_t error_code = device_type == ExecutorDeviceType::GPU ? 0 : start_rowid;
  const auto join_hash_table_ptr = getJoinHashTablePtr(device_type, device_id);
  if (g_enable_dynamic_watchdog && interrupted_) {
    return ERR_INTERRUPTED;
  }
  if (device_type == ExecutorDeviceType::CPU) {
    query_exe_context->launchCpuCode(ra_exe_unit,
                                     compilation_result.native_functions,
                                     hoist_literals,
                                     hoist_buf,
                                     col_buffers,
                                     num_rows,
                                     frag_offsets,
                                     frag_stride,
                                     scan_limit,
                                     query_exe_context->init_agg_vals_,
                                     &error_code,
                                     num_tables,
                                     join_hash_table_ptr);
  } else {
    try {
      query_exe_context->launchGpuCode(ra_exe_unit,
                                       compilation_result.native_functions,
                                       hoist_literals,
                                       hoist_buf,
                                       col_buffers,
                                       num_rows,
                                       frag_offsets,
                                       frag_stride,
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
    } catch (const std::bad_alloc&) {
      return ERR_SPECULATIVE_TOP_OOM;
    } catch (const std::exception& e) {
      LOG(FATAL) << "Error launching the GPU kernel: " << e.what();
    }
  }

  if (error_code == Executor::ERR_OVERFLOW_OR_UNDERFLOW || error_code == Executor::ERR_DIV_BY_ZERO ||
      error_code == Executor::ERR_OUT_OF_TIME || error_code == Executor::ERR_INTERRUPTED) {
    return error_code;
  }

  if (error_code != Executor::ERR_OVERFLOW_OR_UNDERFLOW && error_code != Executor::ERR_DIV_BY_ZERO &&
      !query_exe_context->query_mem_desc_.usesCachedContext() && !render_allocator_map) {
    CHECK(!query_exe_context->query_mem_desc_.sortOnGpu());
    results = query_exe_context->getResult(
        ra_exe_unit, outer_tab_frag_ids, query_exe_context->query_mem_desc_, was_auto_device);
    if (auto rows = boost::get<RowSetPtr>(&results)) {
      (*rows)->holdLiterals(hoist_buf);
    }
  }
  if (error_code && (render_allocator_map || (!scan_limit || check_rows_less_than_needed(results, scan_limit)))) {
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

namespace {

template <class T>
int8_t* insert_one_dict_str(const ColumnDescriptor* cd,
                            const Analyzer::Constant* col_cv,
                            const Catalog_Namespace::Catalog& catalog) {
  auto col_data = reinterpret_cast<T*>(checked_malloc(sizeof(T)));
  if (col_cv->get_is_null()) {
    *col_data = inline_fixed_encoding_null_val(cd->columnType);
  } else {
    const int dict_id = cd->columnType.get_comp_param();
    const auto col_datum = col_cv->get_constval();
    const auto& str = *col_datum.stringval;
    const auto dd = catalog.getMetadataForDict(dict_id);
    CHECK(dd && dd->stringDict);
    int32_t str_id = dd->stringDict->getOrAdd(str);
    const bool invalid = str_id > max_valid_int_value<T>();
    if (invalid || str_id == inline_int_null_value<int32_t>()) {
      if (invalid) {
        LOG(ERROR) << "Could not encode string: " << str << ", the encoded value doesn't fit in " << sizeof(T) * 8
                   << " bits. Will store NULL instead.";
      }
      str_id = inline_fixed_encoding_null_val(cd->columnType);
    }
    *col_data = str_id;
  }
  return reinterpret_cast<int8_t*>(col_data);
}

}  // namespace

void Executor::executeSimpleInsert(const Planner::RootPlan* root_plan) {
  const auto plan = root_plan->get_plan();
  CHECK(plan);
  const auto values_plan = dynamic_cast<const Planner::ValuesScan*>(plan);
  if (!values_plan) {
    throw std::runtime_error("Only simple INSERT of immediate tuples is currently supported");
  }
  row_set_mem_owner_ = std::make_shared<RowSetMemoryOwner>();
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
        *col_data =
            col_cv->get_is_null() ? inline_fixed_encoding_null_val(cd->columnType) : (col_datum.boolval ? 1 : 0);
        col_buffers[col_ids[col_idx]] = col_data;
        break;
      }
      case kSMALLINT: {
        auto col_data = reinterpret_cast<int16_t*>(checked_malloc(sizeof(int16_t)));
        *col_data = col_cv->get_is_null() ? inline_fixed_encoding_null_val(cd->columnType) : col_datum.smallintval;
        col_buffers[col_ids[col_idx]] = reinterpret_cast<int8_t*>(col_data);
        break;
      }
      case kINT: {
        auto col_data = reinterpret_cast<int32_t*>(checked_malloc(sizeof(int32_t)));
        *col_data = col_cv->get_is_null() ? inline_fixed_encoding_null_val(cd->columnType) : col_datum.intval;
        col_buffers[col_ids[col_idx]] = reinterpret_cast<int8_t*>(col_data);
        break;
      }
      case kBIGINT: {
        auto col_data = reinterpret_cast<int64_t*>(checked_malloc(sizeof(int64_t)));
        *col_data = col_cv->get_is_null() ? inline_fixed_encoding_null_val(cd->columnType) : col_datum.bigintval;
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
            switch (cd->columnType.get_size()) {
              case 1:
                col_buffers[col_ids[col_idx]] = insert_one_dict_str<int8_t>(cd, col_cv, cat);
                break;
              case 2:
                col_buffers[col_ids[col_idx]] = insert_one_dict_str<int16_t>(cd, col_cv, cat);
                break;
              case 4:
                col_buffers[col_ids[col_idx]] = insert_one_dict_str<int32_t>(cd, col_cv, cat);
                break;
              default:
                CHECK(false);
            }
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
        *col_data = col_cv->get_is_null() ? inline_fixed_encoding_null_val(cd->columnType) : col_datum.timeval;
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
#if LLVM_VERSION_MAJOR < 4
  CHECK(!owner.getError());
#else
  CHECK(!owner.takeError());
#endif
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
  fetch_ir_builder.SetInsertPoint(&*fetch_bb.begin());
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
    arg_it->setName("col_buf" + std::to_string(i));
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
  row_process_arg_types.push_back(llvm::Type::getInt64PtrTy(context));

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
  if (std::dynamic_pointer_cast<Analyzer::RegexpExpr>(expr)) {
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

void Executor::nukeOldState(const bool allow_lazy_fetch,
                            const JoinInfo& join_info,
                            const std::vector<InputTableInfo>& query_infos,
                            const std::list<std::shared_ptr<Analyzer::Expr>>& outer_join_quals) {
  cgen_state_.reset(new CgenState(query_infos, !outer_join_quals.empty()));
  plan_state_.reset(new PlanState(allow_lazy_fetch && outer_join_quals.empty(), join_info, this));
}

bool Executor::prioritizeQuals(const RelAlgExecutionUnit& ra_exe_unit,
                               std::vector<Analyzer::Expr*>& primary_quals,
                               std::vector<Analyzer::Expr*>& deferred_quals) {
  std::vector<Analyzer::Expr*> unlikely_quals;
  std::vector<Analyzer::Expr*> short_circuited_quals;

  for (auto expr : ra_exe_unit.inner_join_quals) {
    primary_quals.push_back(expr.get());
    short_circuited_quals.push_back(expr.get());
  }

  for (auto expr : ra_exe_unit.simple_quals) {
    if (get_likelihood(expr.get()) < 0.10 && !contains_unsafe_division(expr.get())) {
      unlikely_quals.push_back(expr.get());
      continue;
    }
    if (should_defer_eval(expr)) {
      deferred_quals.push_back(expr.get());
      short_circuited_quals.push_back(expr.get());
      continue;
    }
    primary_quals.push_back(expr.get());
    short_circuited_quals.push_back(expr.get());
  }
  for (auto expr : ra_exe_unit.quals) {
    if (get_likelihood(expr.get()) < 0.10 && !contains_unsafe_division(expr.get())) {
      unlikely_quals.push_back(expr.get());
      continue;
    }
    if (should_defer_eval(expr)) {
      deferred_quals.push_back(expr.get());
      short_circuited_quals.push_back(expr.get());
      continue;
    }
    primary_quals.push_back(expr.get());
    short_circuited_quals.push_back(expr.get());
  }

  if (!unlikely_quals.empty() && !short_circuited_quals.empty()) {
    primary_quals.swap(unlikely_quals);
    deferred_quals.swap(short_circuited_quals);
    return true;
  }

  return false;
}

Executor::CompilationResult Executor::compileWorkUnit(const bool render_output,
                                                      const std::vector<InputTableInfo>& query_infos,
                                                      const RelAlgExecutionUnit& ra_exe_unit,
                                                      const CompilationOptions& co,
                                                      const ExecutionOptions& eo,
                                                      const CudaMgr_Namespace::CudaMgr* cuda_mgr,
                                                      const bool allow_lazy_fetch,
                                                      std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                                                      const size_t max_groups_buffer_entry_guess,
                                                      const size_t small_groups_buffer_entry_count,
                                                      const int8_t crt_min_byte_width,
                                                      const JoinInfo& join_info,
                                                      const bool has_cardinality_estimation) {
  nukeOldState(allow_lazy_fetch, join_info, query_infos, ra_exe_unit.outer_join_quals);

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

  if (query_mem_desc.hash_type == GroupByColRangeType::MultiCol && !query_mem_desc.getSmallBufferSizeBytes() &&
      !has_cardinality_estimation && !render_output && !eo.just_explain) {
    throw CardinalityEstimationRequired();
  }

  const bool output_columnar = group_by_and_aggregate.outputColumnar();

  if (co.device_type_ == ExecutorDeviceType::GPU) {
    for (const auto& count_distinct_descriptor : query_mem_desc.count_distinct_descriptors_) {
      if (count_distinct_descriptor.impl_type_ == CountDistinctImplType::StdSet ||
          (count_distinct_descriptor.impl_type_ != CountDistinctImplType::Invalid && !co.hoist_literals_)) {
        cgen_state_->must_run_on_cpu_ = true;
      }
    }
  }

  if (co.device_type_ == ExecutorDeviceType::GPU &&
      query_mem_desc.hash_type == GroupByColRangeType::MultiColPerfectHash && !cgen_state_->must_run_on_cpu_) {
    const auto grid_size = query_mem_desc.blocksShareMemory() ? 1 : gridSize();
    const size_t required_memory{(grid_size * query_mem_desc.getBufferSizeBytes(ExecutorDeviceType::GPU))};
    CHECK(catalog_->get_dataMgr().cudaMgr_);
    const size_t max_memory{catalog_->get_dataMgr().cudaMgr_->deviceProperties[0].globalMem / 5};
    cgen_state_->must_run_on_cpu_ = required_memory > max_memory;
  }

  // Read the module template and target either CPU or GPU
  // by binding the stream position functions to the right implementation:
  // stride access for GPU, contiguous for CPU
  cgen_state_->module_ = read_template_module(cgen_state_->context_);

  auto agg_fnames = get_agg_fnames(ra_exe_unit.target_exprs, !ra_exe_unit.groupby_exprs.empty());
  const auto agg_slot_count = ra_exe_unit.estimator ? size_t(1) : agg_fnames.size();

  const bool is_group_by{!query_mem_desc.group_col_widths.empty()};
  auto query_func =
      is_group_by ? query_group_by_template(cgen_state_->module_,
                                            is_nested_,
                                            co.hoist_literals_,
                                            query_mem_desc,
                                            co.device_type_,
                                            ra_exe_unit.scan_limit)
                  : query_template(
                        cgen_state_->module_, agg_slot_count, is_nested_, co.hoist_literals_, !!ra_exe_unit.estimator);
  bind_pos_placeholders("pos_start", true, query_func, cgen_state_->module_);
  bind_pos_placeholders("group_buff_idx", false, query_func, cgen_state_->module_);
  bind_pos_placeholders("pos_step", false, query_func, cgen_state_->module_);

  std::vector<llvm::Value*> col_heads;
  std::tie(cgen_state_->row_func_, col_heads) = create_row_function(ra_exe_unit.input_col_descs.size(),
                                                                    is_group_by ? 0 : agg_slot_count,
                                                                    co.hoist_literals_,
                                                                    query_func,
                                                                    cgen_state_->module_,
                                                                    cgen_state_->context_);
  CHECK(cgen_state_->row_func_);

  // make sure it's in-lined, we don't want register spills in the inner loop
  cgen_state_->row_func_->addAttribute(llvm::AttributeSet::FunctionIndex, llvm::Attribute::AlwaysInline);

  auto bb = llvm::BasicBlock::Create(cgen_state_->context_, "entry", cgen_state_->row_func_);
  cgen_state_->ir_builder_.SetInsertPoint(bb);
  preloadFragOffsets(ra_exe_unit.input_descs, query_infos);

  allocateInnerScansIterators(ra_exe_unit.input_descs);

  // generate the code for the filter
  allocateLocalColumnIds(ra_exe_unit.input_col_descs);

  if (isOuterJoin()) {
    cgen_state_->outer_join_cond_lv_ =
        llvm::ConstantInt::get(llvm::IntegerType::getInt1Ty(cgen_state_->context_), true);
    for (auto expr : ra_exe_unit.outer_join_quals) {
      cgen_state_->outer_join_cond_lv_ = cgen_state_->ir_builder_.CreateAnd(
          cgen_state_->outer_join_cond_lv_, toBool(codegen(expr.get(), true, co).front()));
    }
  }

  std::vector<Analyzer::Expr*> primary_quals;
  std::vector<Analyzer::Expr*> deferred_quals;
  bool short_circuited = prioritizeQuals(ra_exe_unit, primary_quals, deferred_quals);
  if (short_circuited) {
    VLOG(1) << "Prioritized " << std::to_string(primary_quals.size()) << " unlikely quals, "
            << "short-circuited " << std::to_string(deferred_quals.size()) << " quals";
  }

  llvm::Value* filter_lv = llvm::ConstantInt::get(llvm::IntegerType::getInt1Ty(cgen_state_->context_), true);
  for (auto expr : primary_quals) {
    // Generate the filter for primary quals
    filter_lv = cgen_state_->ir_builder_.CreateAnd(filter_lv, toBool(codegen(expr, true, co).front()));
  }
  CHECK(filter_lv->getType()->isIntegerTy(1));

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

  if (needs_error_check || cgen_state_->needs_error_check_ || eo.with_dynamic_watchdog) {
    createErrorCheckControlFlow(query_func, eo.with_dynamic_watchdog);
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
  plan_state_->init_agg_vals_ = init_agg_val_vec(ra_exe_unit.target_exprs, ra_exe_unit.quals, query_mem_desc);

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
          : optimizeAndCodegenGPU(query_func,
                                  multifrag_query_func,
                                  live_funcs,
                                  cgen_state_->module_,
                                  is_group_by || ra_exe_unit.estimator,
                                  cuda_mgr,
                                  co),
      cgen_state_->getLiterals(),
      query_mem_desc,
      output_columnar,
      llvm_ir};
}

void Executor::createErrorCheckControlFlow(llvm::Function* query_func, bool run_with_dynamic_watchdog) {
  // check whether the row processing was successful; currently, it can
  // fail by running out of group by buffer slots
  bool done_splitting = false;
  for (auto bb_it = query_func->begin(); bb_it != query_func->end() && !done_splitting; ++bb_it) {
    llvm::Value* pos = nullptr;
    for (auto inst_it = bb_it->begin(); inst_it != bb_it->end(); ++inst_it) {
      if (run_with_dynamic_watchdog && llvm::isa<llvm::PHINode>(*inst_it)) {
        if (inst_it->getName() == "pos") {
          pos = &*inst_it;
        }
        continue;
      }
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
        llvm::Value* err_lv = &*inst_it;
        if (run_with_dynamic_watchdog) {
          CHECK(pos);
          // run watchdog after every 64 rows
          auto and_lv = ir_builder.CreateAnd(pos, uint64_t(0x3f));
          auto call_watchdog_lv = ir_builder.CreateICmp(llvm::ICmpInst::ICMP_EQ, and_lv, ll_int(int64_t(0LL)));

          auto error_check_bb = bb_it->splitBasicBlock(llvm::BasicBlock::iterator(br_instr), ".error_check");
          auto& watchdog_br_instr = bb_it->back();

          auto watchdog_check_bb =
              llvm::BasicBlock::Create(cgen_state_->context_, ".watchdog_check", query_func, error_check_bb);
          llvm::IRBuilder<> watchdog_ir_builder(watchdog_check_bb);
          auto detected_timeout =
              watchdog_ir_builder.CreateCall(cgen_state_->module_->getFunction("dynamic_watchdog"), {});
          auto timeout_err_lv =
              watchdog_ir_builder.CreateSelect(detected_timeout, ll_int(Executor::ERR_OUT_OF_TIME), err_lv);
          watchdog_ir_builder.CreateBr(error_check_bb);

          llvm::ReplaceInstWithInst(&watchdog_br_instr,
                                    llvm::BranchInst::Create(watchdog_check_bb, error_check_bb, call_watchdog_lv));
          ir_builder.SetInsertPoint(&br_instr);
          auto unified_err_lv = ir_builder.CreatePHI(err_lv->getType(), 2);

          unified_err_lv->addIncoming(timeout_err_lv, watchdog_check_bb);
          unified_err_lv->addIncoming(err_lv, &*bb_it);
          err_lv = unified_err_lv;
        }
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

void Executor::preloadFragOffsets(const std::vector<InputDescriptor>& input_descs,
                                  const std::vector<InputTableInfo>& query_infos) {
#ifdef ENABLE_MULTIFRAG_JOIN
  const auto ld_count = input_descs.size();
#else
  const size_t ld_count = 1;
#endif
  auto frag_off_ptr = get_arg_by_name(cgen_state_->row_func_, "frag_row_off");
  for (size_t i = 0; i < ld_count; ++i) {
#ifdef HAVE_CALCITE
    CHECK_LT(i, query_infos.size());
    const auto frag_count = query_infos[i].info.fragments.size();
#else
    const size_t frag_count = 1;
#endif  // HAVE_CALCITE
    if (frag_count > 1) {
      auto input_off_ptr = !i ? frag_off_ptr : cgen_state_->ir_builder_.CreateGEP(frag_off_ptr, ll_int(int32_t(i)));
      cgen_state_->frag_offsets_.push_back(cgen_state_->ir_builder_.CreateLoad(input_off_ptr));
    } else {
      cgen_state_->frag_offsets_.push_back(nullptr);
    }
  }
}

void Executor::allocateInnerScansIterators(const std::vector<InputDescriptor>& input_descs) {
  if (input_descs.size() <= 1) {
    return;
  }
  if (plan_state_->join_info_.join_impl_type_ == JoinImplType::HashOneToOne) {
    return;
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
      auto rows_per_scan_ptr = cgen_state_->ir_builder_.CreateGEP(
          get_arg_by_name(cgen_state_->row_func_, "num_rows_per_scan"), ll_int(int32_t(inner_scan_idx)));
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
                                            const std::vector<InputTableInfo>& query_infos,
                                            const std::list<std::shared_ptr<const InputColDescriptor>>& input_col_descs,
                                            const ExecutorDeviceType device_type) {
  CHECK(device_type != ExecutorDeviceType::Hybrid);
  std::string hash_join_fail_reason{"No equijoin expression found"};

  const MemoryLevel memory_level{device_type == ExecutorDeviceType::GPU ? MemoryLevel::GPU_LEVEL
                                                                        : MemoryLevel::CPU_LEVEL};
  for (auto qual : join_quals) {
    auto qual_bin_oper = std::dynamic_pointer_cast<Analyzer::BinOper>(qual);
    if (!qual_bin_oper) {
      const auto bool_const = std::dynamic_pointer_cast<Analyzer::Constant>(qual);
      if (bool_const) {
        CHECK(bool_const->get_type_info().is_boolean());
      }
      continue;
    }
    if (qual_bin_oper->get_optype() == kEQ) {
      const int device_count =
          device_type == ExecutorDeviceType::GPU ? catalog_->get_dataMgr().cudaMgr_->getDeviceCount() : 1;
      CHECK_GT(device_count, 0);
      try {
        const auto join_hash_table = JoinHashTable::getInstance(
            qual_bin_oper, *catalog_, query_infos, input_col_descs, memory_level, device_count, this);
        CHECK(join_hash_table);
        return Executor::JoinInfo(JoinImplType::HashOneToOne,
                                  std::vector<std::shared_ptr<Analyzer::BinOper>>{qual_bin_oper},
                                  join_hash_table,
                                  "");
      } catch (const HashJoinFail& e) {
        hash_join_fail_reason = e.what();
      }
    }
  }

  return Executor::JoinInfo(
      JoinImplType::Loop, std::vector<std::shared_ptr<Analyzer::BinOper>>{}, nullptr, hash_join_fail_reason);
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
#if LLVM_VERSION_MAJOR < 4
  pass_manager.add(llvm::createAlwaysInlinerPass());
#else
  pass_manager.add(llvm::createAlwaysInlinerLegacyPass());
#endif
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

std::vector<std::pair<void*, void*>> Executor::getCodeFromCache(
    const CodeCacheKey& key,
    const std::map<CodeCacheKey, std::pair<CodeCacheVal, llvm::Module*>>& cache) {
  auto it = cache.find(key);
  if (it != cache.end()) {
    delete cgen_state_->module_;
    cgen_state_->module_ = it->second.second;
    std::vector<std::pair<void*, void*>> native_functions;
    for (auto& native_code : it->second.first) {
      GpuCompilationContext* gpu_context = std::get<2>(native_code).get();
      native_functions.push_back(
          std::make_pair(std::get<0>(native_code), gpu_context ? gpu_context->module() : nullptr));
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

std::vector<std::pair<void*, void*>> Executor::optimizeAndCodegenCPU(llvm::Function* query_func,
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

  return {std::make_pair(native_code, nullptr)};
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

std::string gen_translate_null_key_sigs() {
  std::string result;
  for (const std::string key_type : {"int8_t", "int16_t", "int32_t", "int64_t"}) {
    const auto key_llvm_type = cpp_to_llvm_name(key_type);
    result += "declare i64 @translate_null_key_" + key_type + "(" + key_llvm_type + ", " + key_llvm_type + ", " +
              key_llvm_type + ");\n";
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
declare i64* @get_group_value(i64*, i32, i64*, i32, i32, i32, i64*);
declare i64* @get_group_value_with_watchdog(i64*, i32, i64*, i32, i32, i32, i64*);
declare i64* @get_group_value_fast(i64*, i64, i64, i64, i32);
declare i64* @get_group_value_fast_with_original_key(i64*, i64, i64, i64, i64, i32);
declare i32 @get_columnar_group_bin_offset(i64*, i64, i64, i64);
declare i64* @get_group_value_one_key(i64*, i32, i64*, i32, i64, i64, i32, i64*);
declare i64* @get_group_value_one_key_with_watchdog(i64*, i32, i64*, i32, i64, i64, i32, i64*);
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
declare void @agg_id_double_shared_slow(i64*, double*);
declare void @agg_id_float_shared(i32*, float);
declare i64 @ExtractFromTime(i32, i64);
declare i64 @ExtractFromTimeNullable(i32, i64, i64);
declare i64 @DateTruncate(i32, i64);
declare i64 @DateTruncateNullable(i32, i64, i64);
declare i64 @DateDiff(i32, i64, i64);
declare i64 @DateDiffNullable(i32, i64, i64, i64);
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
declare i1 @regexp_like(i8*, i32, i8*, i32, i8);
declare i8 @regexp_like_nullable(i8*, i32, i8*, i32, i8, i8);
declare void @linear_probabilistic_count(i8*, i32, i8*, i32);
declare void @agg_count_distinct_bitmap_gpu(i64*, i64, i64, i64, i64);
declare void @agg_count_distinct_bitmap_skip_val_gpu(i64*, i64, i64, i64, i64, i64);
declare void @agg_approximate_count_distinct_gpu(i64*, i64, i32, i64, i64);
declare i32 @record_error_code(i32, i32*);
declare i1 @dynamic_watchdog();
declare void @force_sync();
)" + gen_array_any_all_sigs() +
    gen_translate_null_key_sigs();

#ifdef HAVE_CUDA
std::string extension_function_decls() {
  const auto decls = ExtensionFunctionsWhitelist::getLLVMDeclarations();
  return boost::algorithm::join(decls, "\n");
}
#endif  // HAVE_CUDA

}  // namespace

std::vector<std::pair<void*, void*>> Executor::optimizeAndCodegenGPU(llvm::Function* query_func,
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
            get_gv_call.getCalledFunction()->getName() == "get_group_value_with_watchdog" ||
            get_gv_call.getCalledFunction()->getName() == "get_matching_group_value_perfect_hash" ||
            get_gv_call.getCalledFunction()->getName() == "string_decode" ||
            get_gv_call.getCalledFunction()->getName() == "array_size" ||
            get_gv_call.getCalledFunction()->getName() == "linear_probabilistic_count") {
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

  auto cuda_llir = cuda_rt_decls + extension_function_decls() + ss.str();

  std::vector<std::pair<void*, void*>> native_functions;
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
    auto native_module = gpu_context->module();
    CHECK(native_code);
    CHECK(native_module);
    native_functions.push_back(std::make_pair(native_code, native_module));
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
#if LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR == 5
    formatted_os.flush();
#endif
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
  nvptx_target_machine_.reset(
      target->createTargetMachine("nvptx64-nvidia-cuda", "sm_30", "", llvm::TargetOptions(), llvm::Reloc::Static));
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

int64_t Executor::deviceCycles(int milliseconds) const {
  CHECK(catalog_);
  CHECK(catalog_->get_dataMgr().cudaMgr_);
  const auto& dev_props = catalog_->get_dataMgr().cudaMgr_->deviceProperties;
  return static_cast<int64_t>(dev_props.front().clockKhz) * milliseconds;
}

void Executor::registerActiveModule(void* module, const int device_id) const {
#ifdef HAVE_CUDA
  std::lock_guard<std::mutex> lock(gpu_active_modules_mutex_);
  CHECK_LT(device_id, max_gpu_count);
  gpu_active_modules_device_mask_ |= (1 << device_id);
  gpu_active_modules_[device_id] = module;
  VLOG(1) << "Executor " << this << ", mask 0x" << std::hex << gpu_active_modules_device_mask_ << ": Registered module "
          << module << " on device " << std::to_string(device_id);
#endif
}

void Executor::unregisterActiveModule(void* module, const int device_id) const {
#ifdef HAVE_CUDA
  std::lock_guard<std::mutex> lock(gpu_active_modules_mutex_);
  CHECK_LT(device_id, max_gpu_count);
  if ((gpu_active_modules_device_mask_ & (1 << device_id)) == 0)
    return;
  CHECK_EQ(gpu_active_modules_[device_id], module);
  gpu_active_modules_device_mask_ ^= (1 << device_id);
  VLOG(1) << "Executor " << this << ", mask 0x" << std::hex << gpu_active_modules_device_mask_
          << ": Unregistered module " << module << " on device " << std::to_string(device_id);
#endif
}

void Executor::interrupt() {
#ifdef HAVE_CUDA
  std::lock_guard<std::mutex> lock(gpu_active_modules_mutex_);
  VLOG(1) << "Executor " << this << ": Interrupting Active Modules: mask 0x" << std::hex
          << gpu_active_modules_device_mask_;
  CUcontext old_cu_context;
  checkCudaErrors(cuCtxGetCurrent(&old_cu_context));
  for (int device_id = 0; device_id < max_gpu_count; device_id++) {
    if (gpu_active_modules_device_mask_ & (1 << device_id)) {
      void* module = gpu_active_modules_[device_id];
      auto cu_module = static_cast<CUmodule>(module);
      if (!cu_module)
        continue;
      VLOG(1) << "Terminating module " << module << " on device " << std::to_string(device_id)
              << ", gpu_active_modules_device_mask_: " << std::hex << std::to_string(gpu_active_modules_device_mask_);

      catalog_->get_dataMgr().cudaMgr_->setContext(device_id);

      // Create high priority non-blocking communication stream
      CUstream cu_stream1;
      checkCudaErrors(cuStreamCreateWithPriority(&cu_stream1, CU_STREAM_NON_BLOCKING, 1));

      CUevent start, stop;
      cuEventCreate(&start, 0);
      cuEventCreate(&stop, 0);
      cuEventRecord(start, cu_stream1);

      CUdeviceptr dw_abort;
      size_t dw_abort_size;
      if (cuModuleGetGlobal(&dw_abort, &dw_abort_size, cu_module, "dw_abort") == CUDA_SUCCESS) {
        CHECK_EQ(dw_abort_size, sizeof(uint32_t));
        int32_t abort_val = 1;
        checkCudaErrors(cuMemcpyHtoDAsync(dw_abort, reinterpret_cast<void*>(&abort_val), sizeof(int32_t), cu_stream1));

        if (device_id == 0) {
          LOG(INFO) << "GPU: Async Abort submitted to Device " << std::to_string(device_id);
        }
      }

      cuEventRecord(stop, cu_stream1);
      cuEventSynchronize(stop);
      float milliseconds = 0;
      cuEventElapsedTime(&milliseconds, start, stop);
      VLOG(1) << "Device " << std::to_string(device_id)
              << ": submitted async request to abort: " << std::to_string(milliseconds) << " ms\n";
      checkCudaErrors(cuStreamDestroy(cu_stream1));
    }
  }
  checkCudaErrors(cuCtxSetCurrent(old_cu_context));
#endif

  dynamic_watchdog_init(static_cast<unsigned>(DW_ABORT));

  interrupted_ = true;
  VLOG(1) << "INTERRUPT Executor " << this;
}

void Executor::resetInterrupt() {
#ifdef HAVE_CUDA
  std::lock_guard<std::mutex> lock(gpu_active_modules_mutex_);
#endif

  if (!interrupted_)
    return;

  dynamic_watchdog_init(static_cast<unsigned>(DW_RESET));

  interrupted_ = false;
  VLOG(1) << "RESET Executor " << this << " that had previously been interrupted";
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

llvm::Value* Executor::castToTypeIn(llvm::Value* val, const size_t dst_bits) {
  auto src_bits = val->getType()->getScalarSizeInBits();
  if (src_bits == dst_bits) {
    return val;
  }
  if (val->getType()->isIntegerTy()) {
    return cgen_state_->ir_builder_.CreateIntCast(val, get_int_type(dst_bits, cgen_state_->context_), src_bits != 1);
  }
  // real (not dictionary-encoded) strings; store the pointer to the payload
  if (val->getType()->isPointerTy()) {
    const auto val_ptr_type = static_cast<llvm::PointerType*>(val->getType());
    CHECK(val_ptr_type->getElementType()->isIntegerTy(8));
    return cgen_state_->ir_builder_.CreatePointerCast(val, get_int_type(dst_bits, cgen_state_->context_));
  }

  CHECK(val->getType()->isFloatTy() || val->getType()->isDoubleTy());

  llvm::Type* dst_type = nullptr;
  switch (dst_bits) {
    case 64:
      dst_type = llvm::Type::getDoubleTy(cgen_state_->context_);
      break;
    case 32:
      dst_type = llvm::Type::getFloatTy(cgen_state_->context_);
      break;
    default:
      CHECK(false);
  }

  return cgen_state_->ir_builder_.CreateFPCast(val, dst_type);
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

Executor::GroupColLLVMValue Executor::groupByColumnCodegen(Analyzer::Expr* group_by_col,
                                                           const size_t col_width,
                                                           const CompilationOptions& co,
                                                           const bool translate_null_val,
                                                           const int64_t translated_null_val,
                                                           GroupByAndAggregate::DiamondCodegen& diamond_codegen,
                                                           std::stack<llvm::BasicBlock*>& array_loops,
                                                           const bool thread_mem_shared) {
#ifdef ENABLE_KEY_COMPACTION
  CHECK_GE(col_width, sizeof(int32_t));
#else
  CHECK_EQ(col_width, sizeof(int64_t));
#endif
  auto group_key = codegen(group_by_col, true, co).front();
  auto key_to_cache = group_key;
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
        "array_size", ret_ty, {group_key, posArg(arr_expr), ll_int(log2_bytes(elem_ti.get_logical_size()))});
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
                               : get_int_type(elem_ti.get_logical_size() * 8, cgen_state_->context_);
    group_key = cgen_state_->emitExternalCall(array_at_fname, ar_ret_ty, {group_key, posArg(arr_expr), array_idx});
    if (need_patch_unnest_double(elem_ti, isArchMaxwell(co.device_type_), thread_mem_shared)) {
      key_to_cache = spillDoubleElement(group_key, ar_ret_ty);
    } else {
      key_to_cache = group_key;
    }
    CHECK(array_loop_head);
    array_loops.push(array_loop_head);
  }
  cgen_state_->group_by_expr_cache_.push_back(key_to_cache);
  llvm::Value* orig_group_key{nullptr};
  if (translate_null_val) {
    const std::string translator_func_name(col_width == sizeof(int32_t) ? "translate_null_key_i32_"
                                                                        : "translate_null_key_");
    const auto& ti = group_by_col->get_type_info();
    const auto key_type = get_int_type(ti.get_logical_size() * 8, cgen_state_->context_);
    orig_group_key = group_key;
    group_key =
        cgen_state_->emitCall(translator_func_name + numeric_type_name(ti),
                              {group_key,
                               static_cast<llvm::Value*>(llvm::ConstantInt::get(key_type, inline_int_null_val(ti))),
                               static_cast<llvm::Value*>(llvm::ConstantInt::get(key_type, translated_null_val))});
  }
  group_key = cgen_state_->ir_builder_.CreateBitCast(castToTypeIn(group_key, col_width * 8),
                                                     get_int_type(col_width * 8, cgen_state_->context_));
  if (orig_group_key) {
    orig_group_key = cgen_state_->ir_builder_.CreateBitCast(castToTypeIn(orig_group_key, col_width * 8),
                                                            get_int_type(col_width * 8, cgen_state_->context_));
  }
  return {group_key, orig_group_key};
}

void Executor::allocateLocalColumnIds(const std::list<std::shared_ptr<const InputColDescriptor>>& global_col_ids) {
  for (const auto& col_id : global_col_ids) {
    CHECK(col_id);
    const auto local_col_id = plan_state_->global_to_local_col_ids_.size();
    const auto it_ok = plan_state_->global_to_local_col_ids_.insert(std::make_pair(*col_id, local_col_id));
    plan_state_->local_to_global_col_ids_.push_back(col_id->getColId());
    plan_state_->global_to_local_col_ids_.find(*col_id);
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
  CHECK(it != plan_state_->global_to_local_col_ids_.end());
  if (fetch_column) {
    plan_state_->columns_to_fetch_.insert(std::make_pair(table_id, global_col_id));
  }
  return it->second;
}

std::pair<bool, int64_t> Executor::skipFragment(const InputDescriptor& table_desc,
                                                const Fragmenter_Namespace::FragmentInfo& fragment,
                                                const std::list<std::shared_ptr<Analyzer::Expr>>& simple_quals,
                                                const ExecutionDispatch& execution_dispatch,
                                                const size_t frag_idx) {
  const int table_id = table_desc.getTableId();
  if (table_desc.getSourceType() == InputSourceType::RESULT &&
      boost::get<IterTabPtr>(&get_temporary_table(temporary_tables_, table_id))) {
    return {false, -1};
  }
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
    auto chunk_meta_it = fragment.getChunkMetadataMap().find(col_id);
    int64_t chunk_min{0};
    int64_t chunk_max{0};
    bool is_rowid{false};
    size_t start_rowid{0};
    if (chunk_meta_it == fragment.getChunkMetadataMap().end()) {
      auto cd = get_column_descriptor(col_id, table_id, *catalog_);
      CHECK(cd->isVirtualCol && cd->columnName == "rowid");
      const auto& table_generation = getTableGeneration(table_id);
      start_rowid = table_generation.start_rowid;
      const auto& all_frag_row_offsets = execution_dispatch.getFragOffsets();
      chunk_min = all_frag_row_offsets[frag_idx] + start_rowid;
      chunk_max = all_frag_row_offsets[frag_idx + 1] - 1 + start_rowid;
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
          return {false, rhs_val - start_rowid};
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
