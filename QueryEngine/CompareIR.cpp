#include "Execute.h"

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
