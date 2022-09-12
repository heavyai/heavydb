/**
 * Copyright 2021 OmniSci, Inc.
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Context.h"
#include "Type.h"

#include "SchemaMgr/ColumnInfo.h"
#include "Shared/sqltypes.h"

#include <iostream>
#include <list>
#include <memory>
#include <set>
#include <vector>

class RelAlgNode;

namespace hdk::ir {

class Type;
class Expr;

using ExprPtr = std::shared_ptr<Expr>;
using ExprPtrList = std::list<ExprPtr>;
using ExprPtrVector = std::vector<ExprPtr>;

template <typename Tp, typename... Args>
inline
    typename std::enable_if<std::is_base_of<Expr, Tp>::value, std::shared_ptr<Tp>>::type
    makeExpr(Args&&... args) {
  return std::make_shared<Tp>(std::forward<Args>(args)...);
}

template <class T>
bool expr_is(const ExprPtr& expr) {
  return std::dynamic_pointer_cast<T>(expr) != nullptr;
}

template <typename T>
bool isOneOf(const ExprPtr& expr) {
  return std::dynamic_pointer_cast<T>(expr) != nullptr;
}

template <typename T1, typename T2, typename... Ts>
bool isOneOf(const ExprPtr& expr) {
  return std::dynamic_pointer_cast<T1>(expr) != nullptr || isOneOf<T2, Ts...>(expr);
}

template <typename T>
bool isOneOf(const Expr* expr) {
  return dynamic_cast<const T*>(expr);
}

template <typename T1, typename T2, typename... Ts>
bool isOneOf(const Expr* expr) {
  return dynamic_cast<const T1*>(expr) || isOneOf<T2, Ts...>(expr);
}

class ColumnVar;
class TargetEntry;
using DomainSet = std::list<const Expr*>;

/*
 * @type Expr
 * @brief super class for all expressions in parse trees and in query plans
 */
class Expr : public std::enable_shared_from_this<Expr> {
 public:
  Expr(const Type* type, bool has_agg = false);
  Expr(const SQLTypeInfo& ti, bool has_agg = false);
  Expr(SQLTypes t, bool notnull);
  Expr(SQLTypes t, int d, bool notnull);
  Expr(SQLTypes t, int d, int s, bool notnull);
  virtual ~Expr() {}

  ExprPtr get_shared_ptr() { return shared_from_this(); }
  const SQLTypeInfo& get_type_info() const { return type_info; }
  virtual void set_type_info(const SQLTypeInfo& ti);
  virtual void set_type_info(const hdk::ir::Type* new_type);
  const Type* type() const { return type_; }
  bool get_contains_agg() const { return contains_agg; }
  void set_contains_agg(bool a) { contains_agg = a; }
  virtual ExprPtr add_cast(const Type* new_type, bool is_dict_intersection = false);
  ExprPtr add_cast(const SQLTypeInfo& new_type_info);
  virtual void check_group_by(const ExprPtrList& groupby) const {};
  virtual ExprPtr deep_copy() const = 0;  // make a deep copy of self

  /*
   * @brief normalize_simple_predicate only applies to boolean expressions.
   * it checks if it is an expression comparing a column
   * with a constant.  if so, it returns a normalized copy of the predicate with ColumnVar
   * always as the left operand with rte_idx set to the rte_idx of the ColumnVar.
   * it returns nullptr with rte_idx set to -1 otherwise.
   */
  virtual ExprPtr normalize_simple_predicate(int& rte_idx) const {
    rte_idx = -1;
    return nullptr;
  }
  /*
   * @brief seperate conjunctive predicates into scan predicates, join predicates and
   * constant predicates.
   */
  virtual void group_predicates(std::list<const Expr*>& scan_predicates,
                                std::list<const Expr*>& join_predicates,
                                std::list<const Expr*>& const_predicates) const {}
  /*
   * @brief collect_rte_idx collects the indices of all the range table
   * entries involved in an expression
   */
  virtual void collect_rte_idx(std::set<int>& rte_idx_set) const {}
  /*
   * @brief collect_column_var collects all unique ColumnVar nodes in an expression
   * If include_agg = false, it does not include to ColumnVar nodes inside
   * the argument to AggExpr's.  Otherwise, they are included.
   * It does not make copies of the ColumnVar
   */
  virtual void collect_column_var(
      std::set<const ColumnVar*, bool (*)(const ColumnVar*, const ColumnVar*)>&
          colvar_set,
      bool include_agg) const {}
  /*
   * @brief rewrite_with_targetlist rewrite ColumnVar's in expression with entries in a
   * targetlist. targetlist expressions are expected to be only Var's or AggExpr's.
   * returns a new expression copy
   */
  virtual ExprPtr rewrite_with_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const {
    return deep_copy();
  };
  /*
   * @brief rewrite_with_child_targetlist rewrite ColumnVar's in expression with entries
   * in a child plan's targetlist. targetlist expressions are expected to be only Var's or
   * ColumnVar's returns a new expression copy
   */
  virtual ExprPtr rewrite_with_child_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const {
    return deep_copy();
  };
  /*
   * @brief rewrite_agg_to_var rewrite ColumnVar's in expression with entries in an
   * AggPlan's targetlist. targetlist expressions are expected to be only Var's or
   * ColumnVar's or AggExpr's All AggExpr's are written into Var's. returns a new
   * expression copy
   */
  virtual ExprPtr rewrite_agg_to_var(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const {
    return deep_copy();
  }
  virtual bool operator==(const Expr& rhs) const = 0;
  virtual std::string toString() const = 0;
  virtual void print() const;

  virtual void add_unique(std::list<const Expr*>& expr_list) const;
  /*
   * @brief find_expr traverse Expr hierarchy and adds the node pointer to
   * the expr_list if the function f returns true.
   * Duplicate Expr's are not added the list.
   * Cannot use std::set because we don't have an ordering function.
   */
  virtual void find_expr(bool (*f)(const Expr*),
                         std::list<const Expr*>& expr_list) const {
    if (f(this)) {
      add_unique(expr_list);
    }
  }
  /*
   * @brief decompress adds cast operator to decompress encoded result
   */
  ExprPtr decompress();
  /*
   * @brief perform domain analysis on Expr and fill in domain
   * information in domain_set.  Empty domain_set means no information.
   */
  virtual void get_domain(DomainSet& domain_set) const { domain_set.clear(); }

  virtual size_t hash() const;

 protected:
  const Type* type_;
  SQLTypeInfo type_info;  // SQLTypeInfo of the return result of this expression
  bool contains_agg;
  mutable std::optional<size_t> hash_;
};

/*
 * Reference to an output node column.
 */
class ColumnRef : public Expr {
 public:
  ColumnRef(const SQLTypeInfo& ti, const RelAlgNode* node, unsigned idx)
      : Expr(ti), node_(node), idx_(idx) {}

  ExprPtr deep_copy() const override {
    return makeExpr<ColumnRef>(type_info, node_, idx_);
  }

  bool operator==(const Expr& rhs) const override {
    const ColumnRef* rhsp = dynamic_cast<const ColumnRef*>(&rhs);
    return rhsp && node_ == rhsp->node_ && idx_ == rhsp->idx_;
  }

  std::string toString() const override;

  const RelAlgNode* getNode() const { return node_; }

  unsigned getIndex() const { return idx_; }

  size_t hash() const override;

 protected:
  const RelAlgNode* node_;
  unsigned idx_;
};

/*
 * Used in Compound nodes to referene group by keys columns in target
 * expressions. Numbering starts with 1 to be consistent with RexRef.
 */
class GroupColumnRef : public Expr {
 public:
  GroupColumnRef(const SQLTypeInfo& ti, unsigned idx) : Expr(ti), idx_(idx) {}

  ExprPtr deep_copy() const override { return makeExpr<GroupColumnRef>(type_info, idx_); }

  bool operator==(const Expr& rhs) const override {
    const GroupColumnRef* rhsp = dynamic_cast<const GroupColumnRef*>(&rhs);
    return rhsp && idx_ == rhsp->idx_;
  }

  std::string toString() const override {
    return "(GroupColumnRef idx=" + std::to_string(idx_) + ")";
  }

  unsigned getIndex() const { return idx_; }

  size_t hash() const override;

 protected:
  unsigned idx_;
};

/*
 * @type ColumnVar
 * @brief expression that evaluates to the value of a column in a given row from a base
 * table. It is used in parse trees and is only used in Scan nodes in a query plan for
 * scanning a table while Var nodes are used for all other plans.
 */
class ColumnVar : public Expr {
 public:
  ColumnVar(ColumnInfoPtr col_info, int nest_level)
      : Expr(col_info->type), rte_idx(nest_level), col_info_(std::move(col_info)) {}
  ColumnVar(const SQLTypeInfo& ti)
      : Expr(ti)
      , rte_idx(-1)
      , col_info_(std::make_shared<ColumnInfo>(-1, 0, 0, "", type_, false)) {}
  ColumnVar(const SQLTypeInfo& ti,
            int table_id,
            int col_id,
            int nest_level,
            bool is_virtual = false)
      : Expr(ti)
      , rte_idx(nest_level)
      , col_info_(
            std::make_shared<ColumnInfo>(-1, table_id, col_id, "", type_, is_virtual)) {}
  int get_db_id() const { return col_info_->db_id; }
  int get_table_id() const { return col_info_->table_id; }
  int get_column_id() const { return col_info_->column_id; }
  int get_rte_idx() const { return rte_idx; }
  ColumnInfoPtr get_column_info() const { return col_info_; }
  bool is_virtual() const { return col_info_->is_rowid; }
  EncodingType get_compression() const { return type_info.get_compression(); }
  int get_comp_param() const { return type_info.get_comp_param(); }
  void set_type_info(const hdk::ir::Type* new_type) override {
    if (!type_->equal(new_type)) {
      col_info_ = std::make_shared<ColumnInfo>(col_info_->db_id,
                                               col_info_->table_id,
                                               col_info_->column_id,
                                               col_info_->name,
                                               new_type,
                                               col_info_->is_rowid);
      Expr::set_type_info(new_type);
    }
  }

  void check_group_by(const ExprPtrList& groupby) const override;
  ExprPtr deep_copy() const override;
  void group_predicates(std::list<const Expr*>& scan_predicates,
                        std::list<const Expr*>& join_predicates,
                        std::list<const Expr*>& const_predicates) const override;
  void collect_rte_idx(std::set<int>& rte_idx_set) const override {
    rte_idx_set.insert(rte_idx);
  }
  static bool colvar_comp(const ColumnVar* l, const ColumnVar* r) {
    return l->get_table_id() < r->get_table_id() ||
           (l->get_table_id() == r->get_table_id() &&
            l->get_column_id() < r->get_column_id());
  }
  void collect_column_var(
      std::set<const ColumnVar*, bool (*)(const ColumnVar*, const ColumnVar*)>&
          colvar_set,
      bool include_agg) const override {
    colvar_set.insert(this);
  }
  ExprPtr rewrite_with_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override;
  ExprPtr rewrite_with_child_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override;
  ExprPtr rewrite_agg_to_var(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override;
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;

  size_t hash() const override;

 protected:
  int rte_idx;  // 0-based range table index, used for table ordering in multi-joins
  ColumnInfoPtr col_info_;
};

/*
 * @type ExpressionTuple
 * @brief A tuple of expressions on the side of an equi-join on multiple columns.
 * Not to be used in any other context.
 */
class ExpressionTuple : public Expr {
 public:
  ExpressionTuple(const ExprPtrVector& tuple) : Expr(SQLTypeInfo()), tuple_(tuple){};

  const ExprPtrVector& getTuple() const { return tuple_; }

  void collect_rte_idx(std::set<int>& rte_idx_set) const override;

  ExprPtr deep_copy() const override;

  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;

  size_t hash() const override;

 private:
  const ExprPtrVector tuple_;
};

/*
 * @type Var
 * @brief expression that evaluates to the value of a column in a given row generated
 * from a query plan node.  It is only used in plan nodes above Scan nodes.
 * The row can be produced by either the inner or the outer plan in case of a join.
 * It inherits from ColumnVar to keep track of the lineage through the plan nodes.
 * The table_id will be set to 0 if the Var does not correspond to an original column
 * value.
 */
class Var : public ColumnVar {
 public:
  enum WhichRow { kINPUT_OUTER, kINPUT_INNER, kOUTPUT, kGROUPBY };
  Var(const SQLTypeInfo& ti, int r, int c, int i, bool is_virtual, WhichRow o, int v)
      : ColumnVar(ti, r, c, i, is_virtual), which_row(o), varno(v) {}
  Var(ColumnInfoPtr col_info, int i, WhichRow o, int v)
      : ColumnVar(col_info, i), which_row(o), varno(v) {}
  Var(const SQLTypeInfo& ti, WhichRow o, int v) : ColumnVar(ti), which_row(o), varno(v) {}
  WhichRow get_which_row() const { return which_row; }
  void set_which_row(WhichRow r) { which_row = r; }
  int get_varno() const { return varno; }
  void set_varno(int n) { varno = n; }
  ExprPtr deep_copy() const override;
  std::string toString() const override;
  void check_group_by(const ExprPtrList& groupby) const override;
  void collect_rte_idx(std::set<int>& rte_idx_set) const override {
    rte_idx_set.insert(-1);
  }
  ExprPtr rewrite_with_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override {
    return deep_copy();
  }
  ExprPtr rewrite_with_child_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override {
    return deep_copy();
  }
  ExprPtr rewrite_agg_to_var(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override;

  size_t hash() const override;

 private:
  WhichRow which_row;  // indicate which row this Var should project from.  It can be from
                       // the outer input plan or the inner input plan (for joins) or the
                       // output row in the current plan.
  int varno;           // the column number in the row.  1-based
};

/*
 * @type Constant
 * @brief expression for a constant value
 */
class Constant : public Expr {
 public:
  Constant(SQLTypes t, bool n, bool cacheable = true)
      : Expr(t, !n), is_null(n), cacheable_(cacheable) {
    if (n) {
      set_null_value();
    } else {
      type_info.set_notnull(true);
    }
  }
  Constant(SQLTypes t, bool n, Datum v, bool cacheable = true)
      : Expr(t, !n), is_null(n), cacheable_(cacheable), constval(v) {
    if (n) {
      set_null_value();
    } else {
      type_info.set_notnull(true);
    }
  }
  Constant(const SQLTypeInfo& ti, bool n, Datum v, bool cacheable = true)
      : Expr(ti), is_null(n), cacheable_(cacheable), constval(v) {
    if (n) {
      set_null_value();
    } else {
      type_info.set_notnull(true);
    }
  }
  Constant(const Type* type, bool n, Datum v, bool cacheable = true)
      : Expr(type), is_null(n), cacheable_(cacheable), constval(v) {
    if (n) {
      set_null_value();
    } else {
      type_info.set_notnull(true);
    }
  }
  Constant(const SQLTypeInfo& ti, bool n, const ExprPtrList& l, bool cacheable = true)
      : Expr(ti), is_null(n), cacheable_(cacheable), constval(Datum{0}), value_list(l) {}
  ~Constant() override;
  bool get_is_null() const { return is_null; }
  bool cacheable() const { return cacheable_; }
  Datum get_constval() const { return constval; }
  void set_constval(Datum d) { constval = d; }
  int64_t intVal() const { return extract_int_type_from_datum(constval, type_info); }
  double fpVal() const { return extract_fp_type_from_datum(constval, type_info); }
  const ExprPtrList& get_value_list() const { return value_list; }
  ExprPtr deep_copy() const override;
  ExprPtr add_cast(const Type* new_type, bool is_dict_intersection = false) override;
  using Expr::add_cast;
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;

  size_t hash() const override;

  static ExprPtr make(const SQLTypeInfo& ti, int64_t val, bool cacheable = true);

 protected:
  // Constant is NULL
  bool is_null;
  // A hint for DAG caches. Cache hit is unlikely, when set to true
  // (e.g. constant expression represents NOW datetime).
  bool cacheable_;
  Datum constval;  // the constant value
  const ExprPtrList value_list;
  void cast_number(const Type* new_type);
  void cast_string(const Type* new_type);
  void cast_from_string(const Type* new_type);
  void cast_to_string(const Type* new_type);
  void do_cast(const Type* new_type);
  void set_null_value();
};

/*
 * @type UOper
 * @brief represents unary operator expressions.  operator types include
 * kUMINUS, kISNULL, kEXISTS, kCAST
 */
class UOper : public Expr {
 public:
  UOper(const Type* type, bool has_agg, SQLOps o, ExprPtr p)
      : Expr(type, has_agg), optype(o), operand(p), is_dict_intersection_(false) {}
  UOper(const SQLTypeInfo& ti,
        bool has_agg,
        SQLOps o,
        ExprPtr p,
        bool is_dict_intersection = false)
      : Expr(ti, has_agg)
      , optype(o)
      , operand(p)
      , is_dict_intersection_(is_dict_intersection) {}
  UOper(SQLTypes t, SQLOps o, ExprPtr p)
      : Expr(t, o == kISNULL ? true : p->get_type_info().get_notnull())
      , optype(o)
      , operand(p)
      , is_dict_intersection_(false) {}
  SQLOps get_optype() const { return optype; }
  const Expr* get_operand() const { return operand.get(); }
  const ExprPtr get_own_operand() const { return operand; }
  bool is_dict_intersection() const { return is_dict_intersection_; }
  void check_group_by(const ExprPtrList& groupby) const override;
  ExprPtr deep_copy() const override;
  void group_predicates(std::list<const Expr*>& scan_predicates,
                        std::list<const Expr*>& join_predicates,
                        std::list<const Expr*>& const_predicates) const override;
  void collect_rte_idx(std::set<int>& rte_idx_set) const override {
    operand->collect_rte_idx(rte_idx_set);
  }
  void collect_column_var(
      std::set<const ColumnVar*, bool (*)(const ColumnVar*, const ColumnVar*)>&
          colvar_set,
      bool include_agg) const override {
    operand->collect_column_var(colvar_set, include_agg);
  }
  ExprPtr rewrite_with_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override {
    return makeExpr<UOper>(
        type_info, contains_agg, optype, operand->rewrite_with_targetlist(tlist));
  }
  ExprPtr rewrite_with_child_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override {
    return makeExpr<UOper>(
        type_info, contains_agg, optype, operand->rewrite_with_child_targetlist(tlist));
  }
  ExprPtr rewrite_agg_to_var(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override {
    return makeExpr<UOper>(
        type_info, contains_agg, optype, operand->rewrite_agg_to_var(tlist));
  }
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;
  void find_expr(bool (*f)(const Expr*),
                 std::list<const Expr*>& expr_list) const override;
  ExprPtr add_cast(const Type* new_type, bool is_dict_intersection = false) override;

  size_t hash() const override;

 protected:
  SQLOps optype;    // operator type, e.g., kUMINUS, kISNULL, kEXISTS
  ExprPtr operand;  // operand expression
  bool is_dict_intersection_;
};

/*
 * @type BinOper
 * @brief represents binary operator expressions.  it includes all
 * comparison, arithmetic and boolean binary operators.  it handles ANY/ALL qualifiers
 * in case the right_operand is a subquery.
 */
class BinOper : public Expr {
 public:
  BinOper(const SQLTypeInfo& ti,
          bool has_agg,
          SQLOps o,
          SQLQualifier q,
          ExprPtr l,
          ExprPtr r)
      : Expr(ti, has_agg), optype(o), qualifier(q), left_operand(l), right_operand(r) {}
  BinOper(const Type* type, bool has_agg, SQLOps o, SQLQualifier q, ExprPtr l, ExprPtr r)
      : Expr(type, has_agg), optype(o), qualifier(q), left_operand(l), right_operand(r) {}
  BinOper(SQLTypes t, SQLOps o, SQLQualifier q, ExprPtr l, ExprPtr r)
      : Expr(t, l->get_type_info().get_notnull() && r->get_type_info().get_notnull())
      , optype(o)
      , qualifier(q)
      , left_operand(l)
      , right_operand(r) {}
  SQLOps get_optype() const { return optype; }
  SQLQualifier get_qualifier() const { return qualifier; }
  const Expr* get_left_operand() const { return left_operand.get(); }
  const Expr* get_right_operand() const { return right_operand.get(); }
  const ExprPtr get_own_left_operand() const { return left_operand; }
  const ExprPtr get_own_right_operand() const { return right_operand; }

  void check_group_by(const ExprPtrList& groupby) const override;
  ExprPtr deep_copy() const override;
  ExprPtr normalize_simple_predicate(int& rte_idx) const override;
  void group_predicates(std::list<const Expr*>& scan_predicates,
                        std::list<const Expr*>& join_predicates,
                        std::list<const Expr*>& const_predicates) const override;
  void collect_rte_idx(std::set<int>& rte_idx_set) const override {
    left_operand->collect_rte_idx(rte_idx_set);
    right_operand->collect_rte_idx(rte_idx_set);
  }
  void collect_column_var(
      std::set<const ColumnVar*, bool (*)(const ColumnVar*, const ColumnVar*)>&
          colvar_set,
      bool include_agg) const override {
    left_operand->collect_column_var(colvar_set, include_agg);
    right_operand->collect_column_var(colvar_set, include_agg);
  }
  ExprPtr rewrite_with_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override {
    return makeExpr<BinOper>(type_info,
                             contains_agg,
                             optype,
                             qualifier,
                             left_operand->rewrite_with_targetlist(tlist),
                             right_operand->rewrite_with_targetlist(tlist));
  }
  ExprPtr rewrite_with_child_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override {
    return makeExpr<BinOper>(type_info,
                             contains_agg,
                             optype,
                             qualifier,
                             left_operand->rewrite_with_child_targetlist(tlist),
                             right_operand->rewrite_with_child_targetlist(tlist));
  }
  ExprPtr rewrite_agg_to_var(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override {
    return makeExpr<BinOper>(type_info,
                             contains_agg,
                             optype,
                             qualifier,
                             left_operand->rewrite_agg_to_var(tlist),
                             right_operand->rewrite_agg_to_var(tlist));
  }
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;
  void find_expr(bool (*f)(const Expr*),
                 std::list<const Expr*>& expr_list) const override;
  static bool simple_predicate_has_simple_cast(const ExprPtr cast_operand,
                                               const ExprPtr const_operand);

  size_t hash() const override;

 private:
  SQLOps optype;           // operator type, e.g., kLT, kAND, kPLUS, etc.
  SQLQualifier qualifier;  // qualifier kANY, kALL or kONE.  Only relevant with
                           // right_operand is Subquery
  ExprPtr left_operand;    // the left operand expression
  ExprPtr right_operand;   // the right operand expression
};

/**
 * @type RangeOper
 * @brief
 */
class RangeOper : public Expr {
 public:
  RangeOper(const bool l_inclusive, const bool r_inclusive, ExprPtr l, ExprPtr r)
      : Expr(SQLTypeInfo(kNULLT), /*not_null=*/false)
      , left_inclusive_(l_inclusive)
      , right_inclusive_(r_inclusive)
      , left_operand_(l)
      , right_operand_(r) {
    CHECK(left_operand_);
    CHECK(right_operand_);
  }

  const Expr* get_left_operand() const { return left_operand_.get(); }
  const Expr* get_right_operand() const { return right_operand_.get(); }

  ExprPtr deep_copy() const override;
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;

  void collect_rte_idx(std::set<int>& rte_idx_set) const override {
    left_operand_->collect_rte_idx(rte_idx_set);
    right_operand_->collect_rte_idx(rte_idx_set);
  }

  void collect_column_var(
      std::set<const ColumnVar*, bool (*)(const ColumnVar*, const ColumnVar*)>&
          colvar_set,
      bool include_agg) const override {
    left_operand_->collect_column_var(colvar_set, include_agg);
    right_operand_->collect_column_var(colvar_set, include_agg);
  }

  size_t hash() const override;

 private:
  // build a range between these two operands
  bool left_inclusive_;
  bool right_inclusive_;
  ExprPtr left_operand_;
  ExprPtr right_operand_;
};

class ScalarSubquery : public Expr {
 public:
  ScalarSubquery(const SQLTypeInfo& ti, std::shared_ptr<const RelAlgNode> node)
      : Expr(ti), node_(node) {}

  ExprPtr deep_copy() const override {
    return makeExpr<ScalarSubquery>(type_info, node_);
  }

  bool operator==(const Expr& rhs) const override {
    const ScalarSubquery* rhsp = dynamic_cast<const ScalarSubquery*>(&rhs);
    return rhsp && node_ == rhsp->node_;
  }

  std::string toString() const override;

  const RelAlgNode* getNode() const { return node_.get(); }
  std::shared_ptr<const RelAlgNode> getNodeShared() const { return node_; }

  size_t hash() const override;

 private:
  std::shared_ptr<const RelAlgNode> node_;
};

/*
 * @type InValues
 * @brief represents predicate expr IN (v1, v2, ...)
 * v1, v2, ... are can be either Constant or Parameter.
 */
class InValues : public Expr {
 public:
  InValues(ExprPtr a, const ExprPtrList& l);
  const Expr* get_arg() const { return arg.get(); }
  const ExprPtr get_own_arg() const { return arg; }
  const ExprPtrList& get_value_list() const { return value_list; }
  ExprPtr deep_copy() const override;
  void group_predicates(std::list<const Expr*>& scan_predicates,
                        std::list<const Expr*>& join_predicates,
                        std::list<const Expr*>& const_predicates) const override;
  void collect_rte_idx(std::set<int>& rte_idx_set) const override {
    arg->collect_rte_idx(rte_idx_set);
  }
  void collect_column_var(
      std::set<const ColumnVar*, bool (*)(const ColumnVar*, const ColumnVar*)>&
          colvar_set,
      bool include_agg) const override {
    arg->collect_column_var(colvar_set, include_agg);
  }
  ExprPtr rewrite_with_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override;
  ExprPtr rewrite_with_child_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override;
  ExprPtr rewrite_agg_to_var(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override;
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;
  void find_expr(bool (*f)(const Expr*),
                 std::list<const Expr*>& expr_list) const override;

  size_t hash() const override;

 private:
  ExprPtr arg;                   // the argument left of IN
  const ExprPtrList value_list;  // the list of values right of IN
};

/*
 * @type InIntegerSet
 * @brief represents predicate expr IN (v1, v2, ...) for the case where the right
 *        hand side is a list of integers or dictionary-encoded strings generated
 *        by a IN subquery. Avoids the overhead of storing a list of shared pointers
 *        to Constant objects, making it more suitable for IN sub-queries usage.
 * v1, v2, ... are integers
 */
class InIntegerSet : public Expr {
 public:
  InIntegerSet(const std::shared_ptr<const Expr> a,
               const std::vector<int64_t>& values,
               const bool not_null);

  const Expr* get_arg() const { return arg.get(); }

  const std::vector<int64_t>& get_value_list() const { return value_list; }

  ExprPtr deep_copy() const override;

  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;

  size_t hash() const override;

 private:
  const std::shared_ptr<const Expr> arg;  // the argument left of IN
  const std::vector<int64_t> value_list;  // the list of values right of IN
};

class InSubquery : public Expr {
 public:
  InSubquery(const SQLTypeInfo& ti,
             hdk::ir::ExprPtr arg,
             std::shared_ptr<const RelAlgNode> node)
      : Expr(ti), arg_(std::move(arg)), node_(std::move(node)) {}

  ExprPtr deep_copy() const override {
    return makeExpr<InSubquery>(type_info, arg_->deep_copy(), node_);
  }

  bool operator==(const Expr& rhs) const override {
    const InSubquery* rhsp = dynamic_cast<const InSubquery*>(&rhs);
    return rhsp && node_ == rhsp->node_;
  }

  std::string toString() const override;

  hdk::ir::ExprPtr getArg() const { return arg_; }

  const RelAlgNode* getNode() const { return node_.get(); }
  std::shared_ptr<const RelAlgNode> getNodeShared() const { return node_; }

  size_t hash() const override;

 private:
  hdk::ir::ExprPtr arg_;
  std::shared_ptr<const RelAlgNode> node_;
};

/*
 * @type CharLengthExpr
 * @brief expression for the CHAR_LENGTH expression.
 * arg must evaluate to char, varchar or text.
 */
class CharLengthExpr : public Expr {
 public:
  CharLengthExpr(ExprPtr a, bool e)
      : Expr(kINT, a->get_type_info().get_notnull()), arg(a), calc_encoded_length(e) {}
  const Expr* get_arg() const { return arg.get(); }
  const ExprPtr get_own_arg() const { return arg; }
  bool get_calc_encoded_length() const { return calc_encoded_length; }
  ExprPtr deep_copy() const override;
  void group_predicates(std::list<const Expr*>& scan_predicates,
                        std::list<const Expr*>& join_predicates,
                        std::list<const Expr*>& const_predicates) const override;
  void collect_rte_idx(std::set<int>& rte_idx_set) const override {
    arg->collect_rte_idx(rte_idx_set);
  }
  void collect_column_var(
      std::set<const ColumnVar*, bool (*)(const ColumnVar*, const ColumnVar*)>&
          colvar_set,
      bool include_agg) const override {
    arg->collect_column_var(colvar_set, include_agg);
  }
  ExprPtr rewrite_with_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override {
    return makeExpr<CharLengthExpr>(arg->rewrite_with_targetlist(tlist),
                                    calc_encoded_length);
  }
  ExprPtr rewrite_with_child_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override {
    return makeExpr<CharLengthExpr>(arg->rewrite_with_child_targetlist(tlist),
                                    calc_encoded_length);
  }
  ExprPtr rewrite_agg_to_var(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override {
    return makeExpr<CharLengthExpr>(arg->rewrite_agg_to_var(tlist), calc_encoded_length);
  }
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;
  void find_expr(bool (*f)(const Expr*),
                 std::list<const Expr*>& expr_list) const override;

  size_t hash() const override;

 private:
  ExprPtr arg;
  bool calc_encoded_length;
};

/*
 * @type KeyForStringExpr
 * @brief expression for the KEY_FOR_STRING expression.
 * arg must be a dict encoded column, not str literal.
 */
class KeyForStringExpr : public Expr {
 public:
  KeyForStringExpr(ExprPtr a) : Expr(kINT, a->get_type_info().get_notnull()), arg(a) {}
  const Expr* get_arg() const { return arg.get(); }
  const ExprPtr get_own_arg() const { return arg; }
  ExprPtr deep_copy() const override;
  void group_predicates(std::list<const Expr*>& scan_predicates,
                        std::list<const Expr*>& join_predicates,
                        std::list<const Expr*>& const_predicates) const override;
  void collect_rte_idx(std::set<int>& rte_idx_set) const override {
    arg->collect_rte_idx(rte_idx_set);
  }
  void collect_column_var(
      std::set<const ColumnVar*, bool (*)(const ColumnVar*, const ColumnVar*)>&
          colvar_set,
      bool include_agg) const override {
    arg->collect_column_var(colvar_set, include_agg);
  }
  ExprPtr rewrite_with_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override {
    return makeExpr<KeyForStringExpr>(arg->rewrite_with_targetlist(tlist));
  }
  ExprPtr rewrite_with_child_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override {
    return makeExpr<KeyForStringExpr>(arg->rewrite_with_child_targetlist(tlist));
  }
  ExprPtr rewrite_agg_to_var(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override {
    return makeExpr<KeyForStringExpr>(arg->rewrite_agg_to_var(tlist));
  }
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;
  void find_expr(bool (*f)(const Expr*),
                 std::list<const Expr*>& expr_list) const override;

  size_t hash() const override;

 private:
  ExprPtr arg;
};

/*
 * @type SampleRatioExpr
 * @brief expression for the SAMPLE_RATIO expression. Argument range is expected to be
 * between 0 and 1.
 */
class SampleRatioExpr : public Expr {
 public:
  SampleRatioExpr(ExprPtr a) : Expr(kBOOLEAN, a->get_type_info().get_notnull()), arg(a) {}
  const Expr* get_arg() const { return arg.get(); }
  const ExprPtr get_own_arg() const { return arg; }
  ExprPtr deep_copy() const override;
  void group_predicates(std::list<const Expr*>& scan_predicates,
                        std::list<const Expr*>& join_predicates,
                        std::list<const Expr*>& const_predicates) const override;
  void collect_rte_idx(std::set<int>& rte_idx_set) const override {
    arg->collect_rte_idx(rte_idx_set);
  }
  void collect_column_var(
      std::set<const ColumnVar*, bool (*)(const ColumnVar*, const ColumnVar*)>&
          colvar_set,
      bool include_agg) const override {
    arg->collect_column_var(colvar_set, include_agg);
  }
  ExprPtr rewrite_with_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override {
    return makeExpr<SampleRatioExpr>(arg->rewrite_with_targetlist(tlist));
  }
  ExprPtr rewrite_with_child_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override {
    return makeExpr<SampleRatioExpr>(arg->rewrite_with_child_targetlist(tlist));
  }
  ExprPtr rewrite_agg_to_var(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override {
    return makeExpr<SampleRatioExpr>(arg->rewrite_agg_to_var(tlist));
  }
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;
  void find_expr(bool (*f)(const Expr*),
                 std::list<const Expr*>& expr_list) const override;

  size_t hash() const override;

 private:
  ExprPtr arg;
};

/**
 * @brief Expression class for the LOWER (lowercase) string function.
 * The "arg" constructor parameter must be an expression that resolves to a string
 * datatype (e.g. TEXT).
 */
class LowerExpr : public Expr {
 public:
  LowerExpr(ExprPtr arg) : Expr(arg->get_type_info()), arg(arg) {}

  const Expr* get_arg() const { return arg.get(); }

  const ExprPtr get_own_arg() const { return arg; }

  void collect_rte_idx(std::set<int>& rte_idx_set) const override {
    arg->collect_rte_idx(rte_idx_set);
  }

  void collect_column_var(
      std::set<const ColumnVar*, bool (*)(const ColumnVar*, const ColumnVar*)>&
          colvar_set,
      bool include_agg) const override {
    arg->collect_column_var(colvar_set, include_agg);
  }

  ExprPtr rewrite_with_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override {
    return makeExpr<LowerExpr>(arg->rewrite_with_targetlist(tlist));
  }

  ExprPtr rewrite_with_child_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override {
    return makeExpr<LowerExpr>(arg->rewrite_with_child_targetlist(tlist));
  }

  ExprPtr rewrite_agg_to_var(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override {
    return makeExpr<LowerExpr>(arg->rewrite_agg_to_var(tlist));
  }

  ExprPtr deep_copy() const override;

  void group_predicates(std::list<const Expr*>& scan_predicates,
                        std::list<const Expr*>& join_predicates,
                        std::list<const Expr*>& const_predicates) const override;

  bool operator==(const Expr& rhs) const override;

  std::string toString() const override;

  void find_expr(bool (*f)(const Expr*),
                 std::list<const Expr*>& expr_list) const override;

  size_t hash() const override;

 private:
  ExprPtr arg;
};

/*
 * @type CardinalityExpr
 * @brief expression for the CARDINALITY expression.
 * arg must evaluate to array (or multiset when supported).
 */
class CardinalityExpr : public Expr {
 public:
  CardinalityExpr(ExprPtr a) : Expr(kINT, a->get_type_info().get_notnull()), arg(a) {}
  const Expr* get_arg() const { return arg.get(); }
  const ExprPtr get_own_arg() const { return arg; }
  ExprPtr deep_copy() const override;
  void group_predicates(std::list<const Expr*>& scan_predicates,
                        std::list<const Expr*>& join_predicates,
                        std::list<const Expr*>& const_predicates) const override;
  void collect_rte_idx(std::set<int>& rte_idx_set) const override {
    arg->collect_rte_idx(rte_idx_set);
  }
  void collect_column_var(
      std::set<const ColumnVar*, bool (*)(const ColumnVar*, const ColumnVar*)>&
          colvar_set,
      bool include_agg) const override {
    arg->collect_column_var(colvar_set, include_agg);
  }
  ExprPtr rewrite_with_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override {
    return makeExpr<CardinalityExpr>(arg->rewrite_with_targetlist(tlist));
  }
  ExprPtr rewrite_with_child_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override {
    return makeExpr<CardinalityExpr>(arg->rewrite_with_child_targetlist(tlist));
  }
  ExprPtr rewrite_agg_to_var(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override {
    return makeExpr<CardinalityExpr>(arg->rewrite_agg_to_var(tlist));
  }
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;
  void find_expr(bool (*f)(const Expr*),
                 std::list<const Expr*>& expr_list) const override;

  size_t hash() const override;

 private:
  ExprPtr arg;
};

/*
 * @type LikeExpr
 * @brief expression for the LIKE predicate.
 * arg must evaluate to char, varchar or text.
 */
class LikeExpr : public Expr {
 public:
  LikeExpr(ExprPtr a, ExprPtr l, ExprPtr e, bool i, bool s)
      : Expr(kBOOLEAN, a->get_type_info().get_notnull())
      , arg(a)
      , like_expr(l)
      , escape_expr(e)
      , is_ilike(i)
      , is_simple(s) {}
  const Expr* get_arg() const { return arg.get(); }
  const ExprPtr get_own_arg() const { return arg; }
  const Expr* get_like_expr() const { return like_expr.get(); }
  const Expr* get_escape_expr() const { return escape_expr.get(); }
  bool get_is_ilike() const { return is_ilike; }
  bool get_is_simple() const { return is_simple; }
  ExprPtr deep_copy() const override;
  void group_predicates(std::list<const Expr*>& scan_predicates,
                        std::list<const Expr*>& join_predicates,
                        std::list<const Expr*>& const_predicates) const override;
  void collect_rte_idx(std::set<int>& rte_idx_set) const override {
    arg->collect_rte_idx(rte_idx_set);
  }
  void collect_column_var(
      std::set<const ColumnVar*, bool (*)(const ColumnVar*, const ColumnVar*)>&
          colvar_set,
      bool include_agg) const override {
    arg->collect_column_var(colvar_set, include_agg);
  }
  ExprPtr rewrite_with_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override {
    return makeExpr<LikeExpr>(arg->rewrite_with_targetlist(tlist),
                              like_expr->deep_copy(),
                              escape_expr ? escape_expr->deep_copy() : nullptr,
                              is_ilike,
                              is_simple);
  }
  ExprPtr rewrite_with_child_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override {
    return makeExpr<LikeExpr>(arg->rewrite_with_child_targetlist(tlist),
                              like_expr->deep_copy(),
                              escape_expr ? escape_expr->deep_copy() : nullptr,
                              is_ilike,
                              is_simple);
  }
  ExprPtr rewrite_agg_to_var(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override {
    return makeExpr<LikeExpr>(arg->rewrite_agg_to_var(tlist),
                              like_expr->deep_copy(),
                              escape_expr ? escape_expr->deep_copy() : nullptr,
                              is_ilike,
                              is_simple);
  }
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;
  void find_expr(bool (*f)(const Expr*),
                 std::list<const Expr*>& expr_list) const override;

  size_t hash() const override;

 private:
  ExprPtr arg;          // the argument to the left of LIKE
  ExprPtr like_expr;    // expression that evaluates to like string
  ExprPtr escape_expr;  // expression that evaluates to escape string, can be nullptr
  bool is_ilike;        // is this ILIKE?
  bool is_simple;  // is this simple, meaning we can use fast path search (fits '%str%'
                   // pattern with no inner '%','_','[',']'
};

/*
 * @type RegexpExpr
 * @brief expression for REGEXP.
 * arg must evaluate to char, varchar or text.
 */
class RegexpExpr : public Expr {
 public:
  RegexpExpr(ExprPtr a, ExprPtr p, ExprPtr e)
      : Expr(kBOOLEAN, a->get_type_info().get_notnull())
      , arg(a)
      , pattern_expr(p)
      , escape_expr(e) {}
  const Expr* get_arg() const { return arg.get(); }
  const ExprPtr get_own_arg() const { return arg; }
  const Expr* get_pattern_expr() const { return pattern_expr.get(); }
  const Expr* get_escape_expr() const { return escape_expr.get(); }
  ExprPtr deep_copy() const override;
  void group_predicates(std::list<const Expr*>& scan_predicates,
                        std::list<const Expr*>& join_predicates,
                        std::list<const Expr*>& const_predicates) const override;
  void collect_rte_idx(std::set<int>& rte_idx_set) const override {
    arg->collect_rte_idx(rte_idx_set);
  }
  void collect_column_var(
      std::set<const ColumnVar*, bool (*)(const ColumnVar*, const ColumnVar*)>&
          colvar_set,
      bool include_agg) const override {
    arg->collect_column_var(colvar_set, include_agg);
  }
  ExprPtr rewrite_with_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override {
    return makeExpr<RegexpExpr>(arg->rewrite_with_targetlist(tlist),
                                pattern_expr->deep_copy(),
                                escape_expr ? escape_expr->deep_copy() : nullptr);
  }
  ExprPtr rewrite_with_child_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override {
    return makeExpr<RegexpExpr>(arg->rewrite_with_child_targetlist(tlist),
                                pattern_expr->deep_copy(),
                                escape_expr ? escape_expr->deep_copy() : nullptr);
  }
  ExprPtr rewrite_agg_to_var(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override {
    return makeExpr<RegexpExpr>(arg->rewrite_agg_to_var(tlist),
                                pattern_expr->deep_copy(),
                                escape_expr ? escape_expr->deep_copy() : nullptr);
  }
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;
  void find_expr(bool (*f)(const Expr*),
                 std::list<const Expr*>& expr_list) const override;

  size_t hash() const override;

 private:
  ExprPtr arg;           // the argument to the left of REGEXP
  ExprPtr pattern_expr;  // expression that evaluates to pattern string
  ExprPtr escape_expr;   // expression that evaluates to escape string, can be nullptr
};

/*
 * @type WidthBucketExpr
 * @brief expression for width_bucket functions.
 */
class WidthBucketExpr : public Expr {
 public:
  WidthBucketExpr(const ExprPtr target_value,
                  const ExprPtr lower_bound,
                  const ExprPtr upper_bound,
                  const ExprPtr partition_count)
      : Expr(kINT, target_value->get_type_info().get_notnull())
      , target_value_(target_value)
      , lower_bound_(lower_bound)
      , upper_bound_(upper_bound)
      , partition_count_(partition_count)
      , constant_expr_(false)
      , skip_out_of_bound_check_(false) {}
  const Expr* get_target_value() const { return target_value_.get(); }
  const Expr* get_lower_bound() const { return lower_bound_.get(); }
  const Expr* get_upper_bound() const { return upper_bound_.get(); }
  const Expr* get_partition_count() const { return partition_count_.get(); }
  ExprPtr deep_copy() const override;
  void group_predicates(std::list<const Expr*>& scan_predicates,
                        std::list<const Expr*>& join_predicates,
                        std::list<const Expr*>& const_predicates) const override;
  void collect_rte_idx(std::set<int>& rte_idx_set) const override {
    target_value_->collect_rte_idx(rte_idx_set);
  }
  void collect_column_var(
      std::set<const ColumnVar*, bool (*)(const ColumnVar*, const ColumnVar*)>&
          colvar_set,
      bool include_agg) const override {
    target_value_->collect_column_var(colvar_set, include_agg);
  }
  ExprPtr rewrite_with_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override {
    return makeExpr<WidthBucketExpr>(target_value_->rewrite_with_targetlist(tlist),
                                     lower_bound_,
                                     upper_bound_,
                                     partition_count_);
  }
  ExprPtr rewrite_with_child_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override {
    return makeExpr<WidthBucketExpr>(target_value_->rewrite_with_child_targetlist(tlist),
                                     lower_bound_,
                                     upper_bound_,
                                     partition_count_);
  }
  ExprPtr rewrite_agg_to_var(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override {
    return makeExpr<WidthBucketExpr>(target_value_->rewrite_agg_to_var(tlist),
                                     lower_bound_,
                                     upper_bound_,
                                     partition_count_);
  }
  double get_bound_val(const Expr* bound_expr) const;
  int32_t get_partition_count_val() const;
  template <typename T>
  int32_t compute_bucket(T target_const_val, SQLTypeInfo& ti) const {
    // this utility function is useful for optimizing expression range decision
    // for an expression depending on width_bucket expr
    T null_val = ti.is_integer() ? inline_int_null_val(ti) : inline_fp_null_val(ti);
    double lower_bound_val = get_bound_val(lower_bound_.get());
    double upper_bound_val = get_bound_val(upper_bound_.get());
    auto partition_count_val = get_partition_count_val();
    if (target_const_val == null_val) {
      return INT32_MIN;
    }
    float res;
    if (lower_bound_val < upper_bound_val) {
      if (target_const_val < lower_bound_val) {
        return 0;
      } else if (target_const_val >= upper_bound_val) {
        return partition_count_val + 1;
      }
      double dividend = upper_bound_val - lower_bound_val;
      res = ((partition_count_val * (target_const_val - lower_bound_val)) / dividend) + 1;
    } else {
      if (target_const_val > lower_bound_val) {
        return 0;
      } else if (target_const_val <= upper_bound_val) {
        return partition_count_val + 1;
      }
      double dividend = lower_bound_val - upper_bound_val;
      res = ((partition_count_val * (lower_bound_val - target_const_val)) / dividend) + 1;
    }
    return res;
  }
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;
  void find_expr(bool (*f)(const Expr*),
                 std::list<const Expr*>& expr_list) const override;
  bool can_skip_out_of_bound_check() const { return skip_out_of_bound_check_; }
  void skip_out_of_bound_check() const { skip_out_of_bound_check_ = true; }
  void set_constant_expr() const { constant_expr_ = true; }
  bool is_constant_expr() const { return constant_expr_; }

  size_t hash() const override;

 private:
  ExprPtr target_value_;     // target value expression
  ExprPtr lower_bound_;      // lower_bound
  ExprPtr upper_bound_;      // upper_bound
  ExprPtr partition_count_;  // partition_count
  // true if lower, upper and partition count exprs are constant
  mutable bool constant_expr_;
  // true if we can skip oob check and is determined within compile time
  mutable bool skip_out_of_bound_check_;
};

/*
 * @type LikelihoodExpr
 * @brief expression for LIKELY and UNLIKELY boolean identity functions.
 */
class LikelihoodExpr : public Expr {
 public:
  LikelihoodExpr(ExprPtr a, float l = 0.5)
      : Expr(kBOOLEAN, a->get_type_info().get_notnull()), arg(a), likelihood(l) {}
  const Expr* get_arg() const { return arg.get(); }
  const ExprPtr get_own_arg() const { return arg; }
  float get_likelihood() const { return likelihood; }
  ExprPtr deep_copy() const override;
  void group_predicates(std::list<const Expr*>& scan_predicates,
                        std::list<const Expr*>& join_predicates,
                        std::list<const Expr*>& const_predicates) const override;
  void collect_rte_idx(std::set<int>& rte_idx_set) const override {
    arg->collect_rte_idx(rte_idx_set);
  }
  void collect_column_var(
      std::set<const ColumnVar*, bool (*)(const ColumnVar*, const ColumnVar*)>&
          colvar_set,
      bool include_agg) const override {
    arg->collect_column_var(colvar_set, include_agg);
  }
  ExprPtr rewrite_with_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override {
    return makeExpr<LikelihoodExpr>(arg->rewrite_with_targetlist(tlist), likelihood);
  }
  ExprPtr rewrite_with_child_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override {
    return makeExpr<LikelihoodExpr>(arg->rewrite_with_child_targetlist(tlist),
                                    likelihood);
  }
  ExprPtr rewrite_agg_to_var(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override {
    return makeExpr<LikelihoodExpr>(arg->rewrite_agg_to_var(tlist), likelihood);
  }
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;
  void find_expr(bool (*f)(const Expr*),
                 std::list<const Expr*>& expr_list) const override;

  size_t hash() const override;

 private:
  ExprPtr arg;  // the argument to LIKELY, UNLIKELY
  float likelihood;
};

/*
 * @type AggExpr
 * @brief expression for builtin SQL aggregates.
 */
class AggExpr : public Expr {
 public:
  AggExpr(const Type* type, SQLAgg a, ExprPtr g, bool d, std::shared_ptr<Constant> e)
      : Expr(type, true), aggtype(a), arg(g), is_distinct(d), arg1(e) {}
  AggExpr(const SQLTypeInfo& ti, SQLAgg a, ExprPtr g, bool d, std::shared_ptr<Constant> e)
      : Expr(ti, true), aggtype(a), arg(g), is_distinct(d), arg1(e) {}
  AggExpr(SQLTypes t, SQLAgg a, Expr* g, bool d, std::shared_ptr<Constant> e, int idx)
      : Expr(SQLTypeInfo(t, g == nullptr ? true : g->get_type_info().get_notnull()), true)
      , aggtype(a)
      , arg(g)
      , is_distinct(d)
      , arg1(e) {}
  SQLAgg get_aggtype() const { return aggtype; }
  Expr* get_arg() const { return arg.get(); }
  ExprPtr get_own_arg() const { return arg; }
  bool get_is_distinct() const { return is_distinct; }
  std::shared_ptr<Constant> get_arg1() const { return arg1; }
  ExprPtr deep_copy() const override;
  void group_predicates(std::list<const Expr*>& scan_predicates,
                        std::list<const Expr*>& join_predicates,
                        std::list<const Expr*>& const_predicates) const override;
  void collect_rte_idx(std::set<int>& rte_idx_set) const override {
    if (arg) {
      arg->collect_rte_idx(rte_idx_set);
    }
  };
  void collect_column_var(
      std::set<const ColumnVar*, bool (*)(const ColumnVar*, const ColumnVar*)>&
          colvar_set,
      bool include_agg) const override {
    if (include_agg && arg != nullptr) {
      arg->collect_column_var(colvar_set, include_agg);
    }
  }
  ExprPtr rewrite_with_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override;
  ExprPtr rewrite_with_child_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override;
  ExprPtr rewrite_agg_to_var(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override;
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;
  void find_expr(bool (*f)(const Expr*),
                 std::list<const Expr*>& expr_list) const override;

  size_t hash() const override;

 private:
  SQLAgg aggtype;    // aggregate type: kAVG, kMIN, kMAX, kSUM, kCOUNT
  ExprPtr arg;       // argument to aggregate
  bool is_distinct;  // true only if it is for COUNT(DISTINCT x)
  // APPROX_COUNT_DISTINCT error_rate, APPROX_QUANTILE quantile
  std::shared_ptr<Constant> arg1;
};

/*
 * @type CaseExpr
 * @brief the CASE-WHEN-THEN-ELSE expression
 */
class CaseExpr : public Expr {
 public:
  CaseExpr(const SQLTypeInfo& ti,
           bool has_agg,
           const std::list<std::pair<ExprPtr, ExprPtr>>& w,
           ExprPtr e)
      : Expr(ti, has_agg), expr_pair_list(w), else_expr(e) {}
  CaseExpr(const hdk::ir::Type* type,
           bool has_agg,
           std::list<std::pair<ExprPtr, ExprPtr>> expr_pairs,
           ExprPtr e)
      : Expr(type, has_agg), expr_pair_list(std::move(expr_pairs)), else_expr(e) {}
  const std::list<std::pair<ExprPtr, ExprPtr>>& get_expr_pair_list() const {
    return expr_pair_list;
  }
  const Expr* get_else_expr() const { return else_expr.get(); }
  ExprPtr deep_copy() const override;
  void check_group_by(const ExprPtrList& groupby) const override;
  void group_predicates(std::list<const Expr*>& scan_predicates,
                        std::list<const Expr*>& join_predicates,
                        std::list<const Expr*>& const_predicates) const override;
  void collect_rte_idx(std::set<int>& rte_idx_set) const override;
  void collect_column_var(
      std::set<const ColumnVar*, bool (*)(const ColumnVar*, const ColumnVar*)>&
          colvar_set,
      bool include_agg) const override;
  ExprPtr rewrite_with_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override;
  ExprPtr rewrite_with_child_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override;
  ExprPtr rewrite_agg_to_var(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override;
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;
  void find_expr(bool (*f)(const Expr*),
                 std::list<const Expr*>& expr_list) const override;
  ExprPtr add_cast(const Type* new_type, bool is_dict_intersection) override;
  void get_domain(DomainSet& domain_set) const override;

  size_t hash() const override;

 private:
  std::list<std::pair<ExprPtr, ExprPtr>>
      expr_pair_list;  // a pair of expressions for each WHEN expr1 THEN expr2.  expr1
                       // must be of boolean type.  all expr2's must be of compatible
                       // types and will be promoted to the common type.
  ExprPtr else_expr;   // expression for ELSE.  nullptr if omitted.
};

/*
 * @type ExtractExpr
 * @brief the EXTRACT expression
 */
class ExtractExpr : public Expr {
 public:
  ExtractExpr(const SQLTypeInfo& ti, bool has_agg, ExtractField f, ExprPtr e)
      : Expr(ti, has_agg), field_(f), from_expr_(e) {}
  ExtractField get_field() const { return field_; }
  const Expr* get_from_expr() const { return from_expr_.get(); }
  const ExprPtr get_own_from_expr() const { return from_expr_; }
  ExprPtr deep_copy() const override;
  void check_group_by(const ExprPtrList& groupby) const override;
  void group_predicates(std::list<const Expr*>& scan_predicates,
                        std::list<const Expr*>& join_predicates,
                        std::list<const Expr*>& const_predicates) const override;
  void collect_rte_idx(std::set<int>& rte_idx_set) const override;
  void collect_column_var(
      std::set<const ColumnVar*, bool (*)(const ColumnVar*, const ColumnVar*)>&
          colvar_set,
      bool include_agg) const override;
  ExprPtr rewrite_with_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override;
  ExprPtr rewrite_with_child_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override;
  ExprPtr rewrite_agg_to_var(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override;
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;
  void find_expr(bool (*f)(const Expr*),
                 std::list<const Expr*>& expr_list) const override;

  size_t hash() const override;

 private:
  ExtractField field_;
  ExprPtr from_expr_;
};

/*
 * @type DateaddExpr
 * @brief the DATEADD expression
 */
class DateaddExpr : public Expr {
 public:
  DateaddExpr(const SQLTypeInfo& ti,
              const DateaddField f,
              const ExprPtr number,
              const ExprPtr datetime)
      : Expr(ti, false), field_(f), number_(number), datetime_(datetime) {}
  DateaddField get_field() const { return field_; }
  const Expr* get_number_expr() const { return number_.get(); }
  const Expr* get_datetime_expr() const { return datetime_.get(); }
  ExprPtr deep_copy() const override;
  void check_group_by(const ExprPtrList& groupby) const override;
  void group_predicates(std::list<const Expr*>& scan_predicates,
                        std::list<const Expr*>& join_predicates,
                        std::list<const Expr*>& const_predicates) const override;
  void collect_rte_idx(std::set<int>& rte_idx_set) const override;
  void collect_column_var(
      std::set<const ColumnVar*, bool (*)(const ColumnVar*, const ColumnVar*)>&
          colvar_set,
      bool include_agg) const override;
  ExprPtr rewrite_with_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override;
  ExprPtr rewrite_with_child_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override;
  ExprPtr rewrite_agg_to_var(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override;
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;
  void find_expr(bool (*f)(const Expr*),
                 std::list<const Expr*>& expr_list) const override;

  size_t hash() const override;

 private:
  const DateaddField field_;
  const ExprPtr number_;
  const ExprPtr datetime_;
};

/*
 * @type DatediffExpr
 * @brief the DATEDIFF expression
 */
class DatediffExpr : public Expr {
 public:
  DatediffExpr(const SQLTypeInfo& ti,
               const DatetruncField f,
               const ExprPtr start,
               const ExprPtr end)
      : Expr(ti, false), field_(f), start_(start), end_(end) {}
  DatetruncField get_field() const { return field_; }
  const Expr* get_start_expr() const { return start_.get(); }
  const Expr* get_end_expr() const { return end_.get(); }
  ExprPtr deep_copy() const override;
  void check_group_by(const ExprPtrList& groupby) const override;
  void group_predicates(std::list<const Expr*>& scan_predicates,
                        std::list<const Expr*>& join_predicates,
                        std::list<const Expr*>& const_predicates) const override;
  void collect_rte_idx(std::set<int>& rte_idx_set) const override;
  void collect_column_var(
      std::set<const ColumnVar*, bool (*)(const ColumnVar*, const ColumnVar*)>&
          colvar_set,
      bool include_agg) const override;
  ExprPtr rewrite_with_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override;
  ExprPtr rewrite_with_child_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override;
  ExprPtr rewrite_agg_to_var(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override;
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;
  void find_expr(bool (*f)(const Expr*),
                 std::list<const Expr*>& expr_list) const override;

  size_t hash() const override;

 private:
  const DatetruncField field_;
  const ExprPtr start_;
  const ExprPtr end_;
};

/*
 * @type DatetruncExpr
 * @brief the DATE_TRUNC expression
 */
class DatetruncExpr : public Expr {
 public:
  DatetruncExpr(const SQLTypeInfo& ti, bool has_agg, DatetruncField f, ExprPtr e)
      : Expr(ti, has_agg), field_(f), from_expr_(e) {}
  DatetruncField get_field() const { return field_; }
  const Expr* get_from_expr() const { return from_expr_.get(); }
  const ExprPtr get_own_from_expr() const { return from_expr_; }
  ExprPtr deep_copy() const override;
  void check_group_by(const ExprPtrList& groupby) const override;
  void group_predicates(std::list<const Expr*>& scan_predicates,
                        std::list<const Expr*>& join_predicates,
                        std::list<const Expr*>& const_predicates) const override;
  void collect_rte_idx(std::set<int>& rte_idx_set) const override;
  void collect_column_var(
      std::set<const ColumnVar*, bool (*)(const ColumnVar*, const ColumnVar*)>&
          colvar_set,
      bool include_agg) const override;
  ExprPtr rewrite_with_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override;
  ExprPtr rewrite_with_child_targetlist(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override;
  ExprPtr rewrite_agg_to_var(
      const std::vector<std::shared_ptr<TargetEntry>>& tlist) const override;
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;
  void find_expr(bool (*f)(const Expr*),
                 std::list<const Expr*>& expr_list) const override;

  size_t hash() const override;

 private:
  DatetruncField field_;
  ExprPtr from_expr_;
};

class FunctionOper : public Expr {
 public:
  FunctionOper(const SQLTypeInfo& ti, const std::string& name, const ExprPtrVector& args)
      : Expr(ti, false), name_(name), args_(args) {}

  const std::string& getName() const { return name_; }

  size_t getArity() const { return args_.size(); }

  const Expr* getArg(const size_t i) const {
    CHECK_LT(i, args_.size());
    return args_[i].get();
  }

  ExprPtr getOwnArg(const size_t i) const {
    CHECK_LT(i, args_.size());
    return args_[i];
  }

  ExprPtr deep_copy() const override;
  void collect_rte_idx(std::set<int>& rte_idx_set) const override;
  void collect_column_var(
      std::set<const ColumnVar*, bool (*)(const ColumnVar*, const ColumnVar*)>&
          colvar_set,
      bool include_agg) const override;

  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;

  size_t hash() const override;

 private:
  const std::string name_;
  const ExprPtrVector args_;
};

class FunctionOperWithCustomTypeHandling : public FunctionOper {
 public:
  FunctionOperWithCustomTypeHandling(const SQLTypeInfo& ti,
                                     const std::string& name,
                                     const ExprPtrVector& args)
      : FunctionOper(ti, name, args) {}

  ExprPtr deep_copy() const override;

  bool operator==(const Expr& rhs) const override;
};

/*
 * @type OffsetInFragment
 * @brief The offset of a row in the current fragment. To be used by updates.
 */
class OffsetInFragment : public Expr {
 public:
  OffsetInFragment() : Expr(SQLTypeInfo(kBIGINT, true)){};

  ExprPtr deep_copy() const override;

  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;
};

/*
 * @type OrderEntry
 * @brief represents an entry in ORDER BY clause.
 */
struct OrderEntry {
  OrderEntry(int t, bool d, bool nf) : tle_no(t), is_desc(d), nulls_first(nf){};
  ~OrderEntry() {}

  std::string toString() const;
  void print() const;

  size_t hash() const {
    size_t res = 0;
    boost::hash_combine(res, tle_no);
    boost::hash_combine(res, is_desc);
    boost::hash_combine(res, nulls_first);
    return res;
  }

  int tle_no;       /* targetlist entry number: 1-based */
  bool is_desc;     /* true if order is DESC */
  bool nulls_first; /* true if nulls are ordered first.  otherwise last. */
};

/*
 * @type WindowFunction
 * @brief A window function.
 */
class WindowFunction : public Expr {
 public:
  WindowFunction(const SQLTypeInfo& ti,
                 const SqlWindowFunctionKind kind,
                 const ExprPtrVector& args,
                 const ExprPtrVector& partition_keys,
                 const ExprPtrVector& order_keys,
                 const std::vector<OrderEntry>& collation)
      : Expr(ti)
      , kind_(kind)
      , args_(args)
      , partition_keys_(partition_keys)
      , order_keys_(order_keys)
      , collation_(collation){};

  ExprPtr deep_copy() const override;

  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;

  SqlWindowFunctionKind getKind() const { return kind_; }

  const ExprPtrVector& getArgs() const { return args_; }

  const ExprPtrVector& getPartitionKeys() const { return partition_keys_; }

  const ExprPtrVector& getOrderKeys() const { return order_keys_; }

  const std::vector<OrderEntry>& getCollation() const { return collation_; }

  size_t hash() const override;

 private:
  const SqlWindowFunctionKind kind_;
  const ExprPtrVector args_;
  const ExprPtrVector partition_keys_;
  const ExprPtrVector order_keys_;
  const std::vector<OrderEntry> collation_;
};

/*
 * @type ArrayExpr
 * @brief Corresponds to ARRAY[] statements in SQL
 */

class ArrayExpr : public Expr {
 public:
  ArrayExpr(SQLTypeInfo const& array_ti,
            ExprPtrVector const& array_exprs,
            bool is_null = false,
            bool local_alloc = false)
      : Expr(array_ti)
      , contained_expressions_(array_exprs)
      , local_alloc_(local_alloc)
      , is_null_(is_null) {}

  ExprPtr deep_copy() const override;
  std::string toString() const override;
  bool operator==(Expr const& rhs) const override;
  size_t getElementCount() const { return contained_expressions_.size(); }
  bool isLocalAlloc() const { return local_alloc_; }
  bool isNull() const { return is_null_; }

  const Expr* getElement(const size_t i) const {
    CHECK_LT(i, contained_expressions_.size());
    return contained_expressions_[i].get();
  }

  void collect_rte_idx(std::set<int>& rte_idx_set) const override;
  void collect_column_var(
      std::set<const ColumnVar*, bool (*)(const ColumnVar*, const ColumnVar*)>&
          colvar_set,
      bool include_agg) const override;

  size_t hash() const override;

 private:
  ExprPtrVector contained_expressions_;
  bool local_alloc_;
  bool is_null_;  // constant is NULL
};

/*
 * @type TargetEntry
 * @brief Target list defines a relational projection.  It is a list of TargetEntry's.
 */
class TargetEntry {
 public:
  TargetEntry(const std::string& n, ExprPtr e, bool u) : resname(n), expr(e), unnest(u) {}
  virtual ~TargetEntry() {}
  const std::string& get_resname() const { return resname; }
  void set_resname(const std::string& name) { resname = name; }
  Expr* get_expr() const { return expr.get(); }
  std::shared_ptr<Expr> get_own_expr() const { return expr; }
  void set_expr(ExprPtr e) { expr = e; }
  bool get_unnest() const { return unnest; }
  std::string toString() const;
  void print() const;

  size_t hash() const;

 private:
  std::string resname;  // alias name, e.g., SELECT salary + bonus AS compensation,
  ExprPtr expr;         // expression to evaluate for the value
  bool unnest;          // unnest a collection type
};

// Returns true iff the two expression lists are equal (same size and each element are
// equal).
bool expr_list_match(const ExprPtrVector& lhs, const ExprPtrVector& rhs);

}  // namespace hdk::ir
