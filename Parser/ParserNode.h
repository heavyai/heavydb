/*
 * Copyright 2017 MapD Technologies, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file    ParserNode.h
 * @author  Wei Hong <wei@map-d.com>
 * @brief   Classes representing a parse tree
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/
#ifndef PARSER_NODE_H_
#define PARSER_NODE_H_

#include <cstdint>
#include <cstring>
#include <list>
#include <string>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/process/search_path.hpp>

#include "../Analyzer/Analyzer.h"
#include "../Catalog/Catalog.h"
#include "../Distributed/AggregatedResult.h"
#include "../Shared/sqldefs.h"
#include "../Shared/sqltypes.h"
#include "ThriftHandler/QueryState.h"
#include "Utils/DdlUtils.h"

#include "../Fragmenter/InsertDataLoader.h"

#include <ImportExport/Importer.h>
#include <ImportExport/QueryExporter.h>

#include <functional>

namespace query_state {
class QueryState;
class QueryStateProxy;
}  // namespace query_state
using query_state::QueryStateProxy;

namespace Parser {

/*
 * @type Node
 * @brief root super class for all nodes in a pre-Analyzer
 * parse tree.
 */
class Node {
 public:
  virtual ~Node() {}
};

/*
 * @type SQLType
 * @brief class that captures type, predication and scale.
 */
class SQLType : public Node, public ddl_utils::SqlType {
 public:
  explicit SQLType(SQLTypes t) : ddl_utils::SqlType(t, -1, 0, false, -1) {}
  SQLType(SQLTypes t, int p1) : ddl_utils::SqlType(t, p1, 0, false, -1) {}
  SQLType(SQLTypes t, int p1, int p2, bool a) : ddl_utils::SqlType(t, p1, p2, a, -1) {}
  SQLType(SQLTypes t, int p1, int p2, bool a, int array_size)
      : ddl_utils::SqlType(t, p1, p2, a, array_size) {}
  SQLType(SQLTypes t, bool a, int array_size)
      : ddl_utils::SqlType(t, -1, 0, a, array_size) {}
};

/*
 * @type Expr
 * @brief root super class for all expression nodes
 */
class Expr : public Node {
 public:
  enum TlistRefType { TLIST_NONE, TLIST_REF, TLIST_COPY };
  /*
   * @brief Performs semantic analysis on the expression node
   * @param catalog The catalog object for the current database
   * @return An Analyzer::Expr object for the expression post semantic analysis
   */
  virtual std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const = 0;
  virtual std::string to_string() const = 0;
};

/*
 * @type Literal
 * @brief root super class for all literals
 */
class Literal : public Expr {
 public:
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override = 0;
  std::string to_string() const override = 0;
};

/*
 * @type NullLiteral
 * @brief the Literal NULL
 */
class NullLiteral : public Literal {
 public:
  NullLiteral() {}
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  std::string to_string() const override { return "NULL"; }
};

/*
 * @type StringLiteral
 * @brief the literal for string constants
 */
class StringLiteral : public Literal {
 public:
  explicit StringLiteral(std::string* s) : stringval_(s) {}
  const std::string* get_stringval() const { return stringval_.get(); }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  static std::shared_ptr<Analyzer::Expr> analyzeValue(const std::string&);
  std::string to_string() const override { return "'" + *stringval_ + "'"; }

 private:
  std::unique_ptr<std::string> stringval_;
};

/*
 * @type IntLiteral
 * @brief the literal for integer constants
 */
class IntLiteral : public Literal {
 public:
  explicit IntLiteral(int64_t i) : intval_(i) {}
  int64_t get_intval() const { return intval_; }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  static std::shared_ptr<Analyzer::Expr> analyzeValue(const int64_t intval);
  std::string to_string() const override {
    return boost::lexical_cast<std::string>(intval_);
  }

 private:
  int64_t intval_;
};

/*
 * @type FixedPtLiteral
 * @brief the literal for DECIMAL and NUMERIC
 */
class FixedPtLiteral : public Literal {
 public:
  explicit FixedPtLiteral(std::string* n) : fixedptval_(n) {}
  const std::string* get_fixedptval() const { return fixedptval_.get(); }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  static std::shared_ptr<Analyzer::Expr> analyzeValue(const int64_t numericval,
                                                      const int scale,
                                                      const int precision);
  std::string to_string() const override { return *fixedptval_; }

 private:
  std::unique_ptr<std::string> fixedptval_;
};

/*
 * @type FloatLiteral
 * @brief the literal for FLOAT or REAL
 */
class FloatLiteral : public Literal {
 public:
  explicit FloatLiteral(float f) : floatval_(f) {}
  float get_floatval() const { return floatval_; }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  std::string to_string() const override {
    return boost::lexical_cast<std::string>(floatval_);
  }

 private:
  float floatval_;
};

/*
 * @type DoubleLiteral
 * @brief the literal for DOUBLE PRECISION
 */
class DoubleLiteral : public Literal {
 public:
  explicit DoubleLiteral(double d) : doubleval_(d) {}
  double get_doubleval() const { return doubleval_; }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  std::string to_string() const override {
    return boost::lexical_cast<std::string>(doubleval_);
  }

 private:
  double doubleval_;
};

/*
 * @type TimestampLiteral
 * @brief the literal for Timestamp
 */
class TimestampLiteral : public Literal {
 public:
  explicit TimestampLiteral() { time(reinterpret_cast<time_t*>(&timestampval_)); }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  static std::shared_ptr<Analyzer::Expr> get(const int64_t);
  std::string to_string() const override {
    // TODO: Should we convert to a datum and use the datum toString converters to pretty
    // print?
    return boost::lexical_cast<std::string>(timestampval_);
  }

 private:
  int64_t timestampval_;
};

/*
 * @type UserLiteral
 * @brief the literal for USER
 */
class UserLiteral : public Literal {
 public:
  UserLiteral() {}
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  static std::shared_ptr<Analyzer::Expr> get(const std::string&);
  std::string to_string() const override { return "USER"; }
};

/*
 * @type ArrayLiteral
 * @brief the literal for arrays
 */
class ArrayLiteral : public Literal {
 public:
  ArrayLiteral() {}
  ArrayLiteral(std::list<Expr*>* v) {
    CHECK(v);
    for (const auto e : *v) {
      value_list_.emplace_back(e);
    }
    delete v;
  }
  const std::list<std::unique_ptr<Expr>>& get_value_list() const { return value_list_; }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  std::string to_string() const override;

 private:
  std::list<std::unique_ptr<Expr>> value_list_;
};

/*
 * @type OperExpr
 * @brief all operator expressions
 */
class OperExpr : public Expr {
 public:
  OperExpr(SQLOps t, Expr* l, Expr* r)
      : optype_(t), opqualifier_(kONE), left_(l), right_(r) {}
  OperExpr(SQLOps t, SQLQualifier q, Expr* l, Expr* r)
      : optype_(t), opqualifier_(q), left_(l), right_(r) {}
  SQLOps get_optype() const { return optype_; }
  const Expr* get_left() const { return left_.get(); }
  const Expr* get_right() const { return right_.get(); }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  static std::shared_ptr<Analyzer::Expr> normalize(
      const SQLOps optype,
      const SQLQualifier qual,
      std::shared_ptr<Analyzer::Expr> left_expr,
      std::shared_ptr<Analyzer::Expr> right_expr);
  std::string to_string() const override;

 private:
  SQLOps optype_;
  SQLQualifier opqualifier_;
  std::unique_ptr<Expr> left_;
  std::unique_ptr<Expr> right_;
};

// forward reference of QuerySpec
class QuerySpec;

/*
 * @type SubqueryExpr
 * @brief expression for subquery
 */
class SubqueryExpr : public Expr {
 public:
  explicit SubqueryExpr(QuerySpec* q) : query_(q) {}
  const QuerySpec* get_query() const { return query_.get(); }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  std::string to_string() const override;

 private:
  std::unique_ptr<QuerySpec> query_;
};

class IsNullExpr : public Expr {
 public:
  IsNullExpr(bool n, Expr* a) : is_not_(n), arg_(a) {}
  bool get_is_not() const { return is_not_; }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  std::string to_string() const override;

 private:
  bool is_not_;
  std::unique_ptr<Expr> arg_;
};

/*
 * @type InExpr
 * @brief expression for the IS NULL predicate
 */
class InExpr : public Expr {
 public:
  InExpr(bool n, Expr* a) : is_not_(n), arg_(a) {}
  bool get_is_not() const { return is_not_; }
  const Expr* get_arg() const { return arg_.get(); }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override = 0;
  std::string to_string() const override;

 protected:
  bool is_not_;
  std::unique_ptr<Expr> arg_;
};

/*
 * @type InSubquery
 * @brief expression for the IN (subquery) predicate
 */
class InSubquery : public InExpr {
 public:
  InSubquery(bool n, Expr* a, SubqueryExpr* q) : InExpr(n, a), subquery_(q) {}
  const SubqueryExpr* get_subquery() const { return subquery_.get(); }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  std::string to_string() const override;

 private:
  std::unique_ptr<SubqueryExpr> subquery_;
};

/*
 * @type InValues
 * @brief expression for IN (val1, val2, ...)
 */
class InValues : public InExpr {
 public:
  InValues(bool n, Expr* a, std::list<Expr*>* v) : InExpr(n, a) {
    CHECK(v);
    for (const auto e : *v) {
      value_list_.emplace_back(e);
    }
    delete v;
  }
  const std::list<std::unique_ptr<Expr>>& get_value_list() const { return value_list_; }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  std::string to_string() const override;

 private:
  std::list<std::unique_ptr<Expr>> value_list_;
};

/*
 * @type BetweenExpr
 * @brief expression for BETWEEN lower AND upper
 */
class BetweenExpr : public Expr {
 public:
  BetweenExpr(bool n, Expr* a, Expr* l, Expr* u)
      : is_not_(n), arg_(a), lower_(l), upper_(u) {}
  bool get_is_not() const { return is_not_; }
  const Expr* get_arg() const { return arg_.get(); }
  const Expr* get_lower() const { return lower_.get(); }
  const Expr* get_upper() const { return upper_.get(); }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  std::string to_string() const override;

 private:
  bool is_not_;
  std::unique_ptr<Expr> arg_;
  std::unique_ptr<Expr> lower_;
  std::unique_ptr<Expr> upper_;
};

/*
 * @type CharLengthExpr
 * @brief expression to get length of string
 */

class CharLengthExpr : public Expr {
 public:
  CharLengthExpr(Expr* a, bool e) : arg_(a), calc_encoded_length_(e) {}
  const Expr* get_arg() const { return arg_.get(); }
  bool get_calc_encoded_length() const { return calc_encoded_length_; }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  std::string to_string() const override;

 private:
  std::unique_ptr<Expr> arg_;
  bool calc_encoded_length_;
};

/*
 * @type CardinalityExpr
 * @brief expression to get cardinality of an array
 */

class CardinalityExpr : public Expr {
 public:
  CardinalityExpr(Expr* a) : arg_(a) {}
  const Expr* get_arg() const { return arg_.get(); }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  std::string to_string() const override;

 private:
  std::unique_ptr<Expr> arg_;
};

/*
 * @type LikeExpr
 * @brief expression for the LIKE predicate
 */
class LikeExpr : public Expr {
 public:
  LikeExpr(bool n, bool i, Expr* a, Expr* l, Expr* e)
      : is_not_(n), is_ilike_(i), arg_(a), like_string_(l), escape_string_(e) {}
  bool get_is_not() const { return is_not_; }
  const Expr* get_arg() const { return arg_.get(); }
  const Expr* get_like_string() const { return like_string_.get(); }
  const Expr* get_escape_string() const { return escape_string_.get(); }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  static std::shared_ptr<Analyzer::Expr> get(std::shared_ptr<Analyzer::Expr> arg_expr,
                                             std::shared_ptr<Analyzer::Expr> like_expr,
                                             std::shared_ptr<Analyzer::Expr> escape_expr,
                                             const bool is_ilike,
                                             const bool is_not);
  std::string to_string() const override;

 private:
  bool is_not_;
  bool is_ilike_;
  std::unique_ptr<Expr> arg_;
  std::unique_ptr<Expr> like_string_;
  std::unique_ptr<Expr> escape_string_;

  static void check_like_expr(const std::string& like_str, char escape_char);
  static bool test_is_simple_expr(const std::string& like_str, char escape_char);
  static void erase_cntl_chars(std::string& like_str, char escape_char);
};

/*
 * @type RegexpExpr
 * @brief expression for REGEXP
 */
class RegexpExpr : public Expr {
 public:
  RegexpExpr(bool n, Expr* a, Expr* p, Expr* e)
      : is_not_(n), arg_(a), pattern_string_(p), escape_string_(e) {}
  bool get_is_not() const { return is_not_; }
  const Expr* get_arg() const { return arg_.get(); }
  const Expr* get_pattern_string() const { return pattern_string_.get(); }
  const Expr* get_escape_string() const { return escape_string_.get(); }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  static std::shared_ptr<Analyzer::Expr> get(std::shared_ptr<Analyzer::Expr> arg_expr,
                                             std::shared_ptr<Analyzer::Expr> pattern_expr,
                                             std::shared_ptr<Analyzer::Expr> escape_expr,
                                             const bool is_not);
  std::string to_string() const override;

 private:
  bool is_not_;
  std::unique_ptr<Expr> arg_;
  std::unique_ptr<Expr> pattern_string_;
  std::unique_ptr<Expr> escape_string_;

  static void check_pattern_expr(const std::string& pattern_str, char escape_char);
  static bool translate_to_like_pattern(std::string& pattern_str, char escape_char);
};

class WidthBucketExpr : public Expr {
 public:
  WidthBucketExpr(Expr* t, Expr* l, Expr* u, Expr* p)
      : target_value_(t), lower_bound_(l), upper_bound_(u), partition_count_(p) {}
  const Expr* get_target_value() const { return target_value_.get(); }
  const Expr* get_lower_bound() const { return lower_bound_.get(); }
  const Expr* get_upper_bound() const { return upper_bound_.get(); }
  const Expr* get_partition_count() const { return partition_count_.get(); }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  static std::shared_ptr<Analyzer::Expr> get(
      std::shared_ptr<Analyzer::Expr> target_value,
      std::shared_ptr<Analyzer::Expr> lower_bound,
      std::shared_ptr<Analyzer::Expr> upper_bound,
      std::shared_ptr<Analyzer::Expr> partition_count);
  std::string to_string() const override;

 private:
  std::unique_ptr<Expr> target_value_;
  std::unique_ptr<Expr> lower_bound_;
  std::unique_ptr<Expr> upper_bound_;
  std::unique_ptr<Expr> partition_count_;
};

/*
 * @type LikelihoodExpr
 * @brief expression for LIKELY, UNLIKELY
 */
class LikelihoodExpr : public Expr {
 public:
  LikelihoodExpr(bool n, Expr* a, float l) : is_not_(n), arg_(a), likelihood_(l) {}
  bool get_is_not() const { return is_not_; }
  const Expr* get_arg() const { return arg_.get(); }
  float get_likelihood() const { return likelihood_; }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  static std::shared_ptr<Analyzer::Expr> get(std::shared_ptr<Analyzer::Expr> arg_expr,
                                             float likelihood,
                                             const bool is_not);
  std::string to_string() const override;

 private:
  bool is_not_;
  std::unique_ptr<Expr> arg_;
  float likelihood_;
};

/*
 * @type ExistsExpr
 * @brief expression for EXISTS (subquery)
 */
class ExistsExpr : public Expr {
 public:
  explicit ExistsExpr(QuerySpec* q) : query_(q) {}
  const QuerySpec* get_query() const { return query_.get(); }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  std::string to_string() const override;

 private:
  std::unique_ptr<QuerySpec> query_;
};

/*
 * @type ColumnRef
 * @brief expression for a column reference
 */
class ColumnRef : public Expr {
 public:
  explicit ColumnRef(std::string* n1) : table_(nullptr), column_(n1) {}
  ColumnRef(std::string* n1, std::string* n2) : table_(n1), column_(n2) {}
  const std::string* get_table() const { return table_.get(); }
  const std::string* get_column() const { return column_.get(); }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  std::string to_string() const override;

 private:
  std::unique_ptr<std::string> table_;
  std::unique_ptr<std::string> column_;  // can be nullptr in the t.* case
};

/*
 * @type FunctionRef
 * @brief expression for a function call
 */
class FunctionRef : public Expr {
 public:
  explicit FunctionRef(std::string* n) : name_(n), distinct_(false), arg_(nullptr) {}
  FunctionRef(std::string* n, Expr* a) : name_(n), distinct_(false), arg_(a) {}
  FunctionRef(std::string* n, bool d, Expr* a) : name_(n), distinct_(d), arg_(a) {}
  const std::string* get_name() const { return name_.get(); }
  bool get_distinct() const { return distinct_; }
  Expr* get_arg() const { return arg_.get(); }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  std::string to_string() const override;

 private:
  std::unique_ptr<std::string> name_;
  bool distinct_;              // only true for COUNT(DISTINCT x)
  std::unique_ptr<Expr> arg_;  // for COUNT, nullptr means '*'
};

class CastExpr : public Expr {
 public:
  CastExpr(Expr* a, SQLType* t) : arg_(a), target_type_(t) {}
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  std::string to_string() const override {
    return "CAST(" + arg_->to_string() + " AS " + target_type_->to_string() + ")";
  }

 private:
  std::unique_ptr<Expr> arg_;
  std::unique_ptr<SQLType> target_type_;
};

class ExprPair : public Node {
 public:
  ExprPair(Expr* e1, Expr* e2) : expr1_(e1), expr2_(e2) {}
  const Expr* get_expr1() const { return expr1_.get(); }
  const Expr* get_expr2() const { return expr2_.get(); }

 private:
  std::unique_ptr<Expr> expr1_;
  std::unique_ptr<Expr> expr2_;
};

class CaseExpr : public Expr {
 public:
  CaseExpr(std::list<ExprPair*>* w, Expr* e) : else_expr_(e) {
    CHECK(w);
    for (const auto e : *w) {
      when_then_list_.emplace_back(e);
    }
    delete w;
  }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  static std::shared_ptr<Analyzer::Expr> normalize(
      const std::list<
          std::pair<std::shared_ptr<Analyzer::Expr>, std::shared_ptr<Analyzer::Expr>>>&,
      const std::shared_ptr<Analyzer::Expr>);
  std::string to_string() const override;

 private:
  std::list<std::unique_ptr<ExprPair>> when_then_list_;
  std::unique_ptr<Expr> else_expr_;
};

/*
 * @type TableRef
 * @brief table reference in FROM clause
 */
class TableRef : public Node {
 public:
  explicit TableRef(std::string* t) : table_name_(t), range_var_(nullptr) {}
  TableRef(std::string* t, std::string* r) : table_name_(t), range_var_(r) {}
  const std::string* get_table_name() const { return table_name_.get(); }
  const std::string* get_range_var() const { return range_var_.get(); }
  std::string to_string() const;

 private:
  std::unique_ptr<std::string> table_name_;
  std::unique_ptr<std::string> range_var_;
};

/*
 * @type Stmt
 * @brief root super class for all SQL statements
 */
class Stmt : public Node {
  // intentionally empty
};

/*
 * @type DMLStmt
 * @brief DML Statements
 */
class DMLStmt : public Stmt {
 public:
  virtual void analyze(const Catalog_Namespace::Catalog& catalog,
                       Analyzer::Query& query) const = 0;
};

/*
 * @type DDLStmt
 * @brief DDL Statements
 */
class ColumnDef;
class DDLStmt : public Stmt {
 public:
  virtual void execute(const Catalog_Namespace::SessionInfo& session) = 0;
  void setColumnDescriptor(ColumnDescriptor& cd, const ColumnDef* coldef);
};

/*
 * @type TableElement
 * @brief elements in table definition
 */
class TableElement : public Node {
  // intentionally empty
};

/*
 * @type ColumnConstraintDef
 * @brief integrity constraint on a column
 */
class ColumnConstraintDef : public Node {
 public:
  ColumnConstraintDef(bool n, bool u, bool p, Literal* d)
      : notnull_(n), unique_(u), is_primarykey_(p), defaultval_(d) {}
  ColumnConstraintDef(Expr* c)
      : notnull_(false), unique_(false), is_primarykey_(false), check_condition_(c) {}
  ColumnConstraintDef(std::string* t, std::string* c)
      : notnull_(false)
      , unique_(false)
      , is_primarykey_(false)
      , foreign_table_(t)
      , foreign_column_(c) {}
  bool get_notnull() const { return notnull_; }
  bool get_unique() const { return unique_; }
  bool get_is_primarykey() const { return is_primarykey_; }
  const Literal* get_defaultval() const { return defaultval_.get(); }
  const Expr* get_check_condition() const { return check_condition_.get(); }
  const std::string* get_foreign_table() const { return foreign_table_.get(); }
  const std::string* get_foreign_column() const { return foreign_column_.get(); }

 private:
  bool notnull_;
  bool unique_;
  bool is_primarykey_;
  std::unique_ptr<Literal> defaultval_;
  std::unique_ptr<Expr> check_condition_;
  std::unique_ptr<std::string> foreign_table_;
  std::unique_ptr<std::string> foreign_column_;
};

/*
 * @type CompressDef
 * @brief Node for compression scheme definition
 */
class CompressDef : public Node, public ddl_utils::Encoding {
 public:
  CompressDef(std::string* n, int p) : ddl_utils::Encoding(n, p) {}
};

/*
 * @type ColumnDef
 * @brief Column definition
 */
class ColumnDef : public TableElement {
 public:
  ColumnDef(std::string* c, SQLType* t, CompressDef* cp, ColumnConstraintDef* cc)
      : column_name_(c), column_type_(t), compression_(cp), column_constraint_(cc) {}
  const std::string* get_column_name() const { return column_name_.get(); }
  SQLType* get_column_type() const { return column_type_.get(); }
  const CompressDef* get_compression() const { return compression_.get(); }
  const ColumnConstraintDef* get_column_constraint() const {
    return column_constraint_.get();
  }

 private:
  std::unique_ptr<std::string> column_name_;
  std::unique_ptr<SQLType> column_type_;
  std::unique_ptr<CompressDef> compression_;
  std::unique_ptr<ColumnConstraintDef> column_constraint_;
};

/*
 * @type TableConstraintDef
 * @brief integrity constraint for table
 */
class TableConstraintDef : public TableElement {
  // intentionally empty
};

/*
 * @type UniqueDef
 * @brief uniqueness constraint
 */
class UniqueDef : public TableConstraintDef {
 public:
  UniqueDef(bool p, std::list<std::string*>* cl) : is_primarykey_(p) {
    CHECK(cl);
    for (const auto s : *cl) {
      column_list_.emplace_back(s);
    }
    delete cl;
  }
  bool get_is_primarykey() const { return is_primarykey_; }
  const std::list<std::unique_ptr<std::string>>& get_column_list() const {
    return column_list_;
  }

 private:
  bool is_primarykey_;
  std::list<std::unique_ptr<std::string>> column_list_;
};

/*
 * @type ForeignKeyDef
 * @brief foreign key constraint
 */
class ForeignKeyDef : public TableConstraintDef {
 public:
  ForeignKeyDef(std::list<std::string*>* cl, std::string* t, std::list<std::string*>* fcl)
      : foreign_table_(t) {
    CHECK(cl);
    for (const auto s : *cl) {
      column_list_.emplace_back(s);
    }
    delete cl;
    if (fcl) {
      for (const auto s : *fcl) {
        foreign_column_list_.emplace_back(s);
      }
    }
    delete fcl;
  }
  const std::list<std::unique_ptr<std::string>>& get_column_list() const {
    return column_list_;
  }
  const std::string* get_foreign_table() const { return foreign_table_.get(); }
  const std::list<std::unique_ptr<std::string>>& get_foreign_column_list() const {
    return foreign_column_list_;
  }

 private:
  std::list<std::unique_ptr<std::string>> column_list_;
  std::unique_ptr<std::string> foreign_table_;
  std::list<std::unique_ptr<std::string>> foreign_column_list_;
};

/*
 * @type CheckDef
 * @brief Check constraint
 */
class CheckDef : public TableConstraintDef {
 public:
  CheckDef(Expr* c) : check_condition_(c) {}
  const Expr* get_check_condition() const { return check_condition_.get(); }

 private:
  std::unique_ptr<Expr> check_condition_;
};

/*
 * @type SharedDictionaryDef
 * @brief Shared dictionary hint. The underlying string dictionary will be shared with the
 * referenced column.
 */
class SharedDictionaryDef : public TableConstraintDef {
 public:
  SharedDictionaryDef(const std::string& column,
                      const std::string& foreign_table,
                      const std::string foreign_column)
      : column_(column), foreign_table_(foreign_table), foreign_column_(foreign_column) {}

  const std::string& get_column() const { return column_; }

  const std::string& get_foreign_table() const { return foreign_table_; }

  const std::string& get_foreign_column() const { return foreign_column_; }

 private:
  const std::string column_;
  const std::string foreign_table_;
  const std::string foreign_column_;
};

/*
 * @type ShardKeyDef
 * @brief Shard key for a table, mainly useful for distributed joins.
 */
class ShardKeyDef : public TableConstraintDef {
 public:
  ShardKeyDef(const std::string& column) : column_(column) {}

  const std::string& get_column() const { return column_; }

 private:
  const std::string column_;
};

/*
 * @type NameValueAssign
 * @brief Assignment of a string value to a named attribute
 */
class NameValueAssign : public Node {
 public:
  NameValueAssign(std::string* n, Literal* v) : name_(n), value_(v) {}
  const std::string* get_name() const { return name_.get(); }
  const Literal* get_value() const { return value_.get(); }

 private:
  std::unique_ptr<std::string> name_;
  std::unique_ptr<Literal> value_;
};

/*
 * @type CreateTableBaseStmt
 */
class CreateTableBaseStmt : public DDLStmt {
 public:
  virtual const std::string* get_table() const = 0;
  virtual const std::list<std::unique_ptr<TableElement>>& get_table_element_list()
      const = 0;
};

/*
 * @type CreateTableStmt
 * @brief CREATE TABLE statement
 */
class CreateTableStmt : public CreateTableBaseStmt {
 public:
  CreateTableStmt(std::string* tab,
                  const std::string* storage,
                  std::list<TableElement*>* table_elems,
                  bool is_temporary,
                  bool if_not_exists,
                  std::list<NameValueAssign*>* s)
      : table_(tab), is_temporary_(is_temporary), if_not_exists_(if_not_exists) {
    CHECK(table_elems);
    for (const auto e : *table_elems) {
      table_element_list_.emplace_back(e);
    }
    delete table_elems;
    if (s) {
      for (const auto e : *s) {
        storage_options_.emplace_back(e);
      }
      delete s;
    }
  }

  CreateTableStmt(const rapidjson::Value& payload);

  const std::string* get_table() const override { return table_.get(); }
  const std::list<std::unique_ptr<TableElement>>& get_table_element_list()
      const override {
    return table_element_list_;
  }

  void execute(const Catalog_Namespace::SessionInfo& session) override;
  void executeDryRun(const Catalog_Namespace::SessionInfo& session,
                     TableDescriptor& td,
                     std::list<ColumnDescriptor>& columns,
                     std::vector<SharedDictionaryDef>& shared_dict_defs);

 private:
  std::unique_ptr<std::string> table_;
  std::list<std::unique_ptr<TableElement>> table_element_list_;
  bool is_temporary_;
  bool if_not_exists_;
  std::list<std::unique_ptr<NameValueAssign>> storage_options_;
};

struct DistributedConnector
    : public Fragmenter_Namespace::InsertDataLoader::DistributedConnector {
  virtual ~DistributedConnector() {}

  virtual size_t getOuterFragmentCount(QueryStateProxy,
                                       std::string& sql_query_string) = 0;
  virtual std::vector<AggregatedResult> query(QueryStateProxy,
                                              std::string& sql_query_string,
                                              std::vector<size_t> outer_frag_indices,
                                              bool allow_interrupt) = 0;
};

struct LocalConnector : public DistributedConnector {
  virtual ~LocalConnector() {}

  size_t getOuterFragmentCount(QueryStateProxy, std::string& sql_query_string) override;

  AggregatedResult query(QueryStateProxy,
                         std::string& sql_query_string,
                         std::vector<size_t> outer_frag_indices,
                         bool validate_only,
                         bool allow_interrupt);
  std::vector<AggregatedResult> query(QueryStateProxy,
                                      std::string& sql_query_string,
                                      std::vector<size_t> outer_frag_indices,
                                      bool allow_interrupt) override;
  size_t leafCount() override { return 1; };
  void insertDataToLeaf(const Catalog_Namespace::SessionInfo& session,
                        const size_t leaf_idx,
                        Fragmenter_Namespace::InsertData& insert_data) override;
  void checkpoint(const Catalog_Namespace::SessionInfo& session, int tableId) override;
  void rollback(const Catalog_Namespace::SessionInfo& session, int tableId) override;
  std::list<ColumnDescriptor> getColumnDescriptors(AggregatedResult& result,
                                                   bool for_create);
};

/*
 * @type CreateDataframeStmt
 * @brief CREATE DATAFRAME statement
 */
class CreateDataframeStmt : public CreateTableBaseStmt {
 public:
  CreateDataframeStmt(std::string* tab,
                      std::list<TableElement*>* table_elems,
                      std::string* filename,
                      std::list<NameValueAssign*>* s)
      : table_(tab), filename_(filename) {
    CHECK(table_elems);
    for (const auto e : *table_elems) {
      table_element_list_.emplace_back(e);
    }
    delete table_elems;
    if (s) {
      for (const auto e : *s) {
        storage_options_.emplace_back(e);
      }
      delete s;
    }
  }
  CreateDataframeStmt(const rapidjson::Value& payload);

  const std::string* get_table() const override { return table_.get(); }
  const std::list<std::unique_ptr<TableElement>>& get_table_element_list()
      const override {
    return table_element_list_;
  }

  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::unique_ptr<std::string> table_;
  std::list<std::unique_ptr<TableElement>> table_element_list_;
  std::unique_ptr<std::string> filename_;
  std::list<std::unique_ptr<NameValueAssign>> storage_options_;
};

/*
 * @type InsertIntoTableAsSelectStmt
 * @brief INSERT INTO TABLE SELECT statement
 */
class InsertIntoTableAsSelectStmt : public DDLStmt {
 public:
  // ITAS constructor
  InsertIntoTableAsSelectStmt(const rapidjson::Value& payload);
  InsertIntoTableAsSelectStmt(const std::string* table_name,
                              const std::string* select_query,
                              std::list<std::string*>* c)
      : table_name_(*table_name), select_query_(*select_query) {
    if (c) {
      for (auto e : *c) {
        column_list_.emplace_back(e);
      }
      delete c;
    }

    delete table_name;
    delete select_query;
  }

  void populateData(QueryStateProxy,
                    const TableDescriptor* td,
                    bool validate_table,
                    bool for_CTAS = false);
  void execute(const Catalog_Namespace::SessionInfo& session) override;

  std::string& get_table() { return table_name_; }

  std::string& get_select_query() { return select_query_; }

  DistributedConnector* leafs_connector_ = nullptr;

 protected:
  std::vector<std::unique_ptr<std::string>> column_list_;
  std::string table_name_;
  std::string select_query_;
};

/*
 * @type CreateTableAsSelectStmt
 * @brief CREATE TABLE AS SELECT statement
 */
class CreateTableAsSelectStmt : public InsertIntoTableAsSelectStmt {
 public:
  CreateTableAsSelectStmt(const rapidjson::Value& payload);
  CreateTableAsSelectStmt(const std::string* table_name,
                          const std::string* select_query,
                          const bool is_temporary,
                          const bool if_not_exists,
                          std::list<NameValueAssign*>* s)
      : InsertIntoTableAsSelectStmt(table_name, select_query, nullptr)
      , is_temporary_(is_temporary)
      , if_not_exists_(if_not_exists) {
    if (s) {
      for (const auto& e : *s) {
        storage_options_.emplace_back(e);
      }
      delete s;
    }
  }

  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  bool is_temporary_;
  bool if_not_exists_;
  std::list<std::unique_ptr<NameValueAssign>> storage_options_;
};

/*
 * @type AlterTableStmt
 * @brief ALTER TABLE statement
 *
 * AlterTableStmt is more a composite Stmt in that it relies upon several other Stmts to
 * handle the execution.
 */

class AlterTableStmt : public DDLStmt {
 public:
  static std::unique_ptr<Parser::DDLStmt> delegate(const rapidjson::Value& payload);

  const std::string* get_table() const { return table_.get(); }

  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::unique_ptr<std::string> table_;
  const rapidjson::Value payload_;
};

/*
 * @type DropTableStmt
 * @brief DROP TABLE statement
 */
class DropTableStmt : public DDLStmt {
 public:
  DropTableStmt(std::string* tab, bool i) : table_(tab), if_exists_(i) {}
  DropTableStmt(const rapidjson::Value& payload);

  const std::string* get_table() const { return table_.get(); }
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::unique_ptr<std::string> table_;
  bool if_exists_;
};

/*
 * @type TruncateTableStmt
 * @brief TRUNCATE TABLE statement
 */
class TruncateTableStmt : public DDLStmt {
 public:
  TruncateTableStmt(std::string* tab) : table_(tab) {}
  TruncateTableStmt(const rapidjson::Value& payload);
  const std::string* get_table() const { return table_.get(); }
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::unique_ptr<std::string> table_;
};

class OptimizeTableStmt : public DDLStmt {
 public:
  OptimizeTableStmt(std::string* table, std::list<NameValueAssign*>* o) : table_(table) {
    if (!table_) {
      throw std::runtime_error("Table name is required for OPTIMIZE command.");
    }
    if (o) {
      for (const auto e : *o) {
        options_.emplace_back(e);
      }
      delete o;
    }
  }
  OptimizeTableStmt(const rapidjson::Value& payload);

  const std::string getTableName() const { return *(table_.get()); }

  bool shouldVacuumDeletedRows() const {
    for (const auto& e : options_) {
      if (boost::iequals(*(e->get_name()), "VACUUM")) {
        return true;
      }
    }
    return false;
  }

  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::unique_ptr<std::string> table_;
  std::list<std::unique_ptr<NameValueAssign>> options_;
};

class ValidateStmt : public DDLStmt {
 public:
  ValidateStmt(std::string* type, std::list<NameValueAssign*>* with_opts);
  ValidateStmt(const rapidjson::Value& payload);

  bool isRepairTypeRemove() const { return isRepairTypeRemove_; }

  const std::string getType() const { return *(type_.get()); }

  void execute(const Catalog_Namespace::SessionInfo& session) override { UNREACHABLE(); }

 private:
  std::unique_ptr<std::string> type_;
  bool isRepairTypeRemove_ = false;
};

class RenameDBStmt : public DDLStmt {
 public:
  RenameDBStmt(const rapidjson::Value& payload);

  RenameDBStmt(std::string* database_name, std::string* new_database_name)
      : database_name_(database_name), new_database_name_(new_database_name) {}

  auto const& getPreviousDatabaseName() { return database_name_; }
  auto const& getNewDatabaseName() { return new_database_name_; }
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::unique_ptr<std::string> database_name_;
  std::unique_ptr<std::string> new_database_name_;
};

class RenameUserStmt : public DDLStmt {
 public:
  RenameUserStmt(const rapidjson::Value& payload);
  RenameUserStmt(std::string* username, std::string* new_username)
      : username_(username), new_username_(new_username) {}
  auto const& getOldUserName() { return username_; }
  auto const& getNewUserName() { return new_username_; }
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::unique_ptr<std::string> username_;
  std::unique_ptr<std::string> new_username_;
};

class RenameTableStmt : public DDLStmt {
 public:
  using TableNamePair =
      std::pair<std::unique_ptr<std::string>, std::unique_ptr<std::string>>;

  // when created via ddl
  RenameTableStmt(const rapidjson::Value& payload);

  // to rename a single table
  RenameTableStmt(std::string* tab_name, std::string* new_tab_name);

  // to rename multiple tables
  RenameTableStmt(std::list<std::pair<std::string, std::string>> tableNames);

  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::list<TableNamePair> tablesToRename_;
};

class RenameColumnStmt : public DDLStmt {
 public:
  RenameColumnStmt(std::string* tab, std::string* col, std::string* new_col_name)
      : table_(tab), column_(col), new_column_name_(new_col_name) {}
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::unique_ptr<std::string> table_;
  std::unique_ptr<std::string> column_;
  std::unique_ptr<std::string> new_column_name_;
};

class AddColumnStmt : public DDLStmt {
 public:
  AddColumnStmt(std::string* tab, ColumnDef* coldef) : table_(tab), coldef_(coldef) {}
  AddColumnStmt(std::string* tab, std::list<ColumnDef*>* coldefs) : table_(tab) {
    for (const auto coldef : *coldefs) {
      this->coldefs_.emplace_back(coldef);
    }
    delete coldefs;
  }
  void execute(const Catalog_Namespace::SessionInfo& session) override;
  void check_executable(const Catalog_Namespace::SessionInfo& session,
                        const TableDescriptor* td);
  const std::string* get_table() const { return table_.get(); }

 private:
  std::unique_ptr<std::string> table_;
  std::unique_ptr<ColumnDef> coldef_;
  std::list<std::unique_ptr<ColumnDef>> coldefs_;
};

class DropColumnStmt : public DDLStmt {
 public:
  DropColumnStmt(std::string* tab, std::list<std::string*>* cols) : table_(tab) {
    for (const auto col : *cols) {
      this->columns_.emplace_back(col);
    }
    delete cols;
  }
  void execute(const Catalog_Namespace::SessionInfo& session) override;
  const std::string* get_table() const { return table_.get(); }

 private:
  std::unique_ptr<std::string> table_;
  std::list<std::unique_ptr<std::string>> columns_;
};

class AlterTableParamStmt : public DDLStmt {
 public:
  AlterTableParamStmt(std::string* tab, NameValueAssign* p) : table_(tab), param_(p) {}
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::unique_ptr<std::string> table_;
  std::unique_ptr<NameValueAssign> param_;
};

/*
 * @type DumpTableStmt
 * @brief DUMP TABLE table TO archive_file_path
 */
class DumpRestoreTableStmtBase : public DDLStmt {
 public:
  DumpRestoreTableStmtBase(std::string* tab,
                           std::string* path,
                           std::list<NameValueAssign*>* options,
                           const bool is_restore)
      : table_(tab), path_(path) {
    for (const auto& program : {"tar", "rm", "mkdir", "mv", "cat"}) {
      if (boost::process::search_path(program).empty()) {
        throw std::runtime_error{"Required program \"" + std::string{program} +
                                 "\" was not found."};
      }
    }
    auto options_deleter = [](std::list<NameValueAssign*>* options) {
      for (auto option : *options) {
        delete option;
      }
      delete options;
    };

    std::unique_ptr<std::list<NameValueAssign*>, decltype(options_deleter)> options_ptr(
        options, options_deleter);
    // specialize decompressor or break on osx bsdtar...
    if (options) {
      for (const auto option : *options) {
        if (boost::iequals(*option->get_name(), "compression")) {
          if (const auto str_literal =
                  dynamic_cast<const StringLiteral*>(option->get_value())) {
            compression_ = *str_literal->get_stringval();
            validateCompression(compression_, is_restore);
          } else {
            throw std::runtime_error("Compression option must be a string.");
          }
        } else {
          throw std::runtime_error("Invalid WITH option: " + *option->get_name());
        }
      }
    }
  }

  DumpRestoreTableStmtBase(const rapidjson::Value& payload, const bool is_restore);

  void validateCompression(std::string& compression, const bool is_restore);

  const std::string* getTable() const { return table_.get(); }
  const std::string* getPath() const { return path_.get(); }
  const std::string getCompression() const { return compression_; }

 protected:
  std::unique_ptr<std::string> table_;
  std::unique_ptr<std::string> path_;  // dump TO file path
  std::string compression_;
};

class DumpTableStmt : public DumpRestoreTableStmtBase {
 public:
  DumpTableStmt(std::string* tab, std::string* path, std::list<NameValueAssign*>* options)
      : DumpRestoreTableStmtBase(tab, path, options, false) {}
  DumpTableStmt(const rapidjson::Value& payload);
  void execute(const Catalog_Namespace::SessionInfo& session) override;
};

/*
 * @type RestoreTableStmt
 * @brief RESTORE TABLE table FROM archive_file_path
 */
class RestoreTableStmt : public DumpRestoreTableStmtBase {
 public:
  RestoreTableStmt(std::string* tab,
                   std::string* path,
                   std::list<NameValueAssign*>* options)
      : DumpRestoreTableStmtBase(tab, path, options, true) {}
  RestoreTableStmt(const rapidjson::Value& payload);
  void execute(const Catalog_Namespace::SessionInfo& session) override;
};

/*
 * @type CopyTableStmt
 * @brief COPY ... FROM ...
 */
class CopyTableStmt : public DDLStmt {
 public:
  CopyTableStmt(std::string* t, std::string* f, std::list<NameValueAssign*>* o);
  CopyTableStmt(const rapidjson::Value& payload);

  void execute(const Catalog_Namespace::SessionInfo& session) override;
  void execute(const Catalog_Namespace::SessionInfo& session,
               const std::function<std::unique_ptr<import_export::AbstractImporter>(
                   Catalog_Namespace::Catalog&,
                   const TableDescriptor*,
                   const std::string&,
                   const import_export::CopyParams&)>& importer_factory);
  std::unique_ptr<std::string> return_message;

  std::string& get_table() const {
    CHECK(table_);
    return *table_;
  }

  bool get_success() const { return success_; }

  bool was_geo_copy_from() const { return was_geo_copy_from_; }

  void get_geo_copy_from_payload(std::string& geo_copy_from_table,
                                 std::string& geo_copy_from_file_name,
                                 import_export::CopyParams& geo_copy_from_copy_params,
                                 std::string& geo_copy_from_partitions) {
    geo_copy_from_table = *table_;
    geo_copy_from_file_name = geo_copy_from_file_name_;
    geo_copy_from_copy_params = geo_copy_from_copy_params_;
    geo_copy_from_partitions = geo_copy_from_partitions_;
    was_geo_copy_from_ = false;
  }

 private:
  std::unique_ptr<std::string> table_;
  std::unique_ptr<std::string> file_pattern_;
  bool success_;
  std::list<std::unique_ptr<NameValueAssign>> options_;

  bool was_geo_copy_from_ = false;
  std::string geo_copy_from_file_name_;
  import_export::CopyParams geo_copy_from_copy_params_;
  std::string geo_copy_from_partitions_;
};

/*
 * @type CreateRoleStmt
 * @brief CREATE ROLE statement
 */
class CreateRoleStmt : public DDLStmt {
 public:
  CreateRoleStmt(std::string* r) : role_(r) {}
  CreateRoleStmt(const rapidjson::Value& payload);
  const std::string& get_role() const { return *role_; }
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::unique_ptr<std::string> role_;
};

/*
 * @type DropRoleStmt
 * @brief DROP ROLE statement
 */
class DropRoleStmt : public DDLStmt {
 public:
  DropRoleStmt(std::string* r) : role_(r) {}
  DropRoleStmt(const rapidjson::Value& payload);
  const std::string& get_role() const { return *role_; }
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::unique_ptr<std::string> role_;
};

inline void parser_slistval_to_vector(std::list<std::string*>* l,
                                      std::vector<std::string>& v) {
  CHECK(l);
  for (auto str : *l) {
    v.push_back(*str);
    delete str;
  }
  delete l;
}

/*
 * @type GrantPrivilegesStmt
 * @brief GRANT PRIVILEGES statement
 */
class GrantPrivilegesStmt : public DDLStmt {
 public:
  GrantPrivilegesStmt(std::list<std::string*>* p,
                      std::string* t,
                      std::string* o,
                      std::list<std::string*>* g)
      : type_(t), target_(o) {
    parser_slistval_to_vector(p, privileges_);
    parser_slistval_to_vector(g, grantees_);
  }
  GrantPrivilegesStmt(const rapidjson::Value& payload);

  const std::vector<std::string>& get_privs() const { return privileges_; }
  const std::string& get_object_type() const { return *type_; }
  const std::string& get_object() const { return *target_; }
  const std::vector<std::string>& get_grantees() const { return grantees_; }
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::vector<std::string> privileges_;
  std::unique_ptr<std::string> type_;
  std::unique_ptr<std::string> target_;
  std::vector<std::string> grantees_;
};

/*
 * @type RevokePrivilegesStmt
 * @brief REVOKE PRIVILEGES statement
 */
class RevokePrivilegesStmt : public DDLStmt {
 public:
  RevokePrivilegesStmt(std::list<std::string*>* p,
                       std::string* t,
                       std::string* o,
                       std::list<std::string*>* g)
      : type_(t), target_(o) {
    parser_slistval_to_vector(p, privileges_);
    parser_slistval_to_vector(g, grantees_);
  }
  RevokePrivilegesStmt(const rapidjson::Value& payload);

  const std::vector<std::string>& get_privs() const { return privileges_; }
  const std::string& get_object_type() const { return *type_; }
  const std::string& get_object() const { return *target_; }
  const std::vector<std::string>& get_grantees() const { return grantees_; }
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::vector<std::string> privileges_;
  std::unique_ptr<std::string> type_;
  std::unique_ptr<std::string> target_;
  std::vector<std::string> grantees_;
};

/*
 * @type ShowPrivilegesStmt
 * @brief SHOW PRIVILEGES statement
 */
class ShowPrivilegesStmt : public DDLStmt {
 public:
  ShowPrivilegesStmt(std::string* t, std::string* o, std::string* r)
      : object_type_(t), object_(o), role_(r) {}
  const std::string& get_object_type() const { return *object_type_; }
  const std::string& get_object() const { return *object_; }
  const std::string& get_role() const { return *role_; }
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::unique_ptr<std::string> object_type_;
  std::unique_ptr<std::string> object_;
  std::unique_ptr<std::string> role_;
};

/*
 * @type GrantRoleStmt
 * @brief GRANT ROLE statement
 */
class GrantRoleStmt : public DDLStmt {
 public:
  GrantRoleStmt(std::list<std::string*>* r, std::list<std::string*>* g) {
    parser_slistval_to_vector(r, roles_);
    parser_slistval_to_vector(g, grantees_);
  }
  GrantRoleStmt(const rapidjson::Value& payload);

  const std::vector<std::string>& get_roles() const { return roles_; }
  const std::vector<std::string>& get_grantees() const { return grantees_; }
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::vector<std::string> roles_;
  std::vector<std::string> grantees_;
};

/*
 * @type RevokeRoleStmt
 * @brief REVOKE ROLE statement
 */
class RevokeRoleStmt : public DDLStmt {
 public:
  RevokeRoleStmt(std::list<std::string*>* r, std::list<std::string*>* g) {
    parser_slistval_to_vector(r, roles_);
    parser_slistval_to_vector(g, grantees_);
  }
  RevokeRoleStmt(const rapidjson::Value& payload);

  const std::vector<std::string>& get_roles() const { return roles_; }
  const std::vector<std::string>& get_grantees() const { return grantees_; }
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::vector<std::string> roles_;
  std::vector<std::string> grantees_;
};

/*
 * @type QueryExpr
 * @brief query expression
 */
class QueryExpr : public Node {
 public:
  virtual void analyze(const Catalog_Namespace::Catalog& catalog,
                       Analyzer::Query& query) const = 0;
};

/*
 * @type UnionQuery
 * @brief UNION or UNION ALL queries
 */
class UnionQuery : public QueryExpr {
 public:
  UnionQuery(bool u, QueryExpr* l, QueryExpr* r) : is_unionall_(u), left_(l), right_(r) {}
  bool get_is_unionall() const { return is_unionall_; }
  const QueryExpr* get_left() const { return left_.get(); }
  const QueryExpr* get_right() const { return right_.get(); }
  void analyze(const Catalog_Namespace::Catalog& catalog,
               Analyzer::Query& query) const override;

 private:
  bool is_unionall_;
  std::unique_ptr<QueryExpr> left_;
  std::unique_ptr<QueryExpr> right_;
};

class SelectEntry : public Node {
 public:
  SelectEntry(Expr* e, std::string* r) : select_expr_(e), alias_(r) {}
  const Expr* get_select_expr() const { return select_expr_.get(); }
  const std::string* get_alias() const { return alias_.get(); }
  std::string to_string() const;

 private:
  std::unique_ptr<Expr> select_expr_;
  std::unique_ptr<std::string> alias_;
};

/*
 * @type QuerySpec
 * @brief a simple query
 */
class QuerySpec : public QueryExpr {
 public:
  QuerySpec(bool d,
            std::list<SelectEntry*>* s,
            std::list<TableRef*>* f,
            Expr* w,
            std::list<Expr*>* g,
            Expr* h)
      : is_distinct_(d), where_clause_(w), having_clause_(h) {
    if (s) {
      for (const auto e : *s) {
        select_clause_.emplace_back(e);
      }
      delete s;
    }
    CHECK(f);
    for (const auto e : *f) {
      from_clause_.emplace_back(e);
    }
    delete f;
    if (g) {
      for (const auto e : *g) {
        groupby_clause_.emplace_back(e);
      }
      delete g;
    }
  }
  bool get_is_distinct() const { return is_distinct_; }
  const std::list<std::unique_ptr<SelectEntry>>& get_select_clause() const {
    return select_clause_;
  }
  const std::list<std::unique_ptr<TableRef>>& get_from_clause() const {
    return from_clause_;
  }
  const Expr* get_where_clause() const { return where_clause_.get(); }
  const std::list<std::unique_ptr<Expr>>& get_groupby_clause() const {
    return groupby_clause_;
  }
  const Expr* get_having_clause() const { return having_clause_.get(); }
  void analyze(const Catalog_Namespace::Catalog& catalog,
               Analyzer::Query& query) const override;
  std::string to_string() const;

 private:
  bool is_distinct_;
  std::list<std::unique_ptr<SelectEntry>> select_clause_; /* nullptr means SELECT * */
  std::list<std::unique_ptr<TableRef>> from_clause_;
  std::unique_ptr<Expr> where_clause_;
  std::list<std::unique_ptr<Expr>> groupby_clause_;
  std::unique_ptr<Expr> having_clause_;
  void analyze_from_clause(const Catalog_Namespace::Catalog& catalog,
                           Analyzer::Query& query) const;
  void analyze_select_clause(const Catalog_Namespace::Catalog& catalog,
                             Analyzer::Query& query) const;
  void analyze_where_clause(const Catalog_Namespace::Catalog& catalog,
                            Analyzer::Query& query) const;
  void analyze_group_by(const Catalog_Namespace::Catalog& catalog,
                        Analyzer::Query& query) const;
  void analyze_having_clause(const Catalog_Namespace::Catalog& catalog,
                             Analyzer::Query& query) const;
};

/*
 * @type OrderSpec
 * @brief order spec for a column in ORDER BY clause
 */
class OrderSpec : public Node {
 public:
  OrderSpec(int n, ColumnRef* c, bool d, bool f)
      : colno_(n), column_(c), is_desc_(d), nulls_first_(f) {}
  int get_colno() const { return colno_; }
  const ColumnRef* get_column() const { return column_.get(); }
  bool get_is_desc() const { return is_desc_; }
  bool get_nulls_first() const { return nulls_first_; }

 private:
  int colno_; /* 0 means use column name */
  std::unique_ptr<ColumnRef> column_;
  bool is_desc_;
  bool nulls_first_;
};

/*
 * @type SelectStmt
 * @brief SELECT statement
 */
class SelectStmt : public DMLStmt {
 public:
  SelectStmt(QueryExpr* q, std::list<OrderSpec*>* o, int64_t l, int64_t f)
      : query_expr_(q), limit_(l), offset_(f) {
    if (o) {
      for (const auto e : *o) {
        orderby_clause_.emplace_back(e);
      }
      delete o;
    }
  }
  const QueryExpr* get_query_expr() const { return query_expr_.get(); }
  const std::list<std::unique_ptr<OrderSpec>>& get_orderby_clause() const {
    return orderby_clause_;
  }
  void analyze(const Catalog_Namespace::Catalog& catalog,
               Analyzer::Query& query) const override;

 private:
  std::unique_ptr<QueryExpr> query_expr_;
  std::list<std::unique_ptr<OrderSpec>> orderby_clause_;
  int64_t limit_;
  int64_t offset_;
};

/*
 * @type ShowCreateTableStmt
 * @brief shows create table statement to create table
 */
class ShowCreateTableStmt : public DDLStmt {
 public:
  ShowCreateTableStmt(std::string* tab) : table_(tab) {}
  ShowCreateTableStmt(const rapidjson::Value& payload);

  std::string getCreateStmt() { return create_stmt_; }
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::unique_ptr<std::string> table_;
  std::string create_stmt_;
};

/*
 * @type ExportQueryStmt
 * @brief COPY ( query ) TO file ...
 */
class ExportQueryStmt : public DDLStmt {
 public:
  ExportQueryStmt(std::string* q, std::string* p, std::list<NameValueAssign*>* o)
      : select_stmt_(q), file_path_(p) {
    if (o) {
      for (const auto e : *o) {
        options_.emplace_back(e);
      }
      delete o;
    }
  }
  ExportQueryStmt(const rapidjson::Value& payload);
  void execute(const Catalog_Namespace::SessionInfo& session) override;
  const std::string get_select_stmt() const { return *select_stmt_; }

  DistributedConnector* leafs_connector_ = nullptr;

 private:
  std::unique_ptr<std::string> select_stmt_;
  std::unique_ptr<std::string> file_path_;
  std::list<std::unique_ptr<NameValueAssign>> options_;

  void parseOptions(import_export::CopyParams& copy_params,
                    // @TODO(se) move rest to CopyParams when we have a Thrift endpoint
                    import_export::QueryExporter::FileType& file_type,
                    std::string& layer_name,
                    import_export::QueryExporter::FileCompression& file_compression,
                    import_export::QueryExporter::ArrayNullHandling& array_null_handling);
};

/*
 * @type CreateViewStmt
 * @brief CREATE VIEW statement
 */
class CreateViewStmt : public DDLStmt {
 public:
  CreateViewStmt(const std::string& view_name,
                 const std::string& select_query,
                 const bool if_not_exists)
      : view_name_(view_name)
      , select_query_(select_query)
      , if_not_exists_(if_not_exists) {}

  CreateViewStmt(const rapidjson::Value& payload);

  const std::string& get_view_name() const { return view_name_; }
  const std::string& get_select_query() const { return select_query_; }
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::string view_name_;
  std::string select_query_;
  bool if_not_exists_;
};

/*
 * @type DropViewStmt
 * @brief DROP VIEW statement
 */
class DropViewStmt : public DDLStmt {
 public:
  DropViewStmt(std::string* v, bool i) : view_name_(v), if_exists_(i) {}
  DropViewStmt(const rapidjson::Value& payload);

  const std::string* get_view_name() const { return view_name_.get(); }
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::unique_ptr<std::string> view_name_;
  bool if_exists_;
};

/*
 * @type CreateDBStmt
 * @brief CREATE DATABASE statement
 */
class CreateDBStmt : public DDLStmt {
 public:
  CreateDBStmt(const rapidjson::Value& payload);

  CreateDBStmt(std::string* n, std::list<NameValueAssign*>* l, const bool if_not_exists)
      : db_name_(n), if_not_exists_(if_not_exists) {
    if (l) {
      for (const auto e : *l) {
        options_.emplace_back(e);
      }
      delete l;
    }
  }
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::unique_ptr<std::string> db_name_;
  std::list<std::unique_ptr<NameValueAssign>> options_;
  bool if_not_exists_;
};

/*
 * @type DropDBStmt
 * @brief DROP DATABASE statement
 */
class DropDBStmt : public DDLStmt {
 public:
  DropDBStmt(const rapidjson::Value& payload);

  explicit DropDBStmt(std::string* n, bool i) : db_name_(n), if_exists_(i) {}
  auto const& getDatabaseName() { return db_name_; }
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::unique_ptr<std::string> db_name_;
  bool if_exists_;
};

/*
 * @type CreateUserStmt
 * @brief CREATE USER statement
 */
class CreateUserStmt : public DDLStmt {
 public:
  CreateUserStmt(const rapidjson::Value& payload);
  CreateUserStmt(std::string* n, std::list<NameValueAssign*>* l) : user_name_(n) {
    if (l) {
      for (const auto e : *l) {
        options_.emplace_back(e);
      }
      delete l;
    }
  }
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::unique_ptr<std::string> user_name_;
  std::list<std::unique_ptr<NameValueAssign>> options_;
};

/*
 * @type AlterUserStmt
 * @brief ALTER USER statement
 */
class AlterUserStmt : public DDLStmt {
 public:
  AlterUserStmt(const rapidjson::Value& payload);
  AlterUserStmt(std::string* n, std::list<NameValueAssign*>* l) : user_name_(n) {
    if (l) {
      for (const auto e : *l) {
        options_.emplace_back(e);
      }
      delete l;
    }
  }
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::unique_ptr<std::string> user_name_;
  std::list<std::unique_ptr<NameValueAssign>> options_;
};

/*
 * @type DropUserStmt
 * @brief DROP USER statement
 */
class DropUserStmt : public DDLStmt {
 public:
  DropUserStmt(const rapidjson::Value& payload);
  DropUserStmt(std::string* n) : user_name_(n) {}
  auto const& getUserName() { return user_name_; }
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::unique_ptr<std::string> user_name_;
};

/*
 * @type InsertStmt
 * @brief super class for INSERT statements
 */
class InsertStmt : public DMLStmt {
 public:
  InsertStmt(std::string* t, std::list<std::string*>* c) : table_(t) {
    if (c) {
      for (const auto e : *c) {
        column_list_.emplace_back(e);
      }
      delete c;
    }
  }
  const std::string* get_table() const { return table_.get(); }
  const std::list<std::unique_ptr<std::string>>& get_column_list() const {
    return column_list_;
  }
  void analyze(const Catalog_Namespace::Catalog& catalog,
               Analyzer::Query& query) const override = 0;

 protected:
  std::unique_ptr<std::string> table_;
  std::list<std::unique_ptr<std::string>> column_list_;
};

/*
 * @type InsertValuesStmt
 * @brief INSERT INTO ... VALUES ...
 */
class InsertValuesStmt : public InsertStmt {
 public:
  InsertValuesStmt(std::string* t, std::list<std::string*>* c, std::list<Expr*>* v)
      : InsertStmt(t, c) {
    CHECK(v);
    for (const auto e : *v) {
      value_list_.emplace_back(e);
    }
    delete v;
  }
  const std::list<std::unique_ptr<Expr>>& get_value_list() const { return value_list_; }
  void analyze(const Catalog_Namespace::Catalog& catalog,
               Analyzer::Query& query) const override;

  size_t determineLeafIndex(const Catalog_Namespace::Catalog& catalog, size_t num_leafs);

  void execute(const Catalog_Namespace::SessionInfo& session);

 private:
  std::list<std::unique_ptr<Expr>> value_list_;
};

/*
 * @type Assignment
 * @brief assignment in UPDATE statement
 */
class Assignment : public Node {
 public:
  Assignment(std::string* c, Expr* a) : column_(c), assignment_(a) {}
  const std::string* get_column() const { return column_.get(); }
  const Expr* get_assignment() const { return assignment_.get(); }

 private:
  std::unique_ptr<std::string> column_;
  std::unique_ptr<Expr> assignment_;
};

/*
 * @type UpdateStmt
 * @brief UPDATE statement
 */
class UpdateStmt : public DMLStmt {
 public:
  UpdateStmt(std::string* t, std::list<Assignment*>* a, Expr* w)
      : table_(t), where_clause_(w) {
    CHECK(a);
    for (const auto e : *a) {
      assignment_list_.emplace_back(e);
    }
    delete a;
  }
  const std::string* get_table() const { return table_.get(); }
  const std::list<std::unique_ptr<Assignment>>& get_assignment_list() const {
    return assignment_list_;
  }
  const Expr* get_where_clause() const { return where_clause_.get(); }
  void analyze(const Catalog_Namespace::Catalog& catalog,
               Analyzer::Query& query) const override;

 private:
  std::unique_ptr<std::string> table_;
  std::list<std::unique_ptr<Assignment>> assignment_list_;
  std::unique_ptr<Expr> where_clause_;
};

/*
 * @type DeleteStmt
 * @brief DELETE statement
 */
class DeleteStmt : public DMLStmt {
 public:
  DeleteStmt(std::string* t, Expr* w) : table_(t), where_clause_(w) {}
  const std::string* get_table() const { return table_.get(); }
  const Expr* get_where_clause() const { return where_clause_.get(); }
  void analyze(const Catalog_Namespace::Catalog& catalog,
               Analyzer::Query& query) const override;

 private:
  std::unique_ptr<std::string> table_;
  std::unique_ptr<Expr> where_clause_;
};

template <typename LITERAL_TYPE>
struct DefaultValidate {};

template <>
struct DefaultValidate<IntLiteral> {
  template <typename T>
  decltype(auto) operator()(T t) {
    const std::string property_name(boost::to_upper_copy<std::string>(*t->get_name()));
    if (!dynamic_cast<const IntLiteral*>(t->get_value())) {
      throw std::runtime_error(property_name + " must be an integer literal.");
    }
    const auto val = static_cast<const IntLiteral*>(t->get_value())->get_intval();
    if (val <= 0) {
      throw std::runtime_error(property_name + " must be a positive number.");
    }
    return val;
  }
};

struct PositiveOrZeroValidate {
  template <typename T>
  decltype(auto) operator()(T t) {
    const std::string property_name(boost::to_upper_copy<std::string>(*t->get_name()));
    if (!dynamic_cast<const IntLiteral*>(t->get_value())) {
      throw std::runtime_error(property_name + " must be an integer literal.");
    }
    const auto val = static_cast<const IntLiteral*>(t->get_value())->get_intval();
    if (val < 0) {
      throw std::runtime_error(property_name + " must be greater than or equal to 0.");
    }
    return val;
  }
};

template <>
struct DefaultValidate<StringLiteral> {
  template <typename T>
  decltype(auto) operator()(T t) {
    const auto val = static_cast<const StringLiteral*>(t->get_value())->get_stringval();
    CHECK(val);
    const auto val_upper = boost::to_upper_copy<std::string>(*val);
    return val_upper;
  }
};

struct CaseSensitiveValidate {
  template <typename T>
  decltype(auto) operator()(T t) {
    const auto val = static_cast<const StringLiteral*>(t->get_value())->get_stringval();
    CHECK(val);
    return *val;
  }
};

template <typename T>
struct ShouldInvalidateSessionsByDB : public std::false_type {};
template <typename T>
struct ShouldInvalidateSessionsByUser : public std::false_type {};

template <>
struct ShouldInvalidateSessionsByDB<DropDBStmt> : public std::true_type {};
template <>
struct ShouldInvalidateSessionsByUser<DropUserStmt> : public std::true_type {};
template <>
struct ShouldInvalidateSessionsByDB<RenameDBStmt> : public std::true_type {};
template <>
struct ShouldInvalidateSessionsByUser<RenameUserStmt> : public std::true_type {};

/**
 * Helper functions for parsing the DDL returned from calcite as part of the plan result
 * to a parser node in this class. Currently only used in
 * QueryRunner/DistributedQueryRunner, where we do not want to link in the thrift
 * dependencies wich DdlCommandExecutor currently brings along.
 */
std::unique_ptr<Parser::DDLStmt> create_ddl_from_calcite(const std::string& query_json);

void execute_ddl_from_calcite(
    const std::string& query_json,
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

}  // namespace Parser

#endif  // PARSERNODE_H_
