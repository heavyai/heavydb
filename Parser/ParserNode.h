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

#include "../Fragmenter/InsertDataLoader.h"

#include <Import/Importer.h>

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
class SQLType : public Node {
 public:
  explicit SQLType(SQLTypes t)
      : type(t), param1(-1), param2(0), is_array(false), array_size(-1) {}
  SQLType(SQLTypes t, int p1)
      : type(t), param1(p1), param2(0), is_array(false), array_size(-1) {}
  SQLType(SQLTypes t, int p1, int p2, bool a)
      : type(t), param1(p1), param2(p2), is_array(a), array_size(-1) {}
  SQLTypes get_type() const { return type; }
  int get_param1() const { return param1; }
  int get_param2() const { return param2; }
  bool get_is_array() const { return is_array; }
  void set_is_array(bool a) { is_array = a; }
  int get_array_size() const { return array_size; }
  void set_array_size(int s) { array_size = s; }
  std::string to_string() const;
  void check_type();

 private:
  SQLTypes type;
  int param1;  // e.g. for NUMERIC(10).  -1 means unspecified.
  int param2;  // e.g. for NUMERIC(10,3). 0 is default value.
  bool is_array;
  int array_size;
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
  explicit StringLiteral(std::string* s) : stringval(s) {}
  const std::string* get_stringval() const { return stringval.get(); }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  static std::shared_ptr<Analyzer::Expr> analyzeValue(const std::string&);
  std::string to_string() const override { return "'" + *stringval + "'"; }

 private:
  std::unique_ptr<std::string> stringval;
};

/*
 * @type IntLiteral
 * @brief the literal for integer constants
 */
class IntLiteral : public Literal {
 public:
  explicit IntLiteral(int64_t i) : intval(i) {}
  int64_t get_intval() const { return intval; }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  static std::shared_ptr<Analyzer::Expr> analyzeValue(const int64_t intval);
  std::string to_string() const override {
    return boost::lexical_cast<std::string>(intval);
  }

 private:
  int64_t intval;
};

/*
 * @type FixedPtLiteral
 * @brief the literal for DECIMAL and NUMERIC
 */
class FixedPtLiteral : public Literal {
 public:
  explicit FixedPtLiteral(std::string* n) : fixedptval(n) {}
  const std::string* get_fixedptval() const { return fixedptval.get(); }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  static std::shared_ptr<Analyzer::Expr> analyzeValue(const int64_t numericval,
                                                      const int scale,
                                                      const int precision);
  std::string to_string() const override { return *fixedptval; }

 private:
  std::unique_ptr<std::string> fixedptval;
};

/*
 * @type FloatLiteral
 * @brief the literal for FLOAT or REAL
 */
class FloatLiteral : public Literal {
 public:
  explicit FloatLiteral(float f) : floatval(f) {}
  float get_floatval() const { return floatval; }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  std::string to_string() const override {
    return boost::lexical_cast<std::string>(floatval);
  }

 private:
  float floatval;
};

/*
 * @type DoubleLiteral
 * @brief the literal for DOUBLE PRECISION
 */
class DoubleLiteral : public Literal {
 public:
  explicit DoubleLiteral(double d) : doubleval(d) {}
  double get_doubleval() const { return doubleval; }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  std::string to_string() const override {
    return boost::lexical_cast<std::string>(doubleval);
  }

 private:
  double doubleval;
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
      value_list.emplace_back(e);
    }
    delete v;
  }
  const std::list<std::unique_ptr<Expr>>& get_value_list() const { return value_list; }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  std::string to_string() const override;

 private:
  std::list<std::unique_ptr<Expr>> value_list;
};

/*
 * @type OperExpr
 * @brief all operator expressions
 */
class OperExpr : public Expr {
 public:
  OperExpr(SQLOps t, Expr* l, Expr* r)
      : optype(t), opqualifier(kONE), left(l), right(r) {}
  OperExpr(SQLOps t, SQLQualifier q, Expr* l, Expr* r)
      : optype(t), opqualifier(q), left(l), right(r) {}
  SQLOps get_optype() const { return optype; }
  const Expr* get_left() const { return left.get(); }
  const Expr* get_right() const { return right.get(); }
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
  SQLOps optype;
  SQLQualifier opqualifier;
  std::unique_ptr<Expr> left;
  std::unique_ptr<Expr> right;
};

// forward reference of QuerySpec
class QuerySpec;

/*
 * @type SubqueryExpr
 * @brief expression for subquery
 */
class SubqueryExpr : public Expr {
 public:
  explicit SubqueryExpr(QuerySpec* q) : query(q) {}
  const QuerySpec* get_query() const { return query.get(); }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  std::string to_string() const override;

 private:
  std::unique_ptr<QuerySpec> query;
};

class IsNullExpr : public Expr {
 public:
  IsNullExpr(bool n, Expr* a) : is_not(n), arg(a) {}
  bool get_is_not() const { return is_not; }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  std::string to_string() const override;

 private:
  bool is_not;
  std::unique_ptr<Expr> arg;
};

/*
 * @type InExpr
 * @brief expression for the IS NULL predicate
 */
class InExpr : public Expr {
 public:
  InExpr(bool n, Expr* a) : is_not(n), arg(a) {}
  bool get_is_not() const { return is_not; }
  const Expr* get_arg() const { return arg.get(); }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override = 0;
  std::string to_string() const override;

 protected:
  bool is_not;
  std::unique_ptr<Expr> arg;
};

/*
 * @type InSubquery
 * @brief expression for the IN (subquery) predicate
 */
class InSubquery : public InExpr {
 public:
  InSubquery(bool n, Expr* a, SubqueryExpr* q) : InExpr(n, a), subquery(q) {}
  const SubqueryExpr* get_subquery() const { return subquery.get(); }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  std::string to_string() const override;

 private:
  std::unique_ptr<SubqueryExpr> subquery;
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
      value_list.emplace_back(e);
    }
    delete v;
  }
  const std::list<std::unique_ptr<Expr>>& get_value_list() const { return value_list; }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  std::string to_string() const override;

 private:
  std::list<std::unique_ptr<Expr>> value_list;
};

/*
 * @type BetweenExpr
 * @brief expression for BETWEEN lower AND upper
 */
class BetweenExpr : public Expr {
 public:
  BetweenExpr(bool n, Expr* a, Expr* l, Expr* u)
      : is_not(n), arg(a), lower(l), upper(u) {}
  bool get_is_not() const { return is_not; }
  const Expr* get_arg() const { return arg.get(); }
  const Expr* get_lower() const { return lower.get(); }
  const Expr* get_upper() const { return upper.get(); }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  std::string to_string() const override;

 private:
  bool is_not;
  std::unique_ptr<Expr> arg;
  std::unique_ptr<Expr> lower;
  std::unique_ptr<Expr> upper;
};

/*
 * @type CharLengthExpr
 * @brief expression to get length of string
 */

class CharLengthExpr : public Expr {
 public:
  CharLengthExpr(Expr* a, bool e) : arg(a), calc_encoded_length(e) {}
  const Expr* get_arg() const { return arg.get(); }
  bool get_calc_encoded_length() const { return calc_encoded_length; }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  std::string to_string() const override;

 private:
  std::unique_ptr<Expr> arg;
  bool calc_encoded_length;
};

/*
 * @type CardinalityExpr
 * @brief expression to get cardinality of an array
 */

class CardinalityExpr : public Expr {
 public:
  CardinalityExpr(Expr* a) : arg(a) {}
  const Expr* get_arg() const { return arg.get(); }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  std::string to_string() const override;

 private:
  std::unique_ptr<Expr> arg;
};

/*
 * @type LikeExpr
 * @brief expression for the LIKE predicate
 */
class LikeExpr : public Expr {
 public:
  LikeExpr(bool n, bool i, Expr* a, Expr* l, Expr* e)
      : is_not(n), is_ilike(i), arg(a), like_string(l), escape_string(e) {}
  bool get_is_not() const { return is_not; }
  const Expr* get_arg() const { return arg.get(); }
  const Expr* get_like_string() const { return like_string.get(); }
  const Expr* get_escape_string() const { return escape_string.get(); }
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
  bool is_not;
  bool is_ilike;
  std::unique_ptr<Expr> arg;
  std::unique_ptr<Expr> like_string;
  std::unique_ptr<Expr> escape_string;

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
      : is_not(n), arg(a), pattern_string(p), escape_string(e) {}
  bool get_is_not() const { return is_not; }
  const Expr* get_arg() const { return arg.get(); }
  const Expr* get_pattern_string() const { return pattern_string.get(); }
  const Expr* get_escape_string() const { return escape_string.get(); }
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
  bool is_not;
  std::unique_ptr<Expr> arg;
  std::unique_ptr<Expr> pattern_string;
  std::unique_ptr<Expr> escape_string;

  static void check_pattern_expr(const std::string& pattern_str, char escape_char);
  static bool translate_to_like_pattern(std::string& pattern_str, char escape_char);
};

/*
 * @type LikelihoodExpr
 * @brief expression for LIKELY, UNLIKELY
 */
class LikelihoodExpr : public Expr {
 public:
  LikelihoodExpr(bool n, Expr* a, float l) : is_not(n), arg(a), likelihood(l) {}
  bool get_is_not() const { return is_not; }
  const Expr* get_arg() const { return arg.get(); }
  float get_likelihood() const { return likelihood; }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  static std::shared_ptr<Analyzer::Expr> get(std::shared_ptr<Analyzer::Expr> arg_expr,
                                             float likelihood,
                                             const bool is_not);
  std::string to_string() const override;

 private:
  bool is_not;
  std::unique_ptr<Expr> arg;
  float likelihood;
};

/*
 * @type ExistsExpr
 * @brief expression for EXISTS (subquery)
 */
class ExistsExpr : public Expr {
 public:
  explicit ExistsExpr(QuerySpec* q) : query(q) {}
  const QuerySpec* get_query() const { return query.get(); }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  std::string to_string() const override;

 private:
  std::unique_ptr<QuerySpec> query;
};

/*
 * @type ColumnRef
 * @brief expression for a column reference
 */
class ColumnRef : public Expr {
 public:
  explicit ColumnRef(std::string* n1) : table(nullptr), column(n1) {}
  ColumnRef(std::string* n1, std::string* n2) : table(n1), column(n2) {}
  const std::string* get_table() const { return table.get(); }
  const std::string* get_column() const { return column.get(); }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  std::string to_string() const override;

 private:
  std::unique_ptr<std::string> table;
  std::unique_ptr<std::string> column;  // can be nullptr in the t.* case
};

/*
 * @type FunctionRef
 * @brief expression for a function call
 */
class FunctionRef : public Expr {
 public:
  explicit FunctionRef(std::string* n) : name(n), distinct(false), arg(nullptr) {}
  FunctionRef(std::string* n, Expr* a) : name(n), distinct(false), arg(a) {}
  FunctionRef(std::string* n, bool d, Expr* a) : name(n), distinct(d), arg(a) {}
  const std::string* get_name() const { return name.get(); }
  bool get_distinct() const { return distinct; }
  Expr* get_arg() const { return arg.get(); }
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  std::string to_string() const override;

 private:
  std::unique_ptr<std::string> name;
  bool distinct;              // only true for COUNT(DISTINCT x)
  std::unique_ptr<Expr> arg;  // for COUNT, nullptr means '*'
};

class CastExpr : public Expr {
 public:
  CastExpr(Expr* a, SQLType* t) : arg(a), target_type(t) {}
  std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const override;
  std::string to_string() const override {
    return "CAST(" + arg->to_string() + " AS " + target_type->to_string() + ")";
  }

 private:
  std::unique_ptr<Expr> arg;
  std::unique_ptr<SQLType> target_type;
};

class ExprPair : public Node {
 public:
  ExprPair(Expr* e1, Expr* e2) : expr1(e1), expr2(e2) {}
  const Expr* get_expr1() const { return expr1.get(); }
  const Expr* get_expr2() const { return expr2.get(); }

 private:
  std::unique_ptr<Expr> expr1;
  std::unique_ptr<Expr> expr2;
};

class CaseExpr : public Expr {
 public:
  CaseExpr(std::list<ExprPair*>* w, Expr* e) : else_expr(e) {
    CHECK(w);
    for (const auto e : *w) {
      when_then_list.emplace_back(e);
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
  std::list<std::unique_ptr<ExprPair>> when_then_list;
  std::unique_ptr<Expr> else_expr;
};

/*
 * @type TableRef
 * @brief table reference in FROM clause
 */
class TableRef : public Node {
 public:
  explicit TableRef(std::string* t) : table_name(t), range_var(nullptr) {}
  TableRef(std::string* t, std::string* r) : table_name(t), range_var(r) {}
  const std::string* get_table_name() const { return table_name.get(); }
  const std::string* get_range_var() const { return range_var.get(); }
  std::string to_string() const;

 private:
  std::unique_ptr<std::string> table_name;
  std::unique_ptr<std::string> range_var;
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
      : notnull(n), unique(u), is_primarykey(p), defaultval(d) {}
  ColumnConstraintDef(Expr* c)
      : notnull(false), unique(false), is_primarykey(false), check_condition(c) {}
  ColumnConstraintDef(std::string* t, std::string* c)
      : notnull(false)
      , unique(false)
      , is_primarykey(false)
      , foreign_table(t)
      , foreign_column(c) {}
  bool get_notnull() const { return notnull; }
  bool get_unique() const { return unique; }
  bool get_is_primarykey() const { return is_primarykey; }
  const Literal* get_defaultval() const { return defaultval.get(); }
  const Expr* get_check_condition() const { return check_condition.get(); }
  const std::string* get_foreign_table() const { return foreign_table.get(); }
  const std::string* get_foreign_column() const { return foreign_column.get(); }

 private:
  bool notnull;
  bool unique;
  bool is_primarykey;
  std::unique_ptr<Literal> defaultval;
  std::unique_ptr<Expr> check_condition;
  std::unique_ptr<std::string> foreign_table;
  std::unique_ptr<std::string> foreign_column;
};

/*
 * @type CompressDef
 * @brief Node for compression scheme definition
 */
class CompressDef : public Node {
 public:
  CompressDef(std::string* n, int p) : encoding_name(n), encoding_param(p) {}
  const std::string* get_encoding_name() const { return encoding_name.get(); }
  int get_encoding_param() const { return encoding_param; }

 private:
  std::unique_ptr<std::string> encoding_name;
  int encoding_param;
};

/*
 * @type ColumnDef
 * @brief Column definition
 */
class ColumnDef : public TableElement {
 public:
  ColumnDef(std::string* c, SQLType* t, CompressDef* cp, ColumnConstraintDef* cc)
      : column_name(c), column_type(t), compression(cp), column_constraint(cc) {}
  const std::string* get_column_name() const { return column_name.get(); }
  SQLType* get_column_type() const { return column_type.get(); }
  const CompressDef* get_compression() const { return compression.get(); }
  const ColumnConstraintDef* get_column_constraint() const {
    return column_constraint.get();
  }

 private:
  std::unique_ptr<std::string> column_name;
  std::unique_ptr<SQLType> column_type;
  std::unique_ptr<CompressDef> compression;
  std::unique_ptr<ColumnConstraintDef> column_constraint;
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
  UniqueDef(bool p, std::list<std::string*>* cl) : is_primarykey(p) {
    CHECK(cl);
    for (const auto s : *cl) {
      column_list.emplace_back(s);
    }
    delete cl;
  }
  bool get_is_primarykey() const { return is_primarykey; }
  const std::list<std::unique_ptr<std::string>>& get_column_list() const {
    return column_list;
  }

 private:
  bool is_primarykey;
  std::list<std::unique_ptr<std::string>> column_list;
};

/*
 * @type ForeignKeyDef
 * @brief foreign key constraint
 */
class ForeignKeyDef : public TableConstraintDef {
 public:
  ForeignKeyDef(std::list<std::string*>* cl, std::string* t, std::list<std::string*>* fcl)
      : foreign_table(t) {
    CHECK(cl);
    for (const auto s : *cl) {
      column_list.emplace_back(s);
    }
    delete cl;
    if (fcl) {
      for (const auto s : *fcl) {
        foreign_column_list.emplace_back(s);
      }
    }
    delete fcl;
  }
  const std::list<std::unique_ptr<std::string>>& get_column_list() const {
    return column_list;
  }
  const std::string* get_foreign_table() const { return foreign_table.get(); }
  const std::list<std::unique_ptr<std::string>>& get_foreign_column_list() const {
    return foreign_column_list;
  }

 private:
  std::list<std::unique_ptr<std::string>> column_list;
  std::unique_ptr<std::string> foreign_table;
  std::list<std::unique_ptr<std::string>> foreign_column_list;
};

/*
 * @type CheckDef
 * @brief Check constraint
 */
class CheckDef : public TableConstraintDef {
 public:
  CheckDef(Expr* c) : check_condition(c) {}
  const Expr* get_check_condition() const { return check_condition.get(); }

 private:
  std::unique_ptr<Expr> check_condition;
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
  NameValueAssign(std::string* n, Literal* v) : name(n), value(v) {}
  const std::string* get_name() const { return name.get(); }
  const Literal* get_value() const { return value.get(); }

 private:
  std::unique_ptr<std::string> name;
  std::unique_ptr<Literal> value;
};

/*
 * @type CreateTableStmt
 * @brief CREATE TABLE statement
 */
class CreateTableStmt : public DDLStmt {
 public:
  CreateTableStmt(std::string* tab,
                  const std::string* storage,
                  std::list<TableElement*>* table_elems,
                  bool is_temporary,
                  bool if_not_exists,
                  std::list<NameValueAssign*>* s)
      : table_(tab)
      , storage_type_(storage)
      , is_temporary_(is_temporary)
      , if_not_exists_(if_not_exists) {
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
  const std::string* get_table() const { return table_.get(); }
  const std::list<std::unique_ptr<TableElement>>& get_table_element_list() const {
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
  std::unique_ptr<const std::string> storage_type_;
  bool is_temporary_;
  bool if_not_exists_;
  std::list<std::unique_ptr<NameValueAssign>> storage_options_;
};

/*
 * @type InsertIntoTableAsSelectStmt
 * @brief INSERT INTO TABLE SELECT statement
 */
class InsertIntoTableAsSelectStmt : public DDLStmt {
 public:
  // ITAS constructor
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

  void populateData(QueryStateProxy, bool is_temporary, bool validate_table);
  void execute(const Catalog_Namespace::SessionInfo& session) override;

  std::string& get_table() { return table_name_; }

  std::string& get_select_query() { return select_query_; }

  struct DistributedConnector
      : public Fragmenter_Namespace::InsertDataLoader::DistributedConnector {
    virtual AggregatedResult query(QueryStateProxy, std::string& sql_query_string) = 0;
    virtual void checkpoint(const Catalog_Namespace::SessionInfo& parent_session_info,
                            int tableId) = 0;
    virtual void rollback(const Catalog_Namespace::SessionInfo& parent_session_info,
                          int tableId) = 0;
  };

  struct LocalConnector : public DistributedConnector {
    virtual ~LocalConnector() {}
    AggregatedResult query(QueryStateProxy,
                           std::string& sql_query_string,
                           bool validate_only);
    AggregatedResult query(QueryStateProxy, std::string& sql_query_string) override;
    size_t leafCount() override { return 1; };
    void insertDataToLeaf(const Catalog_Namespace::SessionInfo& session,
                          const size_t leaf_idx,
                          Fragmenter_Namespace::InsertData& insert_data) override;
    void checkpoint(const Catalog_Namespace::SessionInfo& session, int tableId) override;
    void rollback(const Catalog_Namespace::SessionInfo& session, int tableId) override;
    std::list<ColumnDescriptor> getColumnDescriptors(AggregatedResult& result,
                                                     bool for_create);
  };

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
  const bool is_temporary_;
  const bool if_not_exists_;
  std::list<std::unique_ptr<NameValueAssign>> storage_options_;
};

/*
 * @type DropTableStmt
 * @brief DROP TABLE statement
 */
class DropTableStmt : public DDLStmt {
 public:
  DropTableStmt(std::string* tab, bool i) : table(tab), if_exists(i) {}
  const std::string* get_table() const { return table.get(); }
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::unique_ptr<std::string> table;
  bool if_exists;
};

/*
 * @type TruncateTableStmt
 * @brief TRUNCATE TABLE statement
 */
class TruncateTableStmt : public DDLStmt {
 public:
  TruncateTableStmt(std::string* tab) : table(tab) {}
  const std::string* get_table() const { return table.get(); }
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::unique_ptr<std::string> table;
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

  const std::string getTableName() const { return *(table_.get()); }

  bool shouldVacuumDeletedRows() const {
    for (const auto& e : options_) {
      if (boost::iequals(*(e->get_name()), "VACUUM")) {
        return true;
      }
    }
    return false;
  }

  void execute(const Catalog_Namespace::SessionInfo& session) override {
    // Should pass optimize params to the table optimizer
    CHECK(false);
  }

 private:
  std::unique_ptr<std::string> table_;
  std::list<std::unique_ptr<NameValueAssign>> options_;
};

class ValidateStmt : public DDLStmt {
 public:
  ValidateStmt(std::string* type, std::list<NameValueAssign*>* with_opts) : type_(type) {
    if (!type_) {
      throw std::runtime_error("Validation Type is required for VALIDATE command.");
    }
    std::list<std::unique_ptr<NameValueAssign>> options;
    if (with_opts) {
      for (const auto e : *with_opts) {
        options.emplace_back(e);
      }
      delete with_opts;

      for (const auto& opt : options) {
        if (boost::iequals(*opt->get_name(), "REPAIR_TYPE")) {
          const auto repair_type =
              static_cast<const StringLiteral*>(opt->get_value())->get_stringval();
          CHECK(repair_type);
          if (boost::iequals(*repair_type, "REMOVE")) {
            isRepairTypeRemove_ = true;
          } else {
            throw std::runtime_error("REPAIR_TYPE must be REMOVE.");
          }
        } else {
          throw std::runtime_error("The only VALIDATE WITH options is REPAIR_TYPE.");
        }
      }
    }
  }

  bool isRepairTypeRemove() const { return isRepairTypeRemove_; }

  const std::string getType() const { return *(type_.get()); }

  void execute(const Catalog_Namespace::SessionInfo& session) override { UNREACHABLE(); }

 private:
  std::unique_ptr<std::string> type_;
  bool isRepairTypeRemove_ = false;
};

class RenameDatabaseStmt : public DDLStmt {
 public:
  RenameDatabaseStmt(std::string* database_name, std::string* new_database_name)
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
  RenameTableStmt(std::string* tab, std::string* new_tab_name)
      : table(tab), new_table_name(new_tab_name) {}

  const std::string* get_prev_table() const { return table.get(); }
  const std::string* get_new_table() const { return new_table_name.get(); }
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::unique_ptr<std::string> table;
  std::unique_ptr<std::string> new_table_name;
};

class RenameColumnStmt : public DDLStmt {
 public:
  RenameColumnStmt(std::string* tab, std::string* col, std::string* new_col_name)
      : table(tab), column(col), new_column_name(new_col_name) {}
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::unique_ptr<std::string> table;
  std::unique_ptr<std::string> column;
  std::unique_ptr<std::string> new_column_name;
};

class AddColumnStmt : public DDLStmt {
 public:
  AddColumnStmt(std::string* tab, ColumnDef* coldef) : table(tab), coldef(coldef) {}
  AddColumnStmt(std::string* tab, std::list<ColumnDef*>* coldefs) : table(tab) {
    for (const auto coldef : *coldefs) {
      this->coldefs.emplace_back(coldef);
    }
    delete coldefs;
  }
  void execute(const Catalog_Namespace::SessionInfo& session) override;
  void check_executable(const Catalog_Namespace::SessionInfo& session);
  const std::string* get_table() const { return table.get(); }

 private:
  std::unique_ptr<std::string> table;
  std::unique_ptr<ColumnDef> coldef;
  std::list<std::unique_ptr<ColumnDef>> coldefs;
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
      : table(tab), path(path) {
    auto options_deleter = [](std::list<NameValueAssign*>* options) {
      for (auto option : *options) {
        delete option;
      }
      delete options;
    };
    std::unique_ptr<std::list<NameValueAssign*>, decltype(options_deleter)> options_ptr(
        options, options_deleter);
    std::vector<std::string> allowed_compression_programs{"lz4", "gzip", "none"};
    // specialize decompressor or break on osx bsdtar...
    if (options) {
      for (const auto option : *options) {
        if (boost::iequals(*option->get_name(), "compression")) {
          if (const auto str_literal =
                  dynamic_cast<const StringLiteral*>(option->get_value())) {
            compression = *str_literal->get_stringval();
            const std::string lowercase_compression =
                boost::algorithm::to_lower_copy(compression);
            if (allowed_compression_programs.end() ==
                std::find(allowed_compression_programs.begin(),
                          allowed_compression_programs.end(),
                          lowercase_compression)) {
              throw std::runtime_error("Compression program " + compression +
                                       " is not supported.");
            }
          } else {
            throw std::runtime_error("Compression option must be a string.");
          }
        } else {
          throw std::runtime_error("Invalid WITH option: " + *option->get_name());
        }
      }
    }
    // default lz4 compression, next gzip, or none.
    if (compression.empty()) {
      if (boost::process::search_path(compression = "gzip").string().empty()) {
        if (boost::process::search_path(compression = "lz4").string().empty()) {
          compression = "none";
        }
      }
    }
    if (boost::iequals(compression, "none")) {
      compression.clear();
    } else {
      std::map<std::string, std::string> decompression{{"lz4", "unlz4"},
                                                       {"gzip", "gunzip"}};
      const auto use_program = is_restore ? decompression[compression] : compression;
      const auto prog_path = boost::process::search_path(use_program);
      if (prog_path.string().empty()) {
        throw std::runtime_error("Compression program " + use_program + " is not found.");
      }
      compression = "--use-compress-program=" + use_program;
    }
  }
  const std::string* getTable() const { return table.get(); }
  const std::string* getPath() const { return path.get(); }
  const std::string getCompression() const { return compression; }

 protected:
  std::unique_ptr<std::string> table;
  std::unique_ptr<std::string> path;  // dump TO file path
  std::string compression;
};

class DumpTableStmt : public DumpRestoreTableStmtBase {
 public:
  DumpTableStmt(std::string* tab, std::string* path, std::list<NameValueAssign*>* options)
      : DumpRestoreTableStmtBase(tab, path, options, false) {}
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
  void execute(const Catalog_Namespace::SessionInfo& session) override;
};

/*
 * @type CopyTableStmt
 * @brief COPY ... FROM ...
 */
class CopyTableStmt : public DDLStmt {
 public:
  CopyTableStmt(std::string* t, std::string* f, std::list<NameValueAssign*>* o)
      : table(t), file_pattern(f) {
    if (o) {
      for (const auto e : *o) {
        options.emplace_back(e);
      }
      delete o;
    }
  }
  void execute(const Catalog_Namespace::SessionInfo& session) override;
  void execute(const Catalog_Namespace::SessionInfo& session,
               const std::function<std::unique_ptr<Importer_NS::Importer>(
                   Catalog_Namespace::Catalog&,
                   const TableDescriptor*,
                   const std::string&,
                   const Importer_NS::CopyParams&)>& importer_factory);
  std::unique_ptr<std::string> return_message;

  std::string& get_table() const {
    CHECK(table);
    return *table;
  }

  bool was_geo_copy_from() const { return _was_geo_copy_from; }

  void get_geo_copy_from_payload(std::string& geo_copy_from_table,
                                 std::string& geo_copy_from_file_name,
                                 Importer_NS::CopyParams& geo_copy_from_copy_params,
                                 std::string& geo_copy_from_partitions) {
    geo_copy_from_table = *table;
    geo_copy_from_file_name = _geo_copy_from_file_name;
    geo_copy_from_copy_params = _geo_copy_from_copy_params;
    geo_copy_from_partitions = _geo_copy_from_partitions;
    _was_geo_copy_from = false;
  }

 private:
  std::unique_ptr<std::string> table;
  std::unique_ptr<std::string> file_pattern;
  std::list<std::unique_ptr<NameValueAssign>> options;

  bool _was_geo_copy_from = false;
  std::string _geo_copy_from_file_name;
  Importer_NS::CopyParams _geo_copy_from_copy_params;
  std::string _geo_copy_from_partitions;
};

/*
 * @type CreateRoleStmt
 * @brief CREATE ROLE statement
 */
class CreateRoleStmt : public DDLStmt {
 public:
  CreateRoleStmt(std::string* r) : role(r) {}
  const std::string& get_role() const { return *role; }
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::unique_ptr<std::string> role;
};

/*
 * @type DropRoleStmt
 * @brief DROP ROLE statement
 */
class DropRoleStmt : public DDLStmt {
 public:
  DropRoleStmt(std::string* r) : role(r) {}
  const std::string& get_role() const { return *role; }
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::unique_ptr<std::string> role;
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
      : object_type(t), object(o) {
    parser_slistval_to_vector(p, privs);
    parser_slistval_to_vector(g, grantees);
  }

  const std::vector<std::string>& get_privs() const { return privs; }
  const std::string& get_object_type() const { return *object_type; }
  const std::string& get_object() const { return *object; }
  const std::vector<std::string>& get_grantees() const { return grantees; }
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::vector<std::string> privs;
  std::unique_ptr<std::string> object_type;
  std::unique_ptr<std::string> object;
  std::vector<std::string> grantees;
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
      : object_type(t), object(o) {
    parser_slistval_to_vector(p, privs);
    parser_slistval_to_vector(g, grantees);
  }

  const std::vector<std::string>& get_privs() const { return privs; }
  const std::string& get_object_type() const { return *object_type; }
  const std::string& get_object() const { return *object; }
  const std::vector<std::string>& get_grantees() const { return grantees; }
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::vector<std::string> privs;
  std::unique_ptr<std::string> object_type;
  std::unique_ptr<std::string> object;
  std::vector<std::string> grantees;
};

/*
 * @type ShowPrivilegesStmt
 * @brief SHOW PRIVILEGES statement
 */
class ShowPrivilegesStmt : public DDLStmt {
 public:
  ShowPrivilegesStmt(std::string* t, std::string* o, std::string* r)
      : object_type(t), object(o), role(r) {}
  const std::string& get_object_type() const { return *object_type; }
  const std::string& get_object() const { return *object; }
  const std::string& get_role() const { return *role; }
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::unique_ptr<std::string> object_type;
  std::unique_ptr<std::string> object;
  std::unique_ptr<std::string> role;
};

/*
 * @type GrantRoleStmt
 * @brief GRANT ROLE statement
 */
class GrantRoleStmt : public DDLStmt {
 public:
  GrantRoleStmt(std::list<std::string*>* r, std::list<std::string*>* g) {
    parser_slistval_to_vector(r, roles);
    parser_slistval_to_vector(g, grantees);
  }
  const std::vector<std::string>& get_roles() const { return roles; }
  const std::vector<std::string>& get_grantees() const { return grantees; }
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::vector<std::string> roles;
  std::vector<std::string> grantees;
};

/*
 * @type RevokeRoleStmt
 * @brief REVOKE ROLE statement
 */
class RevokeRoleStmt : public DDLStmt {
 public:
  RevokeRoleStmt(std::list<std::string*>* r, std::list<std::string*>* g) {
    parser_slistval_to_vector(r, roles);
    parser_slistval_to_vector(g, grantees);
  }
  const std::vector<std::string>& get_roles() const { return roles; }
  const std::vector<std::string>& get_grantees() const { return grantees; }
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::vector<std::string> roles;
  std::vector<std::string> grantees;
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
  UnionQuery(bool u, QueryExpr* l, QueryExpr* r) : is_unionall(u), left(l), right(r) {}
  bool get_is_unionall() const { return is_unionall; }
  const QueryExpr* get_left() const { return left.get(); }
  const QueryExpr* get_right() const { return right.get(); }
  void analyze(const Catalog_Namespace::Catalog& catalog,
               Analyzer::Query& query) const override;

 private:
  bool is_unionall;
  std::unique_ptr<QueryExpr> left;
  std::unique_ptr<QueryExpr> right;
};

class SelectEntry : public Node {
 public:
  SelectEntry(Expr* e, std::string* r) : select_expr(e), alias(r) {}
  const Expr* get_select_expr() const { return select_expr.get(); }
  const std::string* get_alias() const { return alias.get(); }
  std::string to_string() const;

 private:
  std::unique_ptr<Expr> select_expr;
  std::unique_ptr<std::string> alias;
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
      : is_distinct(d), where_clause(w), having_clause(h) {
    if (s) {
      for (const auto e : *s) {
        select_clause.emplace_back(e);
      }
      delete s;
    }
    CHECK(f);
    for (const auto e : *f) {
      from_clause.emplace_back(e);
    }
    delete f;
    if (g) {
      for (const auto e : *g) {
        groupby_clause.emplace_back(e);
      }
      delete g;
    }
  }
  bool get_is_distinct() const { return is_distinct; }
  const std::list<std::unique_ptr<SelectEntry>>& get_select_clause() const {
    return select_clause;
  }
  const std::list<std::unique_ptr<TableRef>>& get_from_clause() const {
    return from_clause;
  }
  const Expr* get_where_clause() const { return where_clause.get(); }
  const std::list<std::unique_ptr<Expr>>& get_groupby_clause() const {
    return groupby_clause;
  }
  const Expr* get_having_clause() const { return having_clause.get(); }
  void analyze(const Catalog_Namespace::Catalog& catalog,
               Analyzer::Query& query) const override;
  std::string to_string() const;

 private:
  bool is_distinct;
  std::list<std::unique_ptr<SelectEntry>> select_clause; /* nullptr means SELECT * */
  std::list<std::unique_ptr<TableRef>> from_clause;
  std::unique_ptr<Expr> where_clause;
  std::list<std::unique_ptr<Expr>> groupby_clause;
  std::unique_ptr<Expr> having_clause;
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
      : colno(n), column(c), is_desc(d), nulls_first(f) {}
  int get_colno() const { return colno; }
  const ColumnRef* get_column() const { return column.get(); }
  bool get_is_desc() const { return is_desc; }
  bool get_nulls_first() const { return nulls_first; }

 private:
  int colno; /* 0 means use column name */
  std::unique_ptr<ColumnRef> column;
  bool is_desc;
  bool nulls_first;
};

/*
 * @type SelectStmt
 * @brief SELECT statement
 */
class SelectStmt : public DMLStmt {
 public:
  SelectStmt(QueryExpr* q, std::list<OrderSpec*>* o, int64_t l, int64_t f)
      : query_expr(q), limit(l), offset(f) {
    if (o) {
      for (const auto e : *o) {
        orderby_clause.emplace_back(e);
      }
      delete o;
    }
  }
  const QueryExpr* get_query_expr() const { return query_expr.get(); }
  const std::list<std::unique_ptr<OrderSpec>>& get_orderby_clause() const {
    return orderby_clause;
  }
  void analyze(const Catalog_Namespace::Catalog& catalog,
               Analyzer::Query& query) const override;

 private:
  std::unique_ptr<QueryExpr> query_expr;
  std::list<std::unique_ptr<OrderSpec>> orderby_clause;
  int64_t limit;
  int64_t offset;
};

/*
 * @type ShowCreateTableStmt
 * @brief shows create table statement to create table
 */
class ShowCreateTableStmt : public DDLStmt {
 public:
  ShowCreateTableStmt(std::string* tab) : table(tab) {}
  std::string get_create_stmt();
  void execute(const Catalog_Namespace::SessionInfo& session) override { CHECK(false); }

 private:
  std::unique_ptr<std::string> table;
};

/*
 * @type ExportQueryStmt
 * @brief COPY ( query ) TO file ...
 */
class ExportQueryStmt : public DDLStmt {
 public:
  ExportQueryStmt(std::string* q, std::string* p, std::list<NameValueAssign*>* o)
      : select_stmt(q), file_path(p) {
    if (o) {
      for (const auto e : *o) {
        options.emplace_back(e);
      }
      delete o;
    }
  }
  void execute(const Catalog_Namespace::SessionInfo& session) override;
  const std::string get_select_stmt() const { return *select_stmt; }

 private:
  std::unique_ptr<std::string> select_stmt;
  std::unique_ptr<std::string> file_path;
  std::list<std::unique_ptr<NameValueAssign>> options;
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
  const std::string& get_view_name() const { return view_name_; }
  const std::string& get_select_query() const { return select_query_; }
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  const std::string view_name_;
  const std::string select_query_;
  const bool if_not_exists_;
};

/*
 * @type DropViewStmt
 * @brief DROP VIEW statement
 */
class DropViewStmt : public DDLStmt {
 public:
  DropViewStmt(std::string* v, bool i) : view_name(v), if_exists(i) {}
  const std::string* get_view_name() const { return view_name.get(); }
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::unique_ptr<std::string> view_name;
  bool if_exists;
};

/*
 * @type CreateDBStmt
 * @brief CREATE DATABASE statement
 */
class CreateDBStmt : public DDLStmt {
 public:
  CreateDBStmt(std::string* n, std::list<NameValueAssign*>* l, const bool if_not_exists)
      : db_name(n), if_not_exists_(if_not_exists) {
    if (l) {
      for (const auto e : *l) {
        name_value_list.emplace_back(e);
      }
      delete l;
    }
  }
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::unique_ptr<std::string> db_name;
  std::list<std::unique_ptr<NameValueAssign>> name_value_list;
  bool if_not_exists_;
};

/*
 * @type DropDBStmt
 * @brief DROP DATABASE statement
 */
class DropDBStmt : public DDLStmt {
 public:
  explicit DropDBStmt(std::string* n, bool if_exists)
      : db_name(n), if_exists_(if_exists) {}
  auto const& getDatabaseName() { return db_name; }
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::unique_ptr<std::string> db_name;
  bool if_exists_;
};

/*
 * @type CreateUserStmt
 * @brief CREATE USER statement
 */
class CreateUserStmt : public DDLStmt {
 public:
  CreateUserStmt(std::string* n, std::list<NameValueAssign*>* l) : user_name(n) {
    if (l) {
      for (const auto e : *l) {
        name_value_list.emplace_back(e);
      }
      delete l;
    }
  }
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::unique_ptr<std::string> user_name;
  std::list<std::unique_ptr<NameValueAssign>> name_value_list;
};

/*
 * @type AlterUserStmt
 * @brief ALTER USER statement
 */
class AlterUserStmt : public DDLStmt {
 public:
  AlterUserStmt(std::string* n, std::list<NameValueAssign*>* l) : user_name(n) {
    if (l) {
      for (const auto e : *l) {
        name_value_list.emplace_back(e);
      }
      delete l;
    }
  }
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::unique_ptr<std::string> user_name;
  std::list<std::unique_ptr<NameValueAssign>> name_value_list;
};

/*
 * @type DropUserStmt
 * @brief DROP USER statement
 */
class DropUserStmt : public DDLStmt {
 public:
  DropUserStmt(std::string* n) : user_name(n) {}
  auto const& getUserName() { return user_name; }
  void execute(const Catalog_Namespace::SessionInfo& session) override;

 private:
  std::unique_ptr<std::string> user_name;
};

/*
 * @type InsertStmt
 * @brief super class for INSERT statements
 */
class InsertStmt : public DMLStmt {
 public:
  InsertStmt(std::string* t, std::list<std::string*>* c) : table(t) {
    if (c) {
      for (const auto e : *c) {
        column_list.emplace_back(e);
      }
      delete c;
    }
  }
  const std::string* get_table() const { return table.get(); }
  const std::list<std::unique_ptr<std::string>>& get_column_list() const {
    return column_list;
  }
  void analyze(const Catalog_Namespace::Catalog& catalog,
               Analyzer::Query& query) const override = 0;

 protected:
  std::unique_ptr<std::string> table;
  std::list<std::unique_ptr<std::string>> column_list;
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
      value_list.emplace_back(e);
    }
    delete v;
  }
  const std::list<std::unique_ptr<Expr>>& get_value_list() const { return value_list; }
  void analyze(const Catalog_Namespace::Catalog& catalog,
               Analyzer::Query& query) const override;

  size_t determineLeafIndex(const Catalog_Namespace::Catalog& catalog, size_t num_leafs);

 private:
  std::list<std::unique_ptr<Expr>> value_list;
};

/*
 * @type InsertQueryStmt
 * @brief INSERT INTO ... SELECT ...
 */
class InsertQueryStmt : public InsertStmt {
 public:
  InsertQueryStmt(std::string* t, std::list<std::string*>* c, QuerySpec* q)
      : InsertStmt(t, c), query(q) {}
  const QuerySpec* get_query() const { return query.get(); }
  void analyze(const Catalog_Namespace::Catalog& catalog,
               Analyzer::Query& query) const override;

 private:
  std::unique_ptr<QuerySpec> query;
};

/*
 * @type Assignment
 * @brief assignment in UPDATE statement
 */
class Assignment : public Node {
 public:
  Assignment(std::string* c, Expr* a) : column(c), assignment(a) {}
  const std::string* get_column() const { return column.get(); }
  const Expr* get_assignment() const { return assignment.get(); }

 private:
  std::unique_ptr<std::string> column;
  std::unique_ptr<Expr> assignment;
};

/*
 * @type UpdateStmt
 * @brief UPDATE statement
 */
class UpdateStmt : public DMLStmt {
 public:
  UpdateStmt(std::string* t, std::list<Assignment*>* a, Expr* w)
      : table(t), where_clause(w) {
    CHECK(a);
    for (const auto e : *a) {
      assignment_list.emplace_back(e);
    }
    delete a;
  }
  const std::string* get_table() const { return table.get(); }
  const std::list<std::unique_ptr<Assignment>>& get_assignment_list() const {
    return assignment_list;
  }
  const Expr* get_where_clause() const { return where_clause.get(); }
  void analyze(const Catalog_Namespace::Catalog& catalog,
               Analyzer::Query& query) const override;

 private:
  std::unique_ptr<std::string> table;
  std::list<std::unique_ptr<Assignment>> assignment_list;
  std::unique_ptr<Expr> where_clause;
};

/*
 * @type DeleteStmt
 * @brief DELETE statement
 */
class DeleteStmt : public DMLStmt {
 public:
  DeleteStmt(std::string* t, Expr* w) : table(t), where_clause(w) {}
  const std::string* get_table() const { return table.get(); }
  const Expr* get_where_clause() const { return where_clause.get(); }
  void analyze(const Catalog_Namespace::Catalog& catalog,
               Analyzer::Query& query) const override;

 private:
  std::unique_ptr<std::string> table;
  std::unique_ptr<Expr> where_clause;
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
struct ShouldInvalidateSessionsByDB<RenameDatabaseStmt> : public std::true_type {};
template <>
struct ShouldInvalidateSessionsByUser<RenameUserStmt> : public std::true_type {};

}  // namespace Parser

#endif  // PARSERNODE_H_
