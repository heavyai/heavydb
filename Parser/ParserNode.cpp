/**
 * @file    ParserNode.cpp
 * @author  Wei Hong <wei@map-d.com>
 * @brief   Functions for ParserNode classes
 * 
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/

#include <cassert>
#include <stdexcept>
#include <typeinfo>
#include <boost/algorithm/string/predicate.hpp>
#include "../Catalog/Catalog.h"
#include "ParserNode.h"
#include "../Planner/Planner.h"
#include "../Fragmenter/InsertOrderFragmenter.h"
#include "parser.h"

namespace Parser {
  SubqueryExpr::~SubqueryExpr() 
  {
    delete query;
  }

  ExistsExpr::~ExistsExpr() 
  {
    delete query;
  }

  InValues::~InValues()
  {
    for (auto p : *value_list)
      delete p;
    delete value_list;
  }

  BetweenExpr::~BetweenExpr() 
  {
    delete arg;
    delete lower;
    delete upper;
  }

  LikeExpr::~LikeExpr() 
  {
    delete arg;
    delete like_string;
    if (escape_string != nullptr)
      delete escape_string;
  }

  ColumnRef::~ColumnRef() 
  {
    if (table != nullptr)
      delete table;
    if (column != nullptr)
      delete column;
  }

  FunctionRef::~FunctionRef() 
  {
    delete name;
    if (arg != nullptr)
      delete arg;
  }
  
  TableRef::~TableRef() 
  {
    delete table_name;
    if (range_var != nullptr)
      delete range_var;
  }

  ColumnConstraintDef::~ColumnConstraintDef() 
  {
    if (defaultval != nullptr)
      delete defaultval;
    if (check_condition != nullptr)
      delete check_condition;
    if (foreign_table != nullptr)
      delete foreign_table;
    if (foreign_column != nullptr)
      delete foreign_column;
  }

  ColumnDef::~ColumnDef() 
  {
    delete column_name;
    delete column_type;
    if (compression != nullptr)
      delete compression;
    if (column_constraint != nullptr)
      delete column_constraint;
  }

  UniqueDef::~UniqueDef() 
  {
    for (auto p : *column_list)
      delete p;
    delete column_list;
  }

  ForeignKeyDef::~ForeignKeyDef() 
  {
    for (auto p : *column_list)
      delete p;
    delete column_list;
    delete foreign_table;
    if (foreign_column_list != nullptr) {
      for (auto p : *foreign_column_list)
        delete p;
      delete foreign_column_list;
    }
  }

  CreateTableStmt::~CreateTableStmt() 
  {
    delete table;
    for (auto p : *table_element_list)
      delete p;
    delete table_element_list;
    if (storage_options != nullptr) {
      for (auto p : *storage_options)
        delete p;
      delete storage_options;
    }
  }

  SelectEntry::~SelectEntry()
  {
    delete select_expr;
    if (alias != nullptr)
      delete alias;
  }

  QuerySpec::~QuerySpec() 
  {
    if (select_clause != nullptr) {
      for (auto p : *select_clause)
        delete p;
      delete select_clause;
    }
    for (auto p : *from_clause)
      delete p;
    delete from_clause;
    if (where_clause != nullptr)
      delete where_clause;
    if (groupby_clause != nullptr)
      delete groupby_clause;
    if (having_clause != nullptr)
      delete having_clause;
  }

  SelectStmt::~SelectStmt()
  {
    delete query_expr;
    if (orderby_clause != nullptr) {
      for (auto p : *orderby_clause)
        delete p;
      delete orderby_clause;
    }
  }

  CreateViewStmt::~CreateViewStmt() 
  {
    delete view_name;
    if (column_list != nullptr) {
      for (auto p : *column_list)
        delete p;
      delete column_list;
    }
    delete query;
    if (matview_options != nullptr) {
      for (auto p : *matview_options)
        delete p;
      delete matview_options;
    }
  }

  InsertStmt::~InsertStmt()
  {
    delete table;
    if (column_list != nullptr) {
      for (auto p : *column_list)
        delete p;
      delete column_list;
    }
  }

  InsertValuesStmt::~InsertValuesStmt()
  {
    for (auto p : *value_list)
      delete p;
    delete value_list;
  }

  UpdateStmt::~UpdateStmt()
  {
    delete table;
    for (auto p : *assignment_list)
      delete p;
    delete assignment_list;
    if (where_clause != nullptr)
      delete where_clause;
  }

  DeleteStmt::~DeleteStmt()
  {
    delete table;
    if (where_clause != nullptr)
      delete where_clause;
  }
  
  Analyzer::Expr *
  NullLiteral::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query, bool allow_tlist_ref) const 
  {
    Analyzer::Constant *c = new Analyzer::Constant(kNULLT, true);
    Datum d;
    d.bigintval = 0;
    c->set_constval(d);
    return c;
  }
  
  Analyzer::Expr *
  StringLiteral::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query, bool allow_tlist_ref) const 
  {
    SQLTypeInfo ti;
    ti.type = kVARCHAR;
    ti.dimension = stringval->length();
    ti.scale = 0;
    Datum d;
    d.stringval = new std::string(*stringval);
    return new Analyzer::Constant(ti, false, d);
  }

  Analyzer::Expr *
  IntLiteral::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query, bool allow_tlist_ref) const 
  {
    SQLTypes t;
    Datum d;
    if (intval >= INT16_MIN && intval <= INT16_MAX) {
      t = kSMALLINT;
      d.smallintval = (int16_t)intval;
    } else if (intval >= INT32_MIN && intval <= INT32_MAX) {
      t = kINT;
      d.intval = (int32_t)intval;
    } else {
      t = kBIGINT;
      d.bigintval = intval;
    }
    Analyzer::Constant *c = new Analyzer::Constant(t, false, d);
    return c;
  }

  Analyzer::Expr *
  FixedPtLiteral::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query, bool allow_tlist_ref) const 
  {
    SQLTypeInfo ti;
    ti.type = kNUMERIC;
    ti.dimension = 0; // to be filled in by StringToDatum()
    ti.scale = 0;
    Datum d = StringToDatum(*fixedptval, ti);
    return new Analyzer::Constant(ti, false, d);
  }

  Analyzer::Expr *
  FloatLiteral::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query, bool allow_tlist_ref) const 
  {
    Datum d;
    d.floatval = floatval;
    return new Analyzer::Constant(kFLOAT, false, d);
  }
  
  Analyzer::Expr *
  DoubleLiteral::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query, bool allow_tlist_ref) const 
  {
    Datum d;
    d.doubleval = doubleval;
    return new Analyzer::Constant(kDOUBLE, false, d);
  }

  Analyzer::Expr *
  UserLiteral::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query, bool allow_tlist_ref) const 
  {
    throw std::runtime_error("USER literal not supported yet.");
    return nullptr;
  }

  Analyzer::Expr *
  OperExpr::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query, bool allow_tlist_ref) const 
  {
    SQLTypeInfo result_type, left_type, right_type;
    SQLTypeInfo new_left_type, new_right_type;
    Analyzer::Expr *left_expr, *right_expr;
    left_expr = left->analyze(catalog, query, allow_tlist_ref);
    left_type = left_expr->get_type_info();
    if (right == nullptr) {
      return new Analyzer::UOper(left_type, left_expr->get_contains_agg(), optype, left_expr);
    }
    SQLQualifier qual = kONE;
    if (typeid(*right) == typeid(SubqueryExpr))
      qual = dynamic_cast<SubqueryExpr*>(right)->get_qualifier();
    right_expr = right->analyze(catalog, query, allow_tlist_ref);
    bool has_agg = (left_expr->get_contains_agg() || right_expr->get_contains_agg());
    right_type = right_expr->get_type_info();
    result_type = Analyzer::BinOper::analyze_type_info(optype, left_type, right_type, &new_left_type, &new_right_type);
    if (left_type != new_left_type)
      left_expr = left_expr->add_cast(new_left_type);
    if (right_type != new_right_type)
      right_expr = right_expr->add_cast(new_right_type);
    return new Analyzer::BinOper(result_type, has_agg, optype, qual, left_expr, right_expr);
  }

  Analyzer::Expr *
  SubqueryExpr::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query, bool allow_tlist_ref) const 
  {
    throw std::runtime_error("Subqueries are not supported yet.");
    return nullptr;
  }

  Analyzer::Expr *
  IsNullExpr::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query, bool allow_tlist_ref) const 
  {
    Analyzer::Expr *arg_expr = arg->analyze(catalog, query, allow_tlist_ref);
    Analyzer::Expr *result = new Analyzer::UOper(kBOOLEAN, kISNULL, arg_expr);
    if (is_not)
      result = new Analyzer::UOper(kBOOLEAN, kNOT, result);
    return result;
  }

  Analyzer::Expr *
  InSubquery::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query, bool allow_tlist_ref) const 
  {
    throw std::runtime_error("Subqueries are not supported yet.");
    return nullptr;
  }

  Analyzer::Expr *
  InValues::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query, bool allow_tlist_ref) const 
  {
    Analyzer::Expr *arg_expr = arg->analyze(catalog, query, allow_tlist_ref);
    std::list<Analyzer::Expr*> *value_exprs = new std::list<Analyzer::Expr*>();
    for (auto p : *value_list) {
      Analyzer::Expr *e = p->analyze(catalog, query, allow_tlist_ref);
      value_exprs->push_back(e->add_cast(arg_expr->get_type_info()));
    }
    Analyzer::Expr *result = new Analyzer::InValues(arg_expr, value_exprs);
    if (is_not)
      result = new Analyzer::UOper(kBOOLEAN, kNOT, result);
    return result;
  }

  Analyzer::Expr *
  BetweenExpr::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query, bool allow_tlist_ref) const 
  {
    Analyzer::Expr *arg_expr = arg->analyze(catalog, query, allow_tlist_ref);
    Analyzer::Expr *lower_expr = lower->analyze(catalog, query, allow_tlist_ref);
    Analyzer::Expr *upper_expr = upper->analyze(catalog, query, allow_tlist_ref);
    SQLTypeInfo new_left_type, new_right_type;
    (void)Analyzer::BinOper::analyze_type_info(kGE, arg_expr->get_type_info(), lower_expr->get_type_info(), &new_left_type, &new_right_type);
    Analyzer::BinOper *lower_pred = new Analyzer::BinOper(kBOOLEAN, kGE, kONE, arg_expr->add_cast(new_left_type), lower_expr->add_cast(new_right_type));
    (void)Analyzer::BinOper::analyze_type_info(kLE, arg_expr->get_type_info(), lower_expr->get_type_info(), &new_left_type, &new_right_type);
    Analyzer::BinOper *upper_pred = new Analyzer::BinOper(kBOOLEAN, kLE, kONE, arg_expr->deep_copy()->add_cast(new_left_type), upper_expr->add_cast(new_right_type));
    Analyzer::Expr *result = new Analyzer::BinOper(kBOOLEAN, kAND, kONE, lower_pred, upper_pred);
    if (is_not)
      result = new Analyzer::UOper(kBOOLEAN, kNOT, result);
    return result;
  }

  Analyzer::Expr *
  LikeExpr::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query, bool allow_tlist_ref) const 
  {
    Analyzer::Expr *arg_expr = arg->analyze(catalog, query, allow_tlist_ref);
    Analyzer::Expr *like_expr = like_string->analyze(catalog, query, allow_tlist_ref);
    Analyzer::Expr *escape_expr = escape_string == nullptr ? nullptr: escape_string->analyze(catalog, query, allow_tlist_ref);
    if (!IS_STRING(arg_expr->get_type_info().type))
      throw std::runtime_error("expression before LIKE must be of a string type.");
    if (!IS_STRING(like_expr->get_type_info().type))
      throw std::runtime_error("expression after LIKE must be of a string type.");
    if (escape_expr != nullptr && !IS_STRING(escape_expr->get_type_info().type))
      throw std::runtime_error("expression after ESCAPE must be of a string type.");
    Analyzer::Expr *result = new Analyzer::LikeExpr(arg_expr, like_expr, escape_expr);
    if (is_not)
      result = new Analyzer::UOper(kBOOLEAN, kNOT, result);
    return result;
  }

  Analyzer::Expr *
  ExistsExpr::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query, bool allow_tlist_ref) const 
  {
    throw std::runtime_error("Subqueries are not supported yet.");
    return nullptr;
  }

  Analyzer::Expr *
  ColumnRef::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query, bool allow_tlist_ref) const 
  {
    int table_id;
    int rte_idx;
    const ColumnDescriptor *cd;
    if (column == nullptr)
      throw std::runtime_error("invalid column name *.");
    if (table != nullptr) {
      rte_idx = query.get_rte_idx(*table);
      if (rte_idx < 0)
        throw std::runtime_error("range variable or table name " + *table + " does not exist.");
      Analyzer::RangeTblEntry *rte = query.get_rte(rte_idx);
      cd = rte->get_column_desc(catalog, *column);
      if (cd == nullptr)
        throw std::runtime_error("Column name " + *column + " does not exist.");
      table_id = rte->get_table_id();
    } else {
      bool found = false;
      int i = 0;
      for (auto rte : query.get_rangetable()) {
        cd = rte->get_column_desc(catalog, *column);
        if (cd != nullptr && !found) {
          found = true;
          rte_idx = i;
          table_id = rte->get_table_id();
        } else if (cd != nullptr && found)
          throw std::runtime_error("Column name " + *column + " is ambiguous.");
        i++;
      }
      if (cd == nullptr && allow_tlist_ref) {
        // check if this is a reference to a targetlist entry
        bool found = false;
        int varno;
        int i = 1;
        Analyzer::TargetEntry *tle;
        for (auto p : query.get_targetlist()) {
          if (*column == p->get_resname() && !found) {
            found = true;
            varno = i;
            tle = p;
          } else if (*column == p->get_resname() && found)
            throw std::runtime_error("Output alias " + *column + " is ambiguous.");
          i++;
        }
        if (found) {
          if (typeid(*tle->get_expr()) == typeid(Analyzer::Var)) {
            Analyzer::Var *v = dynamic_cast<Analyzer::Var*>(tle->get_expr());
            if (v->get_which_row() == Analyzer::Var::kGROUPBY)
              return v->deep_copy();
          }
          return new Analyzer::Var(tle->get_expr()->get_type_info(), Analyzer::Var::kOUTPUT, varno);
        }
      }
      if (cd == nullptr)
        throw std::runtime_error("Column name " + *column + " does not exist.");
    }
    return new Analyzer::ColumnVar(cd->columnType, table_id, cd->columnId, rte_idx, cd->compression, cd->comp_param);
  }

  Analyzer::Expr *
  FunctionRef::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query, bool allow_tlist_ref) const 
  {
    SQLTypeInfo result_type;
    SQLAgg agg_type;
    Analyzer::Expr *arg_expr;
    bool is_distinct = false;
    if (boost::iequals(*name, "count")) {
      result_type.type = kBIGINT;
      agg_type = kCOUNT;
      if (arg == nullptr)
        arg_expr = nullptr;
      else
        arg_expr = arg->analyze(catalog, query, allow_tlist_ref);
      is_distinct = distinct;
    }
    else if (boost::iequals(*name, "min")) {
      agg_type = kMIN;
      arg_expr = arg->analyze(catalog, query, allow_tlist_ref);
      result_type = arg_expr->get_type_info();
    }
    else if (boost::iequals(*name, "max")) {
      agg_type = kMAX;
      arg_expr = arg->analyze(catalog, query, allow_tlist_ref);
      result_type = arg_expr->get_type_info();
    }
    else if (boost::iequals(*name, "avg")) {
      agg_type = kAVG;
      arg_expr = arg->analyze(catalog, query, allow_tlist_ref);
      result_type.type = kDOUBLE;
    }
    else if (boost::iequals(*name, "sum")) {
      agg_type = kSUM;
      arg_expr = arg->analyze(catalog, query, allow_tlist_ref);
      result_type = arg_expr->get_type_info();
    }
    else
      throw std::runtime_error("invalid function name: " + *name);
    int naggs = query.get_num_aggs();
    query.set_num_aggs(naggs+1);
    return new Analyzer::AggExpr(result_type, agg_type, arg_expr, is_distinct);
  }

  Analyzer::Expr *
  CastExpr::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query, bool allow_tlist_ref) const
  {
    target_type->check_type();
    Analyzer::Expr *arg_expr = arg->analyze(catalog, query, allow_tlist_ref);
    SQLTypeInfo ti;
    ti.type = target_type->get_type();
    ti.dimension = target_type->get_param1();
    ti.scale = target_type->get_param2();
    ti.notnull = arg_expr->get_type_info().notnull;
    return arg_expr->add_cast(ti);
  }

  CaseExpr::~CaseExpr()
  {
    for (auto p : *when_then_list)
      delete p;
    delete when_then_list;
    if (else_expr != nullptr)
      delete else_expr;
  }

  Analyzer::Expr *
  CaseExpr::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query, bool allow_tlist_ref) const
  {
    SQLTypeInfo ti;
    ti.type = kNULLT;
    std::list<std::pair<Analyzer::Expr*, Analyzer::Expr*>> expr_pair_list;
    bool has_agg = false;
    for (auto p : *when_then_list) {
      Analyzer::Expr *e1, *e2;
      e1 = p->get_expr1()->analyze(catalog, query, allow_tlist_ref);
      if (e1->get_type_info().type != kBOOLEAN)
        throw std::runtime_error("Only boolean expressions can be used after WHEN.");
      e2 = p->get_expr2()->analyze(catalog, query, allow_tlist_ref);
      if (ti.type == kNULLT)
        ti = e2->get_type_info();
      else if (e2->get_type_info().type == kNULLT)
        e2->set_type_info(ti);
      else if (ti != e2->get_type_info()) {
        if (IS_STRING(ti.type) && IS_STRING(e2->get_type_info().type))
          ti = Analyzer::BinOper::common_string_type(ti, e2->get_type_info());
        else if (IS_NUMBER(ti.type) && IS_NUMBER(e2->get_type_info().type))
          ti = Analyzer::BinOper::common_numeric_type(ti, e2->get_type_info());
        else
          throw std::runtime_error("expressions in THEN clause must be of the same or compatible types.");
      }
      if (e2->get_contains_agg())
        has_agg = true;
      expr_pair_list.push_back(std::make_pair(e1, e2));
    }
    Analyzer::Expr *else_e = nullptr;
    if (else_expr != nullptr) {
      else_e = else_expr->analyze(catalog, query, allow_tlist_ref);
      if (else_e->get_contains_agg())
        has_agg = true;
      if (else_e->get_type_info().type == kNULLT)
        else_e->set_type_info(ti);
      else if (ti != else_e->get_type_info()) {
        if (IS_STRING(ti.type) && IS_STRING(else_e->get_type_info().type))
          ti = Analyzer::BinOper::common_string_type(ti, else_e->get_type_info());
        else if (IS_NUMBER(ti.type) && IS_NUMBER(else_e->get_type_info().type))
          ti = Analyzer::BinOper::common_numeric_type(ti, else_e->get_type_info());
        else
          throw std::runtime_error("expressions in ELSE clause must be of the same or compatible types as those in the THEN clauses.");
      }
    }
    std::list<std::pair<Analyzer::Expr*, Analyzer::Expr*>> cast_expr_pair_list;
    for (auto p : expr_pair_list) {
      cast_expr_pair_list.push_back(std::make_pair(p.first, p.second->add_cast(ti)));;
    }
    if (else_expr != nullptr)
      else_e = else_e->add_cast(ti);
    return new Analyzer::CaseExpr(ti, has_agg, cast_expr_pair_list, else_e);
  }

  std::string
  CaseExpr::to_string() const
  {
    std::string str("CASE ");
    for (auto p : *when_then_list) {
      str += "WHEN " + p->get_expr1()->to_string() + " THEN " + p->get_expr2()->to_string() + " ";
    }
    if (else_expr != nullptr)
      str += "ELSE " + else_expr->to_string();
    str += " END";
    return str;
  }

  Analyzer::Expr *
  ExtractExpr::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query, bool allow_tlist_ref) const
  {
    ExtractField fieldno;
    if (boost::iequals(*field, "year"))
      fieldno = kYEAR;
    else if (boost::iequals(*field, "month"))
      fieldno = kMONTH;
    else if (boost::iequals(*field, "day"))
      fieldno = kDAY;
    else if (boost::iequals(*field, "hour"))
      fieldno = kHOUR;
    else if (boost::iequals(*field, "minute"))
      fieldno = kMINUTE;
    else if (boost::iequals(*field, "second"))
      fieldno = kSECOND;
    else if (boost::iequals(*field, "dow"))
      fieldno = kDOW;
    else if (boost::iequals(*field, "doy"))
      fieldno = kDOY;
    else if (boost::iequals(*field, "epoch"))
      fieldno = kEPOCH;
    else
      throw std::runtime_error("Invalid field in EXTRACT function " + *field);
    Analyzer::Expr *from_expr = from_arg->analyze(catalog, query, allow_tlist_ref);
    if (!IS_TIME(from_expr->get_type_info().type))
      throw std::runtime_error("Only TIME, TIMESTAMP and DATE types can be in EXTRACT function.");
    switch (from_expr->get_type_info().type) {
      case kTIME:
        if (fieldno != kHOUR && fieldno != kMINUTE && fieldno != kSECOND)
          throw std::runtime_error("Cannot EXTRACT " + *field + " from TIME.");
        break;
      case kDATE:
        if (fieldno != kYEAR && fieldno != kMONTH && fieldno != kDAY && fieldno != kDOW && fieldno != kDOY)
          throw std::runtime_error("Cannot EXTRACT " + *field + " from DATE.");
        break;
      default:
        break;
    }
    SQLTypeInfo ti;
    ti.type = kBIGINT; // standard says DOUBLE but int is much more efficient
    ti.dimension = 0;
    ti.scale = 0;
    ti.notnull = from_expr->get_type_info().notnull;
    Analyzer::Constant *c = dynamic_cast<Analyzer::Constant*>(from_expr);
    if (c != nullptr) {
      c->set_type_info(ti);
      Datum d;
      d.bigintval = ExtractFromTime(fieldno, c->get_constval().timeval);
      c->set_constval(d);
      return c;
    }
    return new Analyzer::ExtractExpr(ti, from_expr->get_contains_agg(), fieldno, from_expr);
  }

  std::string
  ExtractExpr::to_string() const
  {
    std::string str("EXTRACT(");
    str += *field + " FROM " + from_arg->to_string() + ")";
    return str;
  }

  void
  UnionQuery::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const
  {
    left->analyze(catalog, query);
    Analyzer::Query *right_query = new Analyzer::Query();
    right->analyze(catalog, *right_query);
    query.set_next_query(right_query);
    query.set_is_unionall(is_unionall);
  }

  void
  QuerySpec::analyze_having_clause(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const
  {
    Analyzer::Expr *p = nullptr;
    if (having_clause != nullptr) {
      p = having_clause->analyze(catalog, query, true);
      if (p->get_type_info().type != kBOOLEAN)
        throw std::runtime_error("Only boolean expressions can be in HAVING clause.");
      p->check_group_by(query.get_group_by());
    }
    query.set_having_predicate(p);
  }

  void
  QuerySpec::analyze_group_by(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const
  {
    std::list<Analyzer::Expr*> *groupby = nullptr;
    if (groupby_clause != nullptr) {
      groupby = new std::list<Analyzer::Expr*>();
      int gexpr_no = 1;
      Analyzer::Expr *gexpr;
      const std::vector<Analyzer::TargetEntry*> &tlist = query.get_targetlist();
      for (auto c : *groupby_clause) {
        // special-case ordinal numbers in GROUP BY
        if (dynamic_cast<Literal*>(c) != nullptr) {
          IntLiteral *i = dynamic_cast<IntLiteral*>(c);
          if (i == nullptr)
            throw std::runtime_error("Invalid literal in GROUP BY clause.");
          int varno = (int)i->get_intval();
          if (varno <= 0 || varno > tlist.size())
            throw std::runtime_error("Invalid ordinal number in GROUP BY clause.");
          if (tlist[varno-1]->get_expr()->get_contains_agg())
            throw std::runtime_error("Ordinal number in GROUP BY cannot reference an expression containing aggregate functions.");
          gexpr = new Analyzer::Var(tlist[varno-1]->get_expr()->get_type_info(), Analyzer::Var::kOUTPUT, varno);
        } else {
          gexpr = c->analyze(catalog, query, true);
        }
        if (typeid(*gexpr) == typeid(Analyzer::Var)) {
          Analyzer::Var *v = dynamic_cast<Analyzer::Var*>(gexpr);
          int n = v->get_varno();
          gexpr = tlist[n - 1]->get_expr();
          v->set_which_row(Analyzer::Var::kGROUPBY);
          v->set_varno(gexpr_no);
          tlist[n - 1]->set_expr(v);
        }
        groupby->push_back(gexpr);
        gexpr_no++;
      }
    }
    if (query.get_num_aggs() > 0 || groupby != nullptr) {
      for (auto t : query.get_targetlist()) {
        Analyzer::Expr *e = t->get_expr();
        e->check_group_by(groupby);
      }
    }
    query.set_group_by(groupby);
  }

  void
  QuerySpec::analyze_where_clause(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const
  {
    if (where_clause == nullptr) {
      query.set_where_predicate(nullptr);
      return;
    }
    Analyzer::Expr *p = where_clause->analyze(catalog, query);
    if (p->get_type_info().type != kBOOLEAN)
      throw std::runtime_error("Only boolean expressions can be in WHERE clause.");
    query.set_where_predicate(p);
  }

  void
  QuerySpec::analyze_select_clause(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const
  {
    std::vector<Analyzer::TargetEntry*> &tlist = query.get_targetlist_nonconst();
    if (select_clause == nullptr) {
      // this means SELECT *
      int rte_idx = 0;
      for (auto rte : query.get_rangetable()) {
        rte->expand_star_in_targetlist(catalog, tlist, rte_idx++);
      }
    }
    else {
      for (auto p : *select_clause) {
        const Parser::Expr *select_expr = p->get_select_expr();
        // look for the case of range_var.*
        if (typeid(*select_expr) == typeid(ColumnRef) &&
            dynamic_cast<const ColumnRef*>(select_expr)->get_column() == nullptr) {
            const std::string *range_var_name = dynamic_cast<const ColumnRef*>(select_expr)->get_table();
            int rte_idx = query.get_rte_idx(*range_var_name);
            if (rte_idx < 0)
              throw std::runtime_error("invalid range variable name: " + *range_var_name);
            Analyzer::RangeTblEntry *rte = query.get_rte(rte_idx);
            rte->expand_star_in_targetlist(catalog, tlist, rte_idx);
        }
        else {
          Analyzer::Expr *e = select_expr->analyze(catalog, query);
          std::string resname;

          if (p->get_alias() != nullptr)
            resname = *p->get_alias();
          else if (typeid(*e) == typeid(Analyzer::ColumnVar)) {
            Analyzer::ColumnVar *colvar = dynamic_cast<Analyzer::ColumnVar*>(e);
            const ColumnDescriptor *col_desc = catalog.getMetadataForColumn(colvar->get_table_id(), colvar->get_column_id());
            resname = col_desc->columnName;
          }
          Analyzer::TargetEntry *tle = new Analyzer::TargetEntry(resname, e);
          tlist.push_back(tle);
        }
      }
    }
  }

  void
  QuerySpec::analyze_from_clause(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const
  {
    Analyzer::RangeTblEntry *rte;
    for (auto p : *from_clause) {
      const TableDescriptor *table_desc;
      table_desc = catalog.getMetadataForTable(*p->get_table_name());
      if (table_desc == nullptr)
        throw std::runtime_error("Table " + *p->get_table_name() + " does not exist." );
      if (table_desc->isView && !table_desc->isMaterialized)
        throw std::runtime_error("Non-materialized view " + *p->get_table_name() + " is not supported yet.");
      std::string range_var;
      if (p->get_range_var() == nullptr)
        range_var = *p->get_table_name();
      else
        range_var = *p->get_range_var();
      rte = new Analyzer::RangeTblEntry(range_var, table_desc, nullptr);
      query.add_rte(rte);
    }
  }

  void
  QuerySpec::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const
  {
    query.set_is_distinct(is_distinct);
    analyze_from_clause(catalog, query);
    analyze_select_clause(catalog, query);
    analyze_where_clause(catalog, query);
    analyze_group_by(catalog, query);
    analyze_having_clause(catalog, query);
  }

  void
  SelectStmt::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const
  {
    query.set_stmt_type(kSELECT);
    query.set_limit(limit);
    query.set_offset(offset);
    query_expr->analyze(catalog, query);
    if (orderby_clause == nullptr) {
      query.set_order_by(nullptr);
      return;
    }
    const std::vector<Analyzer::TargetEntry*> &tlist = query.get_targetlist();
    std::list<Analyzer::OrderEntry> *order_by = new std::list<Analyzer::OrderEntry>();
    for (auto p : *orderby_clause) {
      int tle_no = p->get_colno();
      if (tle_no == 0) {
        // use column name
        // search through targetlist for matching name
        const std::string *name = p->get_column()->get_column();
        tle_no = 1;
        bool found = false;
        for (auto tle : tlist) {
          if (tle->get_resname() == *name) {
            found = true;
            break;
          }
          tle_no++;
        }
        if (!found)
          throw std::runtime_error("invalid name in order by: " + *name);
      }
      order_by->push_back(Analyzer::OrderEntry(tle_no, p->get_is_desc(), p->get_nulls_first()));
    }
    query.set_order_by(order_by);
  }

  std::string
  SQLType::to_string() const
  {
    std::string str;
    switch (type) {
      case kBOOLEAN:
        str = "BOOLEAN";
        break;
      case kCHAR:
        str = "CHAR(" + boost::lexical_cast<std::string>(param1) + ")";
        break;
      case kVARCHAR:
        str = "VARCHAR(" + boost::lexical_cast<std::string>(param1) + ")";
        break;
      case kTEXT:
        str = "TEXT";
        break;
      case kNUMERIC:
        str = "NUMERIC(" + boost::lexical_cast<std::string>(param1);
        if (param2 > 0)
          str += ", " + boost::lexical_cast<std::string>(param2);
        str += ")";
        break;
      case kDECIMAL:
        str = "DECIMAL(" + boost::lexical_cast<std::string>(param1);
        if (param2 > 0)
          str += ", " + boost::lexical_cast<std::string>(param2);
        str += ")";
        break;
      case kBIGINT:
        str = "BIGINT";
        break;
      case kINT:
        str = "INT";
        break;
      case kSMALLINT:
        str = "SMALLINT";
        break;
      case kFLOAT:
        str = "FLOAT";
        break;
      case kDOUBLE:
        str = "DOUBLE";
        break;
      case kTIME:
        str = "TIME";
        if (param1 < 6)
          str += "(" + boost::lexical_cast<std::string>(param1) + ")";
        break;
      case kTIMESTAMP:
        str = "TIMESTAMP";
        if (param1 < 6)
          str += "(" + boost::lexical_cast<std::string>(param1) + ")";
        break;
      case kDATE:
        str = "DATE";
        break;
      default:
        assert(false);
        break;
    }
    return str;
  }

  std::string
  SelectEntry::to_string() const
  {
    std::string str = select_expr->to_string();
    if (alias != nullptr)
      str += " AS " + *alias;
    return str;
  }

  std::string
  TableRef::to_string() const
  {
    std::string str = *table_name;
    if (range_var != nullptr)
      str += " " + *range_var;
    return str;
  }

  std::string
  ColumnRef::to_string() const
  {
    std::string str;
    if (table == nullptr)
      str = *column;
    else if (column == nullptr)
      str = *table + ".*";
    else
      str = *table + "." + *column;
    return str;
  }

  std::string
  OperExpr::to_string() const
  {
    std::string op_str[] = { "=", "<>", "<", ">", "<=", ">=", " AND ", " OR ", "NOT", "-", "+", "*", "/" };
    std::string str;
    if (optype == kUMINUS)
      str = "-(" + left->to_string() + ")";
    else if (optype == kNOT)
      str = "NOT (" + left->to_string() + ")";
    else
      str = "(" + left->to_string() + op_str[optype] + right->to_string() + ")";
    return str;
  }

  std::string
  InExpr::to_string() const
  {
    std::string str = arg->to_string();
    if (is_not)
      str += " NOT IN ";
    else
      str += " IN ";
    return str;
  }

  std::string
  ExistsExpr::to_string() const
  {
    return "EXISTS (" + query->to_string() + ")";
  }

  std::string
  SubqueryExpr::to_string() const
  {
    std::string str;
    if (qualifier == kANY)
      str = "ANY (";
    else if (qualifier == kALL)
      str = "ALL (";
    else
      str = "(";
    str += query->to_string();
    str += ")";
    return str;
  }

  std::string
  IsNullExpr::to_string() const
  {
    std::string str = arg->to_string();
    if (is_not)
      str += " IS NOT NULL";
    else
      str += " IS NULL";
    return str;
  }

  std::string
  InSubquery::to_string() const
  {
    std::string str = InExpr::to_string();
    str += subquery->to_string();
    return str;
  }

  std::string
  InValues::to_string() const
  {
    std::string str = InExpr::to_string() + "(";
    bool notfirst = false;
    for (auto p : *value_list) {
      if (notfirst)
        str += ", ";
      else
        notfirst = true;
      str += p->to_string();
    }
    str += ")";
    return str;
  }

  std::string
  BetweenExpr::to_string() const
  {
    std::string str = arg->to_string();
    if (is_not)
      str += " NOT BETWEEN ";
    else
      str += " BETWEEN ";
    str += lower->to_string() + " AND " + upper->to_string();
    return str;
  }

  std::string
  LikeExpr::to_string() const
  {
    std::string str = arg->to_string();
    if (is_not)
      str += " NOT LIKE ";
    else
      str += " LIKE ";
    str += like_string->to_string();
    if (escape_string != nullptr)
      str += " ESCAPE " + escape_string->to_string();
    return str;
  }

  std::string
  FunctionRef::to_string() const
  {
    std::string str = *name + "(";
    if (distinct)
      str += "DISTINCT ";
    if (arg == nullptr)
      str += "*)";
    else
      str += arg->to_string() + ")";
    return str;
  }

  std::string
  QuerySpec::to_string() const
  {
    std::string query_str = "SELECT ";
    if (is_distinct)
      query_str += "DISTINCT ";
    if (select_clause == nullptr)
      query_str += "* ";
    else {
      bool notfirst = false;
      for (auto p : *select_clause) {
        if (notfirst)
          query_str += ", ";
        else
          notfirst = true;
        query_str += p->to_string();
      }
    }
    query_str += " FROM ";
    bool notfirst = false;
    for (auto p : *from_clause) {
      if (notfirst)
        query_str += ", ";
      else
        notfirst = true;
      query_str += p->to_string();
    }
    if (where_clause != nullptr)
      query_str += " WHERE " + where_clause->to_string();
    if (groupby_clause != nullptr) {
      query_str += " GROUP BY ";
      bool notfirst = false;
      for (auto p : *groupby_clause) {
        if (notfirst)
          query_str += ", ";
        else
          notfirst = true;
        query_str += p->to_string();
      }
    }
    if (having_clause != nullptr) {
      query_str += " HAVING " + having_clause->to_string();
    }
    query_str += ";";
    return query_str;
  }

  void
  InsertStmt::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const
  {
    query.set_stmt_type(kINSERT);
    const TableDescriptor *td = catalog.getMetadataForTable(*table);
    if (td == nullptr)
      throw std::runtime_error("Table " + *table + " does not exist.");
    if (td->isView && !td->isMaterialized)
      throw std::runtime_error("Insert to views is not supported yet.");
    Analyzer::RangeTblEntry *rte = new Analyzer::RangeTblEntry(*table, td, nullptr);
    query.set_result_table_id(td->tableId);
    std::list<int> result_col_list;
    if (column_list == nullptr) {
      const std::list<const ColumnDescriptor *> all_cols = catalog.getAllColumnMetadataForTable(td->tableId);
      for (auto cd : all_cols)
        result_col_list.push_back(cd->columnId);
    } else {
      for (auto c : *column_list) {
        const ColumnDescriptor *cd = catalog.getMetadataForColumn(td->tableId, *c);
        if (cd == nullptr)
          throw std::runtime_error("Column " + *c + " does not exist.");
        result_col_list.push_back(cd->columnId);
      }
    }
    query.set_result_col_list(result_col_list);
  }

  void
  InsertValuesStmt::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const
  {
    InsertStmt::analyze(catalog, query);
    std::vector<Analyzer::TargetEntry*> &tlist = query.get_targetlist_nonconst();
    std::list<int>::const_iterator it = query.get_result_col_list().begin();
    for (auto v : *value_list) {
      Analyzer::Expr *e = v->analyze(catalog, query);
      const ColumnDescriptor *cd = catalog.getMetadataForColumn(query.get_result_table_id(), *it);
      assert (cd != nullptr);
      e = e->add_cast(cd->columnType);
      tlist.push_back(new Analyzer::TargetEntry("", e));
      ++it;
    }
  }

  void
  InsertQueryStmt::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &insert_query) const
  {
    InsertStmt::analyze(catalog, insert_query);
    query->analyze(catalog, insert_query);
  }

  void
  UpdateStmt::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const
  {
    throw std::runtime_error("UPDATE statement not supported yet.");
  }

  void
  DeleteStmt::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const
  {
    throw std::runtime_error("DELETE statement not supported yet.");
  }

  void
  SQLType::check_type()
  {
    switch (type) {
      case kCHAR:
      case kVARCHAR:
        if (param1 <= 0)
          throw std::runtime_error("CHAR and VARCHAR must have a positive dimension.");
        break;
      case kDECIMAL:
      case kNUMERIC:
        if (param1 <= 0)
          throw std::runtime_error("DECIMAL and NUMERIC must have a positive precision.");
        else if (param1 > 19)
          throw std::runtime_error("DECIMAL and NUMERIC precision cannot be larger than 19.");
        else if (param1 <= param2)
          throw std::runtime_error("DECIMAL and NUMERIC must have precision larger than scale.");
        break;
      case kTIMESTAMP:
      case kTIME:
        if (param1 == -1)
          param1 = 6; // default precision is 6
        if (param1 > 0) { // @TODO(wei) support sub-second precision later.
          if (type == kTIMESTAMP)
            throw std::runtime_error("Only TIMESTAMP(0) is supported now.");
          else
            throw std::runtime_error("Only TIME(0) is supported now.");
        }
        break;
      default:
        param1 = 0;
        break;
    }
  }

  void
  CreateTableStmt::execute(Catalog_Namespace::Catalog &catalog)
  {
    if (catalog.getMetadataForTable(*table) != nullptr) {
      if (if_not_exists)
        return;
      throw std::runtime_error("Table " + *table + " already exits.");
    }
    std::list<ColumnDescriptor> columns;
    for (auto e : *table_element_list) {
      if (typeid(*e) != typeid(ColumnDef))
        throw std::runtime_error("Table constraints are not supported yet.");
      ColumnDef *coldef = dynamic_cast<ColumnDef*>(e);
      ColumnDescriptor cd;
      cd.columnName = *coldef->get_column_name();
      SQLType *t = coldef->get_column_type();
      t->check_type();
      cd.columnType.type = t->get_type();
      cd.columnType.dimension = t->get_param1();
      cd.columnType.scale = t->get_param2();
      const ColumnConstraintDef *cc = coldef->get_column_constraint();
      if (cc == nullptr)
        cd.columnType.notnull = false;
      else {
        cd.columnType.notnull = cc->get_notnull();
      }
      const CompressDef *compression = coldef->get_compression();
      if (compression == nullptr) {
        cd.compression = kENCODING_NONE;
        cd.comp_param = 0;
      } else {
        const std::string &comp = *compression->get_encoding_name();
        if (boost::iequals(comp, "fixed")) {
          if (!IS_INTEGER(cd.columnType.type))
            throw std::runtime_error("Fixed encoding is only supported for integer columns.");
          // fixed-bits encoding
          switch (cd.columnType.type) {
            case kSMALLINT:
              if (compression->get_encoding_param() != 8)
                throw std::runtime_error("Compression parameter for Fixed encoding on SMALLINT must be 8.");
              break;
            case kINT:
              if (compression->get_encoding_param() != 8 && compression->get_encoding_param() != 16)
                throw std::runtime_error("Compression parameter for Fixed encoding on INTEGER must be 8 or 16.");
              break;
            case kBIGINT:
              if (compression->get_encoding_param() != 8 && compression->get_encoding_param() != 16 && compression->get_encoding_param() != 32)
                throw std::runtime_error("Compression parameter for Fixed encoding on BIGINT must be 8 or 16 or 32.");
              break;
            default:
              break;
          }
          cd.compression = kENCODING_FIXED;
          cd.comp_param = compression->get_encoding_param();
        } else if (boost::iequals(comp, "rl")) {
          // run length encoding
          cd.compression = kENCODING_RL;
          cd.comp_param = 0;
        } else if (boost::iequals(comp, "diff")) {
          // differential encoding
          cd.compression = kENCODING_DIFF;
          cd.comp_param = 0;
        } else if (boost::iequals(comp, "dict")) {
          if (!IS_STRING(cd.columnType.type))
            throw std::runtime_error("Dictionary encoding is only supported on string columns.");
          // diciontary encoding
          cd.compression = kENCODING_DICT;
          cd.comp_param = 0;
        } else if (boost::iequals(comp, "token_dict")) {
          if (!IS_STRING(cd.columnType.type))
            throw std::runtime_error("Tokenized-Dictionary encoding is only supported on string columns.");
          // tokenized diciontary encoding
          cd.compression = kENCODING_TOKDICT;
          cd.comp_param = 0;
        } else if (boost::iequals(comp, "sparse")) {
          // sparse column encoding with mostly NULL values
          if (cd.columnType.notnull)
            throw std::runtime_error("Cannot do sparse column encoding on a NOT NULL column.");
          if (compression->get_encoding_param() == 0 || compression->get_encoding_param() % 8 != 0 || compression->get_encoding_param() > 48)
            throw std::runtime_error("Must specify number of bits as 8, 16, 24, 32 or 48 as the parameter to sparse-column encoding.");
          cd.compression = kENCODING_SPARSE;
          cd.comp_param = compression->get_encoding_param();
        } else
          throw std::runtime_error("Invalid column compression scheme " + comp);
      }
      columns.push_back(cd);
    }
    TableDescriptor td;
    td.tableName = *table;
    td.nColumns = columns.size();
    td.isView = false;
    td.isMaterialized = false;
    td.storageOption = kDISK;
    td.refreshOption = kMANUAL;
    td.checkOption = false;
    td.isReady = true;
    td.fragmenter = nullptr;
    td.fragType = Fragmenter_Namespace::FragmenterType::INSERT_ORDER;
    td.maxFragRows = DEFAULT_FRAGMENT_SIZE;
    td.fragPageSize = DEFAULT_PAGE_SIZE;
    if (storage_options != nullptr) {
      for (auto p : *storage_options) {
        if (boost::iequals(*p->get_name(), "fragment_size")) {
          if (typeid(*p->get_value()) != typeid(IntLiteral))
            throw std::runtime_error("FRAGMENT_SIZE must be an integer literal.");
          int frag_size = dynamic_cast<const IntLiteral*>(p->get_value())->get_intval();
          if (frag_size <= 0)
            throw std::runtime_error("FRAGMENT_SIZE must be a positive number.");
          td.maxFragRows = frag_size;
        } else if (boost::iequals(*p->get_name(), "page_size")) {
          if (typeid(*p->get_value()) != typeid(IntLiteral))
            throw std::runtime_error("PAGE_SIZE must be an integer literal.");
          int page_size =  dynamic_cast<const IntLiteral*>(p->get_value())->get_intval();
          if (page_size <= 0)
            throw std::runtime_error("PAGE_SIZE must be a positive number.");
          td.fragPageSize = page_size;
        } else
          throw std::runtime_error("Invalid CREATE TABLE option " + *p->get_name() + ".  Should be FRAGMENT_SIZE or PAGE_SIZE.");
      }
    }
    catalog.createTable(td, columns);
  }

  void
  DropTableStmt::execute(Catalog_Namespace::Catalog &catalog)
  {
    const TableDescriptor *td = catalog.getMetadataForTable(*table);
    if (td == nullptr) {
      if (if_exists)
        return;
      throw std::runtime_error("Table " + *table + " does not exist.");
    }
    if (td->isView)
      throw std::runtime_error(*table + " is a view.  Use DROP VIEW.");
    catalog.dropTable(td);
  }

  void
  CreateViewStmt::execute(Catalog_Namespace::Catalog &catalog)
  {
    if (catalog.getMetadataForTable(*view_name) != nullptr) {
      if (if_not_exists)
        return;
      throw std::runtime_error("Table or View " + *view_name + " already exists.");
    }
    StorageOption matview_storage = kDISK;
    ViewRefreshOption matview_refresh = kMANUAL;
    if (matview_options != nullptr) {
      for (auto p : *matview_options) {
        if (boost::iequals(*p->get_name(), "storage")) {
          if (typeid(*p->get_value()) != typeid(StringLiteral))
            throw std::runtime_error("Storage option must be a string literal.");
          const std::string *str = dynamic_cast<const StringLiteral*>(p->get_value())->get_stringval();
          if (boost::iequals(*str, "gpu") || boost::iequals(*str, "mic"))
            matview_storage = kGPU;
          else if (boost::iequals(*str, "cpu"))
            matview_storage = kCPU;
          else if (boost::iequals(*str, "disk"))
            matview_storage = kDISK;
          else
            throw std::runtime_error("Invalid storage option " + *str + ". Should be GPU, MIC, CPU or DISK.");
        } else if (boost::iequals(*p->get_name(), "refresh")) {
          if (typeid(*p->get_value()) != typeid(StringLiteral))
            throw std::runtime_error("Refresh option must be a string literal.");
          const std::string *str = dynamic_cast<const StringLiteral*>(p->get_value())->get_stringval();
          if (boost::iequals(*str, "auto"))
            matview_refresh = kAUTO;
          else if (boost::iequals(*str, "manual"))
            matview_refresh = kMANUAL;
          else if (boost::iequals(*str, "immediate"))
            matview_refresh = kIMMEDIATE;
          else
            throw std::runtime_error("Invalid refresh option " + *str + ". Should be AUTO, MANUAL or IMMEDIATE.");
        } else
          throw std::runtime_error("Invalid CREATE MATERIALIZED VIEW option " + *p->get_name() + ".  Should be STORAGE or REFRESH.");
      }
    }
    Analyzer::Query analyzed_query;
    query->analyze(catalog, analyzed_query);
    const std::vector<Analyzer::TargetEntry*> &tlist = analyzed_query.get_targetlist();
    // @TODO check column name uniqueness.  for now let sqlite enforce.
    if (column_list != nullptr) {
      if (column_list->size() != tlist.size())
        throw std::runtime_error("Number of column names does not match the number of expressions in SELECT clause.");
      std::list<std::string*>::iterator it = column_list->begin();
      for (auto tle : tlist) { 
        tle->set_resname(**it);
        ++it;
      }
    }
    std::list<ColumnDescriptor> columns;
    for (auto tle : tlist) {
      ColumnDescriptor cd;
      if (tle->get_resname().empty())
        throw std::runtime_error("Must specify a column name for expression.");
      cd.columnName = tle->get_resname();
      cd.columnType = tle->get_expr()->get_type_info();
      cd.compression = kENCODING_NONE;
      cd.comp_param = 0;
      columns.push_back(cd);
    }
    TableDescriptor td;
    td.tableName = *view_name;
    td.nColumns = columns.size();
    td.isView = true;
    td.isMaterialized = is_materialized;
    td.viewSQL = query->to_string();
    td.checkOption = checkoption;
    td.storageOption = matview_storage;
    td.refreshOption = matview_refresh;
    td.isReady = !is_materialized;
    td.fragmenter = nullptr;
    td.fragType = Fragmenter_Namespace::FragmenterType::INSERT_ORDER;
    td.maxFragRows = DEFAULT_FRAGMENT_SIZE;
    td.fragPageSize = DEFAULT_PAGE_SIZE;
    catalog.createTable(td, columns);
  }

  void
  RefreshViewStmt::execute(Catalog_Namespace::Catalog &catalog)
  {
    const TableDescriptor *td = catalog.getMetadataForTable(*view_name);
    if (td == nullptr)
      throw std::runtime_error("Materialied view " + *view_name + " does not exist.");
    if (!td->isView)
      throw std::runtime_error(*view_name + " is a table not a materialized view.");
    if (!td->isMaterialized)
      throw std::runtime_error(*view_name + " is not a materialized view.");
    SQLParser parser;
    std::list<Stmt*> parse_trees;
    std::string last_parsed;
    std::string query_str = "INSERT INTO " + *view_name + " " + td->viewSQL;
    int numErrors = parser.parse(query_str, parse_trees, last_parsed);
    if (numErrors > 0)
      throw std::runtime_error("Internal Error: syntax error at: " + last_parsed);
    DMLStmt *view_stmt = dynamic_cast<DMLStmt*>(parse_trees.front());
    std::unique_ptr<Stmt> stmt_ptr(view_stmt); // make sure it's deleted
    Analyzer::Query query;
    view_stmt->analyze(catalog, query);
    Planner::Optimizer optimizer(query, catalog);
    Planner::RootPlan *plan = optimizer.optimize();
    std::unique_ptr<Planner::RootPlan> plan_ptr(plan); // make sure it's deleted
    // @TODO execute plan
    // plan->print();
  }

  void
  DropViewStmt::execute(Catalog_Namespace::Catalog &catalog)
  {
    const TableDescriptor *td = catalog.getMetadataForTable(*view_name);
    if (td == nullptr) {
      if (if_exists)
        return;
      throw std::runtime_error("View " + *view_name + " does not exist.");
    }
    if (!td->isView)
      throw std::runtime_error(*view_name + " is a table.  Use DROP TABLE.");
    catalog.dropTable(td);
  }

  CreateUserStmt::~CreateUserStmt()
  {
    delete user_name;
    if (name_value_list != nullptr) {
      for (auto p : *name_value_list)
        delete p;
      delete name_value_list;
    }
  }

  CreateDBStmt::~CreateDBStmt()
  {
    delete db_name;
    if (name_value_list != nullptr) {
      for (auto p : *name_value_list)
        delete p;
      delete name_value_list;
    }
  }

  void
  CreateDBStmt::execute(Catalog_Namespace::Catalog &catalog)
  {
    if (catalog.get_currentDB().dbName != MAPD_SYSTEM_DB)
      throw std::runtime_error("Must be in the system database to create databases.");
    Catalog_Namespace::SysCatalog &syscat = static_cast<Catalog_Namespace::SysCatalog&>(catalog);
    int ownerId = catalog.get_currentUser().userId;
    if (name_value_list != nullptr) {
      for (auto p : *name_value_list) {
        if (boost::iequals(*p->get_name(), "owner")) {
          if (typeid(*p->get_value()) != typeid(StringLiteral))
            throw std::runtime_error("Owner name must be a string literal.");
          const std::string *str = dynamic_cast<const StringLiteral*>(p->get_value())->get_stringval();
          Catalog_Namespace::UserMetadata user;
          if (!syscat.getMetadataForUser(*str, user))
            throw std::runtime_error("User " + *str + " does not exist.");
          ownerId = user.userId;
        }
        else
          throw std::runtime_error("Invalid CREATE DATABASE option " + *p->get_name() + ". Only OWNER supported.");
      }
    }
    syscat.createDatabase(*db_name, ownerId);
  }

  void
  DropDBStmt::execute(Catalog_Namespace::Catalog &catalog)
  {
    if (catalog.get_currentDB().dbName != MAPD_SYSTEM_DB)
      throw std::runtime_error("Must be in the system database to drop databases.");
    Catalog_Namespace::SysCatalog &syscat = static_cast<Catalog_Namespace::SysCatalog&>(catalog);
    syscat.dropDatabase(*db_name);
  }

  void
  CreateUserStmt::execute(Catalog_Namespace::Catalog &catalog)
  {
    std::string passwd;
    bool is_super = false;
    for (auto p : *name_value_list) {
      if (boost::iequals(*p->get_name(), "password")) {
        if (typeid(*p->get_value()) != typeid(StringLiteral))
          throw std::runtime_error("Password must be a string literal.");
        passwd = *dynamic_cast<const StringLiteral*>(p->get_value())->get_stringval();
      } else if (boost::iequals(*p->get_name(), "is_super")) {
        if (typeid(*p->get_value()) != typeid(StringLiteral))
          throw std::runtime_error("IS_SUPER option must be a string literal.");
        const std::string *str = dynamic_cast<const StringLiteral*>(p->get_value())->get_stringval();
        if (boost::iequals(*str, "true"))
          is_super = true;
        else if (boost::iequals(*str, "false"))
          is_super = false;
        else
          throw std::runtime_error("Value to IS_SUPER must be TRUE or FALSE.");
      } else
        throw std::runtime_error("Invalid CREATE USER option " + *p->get_name() + ".  Should be PASSWORD or IS_SUPER.");
    }
    if (passwd.empty())
      throw std::runtime_error("Must have a password for CREATE USER.");
    if (catalog.get_currentDB().dbName != MAPD_SYSTEM_DB)
      throw std::runtime_error("Must be in the system database to create users.");
    Catalog_Namespace::SysCatalog &syscat = static_cast<Catalog_Namespace::SysCatalog&>(catalog);
    syscat.createUser(*user_name, passwd, is_super);
  }

  AlterUserStmt::~AlterUserStmt()
  {
    delete user_name;
    if (name_value_list != nullptr) {
      for (auto p : *name_value_list)
        delete p;
      delete name_value_list;
    }
  }

  void
  AlterUserStmt::execute(Catalog_Namespace::Catalog &catalog)
  {
    const std::string *passwd = nullptr;
    bool is_super = false;
    bool *is_superp = nullptr;
    for (auto p : *name_value_list) {
      if (boost::iequals(*p->get_name(), "password")) {
        if (typeid(*p->get_value()) != typeid(StringLiteral))
          throw std::runtime_error("Password must be a string literal.");
        passwd = dynamic_cast<const StringLiteral*>(p->get_value())->get_stringval();
      } else if (boost::iequals(*p->get_name(), "is_super")) {
        if (typeid(*p->get_value()) != typeid(StringLiteral))
          throw std::runtime_error("IS_SUPER option must be a string literal.");
        const std::string *str = dynamic_cast<const StringLiteral*>(p->get_value())->get_stringval();
        if (boost::iequals(*str, "true")) {
          is_super = true;
          is_superp = &is_super;
        } else if (boost::iequals(*str, "false")) {
          is_super = false;
          is_superp = &is_super;
        } else
          throw std::runtime_error("Value to IS_SUPER must be TRUE or FALSE.");
      } else
        throw std::runtime_error("Invalid CREATE USER option " + *p->get_name() + ".  Should be PASSWORD or IS_SUPER.");
    }
    Catalog_Namespace::SysCatalog &syscat = static_cast<Catalog_Namespace::SysCatalog&>(catalog);
    syscat.alterUser(*user_name, passwd, is_superp);
  }

  void
  DropUserStmt::execute(Catalog_Namespace::Catalog &catalog)
  {
    if (catalog.get_currentDB().dbName != MAPD_SYSTEM_DB)
      throw std::runtime_error("Must be in the system database to drop users.");
    Catalog_Namespace::SysCatalog &syscat = static_cast<Catalog_Namespace::SysCatalog&>(catalog);
    syscat.dropUser(*user_name);
  }

}
