/**
 * @file 	SQL_RA_Translator.cpp
 * @author	Steven Stewart <steve@map-d.com>
 */
#include "SQL_RA_Translator.h"

#include "../Parse/SQL/ast/Column.h"
#include "../Parse/SQL/ast/Comparison.h"
#include "../Parse/SQL/ast/DdlStmt.h"
#include "../Parse/SQL/ast/DmlStmt.h"
#include "../Parse/SQL/ast/FromClause.h"
#include "../Parse/SQL/ast/InsertStmt.h"
#include "../Parse/SQL/ast/MathExpr.h"
#include "../Parse/SQL/ast/OptWhere.h"
#include "../Parse/SQL/ast/Predicate.h" 
#include "../Parse/SQL/ast/ScalarExpr.h"
#include "../Parse/SQL/ast/ScalarExprList.h"
#include "../Parse/SQL/ast/SearchCondition.h"
#include "../Parse/SQL/ast/Selection.h"
#include "../Parse/SQL/ast/SelectStmt.h"
#include "../Parse/SQL/ast/SqlStmt.h"
#include "../Parse/SQL/ast/Table.h"
#include "../Parse/SQL/ast/TableList.h"

#include "../Parse/RA/ast/AttrList.h"
#include "../Parse/RA/ast/Attribute.h"
#include "../Parse/RA/ast/Comparison.h"
#include "../Parse/RA/ast/MathExpr.h"
#include "../Parse/RA/ast/Predicate.h"
#include "../Parse/RA/ast/ProductOp.h"
#include "../Parse/RA/ast/Program.h"
#include "../Parse/RA/ast/ProjectOp.h"
#include "../Parse/RA/ast/Relation.h"
#include "../Parse/RA/ast/RelExpr.h"
#include "../Parse/RA/ast/RelExprList.h"

using RA_Namespace::AttrList;
using RA_Namespace::Attribute;
using RA_Namespace::ProductOp;
using RA_Namespace::Program;
using RA_Namespace::ProjectOp;
using RA_Namespace::Relation;
using RA_Namespace::RelExpr;
using RA_Namespace::RelExprList;

using std::to_string;

namespace Translate_Namespace {
    
    SQL_RA_Translator::SQL_RA_Translator() {
        
    }
    
    void SQL_RA_Translator::visit(DdlStmt *v) {
        //// printf("<DdlStmt>\n");
    }
    
    void SQL_RA_Translator::visit(DmlStmt *v) {
        // printf("<DmlStmt>\n");
        if (v->n1) v->n1->accept(*this); // InsertStmt
        if (v->n2) v->n2->accept(*this); // SelectStmt
    }
    
    void SQL_RA_Translator::visit(FromClause *v) {
        // printf("<FromClause>\n");
        if (v->n1) v->n1->accept(*this); // TableList
        // if (v->n2) v->n1->accept(*this); // SelectStmt (nested query)
        size_t numTbls = tableNodes_.size();
        
        // Print out collected table ids (for debugging)
        /*printf("numTbls = %d\n", numTbls);
        for (unsigned i = 0; i < tableIds_.size(); ++i)
            printf("%d ", tableIds_[i]);
        printf("\n");*/
        
        // case: 1 table
        if (numTbls == 1) {
            printf("table[0].name = %s\n", tableNodes_[0]->metadata.tableName.c_str());
            nodeFromClause_ = new RelExpr(new Relation(tableNodes_[0]->metadata));
        }
        
        
        // case: > 1 table
        else if (numTbls > 1) {
            ProductOp *n = new ProductOp(new RelExpr(new Relation(tableNodes_.back()->metadata)), new RelExpr(new Relation(tableNodes_.back()->metadata)));
            tableNodes_.pop_back();
            tableNodes_.pop_back();
            
            while (tableNodes_.size() > 0) {
                n = new ProductOp(new RelExpr(new Relation(tableNodes_.back()->metadata)), new RelExpr(n));
                tableNodes_.pop_back();
            }
            nodeFromClause_ = new RelExpr(n);
        }
        
        // case: 0 tables, do nothing
        else
            nodeFromClause_ = NULL;
    }
    
    void SQL_RA_Translator::visit(Column *v) {
        // printf("<Column>\n");
        // columnIds_.push_back(v->column_id);
        columnNodes_.push_back(v);
    }

    void SQL_RA_Translator::visit(Comparison *v) {
        // printf("<Comparison>\n");

        // case: MathExpr OP MathExpr
        
    }
    
    void SQL_RA_Translator::visit(InsertStmt *v) {
        // printf("<InsertStmt>\n");
    }
    
    void SQL_RA_Translator::visit(MathExpr *v) {
        // printf("<MathExpr>\n");

        // case: column


        // case: aggr_expr


        // case: INTVAL


        // case: FLOATVAL


        // case: MathExpr OP MathExpr
    }

    void SQL_RA_Translator::visit(OptWhere *v) {
        // printf("<OptWhere>\n");
        if (v->n1) v->n1->accept(*this);
    }
    
    void SQL_RA_Translator::visit(Predicate *v) {
        // printf("<Predicate op=\"%s\">\n", v->op.c_str());
        
        // case: NOT predicate
        if (v->op == "NOT" && v->n1) {
            RA_Namespace::Predicate *n = new RA_Namespace::Predicate(v->op, nodePredicateVec_.back());
            nodePredicateVec_.pop_back();
            nodePredicateVec_.push_back(n);
        }

        // case: ( predicate )
        else if (v->op == "" && v->n1 && !v->n2 && !v->n3) {
            v->n1->accept(*this);
        }

        // case: predicate OP predicate
        else if (v->op != "" && v->n1 && v->n2 && !v->n3) {
            v->n1->accept(*this);
            v->n2->accept(*this);

            RA_Namespace::Predicate *left = nodePredicateVec_.back();
            nodePredicateVec_.pop_back();
            RA_Namespace::Predicate *right = nodePredicateVec_.back();
            nodePredicateVec_.pop_back();
            nodePredicateVec_.push_back(new RA_Namespace::Predicate(v->op, left, right));
        }

        // case: comparison
        else if (v->n3) {
            assert(nodeComparisonVec_.size() == 1);
            nodePredicateVec_.push_back(new RA_Namespace::Predicate(nodeComparisonVec_.back()));
            nodeComparisonVec_.pop_back();
        }

    }

    void SQL_RA_Translator::visit(ScalarExpr *v) {
        // printf("<ScalarExpr>\n");
        if (v->n4) v->n4->accept(*this); // Column
    }
    
    void SQL_RA_Translator::visit(ScalarExprList *v) {
        // printf("<ScalarExprList>\n");
        if (v->n1) v->n1->accept(*this); // ScalarExprList
        if (v->n2) v->n2->accept(*this); // ScalarExpr
    }

    void SQL_RA_Translator::visit(SearchCondition *v) {
        // printf("<SearchCondition>\n");
        if (v->n1) v->n1->accept(*this);
        
    }
    
    void SQL_RA_Translator::visit(Selection *v) {
        // printf("<Selection>\n");
        if (v->n1) v->n1->accept(*this); // ScalarExprList
        size_t numCols = columnNodes_.size();
        
        // Print out collected columns ids (for debugging)
        /*printf("numCols = %d\n", numCols);
        for (unsigned i = 0; i < columnIds_.size(); ++i)
            printf("%d ", columnIds_[i]);
        printf("\n");*/
        
        // case: 1 column
        AttrList *n = NULL;
        if (numCols == 1) {
            // n = new AttrList(new Attribute(to_string(columnIds_[0])));
            n = new AttrList(new Attribute(columnNodes_[0]->metadata));
        }
        
        // case: > 1 column
        else if (numCols > 1) {
            n = new AttrList(new Attribute(columnNodes_[0]->metadata));
            columnNodes_.pop_back();
            while (columnNodes_.size() > 0) {
                n = new AttrList(n, new Attribute(columnNodes_.back()->metadata));
                columnNodes_.pop_back();
            }
        }
        nodeSelection_ = n;
    }
    
    void SQL_RA_Translator::visit(SelectStmt *v) {
        // printf("<SelectStmt>\n");
        if (v->n2) v->n2->accept(*this); // Selection
        if (v->n3) v->n3->accept(*this); // FromClause
        if (v->n4) v->n4->accept(*this); // OptWhere
        
        assert(nodeSelection_ && nodeFromClause_);
        nodeSelectStmt_ =  new ProjectOp((RelExpr*)nodeFromClause_, (AttrList*)nodeSelection_);
        assert(nodeSelectStmt_);

        // Predicate?
    }
    
    void SQL_RA_Translator::visit(SqlStmt *v) {
        // printf("<SqlStmt>\n");
        if (v->n1) v->n1->accept(*this); // DmlStmt
        if (v->n2) v->n2->accept(*this); // DdlStmt
        
        if (nodeSelectStmt_) {
            root = new Program(new RelExprList((RelExpr*)nodeSelectStmt_));
            assert(root);
        }
    }
    
    void SQL_RA_Translator::visit(TableList *v) {
        // printf("<TableList>\n");
        if (v->n1) v->n1->accept(*this); // TableList
        if (v->n2) v->n2->accept(*this); // Table
    }
    
    void SQL_RA_Translator::visit(Table *v) {
        // printf("<Table>\n");
        // tableIds_.push_back(v->table_id);
        tableNodes_.push_back(v);
    }
    
} // Translate_Namespace
