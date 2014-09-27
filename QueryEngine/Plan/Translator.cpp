/**
 * @file    Translator.h
 * @author  Steven Stewart <steve@map-d.com>
 */
#include "Translator.h"

// SQL nodes
#include "../Parse/SQL/ast/AggrExpr.h"
#include "../Parse/SQL/ast/AlterStmt.h"
#include "../Parse/SQL/ast/Column.h"
#include "../Parse/SQL/ast/ColumnDef.h"
#include "../Parse/SQL/ast/ColumnDefList.h"
#include "../Parse/SQL/ast/ColumnList.h"
#include "../Parse/SQL/ast/Comparison.h"
#include "../Parse/SQL/ast/CreateStmt.h"
#include "../Parse/SQL/ast/DdlStmt.h"
#include "../Parse/SQL/ast/DmlStmt.h"
#include "../Parse/SQL/ast/DropStmt.h"
#include "../Parse/SQL/ast/FromClause.h"
#include "../Parse/SQL/ast/InsertColumnList.h"
#include "../Parse/SQL/ast/InsertStmt.h"
#include "../Parse/SQL/ast/Literal.h"
#include "../Parse/SQL/ast/LiteralList.h"
#include "../Parse/SQL/ast/MapdDataT.h"
#include "../Parse/SQL/ast/MathExpr.h"
#include "../Parse/SQL/ast/OptAllDistinct.h"
#include "../Parse/SQL/ast/OptGroupby.h"
#include "../Parse/SQL/ast/OptHaving.h"
#include "../Parse/SQL/ast/OptOrderby.h"
#include "../Parse/SQL/ast/OptLimit.h"
#include "../Parse/SQL/ast/OptWhere.h"
#include "../Parse/SQL/ast/OrderbyColumn.h"
#include "../Parse/SQL/ast/OrderByColumnList.h"
#include "../Parse/SQL/ast/Predicate.h"
#include "../Parse/SQL/ast/RenameStmt.h"
#include "../Parse/SQL/ast/ScalarExpr.h"
#include "../Parse/SQL/ast/ScalarExprList.h"
#include "../Parse/SQL/ast/SearchCondition.h"
#include "../Parse/SQL/ast/Selection.h"
#include "../Parse/SQL/ast/SelectStmt.h"
#include "../Parse/SQL/ast/SqlStmt.h"
#include "../Parse/SQL/ast/Table.h"
#include "../Parse/SQL/ast/TableList.h"

// RA nodes
#include "../Parse/RA/ast/AggrExpr.h"
#include "../Parse/RA/ast/AggrList.h"
#include "../Parse/RA/ast/AntijoinOp.h"
#include "../Parse/RA/ast/Attribute.h"
#include "../Parse/RA/ast/AttrList.h"
#include "../Parse/RA/ast/Comparison.h"
#include "../Parse/RA/ast/DiffOp.h"
#include "../Parse/RA/ast/Expr.h"
#include "../Parse/RA/ast/ExtendOp.h"
#include "../Parse/RA/ast/GroupbyOp.h"
#include "../Parse/RA/ast/JoinOp.h"
#include "../Parse/RA/ast/MathExpr.h"
#include "../Parse/RA/ast/OuterjoinOp.h"
#include "../Parse/RA/ast/Predicate.h"
#include "../Parse/RA/ast/ProductOp.h"
#include "../Parse/RA/ast/Program.h"
#include "../Parse/RA/ast/ProjectOp.h"
#include "../Parse/RA/ast/Relation.h"
#include "../Parse/RA/ast/RelExpr.h"
#include "../Parse/RA/ast/RelExprList.h"
#include "../Parse/RA/ast/RenameOp.h"
#include "../Parse/RA/ast/ScanOp.h"
#include "../Parse/RA/ast/SelectOp.h"
#include "../Parse/RA/ast/SemijoinOp.h"
#include "../Parse/RA/ast/SortOp.h"
#include "../Parse/RA/ast/UnionOp.h"

using namespace RA_Namespace;

namespace Plan_Namespace {

    RA_Namespace::RelAlgNode* Translator::translate(SQL_Namespace::ASTNode *parseTreeRoot) {
        assert(parseTreeRoot);
        RelAlgNode *queryPlan = nullptr;
        parseTreeRoot->accept(*this);
        
        /*
         for (size_t i = 0; i < tableNames_.size(); ++i)
            printf("tableNames_[%zu] = %s\n", i, tableNames_[i].c_str());
         for (size_t i = 0; i < columnNames_.size(); ++i)
            printf("columnNames[%zu] = %s\n", i, columnNames_[i].second.c_str());
         */
        
        // retieve table metadata from Catalog
        // set error if a table does not exist
        if (stmtType_ == QUERY_STMT) {
            TableRow tableMetadata;
            for (size_t i = 0; i < queryTables_.size(); ++i) {
                mapd_err_t err = c_.getMetadataForTable(queryTables_[i]->name.second, tableMetadata);
                if (err != MAPD_SUCCESS) {
                    catalogError_.first = true;
                    catalogError_.second = "Table '" + tableNames_[i] + "' not found";
                    return nullptr;
                }
                queryTables_[i]->metadata = tableMetadata;
            }
        }
        
        // retrieve column metadata from Catalog
        // set error if column does not exist or there is ambiguity
        std::vector<ColumnRow> columnMetadata;
        mapd_err_t err = c_.getMetadataForColumns(tableNames_, columnNames_, columnMetadata);
        if (err != MAPD_SUCCESS) {
            catalogError_.first = true;
            catalogError_.second = "Catalog error";
            if (err == MAPD_ERR_COLUMN_DOES_NOT_EXIST) {
                std::string colNotFound = columnNames_[columnMetadata.size()].second;
                catalogError_.second = "Column '" + colNotFound + "' not found";
            }
            return nullptr;
        }
        assert(columnMetadata.size() == columnNames_.size());
        
        // annotate SQL column nodes with Catalog metadata
        if (stmtType_ == QUERY_STMT)
            for (size_t i = 0; i < queryColumns_.size(); ++i)
                queryColumns_[i]->metadata = columnMetadata[i];
        
        // translate the SQL AST to an RA query plan tree
        if (stmtType_ == QUERY_STMT) {
            printf("Translate a QUERY_STMT\n");
            queryPlan = translateQuery();
        }
        else if (stmtType_ == INSERT_STMT) {
            printf("Translate an INSERT_STMT\n");
            exit(EXIT_SUCCESS);
        }
        else
            throw std::runtime_error("Unable to translate SQL statement to RA query plan");
        
        // fill out the rest of the query plan
        queryPlan = new Program(new RelExprList((RelExpr*)queryPlan));
        
        return queryPlan;
    }

    RA_Namespace::RelAlgNode* Translator::translateQuery() {
        assert(queryTables_.size() > 0);

        // Step 1:  create Relation nodes for each table
        std::vector<Relation*> relations;
        size_t numTables = queryTables_.size();
        for (size_t i = 0; i < numTables; ++i)
            relations.push_back(new Relation(queryTables_[i]->metadata));
        
        // Step 2:  take the product of the relations from Step 1
        ProductOp* productOfRelations = nullptr;
        if (numTables > 1) {
            productOfRelations = new ProductOp((RelExpr*)relations[0], (RelExpr*)relations[1]);
            for (size_t i = 2; i < numTables; ++i)
                productOfRelations = new ProductOp((RelExpr*)productOfRelations, (RelExpr*)relations[i]);
        }
        
        // Step 3:  select on the predicate in the where clause
        // @todo Implement the translation of an SQL predicate to a query plan
        SelectOp *select = nullptr;
        
        // Step 4:  project on the fields in the selection clause
        size_t numFields = queryColumns_.size();
        if (numFields == 0) {
            throw std::runtime_error("No columns specified. Probably a 'select *'. Not yet supported.");
        }

        AttrList *fields = new AttrList(new Attribute(queryColumns_[0]->metadata));
        for (size_t i = 1; i < numFields; ++i) {
            assert(queryColumns_[0]->name.second != "");
            fields = new AttrList(fields, new Attribute(queryColumns_[i]->metadata));
        }
        
        ProjectOp *project = nullptr;
        if (select) {
            project = new ProjectOp((RelExpr*)select, fields);
        }
        else {
            if (numTables == 1)
                project = new ProjectOp((RelExpr*)relations[0], fields);
            else
                project = new ProjectOp((RelExpr*)productOfRelations, fields);
        }
        
        // Step 6:  return
        assert(project);
        return project;
    }
    
    void Translator::visit(DmlStmt *v) {
        // printf("<DmlStmt>\n");
        if (v->n1) v->n1->accept(*this); // InsertStmt
        if (v->n2) v->n2->accept(*this); // SelectStmt
    }
    
    void Translator::visit(Column *v) {
        columnNames_.push_back(v->name);
        if (stmtType_ == QUERY_STMT)
            queryColumns_.push_back(v);
        else
            throw std::runtime_error("Unsupported SQL feature.");
    }

    void Translator::visit(FromClause *v) {
        // printf("<FromClause>\n");
        if (v->n1)
            v->n1->accept(*this); // TableList
        else
            throw std::runtime_error("Unsupported SQL feature.");
        
    }
    
    void Translator::visit(OptAllDistinct *v) {
        // printf("<OptAllDistinct>\n");
        throw std::runtime_error("Unsupported SQL feature.");
    }
    
    void Translator::visit(OptGroupby *v) {
        // printf("<OptGroupby>\n");
        throw std::runtime_error("Unsupported SQL feature.");
    }
    
    void Translator::visit(OptHaving *v) {
        // printf("<OptHaving>\n");
        throw std::runtime_error("Unsupported SQL feature.");
    }
    
    void Translator::visit(OptLimit *v) {
        // printf("<OptLimit>\n");
        throw std::runtime_error("Unsupported SQL feature.");
    }
    
    void Translator::visit(OptOrderby *v) {
        // printf("<OptOrderby>\n");
        throw std::runtime_error("Unsupported SQL feature.");
    }
    
    void Translator::visit(OptWhere *v) {
        // printf("<OptWhere>\n");
        throw std::runtime_error("Unsupported SQL feature.");
    }
    
    void Translator::visit(SQL_Namespace::Predicate *v) {
        // printf("<Predicate>\n");
        
        if (stmtType_ == QUERY_STMT)
            queryPredicate_ = v;
        else if (stmtType_ == DELETE_STMT)
            deletePredicate_ = v;
        else
            throw std::runtime_error("Unsupported SQL feature.");
    }
    
    
    void Translator::visit(ScalarExpr *v) {
        // printf("<ScalarExpr>\n");
        
        if (v->n4)
            v->n4->accept(*this); // Column
        else
            throw std::runtime_error("Unsupported SQL feature.");
    }
    
    void Translator::visit(ScalarExprList *v) {
        // printf("<ScalarExprList>\n");
        if (v->n1) v->n1->accept(*this); // ScalarExprList
        if (v->n2) v->n2->accept(*this); // ScalarExpr
    }
    
    void Translator::visit(SearchCondition *v) {
        // printf("<SearchCondition>\n");
        if (v->n1) v->n1->accept(*this); // Predicate
    }
    
    void Translator::visit(Selection *v) {
        // printf("<Selection>\n");
        if (v->n1) v->n1->accept(*this); // ScalarExprList
    }

    void Translator::visit(SelectStmt *v) {
        // printf("<SelectStmt>\n");
        stmtType_ = QUERY_STMT;
        
        if (v->n1) v->n1->accept(*this); // OptAllDistinct
        if (v->n2) v->n2->accept(*this); // Selection
        if (v->n3) v->n3->accept(*this); // FromClause
        if (v->n4) v->n4->accept(*this); // OptWhere
        if (v->n5) v->n5->accept(*this); // OptGroupby
        if (v->n6) v->n6->accept(*this); // OptHaving
        if (v->n7) v->n7->accept(*this); // OptOrderby
        if (v->n8) v->n8->accept(*this); // OptLimit
    }
    
    void Translator::visit(SqlStmt *v) {
        // printf("<SqlStmt>\n");
        if (v->n1) v->n1->accept(*this); // DmlStmt
        if (v->n2) v->n2->accept(*this); // DdlStmt
    }
    
    void Translator::visit(Table *v) {
        assert(v->name.first == ""); // @todo this should pass until we support NAME AS NAME
        // printf("<Table>\n");
        tableNames_.push_back(v->name.second);
        if (stmtType_ == QUERY_STMT)
            queryTables_.push_back(v);
        else
            throw std::runtime_error("Unsupported SQL feature.");
    }
    
    void Translator::visit(TableList *v) {
        // printf("<TableList>\n");
        if (v->n1) v->n1->accept(*this); // TableList
        if (v->n2) v->n2->accept(*this); // Table
    }
}
