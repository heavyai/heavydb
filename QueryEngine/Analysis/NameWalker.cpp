/**
 * @file	NameWalker.cpp
 * @author	Steven Stewart <steve@map-d.com>
 *
 * Annotates table and column nodes with metadata obtained from the Catalog.
 * Principally, this refers to translating names to ids, but additional metadata
 * is included as part of the TableRow and ColumnRow objects that are members
 * of the affected AST nodes.
 */
#include "NameWalker.h"
#include "../Parse/SQL/ast/Column.h"
#include "../Parse/SQL/ast/DmlStmt.h"
#include "../Parse/SQL/ast/FromClause.h"
#include "../Parse/SQL/ast/InsertColumnList.h"
#include "../Parse/SQL/ast/InsertStmt.h"
#include "../Parse/SQL/ast/OptGroupby.h"
#include "../Parse/SQL/ast/OptHaving.h"
#include "../Parse/SQL/ast/OptOrderby.h"
#include "../Parse/SQL/ast/OptWhere.h"
#include "../Parse/SQL/ast/ScalarExpr.h"
#include "../Parse/SQL/ast/ScalarExprList.h"
#include "../Parse/SQL/ast/Selection.h"
#include "../Parse/SQL/ast/SelectStmt.h"
#include "../Parse/SQL/ast/SqlStmt.h"
#include "../Parse/SQL/ast/Table.h"
#include "../Parse/SQL/ast/TableList.h"

namespace Analysis_Namespace {
    
    void NameWalker::visit(Column *v) {
        //v->metadata.print();
        colNodes_.push_back(v);
    }
    
    void NameWalker::visit(DmlStmt *v) {
        if (v->n1) v->n1->accept(*this); // InsertStmt
        if (v->n2) v->n2->accept(*this); // SelectStmt
    }
    
    void NameWalker::visit(FromClause *v) {
        if (v->n1) v->n1->accept(*this);
        // if (v->n2) v->n2->accept(*this);
    }
    
    void NameWalker::visit(InsertColumnList *v) {
        if (v->n1) v->n1->accept(*this);
        insertColNodes_.push_back(v);
    }
    
    /**
     * This method visits the Table and InsertColumnList nodes such as to obtain
     * the table and column names for the insert statement. Once having obtained
     * these names, it then checks with the Catalog to ensure that (1) the table
     * exists and (2) the columns belong to that table. Additionally, respective
     * Table and InsertColumnList nodes are annotated with the metadata obtained
     * from the calls Catalog's methods, including the unique identifiers (ids),
     * which are required by the other tree walkers that follow in the pipeline.
     */
    void NameWalker::visit(InsertStmt *v) {
        mapd_err_t err = MAPD_SUCCESS;
        
        // gather table and column nodes
        if (v->n1) v->n1->accept(*this); // Table
        if (v->n2) v->n2->accept(*this); // InsertColumnList

        // should be no more than 1 table name parsed for an insert stmt.
        assert(tblNodes_.size() == 1);
        
        // annotate table node with metadata (return on error)
        std::string tblName = tblNodes_.back()->name.second;
        err = c_.getMetadataForTable(tblName, tblNodes_.back()->metadata);
        if (err != MAPD_SUCCESS) {
			errFlag_ = true;
			errMsg_ = "table '" + tblName + "' does not exist";
            return;
        }
        
        // annotate column nodes with metadata (return on error)
        for (int i = 0; i < insertColNodes_.size(); ++i) {
            err = c_.getMetadataForColumn(tblName, insertColNodes_[i]->name, insertColNodes_[i]->metadata);
            if (err != MAPD_SUCCESS) {
                errFlag_ = true;
                errMsg_ = "column '" + insertColNodes_[i]->name + "' does not exist";
                return;
            }
        }
        
        /*for (int i = 0; i < colNodes_.size(); ++i)
            printf("tableId=%d columnId=%d columnName=%s\n", colNodes_[i]->metadata.tableId, colNodes_[i]->metadata.columnId, colNodes_[i]->metadata.columnName.c_str());*/
    }
    
    void NameWalker::visit(OptGroupby *v) {
        
    }

    void NameWalker::visit(OptHaving *v) {
        
    }
    
    void NameWalker::visit(OptOrderby *v) {
        
    }
    
    void NameWalker::visit(OptWhere *v) {
        
    }

    void NameWalker::visit(ScalarExpr *v) {
        if (v->n4) v->n4->accept(*this); // Column
    }
    
    void NameWalker::visit(ScalarExprList *v) {
        if (v->n2) v->n2->accept(*this); // ScalarExpr
        if (v->n1) v->n1->accept(*this); // ScalarExprList
    }
    
    void NameWalker::visit(Selection *v) {
        if (v->n1) v->n1->accept(*this);
    }
    
    /**
     * This method will annotate reachable Table/Column AST nodes with metadata,
     * and it will validate the existence of parsed table/column names. If there
     * are invalid names found, then the error flag and message will be set. The
     * Catalog is called in order to both obtain metadata and to validate names.
     */
    void NameWalker::visit(SelectStmt *v) {
        mapd_err_t err = MAPD_SUCCESS;
        
        // gather table nodes by visiting the "from clause"
        if (v->n3) v->n3->accept(*this); // FromClause
        
        // there should be at least one table node
        assert(tblNodes_.size() > 0);

        // annotate table nodes with metadata (return on error)
        // @todo It'd be nice if we could annotate all table nodes with just one method call
        size_t numTbls = tblNodes_.size();
        for (size_t i = 0; i < numTbls; ++i) {
            std::string tblName = tblNodes_[i]->name.second;
            err = c_.getMetadataForTable(tblName, tblNodes_[i]->metadata);
            if (err != MAPD_SUCCESS) {
                errFlag_ = true;
                errMsg_ = "table '" + tblName + "' does not exist";
                return;
            }
        }
        
        // visit the Selection node -- handles case of "select *" if necessary
        if (v->n2) { // Selection

            // case: "select *"
            if (v->n2->all) {
                assert(v->n2->n1 == NULL); // there should not be a ScalarExprList here
                
                // obtain metadata for the columns of all the tables
                std::vector<ColumnRow> columns;
                for (size_t i = 0; i < numTbls; ++i)
                    c_.getAllColumnMetadataForTable(tblNodes_[i]->name.second, columns);

                /*for (int i = 0; i < columns.size(); ++i) {
                    columns[i].print();
                }*/
                
                // there should be more than 0 columns
                assert(columns.size() > 0);
                
                // replace the * with a ScalarExprList composed of column nodes
                v->n2->all = false;
                v->n2->n1 = new ScalarExprList(new ScalarExpr(new Column(columns[0].columnName)));
                ScalarExprList *cur = v->n2->n1;
                for (int i = 1; i < columns.size(); ++i) {
                    cur->n1 = new ScalarExprList(new ScalarExpr(new Column(columns[i].columnName)));
                    cur->n1->n2->n4->metadata = columns[i];
                    cur = cur->n1;
                }
            }
            else {
                v->n2->accept(*this); // Selection
            }
            
            // visiting these optional children will gather up more Column node pointers,
            // which get inserted into colNodes_
            if (v->n4) v->n4->accept(*this);
            if (v->n5) v->n5->accept(*this);
            if (v->n6) v->n6->accept(*this);
            if (v->n7) v->n7->accept(*this);
            
            // annotate each column node in colNodes_
            
        }
    }
    
    void NameWalker::visit(SqlStmt *v) {
        if (v->n1) v->n1->accept(*this); // DmlStmt
    }
    
    void NameWalker::visit(Table *v) {
        assert(v->name.first == ""); // @todo this should pass until we support NAME AS NAME
        tblNodes_.push_back(v);
    }
    
    void NameWalker::visit(TableList *v) {
        if (v->n1) v->n1->accept(*this); // TableList
        if (v->n2) v->n2->accept(*this); // Table
    }
    
} // Analysis_Namespace

