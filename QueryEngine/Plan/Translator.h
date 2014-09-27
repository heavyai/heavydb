/**
 * @file    Translator.h
 * @author  Steven Stewart <steve@map-d.com>
 */
#ifndef QueryEngine_Plan_Translator_h
#define QueryEngine_Plan_Translator_h

#include <vector>
#include "../Parse/SQL/visitor/Visitor.h"
#include "../Parse/SQL/ast/ASTNode.h"
#include "../Parse/RA/ast/RelAlgNode.h"
#include "../../DataMgr/Metadata/Catalog.h"

using namespace SQL_Namespace;
using Metadata_Namespace::Catalog;

namespace Plan_Namespace {
    
    /**
     * @brief Indicates the type of query represented by the SQL parse tree.
     */
    enum QueryStmtType {
        UNKNOWN_STMT, QUERY_STMT, INSERT_STMT, DELETE_STMT, UPDATE_STMT, CREATE_STMT
    };
    
    /**
     * @class Translator
     * @brief An SQL visitor class that translates an SQL AST into an RA AST (query plan).
     */
    class Translator : public SQL_Namespace::Visitor {

    public:
        /// Constructor
        Translator(Catalog &c) : c_(c) {}
        
        /// Destructor
        ~Translator() {}
        
        RA_Namespace::RelAlgNode* translate(SQL_Namespace::ASTNode *parseTreeRoot);
        
        inline std::pair<bool, std::string> checkError() {
            return catalogError_;
        }
        
        // virtual void visit(AggrExpr *v);
        // virtual void visit(AlterStmt *v);
        virtual void visit(Column *v);
        // virtual void visit(ColumnDef *v);
        // virtual void visit(ColumnDefList *v);
        // virtual void visit(ColumnList *v);
        // virtual void visit(Comparison *v);
        // virtual void visit(CreateStmt *v);
        // virtual void visit(DdlStmt *v);
        virtual void visit(DmlStmt *v);
        // virtual void visit(DropStmt *v);
        virtual void visit(FromClause *v);
        // virtual void visit(InsertColumnList *v);
        // virtual void visit(InsertStmt *v);
        // virtual void visit(Literal *v);
        // virtual void visit(LiteralList *v);
        // virtual void visit(MapdDataT *v);
        // virtual void visit(MathExpr *v);
        virtual void visit(OptAllDistinct *v);
        virtual void visit(OptGroupby *v);
        virtual void visit(OptHaving *v);
        virtual void visit(OptLimit *v);
        virtual void visit(OptOrderby *v);
        virtual void visit(OptWhere *v);
        // virtual void visit(OrderbyColumn *v);
        // virtual void visit(OrderbyColumnList *v);
        virtual void visit(SQL_Namespace::Predicate *v);
        // virtual void visit(RenameStmt *v);
        virtual void visit(ScalarExpr *v);
        virtual void visit(ScalarExprList *v);
        virtual void visit(SearchCondition *v);
        virtual void visit(SelectStmt *v);
        virtual void visit(Selection *v);
        virtual void visit(SqlStmt *v);
        virtual void visit(Table *v);
        virtual void visit(TableList *v);
        
    private:
        Catalog &c_; /// a reference to a Catalog, which holds table/column metadata
        
        // type of query; initialized to "unknown"
        QueryStmtType stmtType_ = UNKNOWN_STMT;
        
        // query data (sql: select)
        std::vector<SQL_Namespace::Table*> queryTables_;
        std::vector<SQL_Namespace::Column*> queryColumns_;
        SQL_Namespace::Predicate *queryPredicate_ = nullptr;
        
        // insert data (sql: insert into)
        SQL_Namespace::Table *insertTableName_ = nullptr;
        std::vector<SQL_Namespace::Column*> insertColumns_;
        std::vector<SQL_Namespace::MapdDataT*> insertValues_;
        
        // delete data (sql: delete from)
        SQL_Namespace::Table *deleteTableName_ = nullptr;
        SQL_Namespace::Predicate *deletePredicate_ = nullptr;
        
        // update data (sql: update)
        SQL_Namespace::Table *updateTableName_ = nullptr;
        std::vector<SQL_Namespace::Column*> updateColumns_;
        std::vector<SQL_Namespace::MapdDataT*> updateValues_;
        
        // create table data (sql: create table)
        SQL_Namespace::Table *createTableName_ = nullptr;
        std::vector<SQL_Namespace::Column*> createColumns_;
        std::vector<SQL_Namespace::MapdDataT*> createValues_;
        
        // collect table and column names (passed to Catalog for
        // optional annotation of nodes)
        std::vector<std::string> tableNames_;
        std::vector<std::pair<std::string, std::string>> columnNames_;
        
        // sets an error (used to indicate Catalog errors)
        std::pair<bool, std::string> catalogError_;
        
        /**
         * @brief Returns a query plan for a query (sql select statement)
         */
        RA_Namespace::RelAlgNode* translateQuery();
        
    };
    
} // Plan_Namespace

#endif // QueryEngine_Plan_Planner_h