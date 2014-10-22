/**
 * @file    Translator.h
 * @author  Steven Stewart <steve@map-d.com>
 */
#ifndef QueryEngine_Plan_Translator_h
#define QueryEngine_Plan_Translator_h

#include <vector>
#include "Plan.h"
#include "../Parse/SQL/visitor/Visitor.h"
#include "../Parse/SQL/ast/ASTNode.h"
#include "../Parse/RA/ast/RelAlgNode.h"
#include "../../DataMgr/Metadata/Catalog.h"
#include "../../DataMgr/Partitioner/Partitioner.h"

using namespace SQL_Namespace;
using Metadata_Namespace::Catalog;

namespace Plan_Namespace {
    
    /**
     * @brief Indicates the type of query represented by the SQL parse tree.
     */
    enum QueryStmtType {
        UNKNOWN_STMT, QUERY_STMT, INSERT_STMT, DELETE_STMT, UPDATE_STMT, CREATE_STMT, DROP_STMT, ALTER_STMT, RENAME_STMT
    };
    
    /**
     * @class Translator
     * @brief An SQL visitor class that translates an SQL AST into an RA AST (query plan).
     */
    class Translator : public SQL_Namespace::Visitor {

    public:
        /// Constructor
        Translator(Catalog &c);
        
        /// Destructor
        ~Translator() {}
        
        /**
         * @brief The translate method takes an SQL AST parse tree and produces a query plan. Nifty.
         */
        AbstractPlan* translate(SQL_Namespace::ASTNode *parseTreeRoot);
        
        /**
         * @brief Returns whether or not the Translator is in an error state
         */
        inline bool isError() { return error_ ; }
        
        /**
         * @returns If Translator is in an error state, this returns the error message
         */
        inline std::string errorMsg() { return errorMsg_; }
        
        /**
         * @brief Returns an InsertData object (for sql insert statements)
         */
        inline Partitioner_Namespace::InsertData getInsertData() { return insertData_; }
        
        /**
         * @brief Returns the type of the parsed query
         */
        inline QueryStmtType getType() { return stmtType_; }
        
        // virtual void visit(AggrExpr *v);
        virtual void visit(AlterStmt *v);           /// visit method for AlterStmt
        virtual void visit(Column *v);              /// visit method for Column
        virtual void visit(ColumnDef *v);           /// visit method for ColumnDef
        virtual void visit(ColumnDefList *v);       /// visit method for ColumnDefList
        virtual void visit(ColumnList *v);          /// visit method for ColumnList
        virtual void visit(Comparison *v);          /// visit method for Comparison
        virtual void visit(CreateStmt *v);          /// visit method for CreateStmt
        virtual void visit(DdlStmt *v);             /// visit method for DdlStmt
        virtual void visit(DeleteStmt *v);          /// visit method for DeleteStmt
        virtual void visit(DmlStmt *v);             /// visit method for DmlStmt
        virtual void visit(DropStmt *v);            /// visit method for DropStmt
        virtual void visit(FromClause *v);          /// visit method for FromClause
        virtual void visit(InsertColumnList *v);    /// visit method for InsertColumnList
        virtual void visit(InsertStmt *v);          /// visit method for InsertStmt
        virtual void visit(Literal *v);             /// visit method for Literal
        virtual void visit(LiteralList *v);         /// visit method for LiteralList
        virtual void visit(MapdDataT *v);           /// visit method for MapdDataT
        virtual void visit(MathExpr *v);            /// visit method for MathExpr
        virtual void visit(OptAllDistinct *v);      /// visit method for OptAllDistinct
        virtual void visit(OptGroupby *v);          /// visit method for OptGroupby
        virtual void visit(OptHaving *v);           /// visit method for OptHaving
        virtual void visit(OptLimit *v);            /// visit method for OptLimit
        virtual void visit(OptOrderby *v);          /// visit method for OptOrderby
        virtual void visit(OptWhere *v);            /// visit method for OptWhere
        // virtual void visit(OrderbyColumn *v);
        // virtual void visit(OrderbyColumnList *v);
        virtual void visit(SQL_Namespace::Predicate *v);    /// visit method for Predicate
        virtual void visit(SQL_Namespace::RenameStmt *v);   /// visit method for RenameStmt
        virtual void visit(ScalarExpr *v);          /// visit method for ScalarExpr
        virtual void visit(ScalarExprList *v);      /// visit method for ScalarExprList
        virtual void visit(SearchCondition *v);     /// visit method for SearchCondition
        virtual void visit(SelectStmt *v);          /// visit method for SelectStmt
        virtual void visit(Selection *v);           /// visit method for Selection
        virtual void visit(SqlStmt *v);             /// visit method for SqlStmt
        virtual void visit(Table *v);               /// visit method for Table
        virtual void visit(TableList *v);           /// visit method for TableList
        
        // non-void visitor methods for translating predicates, math expressions,
        // and comparisons expressions
        RA_Namespace::Comparison* translateComparison(SQL_Namespace::Comparison*);
        RA_Namespace::MathExpr* translateMathExpr(SQL_Namespace::MathExpr*);
        RA_Namespace::Predicate* translatePredicate(SQL_Namespace::Predicate*);
        RA_Namespace::Attribute* translateColumn(SQL_Namespace::Column*);
        void* translateMapdDataT(SQL_Namespace::MapdDataT*);
        
    private:
        Catalog &c_; /// a reference to a Catalog, which holds table/column metadata
        
        /// a pointer to the plan object for the translated statement
        AbstractPlan *plan_ = nullptr;
        
        // type of query; initialized to "unknown"
        QueryStmtType stmtType_ = UNKNOWN_STMT;
        
        /// indicates state; i.e., whether or not the visitor is inside a predicate
        bool queryInsidePredicate_ = false;
        
        // member variables for a query (sql: select)
        std::vector<SQL_Namespace::Table*> queryTables_;
        std::vector<SQL_Namespace::Column*> queryColumns_;
        std::vector<std::pair<std::string, std::string>> queryPredicateColumnNames_;
        SQL_Namespace::Predicate *queryPredicate_ = nullptr;
        bool querySelectAllFields_;
        
        // member variables for an 'insert' (sql: insert into)
        SQL_Namespace::Table *insertTable_ = nullptr;
        std::vector<SQL_Namespace::InsertColumnList*> insertColumns_;
        std::vector<SQL_Namespace::Literal*> insertValues_;
        Partitioner_Namespace::InsertData insertData_;
        size_t byteCount_ = 0; // total number of bytes to be inserted
        
        // member variables for 'delete' (sql: delete from)
        SQL_Namespace::Table *deleteTableName_ = nullptr;
        SQL_Namespace::Predicate *deletePredicate_ = nullptr;
        
        // member variables for 'update' (sql: update)
        SQL_Namespace::Table *updateTableName_ = nullptr;
        std::vector<SQL_Namespace::Column*> updateColumns_;
        std::vector<SQL_Namespace::MapdDataT*> updateValues_;
        
        // member variables for 'create table' (sql: create table)
        SQL_Namespace::Table *createTableName_ = nullptr;
        std::vector<SQL_Namespace::Column*> createColumns_;
        std::vector<SQL_Namespace::MapdDataT*> createTypes_;
        
        // member variables for 'drop table' (sql: drop table)
        SQL_Namespace::Table *dropTableName_ = nullptr;
        
        // member variables for 'alter table' (sql: alter table
        bool alterDrop_;
        mapd_data_t alterColumnType_;
        
        // member variables for 'rename table' (sql: rename table)
        // uses tableNames_ vector for the source table name
        std::string renameTableNewName_;
        
        // collect table and column names (passed to Catalog for
        // optional annotation of nodes)
        std::vector<std::string> tableNames_;
        std::vector<std::pair<std::string, std::string>> columnNames_;

        // sets an error (used to indicate Catalog errors)
        bool error_ = false;
        std::string errorMsg_;

        /// Clears (resets or nullifies) the internal state of Translator
        void clearState();
        
        /**
         * @brief Returns an RA predicate subtree translated from an SQL predicate subtree
         */
        RA_Namespace::Predicate* translatePredicate(const SQL_Namespace::Predicate &sqlPred);
        
        /**
         * @brief Returns a query plan for a query (sql select statement)
         */
        QueryPlan* translateQuery();
        
        /**
         * @brief Annotates SQL AST nodes (Table, Column) with Catalog metadata
         */
        void annotateQuery();
        
        /**
         * @brief Sets the insertData_ object for an sql insert statement
         */
        InsertPlan* translateInsert();
        
        /**
         * @brief Type checks an insert statement.
         */
        void typeCheckInsert();
        
        /**
         * @brief Translates a create statement into a plan.
         */
        CreatePlan* translateCreate();
        
        /**
         * @brief Translates a drop statement into a plan.
         */
        DropPlan* translateDrop();
        
        /**
         * @brief Translates a delete statement into a plan.
         */
        DeletePlan* translateDelete();
        
        /**
         * @brief Translates an alter table statement into a plan.
         */
        AlterPlan* translateAlter();
        
        /**
         * @brief Translates a rename table statement into a plan.
         */
        RenamePlan* translateRename();
        
    };
    
} // Plan_Namespace

#endif // QueryEngine_Plan_Planner_h
