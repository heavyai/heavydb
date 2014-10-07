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
        
        AbstractPlan* translate(SQL_Namespace::ASTNode *parseTreeRoot);
        
        inline bool isError() { return error_ ; }
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
        virtual void visit(AlterStmt *v);
        virtual void visit(Column *v);
        virtual void visit(ColumnDef *v);
        virtual void visit(ColumnDefList *v);
        // virtual void visit(ColumnList *v);
        // virtual void visit(Comparison *v);
        virtual void visit(CreateStmt *v);
        virtual void visit(DdlStmt *v);
        virtual void visit(DeleteStmt *v);
        virtual void visit(DmlStmt *v);
        virtual void visit(DropStmt *v);
        virtual void visit(FromClause *v);
        virtual void visit(InsertColumnList *v);
        virtual void visit(InsertStmt *v);
        virtual void visit(Literal *v);
        virtual void visit(LiteralList *v);
        virtual void visit(MapdDataT *v);
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
        virtual void visit(SQL_Namespace::RenameStmt *v);
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
        
        // the plan
        AbstractPlan *plan_;
        
        // type of query; initialized to "unknown"
        QueryStmtType stmtType_ = UNKNOWN_STMT;
        
        // query (sql: select)
        std::vector<SQL_Namespace::Table*> queryTables_;
        std::vector<SQL_Namespace::Column*> queryColumns_;
        SQL_Namespace::Predicate *queryPredicate_ = nullptr;
        
        // insert (sql: insert into)
        SQL_Namespace::Table *insertTable_ = nullptr;
        std::vector<SQL_Namespace::InsertColumnList*> insertColumns_;
        std::vector<SQL_Namespace::Literal*> insertValues_;
        Partitioner_Namespace::InsertData insertData_;
        size_t byteCount_ = 0; // total number of bytes to be inserted
        
        // delete (sql: delete from)
        SQL_Namespace::Table *deleteTableName_ = nullptr;
        SQL_Namespace::Predicate *deletePredicate_ = nullptr;
        
        // update (sql: update)
        SQL_Namespace::Table *updateTableName_ = nullptr;
        std::vector<SQL_Namespace::Column*> updateColumns_;
        std::vector<SQL_Namespace::MapdDataT*> updateValues_;
        
        // create table (sql: create table)
        SQL_Namespace::Table *createTableName_ = nullptr;
        std::vector<SQL_Namespace::Column*> createColumns_;
        std::vector<SQL_Namespace::MapdDataT*> createTypes_;
        
        // drop table (sql: drop table)
        SQL_Namespace::Table *dropTableName_ = nullptr;
        
        // alter table (sql: alter table
        bool alterDrop_;
        mapd_data_t alterColumnType_;
        
        // rename table (sql: rename table)
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
         * @brief
         */
        CreatePlan* translateCreate();
        
        /**
         * @brief
         */
        DropPlan* translateDrop();
        
        /**
         * @brief
         */
        DeletePlan* translateDelete();
        
        /**
         * @brief
         */
        AlterPlan* translateAlter();
        
        /**
         * @brief
         */
        RenamePlan* translateRename();
        
    };
    
} // Plan_Namespace

#endif // QueryEngine_Plan_Planner_h