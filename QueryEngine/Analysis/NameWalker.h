/**
 * @file	TypeChecker.h
 * @author	Steven Stewart <steve@map-d.com>
 */
#ifndef QUERYENGINE_ANALYSIS_NAMEWALKER
#define QUERYENGINE_ANALYSIS_NAMEWALKER

#include <iostream>
#include "../../Shared/types.h"
#include "../Parse/SQL/visitor/Visitor.h"
#include "../../DataMgr/Metadata/Catalog.h"

using namespace SQL_Namespace;
using namespace Metadata_Namespace;

namespace Analysis_Namespace {
    
    /**
     * @class 	NameWalker
     * @brief	This class is a visitor that resolves names of tables and columns to their ids.
     */
    class NameWalker : public Visitor {
        
    public:
        /// Constructor
        NameWalker(Catalog &c) : c_(c), errFlag_(false) {}
        
        /// Returns an error message if an error was encountered
        inline std::pair<bool, std::string> isError() { return std::pair<bool, std::string>(errFlag_, errMsg_); }

        /// @brief Visit an Column node
        virtual void visit(Column *v);
        
        /// @brief Visit an DmlStmt node
        virtual void visit(DmlStmt *v);

        /// @brief Visit an FromClause node
        virtual void visit(FromClause *v);
        
        /// @brief Visit an InsertColumnList node
        virtual void visit(InsertColumnList *v);
        
        /// @brief Visit an InsertStmt node
        virtual void visit(InsertStmt *v);

        /// @brief Visit an OptGroupby node
        virtual void visit(OptGroupby *v);

        /// @brief Visit an OptHaving node
        virtual void visit(OptHaving *v);
        
        /// @brief Visit an OptOrderby node
        virtual void visit(OptOrderby *v);
        
        /// @brief Visit an OptWhere node
        virtual void visit(OptWhere *v);

        /// @brief Visit an ScalarExpr node
        virtual void visit(ScalarExpr *v);
        
        /// @brief Visit an ScalarExprList node
        virtual void visit(ScalarExprList *v);
        
        /// @brief Visit an Selection node
        virtual void visit(Selection *v);
        
        /// @brief Visit an SelectStmt node
        virtual void visit(SelectStmt *v);
        
        /// @brief Visit an SqlStmt node
        virtual void visit(SqlStmt *v);
        
        /// @brief Visit an Table node
        virtual void visit(Table *v);
        
        /// @brief Visit an TableList node
        virtual void visit(TableList *v);
        
    private:
        Catalog &c_;			/// a reference to a Catalog, which holds table/column metadata
        std::string errMsg_;	/// holds an error message, if applicable; otherwise, it is ""
        bool errFlag_ = false;	/// indicates the existence of an error when true
        
        std::vector<Table*> tblNodes_;
        std::vector<Column*> colNodes_;
        std::vector<InsertColumnList*> insertColNodes_;
    };
    
} // Analysis_Namespace

#endif // QUERYENGINE_ANALYSIS_NAMEWALKER
