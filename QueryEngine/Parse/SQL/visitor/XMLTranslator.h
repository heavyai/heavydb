/**
 * @file	XMLTranslator.h
 * @author	Steven Stewart <steve@map-d.com>
 */
#ifndef QUERYENGINE_SQL_VISITOR_XMLVISITOR_H
#define QUERYENGINE_SQL_VISITOR_XMLVISITOR_H

#include <iostream>
#include <vector>
#include "../../../Parse/SQL/visitor/Visitor.h"

namespace SQL_Namespace {
    
    class XMLTranslator : public SQL_Namespace::Visitor {
        
    public:
        
        /// Constructor
        XMLTranslator() {}
        
        /// Destructor
        ~XMLTranslator() {}
        
        virtual void visit(AggrExpr *v);
        virtual void visit(AlterStmt *v);
        virtual void visit(Column *v);
        virtual void visit(ColumnDef *v);
        virtual void visit(ColumnDefList *v);
        virtual void visit(ColumnList *v);
        virtual void visit(Comparison *v);
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
        virtual void visit(MathExpr *v);
        virtual void visit(OptAllDistinct *v);
        virtual void visit(OptGroupby *v);
        virtual void visit(OptHaving *v);
        virtual void visit(OptLimit *v);
        virtual void visit(OptOrderby *v);
        virtual void visit(OptWhere *v);
        virtual void visit(OrderbyColumn *v);
        virtual void visit(OrderbyColumnList *v);
        virtual void visit(Predicate *v);
        virtual void visit(RenameStmt *v);
        virtual void visit(ScalarExpr *v);
        virtual void visit(ScalarExprList *v);
        virtual void visit(SearchCondition *v);
        virtual void visit(SelectStmt *v);
        virtual void visit(Selection *v);
        virtual void visit(SqlStmt *v);
        virtual void visit(Table *v);
        virtual void visit(TableList *v);

    private:
        int tabCount_ = 0;
        
        inline void printTabs() {
            for (int i = 0; i < tabCount_; ++i)
                std::cout << "   ";
        }
    };
    
} // SQL_Namespace

#endif // QUERYENGINE_SQL_VISITOR_XMLVISITOR_H
