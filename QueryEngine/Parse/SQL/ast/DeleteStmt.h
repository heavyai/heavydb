#ifndef QueryEngine_Parse_ast_DeleteStmt_h
#define QueryEngine_Parse_ast_DeleteStmt_h

#include <cassert>
#include "Statement.h"

namespace SQL_Namespace {
    
    class DeleteStmt : public Statement {
        
    public:
        
        Table *n1 = nullptr;
        Predicate *n2 = nullptr;
        
        explicit DeleteStmt(Table *n1) {
            assert(n1);
            this->n1 = n1;
        }

        DeleteStmt(Table *n1, Predicate *n2) {
            assert(n1 && n2);
            this->n1 = n1;
            this->n2 = n2;
        }

        virtual void accept(Visitor &v) {
            v.visit(this);
        }
        
    };
    
} // SQL_Namespace

#endif // QueryEngine_Parse_ast_DeleteStmt
