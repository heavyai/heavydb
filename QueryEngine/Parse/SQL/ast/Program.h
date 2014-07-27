#ifndef AST_PROGRAM_H
#define AST_PROGRAM_H

#include <cassert>
#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {

class Program : public ASTNode {
    
public:
    SQLList *sqlList;

    /**< Constructor */
    explicit Program(SQLList *n) {
    	assert(n);
    	sqlList = n;
    }
    
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

} // SQL_Namespace

#endif // AST_PROGRAM_H
