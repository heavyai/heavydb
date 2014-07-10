#ifndef AST_PROGRAM_H
#define AST_PROGRAM_H

#include <iostream>
#include "RelAlgNode.h"
#include "RelExprList.h"
#include "../visitor/Visitor.h"

using std::string;

class Program : public ASTNode {
    
public:
    RelExprList* rel;

    /**< Constructor */
    explicit Program(RelExprList *n) : rel(n) {}
    
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};




#endif // AST_PROGRAM_H
