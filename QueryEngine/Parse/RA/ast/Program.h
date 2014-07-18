#ifndef AST_PROGRAM_H
#define AST_PROGRAM_H

#include <iostream>
#include "RelAlgNode.h"
#include "RelExprList.h"
#include "../visitor/Visitor.h"

using std::string;

namespace RA_Namespace {
class RA_Program : public RelAlgNode {
    
public:
    RelExprList* rel;

    /**< Constructor */
    explicit RA_Program(RelExprList *n) : rel(n) {}
    
    /**< Accepts the given void visitor by calling v.visit(this) */
    virtual void accept(Visitor &v) {
        v.visit(this);
    }
    
	};
}
#endif // AST_PROGRAM_H