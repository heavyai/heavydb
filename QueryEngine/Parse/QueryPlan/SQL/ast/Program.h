#ifndef AST_PROGRAM_H
#define AST_PROGRAM_H

#include <iostream>
#include "ASTNode.h"
#include "SQLList.h"
#include "../visitor/Visitor.h"

using std::string;

namespace SQL_Namespace {
	class  Program : public ASTNode {
    
public:
    SQLList *sqlList;

    /**< Constructor */
    explicit Program(SQLList *n) : sqlList(n) {}
    
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
	};
}




#endif // AST_PROGRAM_H
