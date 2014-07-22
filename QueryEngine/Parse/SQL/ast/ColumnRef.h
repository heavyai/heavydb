/**
 * @file    ColumnRef.h
 * @author  Steven Stewart <steve@map-d.com>
 * @author  Gil Walzer <gil@map-d.com>
 */
#ifndef AST_COLUMN_REF_H
#define AST_COLUMN_REF_H

#include "ASTNode.h"
#include "AbstractScalarExpr.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {
	class  ColumnRef : public AbstractScalarExpr {
    
public:
	int args = -1;
    std::string name1 = "";
    std::string name2 = "";
    std::string name3 = "";
    
    int rule_Flag = -1;
    /* Rules:
    0 '.'
    1 AS */

    /**< Constructor */
    ColumnRef(const std::string &n1) { 
        assert(n1 != "");
        name1 = n1; 
        args = 1; 
    }
    ColumnRef(int rF, const std::string &n1, const std::string &n2) { 
        assert(((rF == 0) || (rF == 1)) && (n1 != "") && (n2 != ""));
        rule_Flag = rF; 
        name1 = n1; 
        name2 = n2; 
        args = 2; 
    }
    ColumnRef(const std::string &n1, const std::string &n2, const std::string &n3) { 
        assert((n1 != "") && (n2 != "") && (n3 != ""));
        name1 = n1; 
        name2 = n2; 
        name3 = n3; 
        args = 3; 
    }

    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
	};
}

#endif // AST_COLUMN_REF_H
