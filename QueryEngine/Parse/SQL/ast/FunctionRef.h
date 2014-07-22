/**
 * @file    FunctionRef.h
 * @author  Steven Stewart <steve@map-d.com>
 * @author  Gil Walzer <gil@map-d.com>
 */
#ifndef FUNCTION_REF_NODE_H
#define FUNCTION_REF_NODE_H

#include "ASTNode.h"
#include "AbstractScalarExpr.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {
	class  FunctionRef : public AbstractScalarExpr {
    
public:
    int rule_Flag = -1;
    
    /* Rules:
    0 (ALL scalar_exp)
    1 (scalar_exp) */

    Ammsc* am = NULL;
    ColumnRef* cr = NULL;
    ScalarExp* se1 = NULL;
    ScalarExp* se2 = NULL;
    ScalarExp* se3 = NULL;
    std::string func_name = "";
    ScalarExpCommalist* sec = NULL;

    /* constructor */
    FunctionRef(Ammsc* n) {
        assert(n);
        am = n; 
    }
    FunctionRef(Ammsc* n1, ColumnRef* n2) {
        assert(n1 && n2);
        am = n1; 
        cr = n2;
    }
    FunctionRef(int rF, Ammsc *n1, ScalarExp* n2) { 
        assert(((rF == 0) || (rF == 1)) && n1 && n2);
        rule_Flag = rF;
        am = n1; 
        se1 = n2; 
    }
    FunctionRef(const std::string &n, ScalarExpCommalist* n1) { 
        assert(n1 && (n != ""));
        sec = n1; 
        this->func_name = n; 
    }
    FunctionRef(const std::string &n, ScalarExp* n1, ScalarExp* n2) { 
        assert((n != "") && n1 && n2);
        se1 = n1; 
        se2 = n2; 
        func_name = n; 
    }
    FunctionRef(const std::string &n, ScalarExp* n1, ScalarExp* n2, ScalarExp* n3) { 
        assert((n != "") && n1 && n2 && n3);
        se1 = n1; 
        se2 = n2; 
        se3 = n3; 
        func_name = n; 
    }

    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
	};
}
#endif // FUNCTION_REF_NODE_H
