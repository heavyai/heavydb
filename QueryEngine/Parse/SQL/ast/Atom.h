/**
 * @file    Atom.h
 * @author  Steven Stewart <steve@map-d.com>
 * @author  Gil Walzer <gil@map-d.com>
 */
#ifndef SQL_AST_ATOM_H
#define SQL_AST_ATOM_H

#include <cassert>
#include "ASTNode.h"
#include "AbstractScalarExpr.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {

class Atom : public AbstractScalarExpr {
    
public:
    std::string user = "";
    Literal *lit = NULL;

    /**< Constructor */
    Atom(Literal *n) {
        assert(n);
        this->lit = n;
    }

    Atom(std::string n) {
        assert(n != "");
        user = n;
    }

    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

} // SQL_Namespace

#endif // SQL_AST_ATOM_H
