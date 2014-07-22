/**
 * @file    ComparisonPredicate.h
 * @author  Steven Stewart <steve@map-d.com>
 * @author  Gil Walzer <gil@map-d.com>
 */
#ifndef COMPARISON_PREDICATE_NODE_H
#define COMPARISON_PREDICATE_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {

class  ComparisonPredicate : public ASTNode {
    
public:

    ScalarExp* se1 = NULL;
    ScalarExp* se2 = NULL;
    Subquery* s = NULL;
    std::string comparison = "";

    /* constructor */
    ComparisonPredicate(std::string &n, ScalarExp* n1, ScalarExp* n2) { comparison = n; se1 = n1; se2 = n2; }
    ComparisonPredicate(std::string &n, ScalarExp* n1, Subquery* n2) { comparison = n; se1 = n1; s = n2; }

	/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
	};
}

#endif // COMPARISON_PREDICATE_NODE_H
