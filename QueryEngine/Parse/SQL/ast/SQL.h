#ifndef SQL_NODE_H
#define SQL_NODE_H

#include <cassert>
#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {

class SQL : public ASTNode {
    
public:
    Schema *sch = NULL;
    ManipulativeStatement *manSta = NULL;

    /**< Constructor */
    explicit SQL(Schema *n) {
        assert(n);
        this->sch = n;
    }
    
    SQL(ManipulativeStatement *n) {
        assert(n);
        this->manSta = n;
    }

    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

} // SQL_Namespace

#endif // SQL_NODE_H
