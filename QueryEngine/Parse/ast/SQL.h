#ifndef SQL_NODE_H
#define SQL_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class SQL : public ASTNode {
    
public:
    Schema *sch;
    ManipulativeStatement *manSta;

    /**< Constructor */
    explicit SQL(Schema *n) : sch(n), manSta(NULL) {}
    SQL(ManipulativeStatement *n) : manSta(n), sch(NULL) {}

    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // SQL_NODE_H
