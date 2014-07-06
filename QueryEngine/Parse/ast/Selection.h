#ifndef SELECTION_NODE_H
#define SELECTION_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class Selection : public ASTNode {
    
public:
    std::string selectAll; // if selection is '*'
    //ScalarExpCommalist *sEC;
    
    /**< Constructor */
    explicit Selection(const std::string &n1) : selectAll(n1) {}
 //   Selection(ScalarExpCommalist *n1) : sEC(n1) {}

    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // SELECTION_NODE_H
