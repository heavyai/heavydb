#ifndef OPT_ALL_DISTINCT_NODE_H
#define OPT_ALL_DISTINCT_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class OptAllDistinct : public ASTNode {
    
public:
    std::string ddlCmd; // Should be ALL, DISTINCT, or empty
    
    /**< Constructor */
    OptAllDistinct(const std::string &n1) : ddlCmd(n1) {}
    
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // OPT_ALL_DISTINCT_NODE_H
