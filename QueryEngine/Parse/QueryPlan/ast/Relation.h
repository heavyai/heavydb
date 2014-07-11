#ifndef RELATION_NODE_H
#define RELATION_NODE_H

#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

class Relation : public RelAlgNode {
    
public:
    std::string name1;

    /**< Constructor */
    explicit Relation(const std::string &n1) : name1(n1) {}

    void accept(Visitor &v) {
        v.visit(this);
    }    
};

#endif // RELATION_NODE_H