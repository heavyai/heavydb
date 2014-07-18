#ifndef RELATION_NODE_H
#define RELATION_NODE_H

#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

namespace RA_Namespace {
class Relation : public RelAlgNode {
    
public:
    std::string name1;
    RA_Table* tbl;

    /**< Constructor */
    explicit Relation(const std::string &n1) : name1(n1), tbl(NULL) {}
    explicit Relation(RA_Table* n) : tbl(n), name1("") {}

    void accept(Visitor &v) {
        v.visit(this);
    }    
	};
}

#endif // RELATION_NODE_H