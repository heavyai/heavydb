#ifndef REL_ALG_TABLE_H
#define REL_ALG_TABLE_H

#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

namespace RA_Namespace {
class Table : public RelAlgNode {
    
public:
    std::string name1;

    /**< Constructor */
    explicit Table(const std::string &n1) : name1(n1) {
 //   	name2 = "";
    }
    /* copy fields of another table */
    explicit Table(Table* n) : name1(n->name1) {}

    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
	};
}

#endif // REL_ALG_TABLE_H