#ifndef REL_ALG_TABLE_H
#define REL_ALG_TABLE_H

#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

namespace RA_Namespace {
class RA_Table : public RelAlgNode {
    
public:
    std::string name1;
    std::string name2;

    /**< Constructor */
    explicit RA_Table(const std::string &n1) : name1(n1) {
    	name2 = "";
    }
    RA_Table(const std::string &n1, const std::string &n2) : name1(n1), name2(n2) {}

    /* copy fields of another table */
    explicit RA_Table(RA_Table* n) : name1(n->name1), name2(n->name2) {}

    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
	};
}

#endif // REL_ALG_TABLE_H