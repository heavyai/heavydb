#ifndef OPT_COLUMN_COMMALIST_NODE_H
#define OPT_COLUMN_COMMALIST_NODE_H

#include <cstddef>
#include <cassert>
#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {

class OptColumnCommalist : public ASTNode {
    
public:
    ColumnCommalist* cc = NULL;
    
    /**< Constructor */
    explicit OptColumnCommalist(ColumnCommalist *n) {
    	assert(n);
    	this->cc = n;
    }
        
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

} // SQL_Namespace

#endif // OPT_COLUMN_COMMALIST_NODE_H
