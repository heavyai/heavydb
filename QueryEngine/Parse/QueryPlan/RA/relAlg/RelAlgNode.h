#ifndef REL_ALG_NODE_H
#define REL_ALG_NODE_H

namespace RA_Namespace {
class RelAlgNode {
    
public:

    /**< Accepts the given void visitor by calling v.visit(this) */
    virtual void accept(class Visitor &v) = 0;
};
}

#endif // REL_ALG_NODE_H